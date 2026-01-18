"""
Elite HRVデータローダー

Elite HRVアプリから出力されたRR間隔データの読み込みと処理を行う
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .base import add_timestamp_column, apply_warmup, normalize_dataframe, clean_rr_intervals


def parse_elite_hrv_timestamp(filename: str) -> Optional[datetime]:
    """
    Elite HRVファイル名からタイムスタンプをパース

    Args:
        filename: ファイル名（例: "2026-01-17 17-47-13.txt"）

    Returns:
        パースされたdatetimeオブジェクト、失敗時はNone

    Examples:
        >>> parse_elite_hrv_timestamp("2026-01-17 17-47-13.txt")
        datetime(2026, 1, 17, 17, 47, 13)
    """
    try:
        # .txtを除去
        filename_no_ext = Path(filename).stem

        # "YYYY-MM-DD HH-MM-SS" 形式をパース
        # ファイル名の制約上、時刻部分はハイフン区切りの可能性もあるため両方対応
        parts = filename_no_ext.split()
        if len(parts) != 2:
            return None

        date_part = parts[0]  # "2026-01-17"
        time_part = parts[1]  # "17-47-13" または "17:47:13"

        # 日付パース
        date_parts = date_part.split('-')
        if len(date_parts) != 3:
            return None
        year = int(date_parts[0])
        month = int(date_parts[1])
        day = int(date_parts[2])

        # 時刻パース（ハイフンまたはコロン区切り）
        time_part = time_part.replace('-', ':')
        time_parts = time_part.split(':')
        if len(time_parts) != 3:
            return None
        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])

        return datetime(year, month, day, hour, minute, second)

    except (ValueError, IndexError):
        return None


def load_elite_hrv_txt(
    txt_path: str,
    warmup_seconds: float = 0.0,
    breathing_rate: Optional[float] = None
) -> pd.DataFrame:
    """
    Elite HRV TXTファイルを読み込む

    Elite HRV固有の仕様:
    - ファイル名: "YYYY-MM-DD HH-MM-SS.txt" 形式でタイムスタンプを含む
    - 各行: 1つのRR間隔（ms）

    Parameters
    ----------
    txt_path : str
        TXTファイルのパス
    warmup_seconds : float, default 0.0
        記録開始からの除外期間（秒）
        測定開始直後の不安定なデータを除外
    breathing_rate : float, optional
        呼吸レート（breaths/min）
        指定された場合、DataFrameのattrsに保存

    Returns
    -------
    df : pd.DataFrame
        読み込んだデータフレーム（標準形式）
        - Time_sec: 相対時間（秒）
        - R-R (ms): R-R間隔
        - HR (bpm): 心拍数
        - TimeStamp: 絶対時刻（pandas Timestamp）

        DataFrameのattrsに以下を保存:
        - session_start: セッション開始時刻（datetime）
        - breathing_rate: 呼吸レート（breaths/min）、指定された場合のみ

    Examples
    --------
    >>> df = load_elite_hrv_txt('data/2026-01-17 17-47-13.txt', breathing_rate=6.5)
    >>> print(df.attrs['session_start'])
    2026-01-17 17:47:13
    >>> print(df.attrs['breathing_rate'])
    6.5
    >>> 'TimeStamp' in df.columns
    True
    """
    # ファイル名からタイムスタンプを取得
    filename = Path(txt_path).name
    session_start = parse_elite_hrv_timestamp(filename)

    if session_start is None:
        # タイムスタンプがパースできない場合、現在時刻を使用
        session_start = datetime.now()
        print(f"⚠️  ファイル名からタイムスタンプを取得できませんでした: {filename}")
        print(f"    現在時刻を使用します: {session_start}")

    # RR間隔データを読み込み
    rr_intervals = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rr_intervals.append(float(line))
                except ValueError:
                    # 数値でない行はスキップ
                    continue

    rr_intervals = np.array(rr_intervals)

    if len(rr_intervals) == 0:
        raise ValueError(f"RR間隔データが見つかりませんでした: {txt_path}")

    # RR間隔から累積時間を計算
    rr_sec = rr_intervals / 1000.0  # ミリ秒を秒に変換
    cumulative_time = np.cumsum(rr_sec)
    time_sec = np.insert(cumulative_time, 0, 0)[:-1]  # 時刻を調整

    # 心拍数を計算
    hr = 60000.0 / rr_intervals  # bpm

    # DataFrameを作成
    df = pd.DataFrame({
        'Time_sec': time_sec,
        'R-R (ms)': rr_intervals,
        'HR (bpm)': hr
    })

    # セッション開始時刻を保存
    df.attrs['session_start'] = session_start

    # 呼吸レートを保存（指定された場合）
    if breathing_rate is not None:
        df.attrs['breathing_rate'] = breathing_rate

    # 標準形式に正規化（TimeStamp列追加、warmup処理）
    df = normalize_dataframe(
        df,
        session_start=session_start,
        warmup_seconds=warmup_seconds,
        reset_time=True  # warmup後にTime_secを0にリセット
    )

    # attrsを再設定（copyで失われる可能性があるため）
    df.attrs['session_start'] = session_start
    if breathing_rate is not None:
        df.attrs['breathing_rate'] = breathing_rate

    return df


def get_hrv_data(
    df: pd.DataFrame,
    clean_artifacts: bool = True
) -> Dict[str, Any]:
    """
    HRVデータを取得（NeuroKit2互換形式）

    Parameters
    ----------
    df : pd.DataFrame
        Elite HRVデータフレーム（load_elite_hrv_txt()の戻り値）
    clean_artifacts : bool, default True
        True: 外れ値・アーティファクトを除外（推奨）
        False: 生データをそのまま使用

    Returns
    -------
    hrv_dict : dict
        {
            'rr_intervals': np.ndarray,     # R-R間隔（ms）- 生データ
            'rr_intervals_clean': np.ndarray,  # R-R間隔（ms）- クリーニング済み
            'hr': np.ndarray,               # 心拍数（bpm）
            'time': np.ndarray,             # 相対時間（秒）
            'session_start': datetime,      # セッション開始時刻
            'breathing_rate': float,        # 呼吸レート（指定された場合のみ）
            'sampling_rate': int            # 仮想サンプリングレート（Hz）
        }

    Examples
    --------
    >>> df = load_elite_hrv_txt('data/2026-01-17 17-47-13.txt', breathing_rate=6.5)
    >>> hrv_data = get_hrv_data(df)
    >>> print(hrv_data['hr'].mean())
    75.5
    >>> print(hrv_data['breathing_rate'])
    6.5
    """
    rr_intervals = df['R-R (ms)'].values

    if clean_artifacts:
        # R-R間隔の外れ値を検出・補正
        rr_intervals_clean = clean_rr_intervals(
            rr_intervals,
            min_rr=300,   # 最小R-R間隔 (ms) → 200 bpm上限
            max_rr=2000,  # 最大R-R間隔 (ms) → 30 bpm下限
            max_diff_percent=20  # 前の値との最大変化率 (%)
        )
        # クリーニング済みR-R間隔から心拍数を計算
        hr = 60000.0 / rr_intervals_clean
    else:
        # 生データをそのまま使用
        rr_intervals_clean = rr_intervals
        hr = 60000.0 / rr_intervals

    result = {
        'rr_intervals': rr_intervals,           # 生データ
        'rr_intervals_clean': rr_intervals_clean,  # クリーニング済み
        'hr': hr,
        'time': df['Time_sec'].values,
        'session_start': df.attrs.get('session_start'),
        'sampling_rate': 1000  # R-R間隔はms単位なので1000Hz相当
    }

    # 呼吸レートがある場合は追加
    if 'breathing_rate' in df.attrs:
        result['breathing_rate'] = df.attrs['breathing_rate']

    return result
