"""
Selfloops HRVデータローダー

Selfloopsアプリから出力されたHRVデータの読み込みと処理を行う
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from .base import add_timestamp_column, apply_warmup, normalize_dataframe, clean_rr_intervals


def parse_selfloops_timestamp(timestamp_line: str) -> Optional[datetime]:
    """
    Selfloopsファイルの1行目からタイムスタンプをパース

    Args:
        timestamp_line: タイムスタンプ行（例: "10 1月 2026 16:08:50"）

    Returns:
        パースされたdatetimeオブジェクト、失敗時はNone
    """
    # 日本語の月名マッピング
    month_map = {
        '1月': 1, '2月': 2, '3月': 3, '4月': 4,
        '5月': 5, '6月': 6, '7月': 7, '8月': 8,
        '9月': 9, '10月': 10, '11月': 11, '12月': 12
    }

    try:
        parts = timestamp_line.strip().split()
        if len(parts) < 4:
            return None

        day = int(parts[0])
        month = month_map.get(parts[1])
        year = int(parts[2])
        time_parts = parts[3].split(':')

        if month is None or len(time_parts) != 3:
            return None

        hour = int(time_parts[0])
        minute = int(time_parts[1])
        second = int(time_parts[2])

        return datetime(year, month, day, hour, minute, second)

    except (ValueError, IndexError):
        return None


def generate_selfloops_filename(timestamp: datetime) -> str:
    """
    タイムスタンプからSelfloopsファイル名を生成

    Args:
        timestamp: タイムスタンプ

    Returns:
        ファイル名（例: "selfloops_2026-01-10--16-08-50.csv"）
    """
    return timestamp.strftime("selfloops_%Y-%m-%d--%H-%M-%S.csv")


def rename_selfloops_file(file_path: str) -> str:
    """
    Selfloopsファイルを1行目のタイムスタンプから適切な名前にリネーム

    1行目に記録されているタイムスタンプ（例: "10 1月 2026 16:08:50"）を
    パースして、Muse形式に準じたファイル名（selfloops_YYYY-MM-DD--HH-MM-SS.csv）
    にリネームする。

    Args:
        file_path: Selfloopsファイルのパス

    Returns:
        リネーム後のファイルパス（リネーム不要/失敗時は元のパス）

    Examples:
        >>> rename_selfloops_file("data/SelfLoops HRV data.csv")
        "data/selfloops_2026-01-10--16-08-50.csv"
    """
    path = Path(file_path)

    # Selfloopsファイルかチェック（SelfLoopsという文字列を含む）
    if 'SelfLoops' not in path.name and 'selfloops' not in path.name:
        return file_path

    # すでに適切な名前の場合はスキップ
    if path.name.startswith('selfloops_') and '--' in path.name:
        return file_path

    print(f"\nSelfloopsファイルをリネーム中: {path.name}")

    try:
        # 1行目を読む
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()

        # タイムスタンプをパース
        timestamp = parse_selfloops_timestamp(first_line)
        if timestamp is None:
            print("⚠️  タイムスタンプのパースに失敗しました。ファイル名はそのままです。")
            return file_path

        # 新しいファイル名を生成
        new_name = generate_selfloops_filename(timestamp)
        new_path = path.parent / new_name

        # リネーム
        path.rename(new_path)
        print(f"✅ リネーム完了: {new_name}")
        return str(new_path)

    except Exception as e:
        print(f"⚠️  リネームエラー: {e}。ファイル名はそのままです。")
        return file_path


def load_selfloops_csv(csv_path: str, warmup_seconds: float = 0.0) -> pd.DataFrame:
    """
    Selfloops HRV CSVファイルを読み込む

    Selfloops固有の仕様:
    - 1行目: タイムスタンプ（例: "10 1月 2026 16:08:50"）
    - 2行目: ヘッダー（Time (ms),HR (bpm),R-R (ms)）
    - 3行目以降: データ

    Parameters
    ----------
    csv_path : str
        CSVファイルのパス
    warmup_seconds : float, default 0.0
        記録開始からの除外期間（秒）
        HRV測定開始直後の不安定なデータを除外

    Returns
    -------
    df : pd.DataFrame
        読み込んだデータフレーム（標準形式）
        - Time (ms): 累積時間（ミリ秒）
        - HR (bpm): 心拍数
        - R-R (ms): R-R間隔
        - Time_sec: 相対時間（秒）
        - TimeStamp: 絶対時刻（pandas Timestamp）

        DataFrameのattrsに以下を保存:
        - session_start: セッション開始時刻（datetime）

    Examples
    --------
    >>> df = load_selfloops_csv('data/selfloops_2026-01-10--16-08-50.csv')
    >>> print(df.attrs['session_start'])
    2026-01-10 16:08:50
    >>> 'TimeStamp' in df.columns
    True
    """
    # 1行目のタイムスタンプを読む
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline()

    session_start = parse_selfloops_timestamp(first_line)

    # 2行目以降をDataFrameとして読み込み（1行目スキップ、2行目がヘッダー）
    df = pd.read_csv(csv_path, skiprows=1)

    # 相対時間（秒）を計算
    df['Time_sec'] = df['Time (ms)'] / 1000.0

    # セッション開始時刻を保存
    df.attrs['session_start'] = session_start

    # 標準形式に正規化（TimeStamp列追加、warmup処理）
    df = normalize_dataframe(
        df,
        session_start=session_start,
        warmup_seconds=warmup_seconds,
        reset_time=True  # Selfloopsはwarmup後にTime_secを0にリセット
    )

    # attrsを再設定（copyで失われる可能性があるため）
    df.attrs['session_start'] = session_start

    return df


def get_hrv_data(df: pd.DataFrame,
                 use_device_hr: bool = False,
                 clean_artifacts: bool = True) -> Dict[str, Any]:
    """
    HRVデータを取得（NeuroKit2互換形式）

    Parameters
    ----------
    df : pd.DataFrame
        Selfloopsデータフレーム（load_selfloops_csv()の戻り値）
    use_device_hr : bool, default False
        True: SelfLoopsのHR列を使用（デバイス固有、後方互換性のみ）
        False: R-R間隔から計算（推奨、汎用的、他のHRVアプリにも対応可能）
    clean_artifacts : bool, default True
        True: NeuroKit2で外れ値・アーティファクトを除外（推奨）
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
            'sampling_rate': int            # 仮想サンプリングレート（Hz）
        }

    Examples
    --------
    >>> # 推奨: R-R間隔から計算、外れ値除外
    >>> df = load_selfloops_csv('data/selfloops_2026-01-10--16-08-50.csv')
    >>> hrv_data = get_hrv_data(df)
    >>> print(hrv_data['hr'].mean())  # 外れ値除外済みの心拍数
    75.5

    >>> # 生データのまま（外れ値含む）
    >>> hrv_data = get_hrv_data(df, clean_artifacts=False)
    >>> print(hrv_data['hr'].max())  # 外れ値を含む可能性
    214.3

    >>> # デバイスのHR列を使用（SelfLoops固有、非推奨）
    >>> hrv_data = get_hrv_data(df, use_device_hr=True)

    Notes
    -----
    - use_device_hr=False（デフォルト）を推奨
      理由: Elite HRV、Polar等の標準的なHRVアプリと互換性があり、
            将来的に他のデバイスにも同じコードで対応可能
    - clean_artifacts=True（デフォルト）を推奨
      理由: NeuroKit2の外れ値処理は学術研究で検証済みの標準手法
    """
    rr_intervals = df['R-R (ms)'].values

    if use_device_hr:
        # 後方互換性のため残す（非推奨）
        # SelfLoops固有のHR列を使用
        hr = df['HR (bpm)'].values
        rr_intervals_clean = rr_intervals  # クリーニングなし
    else:
        # 推奨: R-R間隔から計算（汎用的）
        if clean_artifacts:
            # R-R間隔の外れ値を検出・補正
            # HRV Task Force (1996) の基準に基づく
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

    return {
        'rr_intervals': rr_intervals,           # 生データ
        'rr_intervals_clean': rr_intervals_clean,  # クリーニング済み
        'hr': hr,
        'time': df['Time_sec'].values,
        'session_start': df.attrs.get('session_start'),
        'sampling_rate': 1000  # R-R間隔はms単位なので1000Hz相当
    }
