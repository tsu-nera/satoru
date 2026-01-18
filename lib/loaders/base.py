"""
データローダー共通基盤

全データソース（Mind Monitor, Selfloops等）で共通のユーティリティ関数
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any


def add_timestamp_column(
    df: pd.DataFrame,
    session_start: datetime,
    time_sec_column: str = 'Time_sec'
) -> pd.DataFrame:
    """
    相対時間(Time_sec)からTimeStamp列を生成

    全データソースで統一された形式のDataFrameを作成するため、
    TimeStamp列がない場合にsession_startとTime_secから生成する。

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（Time_sec列を持つ）
    session_start : datetime
        セッション開始時刻
    time_sec_column : str, default 'Time_sec'
        相対時間のカラム名

    Returns
    -------
    df : pd.DataFrame
        TimeStamp列を追加したデータフレーム

    Examples
    --------
    >>> df = pd.DataFrame({'Time_sec': [0.0, 1.0, 2.0]})
    >>> session_start = datetime(2026, 1, 10, 16, 8, 50)
    >>> df = add_timestamp_column(df, session_start)
    >>> df['TimeStamp'].iloc[0]
    Timestamp('2026-01-10 16:08:50')
    """
    if 'TimeStamp' in df.columns:
        # すでにTimeStamp列がある場合は何もしない
        return df

    if time_sec_column not in df.columns:
        raise ValueError(f"'{time_sec_column}' column not found in DataFrame")

    # session_startをpandas Timestampに変換
    session_start_ts = pd.Timestamp(session_start)

    # Time_secから絶対時刻を計算
    df['TimeStamp'] = session_start_ts + pd.to_timedelta(df[time_sec_column], unit='s')

    return df


def apply_warmup(
    df: pd.DataFrame,
    warmup_seconds: float,
    timestamp_column: str = 'TimeStamp',
    reset_time: bool = False
) -> pd.DataFrame:
    """
    ウォームアップ期間を除外

    データ収集開始直後の不安定な期間を除外する。
    全データソースで統一された処理を提供。

    Parameters
    ----------
    df : pd.DataFrame
        データフレーム（TimeStampまたはTime_sec列を持つ）
    warmup_seconds : float
        除外する期間（秒）、0の場合は除外しない
    timestamp_column : str, default 'TimeStamp'
        タイムスタンプカラム名
    reset_time : bool, default False
        Trueの場合、warmup後のTime_secを0にリセット
        Falseの場合、元のTime_secを保持

    Returns
    -------
    df : pd.DataFrame
        ウォームアップ期間を除外したデータフレーム

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'TimeStamp': pd.date_range('2026-01-10 16:08:50', periods=100, freq='1s'),
    ...     'Time_sec': range(100),
    ...     'value': range(100)
    ... })
    >>> df_warmup = apply_warmup(df, warmup_seconds=10.0)
    >>> len(df_warmup)
    90
    >>> df_warmup['Time_sec'].iloc[0]  # reset_time=Falseなので10秒
    10
    """
    if warmup_seconds <= 0:
        return df

    if len(df) == 0:
        return df

    # TimeStamp列ベースでフィルタリング
    if timestamp_column in df.columns:
        start_time = df[timestamp_column].min()
        warmup_cutoff = start_time + pd.Timedelta(seconds=warmup_seconds)
        df = df[df[timestamp_column] >= warmup_cutoff].copy()
    # Time_sec列ベースでフィルタリング（フォールバック）
    elif 'Time_sec' in df.columns:
        df = df[df['Time_sec'] >= warmup_seconds].copy()
    else:
        raise ValueError(f"'{timestamp_column}' or 'Time_sec' column not found in DataFrame")

    # Time_secをリセット
    if reset_time and 'Time_sec' in df.columns and len(df) > 0:
        offset = df['Time_sec'].iloc[0]
        df['Time_sec'] = df['Time_sec'] - offset

    return df


def normalize_dataframe(
    df: pd.DataFrame,
    session_start: Optional[datetime] = None,
    warmup_seconds: float = 0.0,
    reset_time: bool = False
) -> pd.DataFrame:
    """
    データフレームを標準形式に正規化

    全データソースで統一された形式のDataFrameを作成：
    - TimeStamp列の追加（存在しない場合）
    - Time_sec列の確認
    - Warmup期間の除外

    Parameters
    ----------
    df : pd.DataFrame
        元のデータフレーム
    session_start : datetime, optional
        セッション開始時刻（TimeStamp列がない場合に必要）
    warmup_seconds : float, default 0.0
        除外する期間（秒）
    reset_time : bool, default False
        warmup後のTime_secを0にリセットするか

    Returns
    -------
    df : pd.DataFrame
        正規化されたデータフレーム

    Raises
    ------
    ValueError
        Time_sec列がない、またはTimeStamp列がなくsession_startも指定されていない場合

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'Time_sec': range(100),
    ...     'value': range(100)
    ... })
    >>> session_start = datetime(2026, 1, 10, 16, 8, 50)
    >>> df_norm = normalize_dataframe(df, session_start, warmup_seconds=10.0)
    >>> 'TimeStamp' in df_norm.columns
    True
    """
    if 'Time_sec' not in df.columns:
        raise ValueError("'Time_sec' column is required")

    # TimeStamp列を追加
    if 'TimeStamp' not in df.columns:
        if session_start is None:
            raise ValueError("session_start is required when TimeStamp column does not exist")
        df = add_timestamp_column(df, session_start)

    # Warmup期間を除外
    if warmup_seconds > 0:
        df = apply_warmup(df, warmup_seconds, reset_time=reset_time)

    return df


def merge_multimodal_data(
    primary_df: pd.DataFrame,
    secondary_df: pd.DataFrame,
    tolerance_seconds: float = 0.5,
    direction: str = 'nearest',
    suffixes: tuple = ('_primary', '_secondary')
) -> pd.DataFrame:
    """
    異なるデータソースを時系列で統合

    TimeStampを基準に、異なるサンプリングレートのデータを結合する。
    Mind MonitorとSelfloopsのような異なるデバイスからのデータを統合する際に使用。

    Parameters
    ----------
    primary_df : pd.DataFrame
        主となるデータフレーム（TimeStamp列を持つ）
        通常はサンプリングレートの高いデータ（例: Mind Monitor, 256Hz）
    secondary_df : pd.DataFrame
        副次的なデータフレーム（TimeStamp列を持つ）
        通常はサンプリングレートの低いデータ（例: Selfloops, 1Hz）
    tolerance_seconds : float, default 0.5
        マッチング許容時間（秒）
        この時間以内の最も近いデータポイントを結合
    direction : str, default 'nearest'
        マッチング方向
        - 'nearest': 最も近いポイント
        - 'backward': 前方向のみ
        - 'forward': 後方向のみ
    suffixes : tuple, default ('_primary', '_secondary')
        重複列名に付与するサフィックス

    Returns
    -------
    merged_df : pd.DataFrame
        統合されたデータフレーム
        - primary_dfのすべての行を保持
        - secondary_dfの値は最も近いタイムスタンプでマッチング
        - マッチングできない場合はNaN

    Examples
    --------
    >>> mm_df = load_mind_monitor_csv('data/muse/file.csv')
    >>> sl_df = load_selfloops_csv('data/selfloops/file.csv')
    >>> merged = merge_multimodal_data(mm_df, sl_df, tolerance_seconds=0.5)
    >>> # Mind MonitorのすべてのサンプルにSelfloopsのHRデータが付与される
    >>> merged[['TimeStamp', 'Alpha_TP9', 'HR (bpm)']].head()

    Notes
    -----
    - primary_dfの時系列が保持される
    - secondary_dfはprimary_dfのタイムスタンプに合わせて補間される
    - 5秒程度の記録開始時間差は自動的に処理される
    """
    if 'TimeStamp' not in primary_df.columns or 'TimeStamp' not in secondary_df.columns:
        raise ValueError("Both DataFrames must have 'TimeStamp' column")

    # TimeStampでソート
    primary_df = primary_df.sort_values('TimeStamp').reset_index(drop=True)
    secondary_df = secondary_df.sort_values('TimeStamp').reset_index(drop=True)

    # pd.merge_asofで時系列マージ
    merged = pd.merge_asof(
        primary_df,
        secondary_df,
        on='TimeStamp',
        direction=direction,
        tolerance=pd.Timedelta(seconds=tolerance_seconds),
        suffixes=suffixes
    )

    return merged


def resample_to_common_timebase(
    dfs: list,
    freq: str = '1s',
    aggregation: str = 'mean'
) -> list:
    """
    複数のデータフレームを共通の時間軸にリサンプリング

    異なるサンプリングレートのデータを統一した時間間隔に集約する。
    統計的な分析や可視化に適した形式に変換。

    Parameters
    ----------
    dfs : list of pd.DataFrame
        リサンプリングするデータフレームのリスト
        各DataFrameはTimeStamp列を持つ必要がある
    freq : str, default '1s'
        リサンプリング頻度（pandas freq文字列）
        例: '1s'=1秒, '500ms'=0.5秒, '10s'=10秒
    aggregation : str, default 'mean'
        集約方法
        - 'mean': 平均値
        - 'median': 中央値
        - 'first': 最初の値
        - 'last': 最後の値

    Returns
    -------
    resampled_dfs : list of pd.DataFrame
        リサンプリングされたデータフレームのリスト
        すべて同じ時間軸（TimeStamp）を持つ

    Examples
    --------
    >>> mm_df = load_mind_monitor_csv('data/muse/file.csv')
    >>> sl_df = load_selfloops_csv('data/selfloops/file.csv')
    >>> mm_1s, sl_1s = resample_to_common_timebase([mm_df, sl_df], freq='1s')
    >>> # 両方とも1秒間隔のデータになる
    >>> mm_1s['TimeStamp'].diff().mean()
    Timedelta('0 days 00:00:01')

    Notes
    -----
    - 高サンプリングレートのデータ（Mind Monitor）は集約される
    - 低サンプリングレートのデータ（Selfloops）は補間される可能性がある
    """
    resampled_dfs = []

    for df in dfs:
        if 'TimeStamp' not in df.columns:
            raise ValueError("All DataFrames must have 'TimeStamp' column")

        # TimeStampをインデックスに設定
        df_indexed = df.set_index('TimeStamp')

        # リサンプリング
        if aggregation == 'mean':
            df_resampled = df_indexed.resample(freq).mean()
        elif aggregation == 'median':
            df_resampled = df_indexed.resample(freq).median()
        elif aggregation == 'first':
            df_resampled = df_indexed.resample(freq).first()
        elif aggregation == 'last':
            df_resampled = df_indexed.resample(freq).last()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # TimeStampを列に戻す
        df_resampled = df_resampled.reset_index()

        resampled_dfs.append(df_resampled)

    return resampled_dfs


def clean_rr_intervals(
    rr_intervals: np.ndarray,
    min_rr: float = 300,
    max_rr: float = 2000,
    max_diff_percent: float = 20
) -> np.ndarray:
    """
    R-R間隔の外れ値を検出・補正

    HRV Task Force (1996) の基準に基づく外れ値除外：
    - 生理学的に不可能な値を除外
    - 急激な変化を補間

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R間隔配列（ms）
    min_rr : float, default 300
        最小R-R間隔（ms）これより小さい値は外れ値
        300ms = 200 bpm
    max_rr : float, default 2000
        最大R-R間隔（ms）これより大きい値は外れ値
        2000ms = 30 bpm
    max_diff_percent : float, default 20
        前の値との最大変化率（%）
        これを超える変化は外れ値の可能性

    Returns
    -------
    rr_clean : np.ndarray
        クリーニング済みR-R間隔配列

    References
    ----------
    Task Force of the European Society of Cardiology and the North American
    Society of Pacing and Electrophysiology (1996). Heart rate variability:
    standards of measurement, physiological interpretation and clinical use.
    Circulation, 93(5), 1043-1065.
    """
    rr_clean = rr_intervals.copy()

    # 1. 絶対的な閾値による外れ値除外
    outliers = (rr_clean < min_rr) | (rr_clean > max_rr)

    # 2. 前の値との変化率による外れ値検出
    if len(rr_clean) > 1:
        diff_percent = np.abs(np.diff(rr_clean) / rr_clean[:-1]) * 100
        # 最初の要素はチェックできないので、2番目以降にTrueを追加
        sudden_change = np.concatenate([[False], diff_percent > max_diff_percent])
        outliers = outliers | sudden_change

    # 3. 外れ値を補間
    if np.any(outliers):
        # 線形補間で外れ値を置き換え
        valid_indices = np.where(~outliers)[0]
        if len(valid_indices) > 1:
            # 有効な値のみで補間
            rr_clean[outliers] = np.interp(
                np.where(outliers)[0],
                valid_indices,
                rr_clean[valid_indices]
            )
        elif len(valid_indices) == 1:
            # 有効な値が1つだけの場合、その値で埋める
            rr_clean[outliers] = rr_clean[valid_indices[0]]
        else:
            # すべて外れ値の場合、中央値で埋める
            median_rr = np.median(rr_intervals)
            if min_rr <= median_rr <= max_rr:
                rr_clean[:] = median_rr
            else:
                # 中央値も外れ値の場合、安全な値を使用
                rr_clean[:] = (min_rr + max_rr) / 2

    return rr_clean


def get_heart_rate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    HRVデータから心拍数データを抽出（Muse形式互換）

    lib/loaders/mind_monitor.py:get_heart_rate_data()と同じ形式で返す。
    Elite HRV、Selfloops等のHRVデータに対応。

    Parameters
    ----------
    df : pd.DataFrame
        HRVデータフレーム（TimeStamp列とHR (bpm)列を含む）

    Returns
    -------
    hr_dict : dict
        {
            'heart_rate': np.ndarray,   # 心拍数（bpm）
            'time': np.ndarray,         # 相対時間（秒）
            'timestamps': np.ndarray    # 絶対時刻（pandas Timestamp）
        }

    Examples
    --------
    >>> df = load_elite_hrv_txt('data/2026-01-17 17-47-13.txt')
    >>> hr_data = get_heart_rate_data(df)
    >>> print(hr_data['heart_rate'].mean())
    75.5
    """
    if 'HR (bpm)' not in df.columns:
        raise ValueError("'HR (bpm)' column not found")

    # 心拍数が0より大きいデータのみ抽出
    df_hr = df[df['HR (bpm)'] > 0].copy()

    return {
        'heart_rate': df_hr['HR (bpm)'].values,
        'time': df_hr['Time_sec'].values,
        'timestamps': df_hr['TimeStamp'].values
    }
