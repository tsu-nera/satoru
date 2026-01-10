"""
データローダー共通基盤

全データソース（Mind Monitor, Selfloops等）で共通のユーティリティ関数
"""

import pandas as pd
from datetime import datetime
from typing import Optional


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
