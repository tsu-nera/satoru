"""
ヒルベルト変換ベースのバンドパワー計算モジュール

MNE-Pythonを使用してバンドパスフィルタ→ヒルベルト包絡線→パワー計算を行う
共通パイプラインを提供する。

使用例:
- Frontal Midline Theta (4-8Hz)
- Frontal Alpha Asymmetry (8-13Hz)
- SMR (12-15Hz)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import mne


def calculate_hilbert_band_power(
    raw: mne.io.BaseRaw,
    band: Tuple[float, float],
    channels: Sequence[str],
    start_time: pd.Timestamp,
    resample_interval: str = '10s',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    outlier_percentile: float = 0.90,
) -> pd.DataFrame:
    """
    ヒルベルト変換でバンドパワー時系列を計算する。

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Rawオブジェクト（対象チャネルを含む）。
    band : Tuple[float, float]
        抽出する周波数帯域 (low_freq, high_freq) Hz。
    channels : Sequence[str]
        使用するチャネル名のリスト（例: ['RAW_AF7', 'RAW_AF8']）。
    start_time : pd.Timestamp
        セッション開始時刻（タイムインデックス生成用）。
    resample_interval : str
        リサンプル間隔（例: '10s'）。Noneの場合はリサンプルしない。
    smoothing_seconds : float
        ローリング平均による平滑化時定数（秒）。0以下で無効。
    rolling_window_seconds : float
        メディアンフィルタのウィンドウ幅（秒）。0以下で無効。
    outlier_percentile : float
        外れ値除去のパーセンタイル（0-1）。0.90なら上位10%を除去。

    Returns
    -------
    pd.DataFrame
        columns: チャネル名
        index: pd.DatetimeIndex
        values: パワー (dB, 10*log10(μV²))
    """
    channel_list = list(channels)

    # チャネル存在確認
    available = set(raw.ch_names)
    missing = [ch for ch in channel_list if ch not in available]
    if missing:
        raise ValueError(f'Specified channels not found: {missing}')

    # 対象チャネルのみ抽出
    raw_picked = raw.copy().pick_channels(channel_list)

    # バンドパスフィルタ
    raw_filtered = raw_picked.copy().filter(
        l_freq=band[0],
        h_freq=band[1],
        fir_design='firwin',
        phase='zero',
        verbose=False,
    )

    # ヒルベルト変換でエンベロープ抽出
    raw_envelope = raw_filtered.copy().apply_hilbert(envelope=True, verbose=False)
    env_data_uv = raw_envelope.get_data(units='uV')
    times = raw_envelope.times

    # タイムインデックス作成
    time_index = start_time + pd.to_timedelta(times, unit='s')

    # パワー計算: エンベロープの二乗をdBに変換
    power_uv2 = env_data_uv.T ** 2
    epsilon = 1e-12  # ゼロ除算防止
    power_db = 10 * np.log10(power_uv2 + epsilon)

    power_df = pd.DataFrame(
        power_db,
        index=time_index,
        columns=channel_list,
    )

    # 外れ値除去（各チャネル独立にパーセンタイルでクリップ）
    if outlier_percentile and 0 < outlier_percentile < 1:
        for ch in channel_list:
            upper_bound = power_df[ch].quantile(outlier_percentile)
            lower_bound = 10 * np.log10(epsilon)  # dBの下限
            power_df[ch] = power_df[ch].clip(lower=lower_bound, upper=upper_bound)

    # リサンプリング
    if resample_interval:
        power_df = power_df.resample(resample_interval).median()

    # 平滑化（ローリング平均）
    if smoothing_seconds and smoothing_seconds > 0:
        window = f'{max(int(smoothing_seconds), 1)}s'
        power_df = power_df.rolling(window=window, min_periods=1).mean()

    # メディアンフィルタ（ローリングウィンドウ）
    if rolling_window_seconds and rolling_window_seconds > 0:
        window = f'{max(int(rolling_window_seconds), 1)}s'
        power_df = power_df.rolling(window=window, min_periods=1).median()

    return power_df.dropna()


def calculate_channel_average_power(
    raw: mne.io.BaseRaw,
    band: Tuple[float, float],
    channels: Sequence[str],
    start_time: pd.Timestamp,
    resample_interval: str = '10s',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    outlier_percentile: float = 0.90,
) -> pd.Series:
    """
    ヒルベルト変換でバンドパワーを計算し、チャネル平均を返す。

    Parameters
    ----------
    raw : mne.io.BaseRaw
        MNE Rawオブジェクト。
    band : Tuple[float, float]
        抽出する周波数帯域 (low_freq, high_freq) Hz。
    channels : Sequence[str]
        使用するチャネル名のリスト。
    start_time : pd.Timestamp
        セッション開始時刻。
    resample_interval : str
        リサンプル間隔。
    smoothing_seconds : float
        平滑化時定数（秒）。
    rolling_window_seconds : float
        メディアンフィルタのウィンドウ幅（秒）。
    outlier_percentile : float
        外れ値除去のパーセンタイル。

    Returns
    -------
    pd.Series
        チャネル平均パワー時系列 (dB)
    """
    power_df = calculate_hilbert_band_power(
        raw=raw,
        band=band,
        channels=channels,
        start_time=start_time,
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        outlier_percentile=outlier_percentile,
    )

    return power_df.mean(axis=1)
