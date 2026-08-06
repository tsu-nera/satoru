"""
帯域パワー計算の共通基盤 - Frontal Midline Theta (Fmθ) と SMR の共通ロジック。

このモジュールは、frontal_theta.py と smr.py に共通する以下の処理を集約する:
raw準備 → band解決 → Hilbert変換によるチャネル平均パワー計算 → 統計DataFrame作成 → メタデータ作成
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import mne
import pandas as pd

from .core.hilbert_power import calculate_channel_average_power
from .core.statistics import calculate_half_comparison, create_metadata, create_statistics_dataframe
from .preprocessing import prepare_mne_raw


@dataclass
class BandPowerComputation:
    """帯域パワー計算結果を保持するデータクラス。"""

    time_series: pd.Series
    statistics: pd.DataFrame
    metadata: dict
    raw: mne.io.BaseRaw
    channels: list
    start_time: pd.Timestamp


def prepare_raw_for_channels(
    df: pd.DataFrame,
    channels: Sequence[str],
    sfreq: Optional[float] = None,
) -> mne.io.BaseRaw:
    """指定チャネルのみ抽出したRawオブジェクトを取得。"""
    mne_dict = prepare_mne_raw(df, sfreq=sfreq)
    if not mne_dict:
        raise ValueError('Failed to construct RAW data.')

    raw = mne_dict['raw'].copy()
    available = set(raw.ch_names)
    missing = [ch for ch in channels if ch not in available]
    if missing:
        raise ValueError(f'Specified channels not found: {missing}')

    return raw


def resolve_band(
    band: Optional[Tuple[float, float]],
    band_key: Optional[str],
    band_options: Dict[str, Tuple[float, float]],
    default_key: str,
    band_label_for_error: str,
) -> Tuple[Tuple[float, float], str]:
    """band指定を優先し、なければband_key（未指定時はdefault_key）をband_optionsから解決する。"""
    if band is not None:
        return band, 'custom'

    key = band_key or default_key
    if key not in band_options:
        raise ValueError(f'未定義の{band_label_for_error}帯域キーです: {key}')
    return band_options[key], key


def calculate_band_power(
    df: pd.DataFrame,
    *,
    channels: Optional[Sequence[str]] = None,
    band: Optional[Tuple[float, float]] = None,
    band_key: Optional[str] = None,
    band_options: Dict[str, Tuple[float, float]],
    default_band_key: str,
    band_label_for_error: str,
    empty_series_message: str,
    metadata_extra: Optional[dict] = None,
    resample_interval: str = '10s',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
    default_channels: Sequence[str] = ('RAW_AF7', 'RAW_AF8'),
) -> BandPowerComputation:
    """
    帯域パワー（dB単位）の時系列・統計・メタデータを計算する共通処理。

    Fmθ/SMRいずれの計算も、band解決 → Raw準備 → チャネル平均パワー計算 → 統計DataFrame作成
    → メタデータ作成、という共通手順を辿るため、その部分をここに集約する。
    """
    if channels is None:
        channels = default_channels

    channel_list = list(channels)

    band_tuple, band_label = resolve_band(
        band, band_key, band_options, default_band_key, band_label_for_error
    )

    # RAWデータ準備
    if raw is None:
        raw = prepare_raw_for_channels(df, channel_list)
    else:
        raw = raw.copy()

    # セッション開始時刻
    start_time = pd.to_datetime(df['TimeStamp'].min())

    # 処理パラメータ
    processing_params = {
        'resample_interval': resample_interval,
        'smoothing_seconds': smoothing_seconds,
        'rolling_window_seconds': rolling_window_seconds,
    }

    # ヒルベルト変換でバンドパワー計算（チャネル平均）
    series = calculate_channel_average_power(
        raw=raw,
        band=band_tuple,
        channels=channel_list,
        start_time=start_time,
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        outlier_percentile=0.90,
    )

    if series.empty:
        raise ValueError(empty_series_message)

    # 統計DataFrame作成
    stats_df = create_statistics_dataframe(
        series,
        name='Value',
        unit='dB',
        include_half_comparison=True,
    )
    # Unit列を削除（後方互換性のため）
    if 'Unit' in stats_df.columns:
        stats_df = stats_df.drop(columns=['Unit'])

    # メタデータ作成
    metadata = create_metadata(
        series=series,
        band=band_tuple,
        channels=channel_list,
        sfreq=float(raw.info['sfreq']),
        processing_params=processing_params,
        extra={'band_key': band_label, **(metadata_extra or {})},
    )

    # 後方互換性のためのキー追加
    half_stats = calculate_half_comparison(series)
    metadata['increase_db'] = half_stats['change_db']
    metadata['increase_rate_percent'] = half_stats['change_percent']

    return BandPowerComputation(
        time_series=series,
        statistics=stats_df,
        metadata=metadata,
        raw=raw,
        channels=channel_list,
        start_time=start_time,
    )
