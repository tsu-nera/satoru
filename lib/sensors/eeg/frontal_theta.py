"""
Frontal Midline Theta (Fmθ) 解析モジュール

AF7/AF8チャネルをMNE-Pythonの処理パイプラインでバンドパス→ヒルベルト包絡へ変換し、
Fmθパワーの時系列と統計指標を算出する。

パワーは dB 単位（10*log10(μV²)）で出力される。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import mne

from .preprocessing import prepare_mne_raw
from .core.hilbert_power import calculate_hilbert_band_power, calculate_channel_average_power
from .core.statistics import calculate_half_comparison, create_statistics_dataframe, create_metadata

# 代表的なFmθ帯域プリセット（必要に応じて切り替え可能）
FMTHETA_BAND_OPTIONS: Dict[str, Tuple[float, float]] = {
    'narrow': (6.0, 7.0),
    'medium': (5.0, 7.0),
    'wide': (4.0, 8.0),
}


@dataclass
class FrontalThetaResult:
    """Fmθ解析結果を保持するデータクラス。"""

    time_series: pd.Series
    statistics: pd.DataFrame
    metadata: dict
    alpha_series: Optional[pd.Series] = None  # アルファ波時系列（オプション）


def _prepare_raw_for_channels(
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


def calculate_frontal_theta(
    df: pd.DataFrame,
    channels: Optional[Iterable[str]] = None,
    band: Optional[Tuple[float, float]] = None,
    band_key: Optional[str] = None,
    resample_interval: str = '10s',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
    include_alpha: bool = True,
    alpha_band: Tuple[float, float] = (8.0, 12.0),
) -> FrontalThetaResult:
    """
    Frontal Midline Theta (Fmθ) の指標を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorの生データ（TimeStamp, RAW_AF7, RAW_AF8などを含む）。
    channels : iterable, optional
        Fmθ解析に使用するRAWチャネル。デフォルトはAF7/AF8。
    band : tuple, optional
        抽出する周波数帯域 (Hz)。指定しない場合はband_keyかnarrowを使用。
    band_key : str, optional
        `FMTHETA_BAND_OPTIONS` に定義された帯域キー。例: 'narrow', 'medium', 'wide'
    resample_interval : str
        可視化用にリサンプルする間隔。デフォルトは10秒。
    smoothing_seconds : float
        ローリング平均による平滑化時定数（秒）。
    raw : mne.io.BaseRaw, optional
        既存のRawオブジェクト。Noneの場合は新規作成。
    include_alpha : bool
        Trueの場合、アルファ波の時系列も計算して返す。
    alpha_band : tuple
        アルファ波の周波数帯域 (Hz)。デフォルトは (8.0, 12.0)。

    Returns
    -------
    FrontalThetaResult
        時系列・統計情報・メタデータを含む解析結果。
        時系列パワーはdB単位（10*log10(μV²)）で出力される。
        include_alpha=Trueの場合、alpha_seriesにアルファ波時系列も含まれる。
    """
    if channels is None:
        channels = ('RAW_AF7', 'RAW_AF8')

    channel_list = list(channels)

    if band is not None:
        band_tuple = band
        band_label = 'custom'
    else:
        key = band_key or 'narrow'
        if key not in FMTHETA_BAND_OPTIONS:
            raise ValueError(f'未定義のFmθ帯域キーです: {key}')
        band_tuple = FMTHETA_BAND_OPTIONS[key]
        band_label = key

    # RAWデータ準備
    if raw is None:
        raw = _prepare_raw_for_channels(df, channel_list)
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
    theta_series = calculate_channel_average_power(
        raw=raw,
        band=band_tuple,
        channels=channel_list,
        start_time=start_time,
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        outlier_percentile=0.90,
    )

    if theta_series.empty:
        raise ValueError('Fmθ time series is empty.')

    # 統計DataFrame作成
    stats_df = create_statistics_dataframe(
        theta_series,
        name='Value',
        unit='dB',
        include_half_comparison=True,
    )
    # Unit列を削除（後方互換性のため）
    if 'Unit' in stats_df.columns:
        stats_df = stats_df.drop(columns=['Unit'])

    # メタデータ作成
    metadata = create_metadata(
        series=theta_series,
        band=band_tuple,
        channels=channel_list,
        sfreq=float(raw.info['sfreq']),
        processing_params=processing_params,
        extra={'band_key': band_label},
    )

    # 後方互換性のためのキー追加
    half_stats = calculate_half_comparison(theta_series)
    metadata['increase_db'] = half_stats['change_db']
    metadata['increase_rate_percent'] = half_stats['change_percent']

    # アルファ波の計算（オプション）
    alpha_series_final = None
    if include_alpha:
        alpha_series_final = calculate_channel_average_power(
            raw=raw,
            band=alpha_band,
            channels=channel_list,
            start_time=start_time,
            resample_interval=resample_interval,
            smoothing_seconds=smoothing_seconds,
            rolling_window_seconds=rolling_window_seconds,
            outlier_percentile=0.90,
        )
        metadata['alpha_band'] = alpha_band

    return FrontalThetaResult(
        time_series=theta_series,
        statistics=stats_df,
        metadata=metadata,
        alpha_series=alpha_series_final,
    )
