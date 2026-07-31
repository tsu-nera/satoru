"""
Frontal Midline Theta (Fmθ) 解析モジュール

AF7/AF8チャネルをMNE-Pythonの処理パイプラインでバンドパス→ヒルベルト包絡へ変換し、
Fmθパワーの時系列と統計指標を算出する。

パワーは dB 単位（10*log10(μV²)）で出力される。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import mne

from .core.hilbert_power import calculate_channel_average_power
from ._band_power_base import calculate_band_power

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
    computation = calculate_band_power(
        df,
        channels=channels,
        band=band,
        band_key=band_key,
        band_options=FMTHETA_BAND_OPTIONS,
        default_band_key='narrow',
        band_label_for_error='Fmθ',
        empty_series_message='Fmθ time series is empty.',
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        raw=raw,
    )

    # アルファ波の計算（オプション）
    alpha_series_final = None
    if include_alpha:
        alpha_series_final = calculate_channel_average_power(
            raw=computation.raw,
            band=alpha_band,
            channels=computation.channels,
            start_time=computation.start_time,
            resample_interval=resample_interval,
            smoothing_seconds=smoothing_seconds,
            rolling_window_seconds=rolling_window_seconds,
            outlier_percentile=0.90,
        )
        computation.metadata['alpha_band'] = alpha_band

    return FrontalThetaResult(
        time_series=computation.time_series,
        statistics=computation.statistics,
        metadata=computation.metadata,
        alpha_series=alpha_series_final,
    )
