"""
Posterior Alpha (後方2chアルファ) 解析モジュール

TP9/TP10チャネル（後方/側頭部）をMNE-Pythonの処理パイプラインで
バンドパス→ヒルベルト包絡へ変換し、後方Alphaパワーの時系列と統計指標を算出する。

パワーは dB 単位（10*log10(μV²)）で出力される。

frontal_theta.pyの実装パターンを参考に、後方2チャネルのAlpha計算を実現。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import mne
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from lib.sensors.eeg.preprocessing import prepare_mne_raw
from lib.sensors.eeg.core.hilbert_power import calculate_channel_average_power
from lib.sensors.eeg.core.statistics import create_statistics_dataframe, create_metadata


@dataclass
class PosteriorAlphaResult:
    """後方Alpha解析結果を保持するデータクラス。"""

    time_series: pd.Series
    statistics: pd.DataFrame
    metadata: dict


def _prepare_raw_for_channels(
    df: pd.DataFrame,
    channels: tuple[str, ...],
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


def calculate_posterior_alpha(
    df: pd.DataFrame,
    channels: Optional[Iterable[str]] = None,
    band: Tuple[float, float] = (8.0, 13.0),
    resample_interval: str = '10S',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
) -> PosteriorAlphaResult:
    """
    Posterior Alpha（後方2チャネルのアルファ波）の指標を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorの生データ（TimeStamp, RAW_TP9, RAW_TP10などを含む）。
    channels : iterable, optional
        後方Alpha解析に使用するRAWチャネル。デフォルトはTP9/TP10。
    band : tuple, optional
        抽出する周波数帯域 (Hz)。デフォルトは (8.0, 13.0)。
    resample_interval : str
        可視化用にリサンプルする間隔。デフォルトは10秒。
    smoothing_seconds : float
        ローリング平均による平滑化時定数（秒）。
    rolling_window_seconds : float
        ローリング平均のウィンドウサイズ（秒）。
    raw : mne.io.BaseRaw, optional
        既存のRawオブジェクト。Noneの場合は新規作成。

    Returns
    -------
    PosteriorAlphaResult
        時系列・統計情報・メタデータを含む解析結果。
        時系列パワーはdB単位（10*log10(μV²)）で出力される。
    """
    if channels is None:
        channels = ('RAW_TP9', 'RAW_TP10')

    channel_list = list(channels)

    # RAWデータ準備
    if raw is None:
        raw = _prepare_raw_for_channels(df, tuple(channel_list))
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
    alpha_series = calculate_channel_average_power(
        raw=raw,
        band=band,
        channels=channel_list,
        start_time=start_time,
        resample_interval=resample_interval,
        smoothing_seconds=smoothing_seconds,
        rolling_window_seconds=rolling_window_seconds,
        outlier_percentile=0.90,
    )

    if alpha_series.empty:
        raise ValueError('Posterior Alpha time series is empty.')

    # 統計DataFrame作成
    stats_df = create_statistics_dataframe(
        alpha_series,
        name='Posterior Alpha',
        unit='dB',
        include_half_comparison=True,
    )

    # メタデータ作成
    metadata = create_metadata(
        series=alpha_series,
        band=band,
        channels=channel_list,
        sfreq=raw.info['sfreq'],
        processing_params=processing_params,
    )

    return PosteriorAlphaResult(
        time_series=alpha_series,
        statistics=stats_df,
        metadata=metadata,
    )
