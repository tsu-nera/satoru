"""
Frontal Alpha Asymmetry (FAA) 解析モジュール

AF7 (左前頭部) とAF8 (右前頭部) のアルファ波パワーを比較し、
左右の非対称性を定量化する。FAA = 10*log10(右) - 10*log10(左) で算出（Bels差分）。

正のFAA: 左半球優位 (接近動機、ポジティブ感情)
負のFAA: 右半球優位 (回避動機、ネガティブ感情)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import mne

from .preprocessing import prepare_mne_raw


@dataclass
class FrontalAsymmetryResult:
    """FAA解析結果を保持するデータクラス。"""

    time_series: pd.Series  # FAA時系列 (Bels差分: 10*log10(右) - 10*log10(左))
    left_power: pd.Series  # 左前頭部アルファパワー (Bels)
    right_power: pd.Series  # 右前頭部アルファパワー (Bels)
    statistics: pd.DataFrame
    metadata: dict


def calculate_frontal_asymmetry(
    df: pd.DataFrame,
    left_channel: str = 'RAW_AF7',
    right_channel: str = 'RAW_AF8',
    alpha_band: Tuple[float, float] = (8.0, 13.0),
    resample_interval: str = '2S',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
) -> FrontalAsymmetryResult:
    """
    Frontal Alpha Asymmetry (FAA) を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorの生データ (TimeStamp, RAW_AF7, RAW_AF8を含む)。
    left_channel : str
        左前頭部チャネル (デフォルト: RAW_AF7)。
    right_channel : str
        右前頭部チャネル (デフォルト: RAW_AF8)。
    alpha_band : tuple
        アルファ波帯域 (Hz)。デフォルトは (8.0, 13.0)。
    resample_interval : str
        リサンプル間隔。
    smoothing_seconds : float
        平滑化時定数（秒）。
    rolling_window_seconds : float
        ローリングウィンドウ（秒）。
    raw : mne.io.BaseRaw, optional
        既存のRawオブジェクト。Noneの場合は新規作成。

    Returns
    -------
    FrontalAsymmetryResult
        FAA時系列（Bels差分）、左右パワー（Bels）、統計情報を含む解析結果。
    """
    channels = [left_channel, right_channel]

    # RAWデータ準備
    if raw is None:
        mne_dict = prepare_mne_raw(df)
        if not mne_dict:
            raise ValueError('Failed to construct RAW data.')
        raw = mne_dict['raw']

    raw = raw.copy()
    available = set(raw.ch_names)
    missing = [ch for ch in channels if ch not in available]
    if missing:
        raise ValueError(f'Specified channels not found: {missing}')

    raw.pick_channels(channels)

    # アルファ帯域フィルタリング
    raw_filtered = raw.copy().filter(
        l_freq=alpha_band[0],
        h_freq=alpha_band[1],
        fir_design='firwin',
        phase='zero',
        verbose=False,
    )

    # ヒルベルト変換でエンベロープ抽出
    raw_envelope = raw_filtered.copy().apply_hilbert(envelope=True, verbose=False)
    env_data_uv = raw_envelope.get_data(units='uV')
    times = raw_envelope.times

    # タイムスタンプ作成
    start_time = pd.to_datetime(df['TimeStamp'].min())
    time_index = start_time + pd.to_timedelta(times, unit='s')

    # パワー計算: エンベロープの二乗をBelsに変換
    power_uv2 = env_data_uv.T ** 2
    epsilon = 1e-12  # ゼロ除算防止
    power_bels = 10 * np.log10(power_uv2 + epsilon)

    power_df = pd.DataFrame(
        power_bels,
        index=time_index,
        columns=channels,
    )

    # 外れ値除去（各チャネル独立に90パーセンタイルでクリップ）
    # 256Hzの高サンプリングレートでは極端な値がより顕著に現れるため必須
    # Z-scoreではなくパーセンタイルベースの方が、装着直後の集中的なスパイクに効果的
    # 90パーセンタイル: 上位10%の極端な値を除去（装着直後の不安定な信号を積極的に除外）
    for ch in channels:
        upper_bound = power_df[ch].quantile(0.90)
        # Belsの場合、下限はlog10(epsilon)の値
        lower_bound = 10 * np.log10(epsilon)
        power_df[ch] = power_df[ch].clip(lower=lower_bound, upper=upper_bound)

    # リサンプリング
    if resample_interval:
        power_df = power_df.resample(resample_interval).median()

    # 平滑化
    if smoothing_seconds and smoothing_seconds > 0:
        window = f'{max(int(smoothing_seconds), 1)}S'
        power_df = power_df.rolling(window=window, min_periods=1).mean()

    # ローリングウィンドウ
    if rolling_window_seconds and rolling_window_seconds > 0:
        window = f'{max(int(rolling_window_seconds), 1)}S'
        power_df = power_df.rolling(window=window, min_periods=1).median()

    # 左右パワーを抽出（既にBels単位）
    left_power = power_df[left_channel].dropna()
    right_power = power_df[right_channel].dropna()

    if left_power.empty or right_power.empty:
        raise ValueError('Failed to calculate left and right alpha power.')

    # FAA計算: Bels差分 = 10*log10(右) - 10*log10(左)
    # 既にBelsに変換済みなので、単純に差分を取る
    faa_series = right_power - left_power

    faa_series = faa_series.dropna()
    if faa_series.empty:
        raise ValueError('FAA time series is empty.')

    # 統計計算
    faa_mean = faa_series.mean()
    faa_median = faa_series.median()
    faa_std = faa_series.std()

    # 前半・後半比較
    midpoint = faa_series.index[0] + (faa_series.index[-1] - faa_series.index[0]) / 2
    first_half = faa_series[faa_series.index <= midpoint]
    second_half = faa_series[faa_series.index > midpoint]

    first_mean = first_half.mean() if not first_half.empty else np.nan
    second_mean = second_half.mean() if not second_half.empty else np.nan

    # FAA解釈（Bels差分）
    # 0.2 Bels ≈ ln(1.05) ≈ 5%の差に相当
    if faa_mean > 0.2:
        interpretation = 'Left hemisphere dominant (Approach motivation/Positive)'
    elif faa_mean < -0.2:
        interpretation = 'Right hemisphere dominant (Avoidance motivation/Negative)'
    else:
        interpretation = 'Balanced'

    stats_df = pd.DataFrame(
        [
            {'Metric': 'Mean FAA', 'Value': faa_mean, 'Unit': 'Bels'},
            {'Metric': 'Median', 'Value': faa_median, 'Unit': 'Bels'},
            {'Metric': 'Std Dev', 'Value': faa_std, 'Unit': 'Bels'},
            {'Metric': 'First Half Mean', 'Value': first_mean, 'Unit': 'Bels'},
            {'Metric': 'Second Half Mean', 'Value': second_mean, 'Unit': 'Bels'},
            {'Metric': 'Interpretation', 'Value': interpretation, 'Unit': ''},
        ]
    )

    metadata = {
        'left_channel': left_channel,
        'right_channel': right_channel,
        'alpha_band': alpha_band,
        'sfreq': float(raw.info['sfreq']),
        'first_half_mean': first_mean,
        'second_half_mean': second_mean,
        'interpretation': interpretation,
        'unit': 'Bels',
        'method': 'mne_hilbert_bels',
        'processing': {
            'resample_interval': resample_interval,
            'smoothing_seconds': smoothing_seconds,
            'rolling_window_seconds': rolling_window_seconds,
        },
    }

    return FrontalAsymmetryResult(
        time_series=faa_series,
        left_power=left_power,
        right_power=right_power,
        statistics=stats_df,
        metadata=metadata,
    )
