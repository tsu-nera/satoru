"""
Frontal Alpha Asymmetry (FAA) 解析モジュール

AF7 (左前頭部) とAF8 (右前頭部) のアルファ波パワーを比較し、
左右の非対称性を定量化する。FAA = 10*log10(右) - 10*log10(左) で算出（dB差分）。

正のFAA: 左半球優位 (接近動機、ポジティブ感情)
負のFAA: 右半球優位 (回避動機、ネガティブ感情)

参考文献:
Cannard, C., Wahbeh, H., & Delorme, A. (2021).
"Validating the wearable MUSE headset for EEG spectral analysis and Frontal Alpha Asymmetry"
IEEE International Conference on Bioinformatics and Biomedicine (BIBM).
https://doi.org/10.1109/BIBM52615.2021.9669778

論文の推奨事項:
- MUSEのデフォルトリファレンス(Fpz)は前頭チャネルに近すぎるため、
  TP9/TP10をlinked mastoid referenceとして使用することを推奨
- 8-13Hz全帯域での従来法がIAFベースの方法と同等以上の信頼性
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

    time_series: pd.Series  # FAA時系列 (dB差分: 10*log10(右) - 10*log10(左))
    left_power: pd.Series  # 左前頭部アルファパワー (dB)
    right_power: pd.Series  # 右前頭部アルファパワー (dB)
    statistics: pd.DataFrame
    metadata: dict


def calculate_frontal_asymmetry(
    df: pd.DataFrame,
    left_channel: str = 'RAW_AF7',
    right_channel: str = 'RAW_AF8',
    alpha_band: Tuple[float, float] = (8.0, 13.0),
    resample_interval: str = '10S',
    smoothing_seconds: float = 6.0,
    rolling_window_seconds: float = 8.0,
    raw: Optional[mne.io.BaseRaw] = None,
    use_mastoid_reference: bool = True,
    left_mastoid: str = 'RAW_TP9',
    right_mastoid: str = 'RAW_TP10',
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
    use_mastoid_reference : bool
        Trueの場合、TP9/TP10をlinked mastoid referenceとして使用。
        Cannard et al. (2021) の推奨に基づく。デフォルトはTrue。
    left_mastoid : str
        左側マストイドチャネル (デフォルト: RAW_TP9)。
    right_mastoid : str
        右側マストイドチャネル (デフォルト: RAW_TP10)。

    Returns
    -------
    FrontalAsymmetryResult
        FAA時系列（dB差分）、左右パワー（dB）、統計情報を含む解析結果。

    Notes
    -----
    Mastoid referenceについて:
    MUSEのデフォルトリファレンス(Fpz)は前頭チャネル(AF7/AF8)に近いため、
    信号振幅が低くなる。TP9/TP10へのre-referenceにより、
    研究グレードEEGシステムとの相関が向上する (r=.67, Cannard et al., 2021)。
    """
    frontal_channels = [left_channel, right_channel]

    # RAWデータ準備
    if raw is None:
        mne_dict = prepare_mne_raw(df)
        if not mne_dict:
            raise ValueError('Failed to construct RAW data.')
        raw = mne_dict['raw']

    raw = raw.copy()
    available = set(raw.ch_names)

    # Mastoid referenceを使用する場合、4チャネル全て必要
    if use_mastoid_reference:
        all_channels = frontal_channels + [left_mastoid, right_mastoid]
        missing = [ch for ch in all_channels if ch not in available]
        if missing:
            raise ValueError(
                f'Mastoid reference requires channels {all_channels}, '
                f'but missing: {missing}'
            )
        raw.pick_channels(all_channels)

        # Linked mastoid reference: (TP9 + TP10) / 2
        # AF7_reref = AF7 - mastoid_ref
        # AF8_reref = AF8 - mastoid_ref
        data = raw.get_data()
        ch_names = raw.ch_names

        left_idx = ch_names.index(left_channel)
        right_idx = ch_names.index(right_channel)
        left_mast_idx = ch_names.index(left_mastoid)
        right_mast_idx = ch_names.index(right_mastoid)

        mastoid_ref = (data[left_mast_idx] + data[right_mast_idx]) / 2
        data[left_idx] = data[left_idx] - mastoid_ref
        data[right_idx] = data[right_idx] - mastoid_ref

        # Re-referenced dataで新しいRawオブジェクトを作成
        info = mne.create_info(
            ch_names=[left_channel, right_channel],
            sfreq=raw.info['sfreq'],
            ch_types='eeg',
        )
        raw = mne.io.RawArray(data[[left_idx, right_idx]], info, verbose=False)
        reference_method = 'linked_mastoid'
    else:
        missing = [ch for ch in frontal_channels if ch not in available]
        if missing:
            raise ValueError(f'Specified channels not found: {missing}')
        raw.pick_channels(frontal_channels)
        reference_method = 'default_fpz'

    channels = [left_channel, right_channel]

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

    # パワー計算: エンベロープの二乗をdBに変換
    power_uv2 = env_data_uv.T ** 2
    epsilon = 1e-12  # ゼロ除算防止
    power_db = 10 * np.log10(power_uv2 + epsilon)

    power_df = pd.DataFrame(
        power_db,
        index=time_index,
        columns=channels,
    )

    # 外れ値除去（各チャネル独立に90パーセンタイルでクリップ）
    # 256Hzの高サンプリングレートでは極端な値がより顕著に現れるため必須
    # Z-scoreではなくパーセンタイルベースの方が、装着直後の集中的なスパイクに効果的
    # 90パーセンタイル: 上位10%の極端な値を除去（装着直後の不安定な信号を積極的に除外）
    for ch in channels:
        upper_bound = power_df[ch].quantile(0.90)
        # dBの場合、下限はlog10(epsilon)の値
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

    # 左右パワーを抽出（既にdB単位）
    left_power = power_df[left_channel].dropna()
    right_power = power_df[right_channel].dropna()

    if left_power.empty or right_power.empty:
        raise ValueError('Failed to calculate left and right alpha power.')

    # FAA計算: dB差分 = 10*log10(右) - 10*log10(左)
    # 既にdBに変換済みなので、単純に差分を取る
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

    # FAA解釈（dB差分）
    # 2 dB ≈ ln(1.05) ≈ 5%の差に相当
    if faa_mean > 2.0:
        interpretation = 'Left hemisphere dominant (Approach motivation/Positive)'
    elif faa_mean < -2.0:
        interpretation = 'Right hemisphere dominant (Avoidance motivation/Negative)'
    else:
        interpretation = 'Balanced'

    stats_df = pd.DataFrame(
        [
            {'Metric': 'Mean FAA', 'Value': faa_mean, 'Unit': 'dB'},
            {'Metric': 'Median', 'Value': faa_median, 'Unit': 'dB'},
            {'Metric': 'Std Dev', 'Value': faa_std, 'Unit': 'dB'},
            {'Metric': 'First Half Mean', 'Value': first_mean, 'Unit': 'dB'},
            {'Metric': 'Second Half Mean', 'Value': second_mean, 'Unit': 'dB'},
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
        'unit': 'dB',
        'method': 'mne_hilbert_db',
        'reference_method': reference_method,
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
