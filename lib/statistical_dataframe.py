"""
統一的なStatistical DataFrameレイヤー

EEG解析のための統一的なバンドパワーおよび比率計算を提供する。
全ての解析でこのレイヤーを使用することで、計算の一貫性と効率性を確保する。
"""

from __future__ import annotations

from typing import Dict, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

# Spectral Entropy計算関数をインポート
from .sensors.eeg.spectral_entropy import _calculate_shannon_entropy

if TYPE_CHECKING:
    import mne

try:
    import mne
    from mne import Epochs, make_fixed_length_events
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


def create_statistical_dataframe(
    raw: 'mne.io.RawArray',
    segment_minutes: int = 3,
    warmup_minutes: float = 0.0,
    session_start: Optional[pd.Timestamp] = None,
) -> Dict[str, pd.DataFrame]:
    """
    統一的なStatistical DataFrameを生成する。

    MNE Epochsを使用してセグメントごとのバンドパワーと比率を計算し、
    統計量とともに返す。全ての解析でこの関数を使用することで、
    一貫した計算結果を保証する。

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE RawArrayオブジェクト
    segment_minutes : int, default 3
        セグメント長（分単位）
    warmup_minutes : float, default 0.0
        ウォームアップ期間（分単位）
    session_start : pd.Timestamp, optional
        セッション開始時刻（Noneの場合は現在時刻を使用）

    Returns
    -------
    dict
        {
            'band_powers': DataFrame,      # セグメント別バンドパワー時系列（Bels）
            'band_ratios': DataFrame,      # セグメント別バンド比率時系列
            'spectral_entropy': DataFrame, # セグメント別Spectral Entropy時系列（正規化済み）
            'statistics': DataFrame        # 統計サマリー（縦長形式）
        }

    Notes
    -----
    - バンドパワーはBels（10*log10(μV²)）で表現
    - 比率（対数）はBels差分（log10(A/B) = log10(A) - log10(B)）
    - 比率（実数）は10^(Bels差分)で計算
    - 統計量計算時にZ-score外れ値除去（閾値3.0）を適用
    """
    if not MNE_AVAILABLE:
        raise ImportError('MNE-Pythonが必要です。pip install mne でインストールしてください。')

    # セッション開始時刻のデフォルト値
    if session_start is None:
        session_start = pd.Timestamp.now()

    # ウォームアップ期間をスキップ
    tmin_sec = warmup_minutes * 60.0
    tmax_sec = raw.times[-1]

    if tmin_sec >= tmax_sec:
        raise ValueError(f'ウォームアップ期間（{warmup_minutes}分）がデータ長を超えています。')

    # ウォームアップ後のデータをクロップ
    raw_cropped = raw.copy().crop(tmin=tmin_sec, tmax=tmax_sec)

    # 固定長イベント作成
    duration_sec = segment_minutes * 60.0
    events = make_fixed_length_events(raw_cropped, duration=duration_sec)

    if len(events) == 0:
        raise ValueError('セグメント用のイベントが生成できませんでした。')

    # Epochsオブジェクト作成
    epochs = Epochs(
        raw_cropped,
        events,
        tmin=0,
        tmax=duration_sec,
        baseline=None,
        preload=True,
        verbose=False,
    )

    # PSD計算（Welch法）
    sfreq = raw_cropped.info['sfreq']
    nyquist = sfreq / 2.0
    fmax = min(50.0, nyquist * 0.95)  # 安全マージン5%

    spectrum = epochs.compute_psd(method='welch', fmin=1.0, fmax=fmax, verbose=False)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # バンド定義（全バンド）
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50),
    }

    # タイムスタンプ生成
    timestamps = [
        session_start + pd.Timedelta(minutes=warmup_minutes) + pd.Timedelta(seconds=i * duration_sec)
        for i in range(len(epochs))
    ]

    # バンドパワー計算（全チャネル平均、Bels変換）
    band_powers_dict = {}
    for band_name, (fmin_band, fmax_band) in bands.items():
        freq_mask = (freqs >= fmin_band) & (freqs < fmax_band)
        # shape: (n_epochs, n_channels, n_freqs) -> (n_epochs,)
        band_power = psds[:, :, freq_mask].mean(axis=(1, 2))

        # Bels変換（10*log10）
        band_power_bels = 10 * np.log10(band_power + 1e-12)  # ゼロ除算回避

        band_powers_dict[band_name] = band_power_bels

    # DataFrameに変換
    band_powers_df = pd.DataFrame(band_powers_dict, index=timestamps)

    # Spectral Entropy計算（全チャネル平均）
    # 周波数範囲でマスク（1-40Hz）
    freq_range = (1.0, 40.0)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    se_values = []
    for epoch_idx in range(len(epochs)):
        # このエポックのPSD (n_channels, n_freqs)
        psd_epoch = psds[epoch_idx]

        # 全チャネルの平均PSD
        psd_avg = psd_epoch.mean(axis=0)

        # 周波数範囲でマスク
        psd_masked = psd_avg[freq_mask]

        # Shannon Entropy計算（spectral_entropy.pyの関数を使用）
        se = _calculate_shannon_entropy(psd_masked, normalize=True)
        se_values.append(se)

    # DataFrameに変換
    se_df = pd.DataFrame({'spectral_entropy': se_values}, index=timestamps)

    # バンド比率計算
    ratios_dict = {}

    # α/β比（リラックス度）
    ratios_dict['alpha_beta_bels'] = band_powers_df['Alpha'] - band_powers_df['Beta']
    ratios_dict['alpha_beta'] = 10 ** ratios_dict['alpha_beta_bels']

    # β/θ比（覚醒度・注意）
    ratios_dict['beta_theta_bels'] = band_powers_df['Beta'] - band_powers_df['Theta']
    ratios_dict['beta_theta'] = 10 ** ratios_dict['beta_theta_bels']

    # θ/α比（瞑想深度）
    ratios_dict['theta_alpha_bels'] = band_powers_df['Theta'] - band_powers_df['Alpha']
    ratios_dict['theta_alpha'] = 10 ** ratios_dict['theta_alpha_bels']

    # δ/β比（睡眠傾向）
    ratios_dict['delta_beta_bels'] = band_powers_df['Delta'] - band_powers_df['Beta']
    ratios_dict['delta_beta'] = 10 ** ratios_dict['delta_beta_bels']

    # γ/θ比（認知負荷）
    ratios_dict['gamma_theta_bels'] = band_powers_df['Gamma'] - band_powers_df['Theta']
    ratios_dict['gamma_theta'] = 10 ** ratios_dict['gamma_theta_bels']

    # DataFrameに変換
    band_ratios_df = pd.DataFrame(ratios_dict, index=timestamps)

    # 統計量計算（縦長形式）
    statistics_rows = []

    # バンドパワー統計
    for band_name in bands.keys():
        values = band_powers_df[band_name].dropna()
        if len(values) == 0:
            continue

        # Z-score外れ値除去（閾値3.0）
        if len(values) > 3:
            z_scores = np.abs(stats.zscore(values))
            filtered_values = values[z_scores < 3.0]
            if len(filtered_values) > 0:
                values = filtered_values

        statistics_rows.extend([
            {
                'Category': 'BandPower',
                'Metric': f'{band_name}_Mean',
                'Value': values.mean(),
                'Unit': 'Bels',
                'DisplayName': f'{band_name}平均 (Bels)',
            },
            {
                'Category': 'BandPower',
                'Metric': f'{band_name}_Median',
                'Value': values.median(),
                'Unit': 'Bels',
                'DisplayName': f'{band_name}中央値 (Bels)',
            },
            {
                'Category': 'BandPower',
                'Metric': f'{band_name}_Std',
                'Value': values.std(),
                'Unit': 'Bels',
                'DisplayName': f'{band_name}標準偏差 (Bels)',
            },
        ])

    # Spectral Entropy統計
    se_values_clean = se_df['spectral_entropy'].dropna()
    if len(se_values_clean) > 0:
        # Z-score外れ値除去（閾値3.0）
        if len(se_values_clean) > 3:
            z_scores = np.abs(stats.zscore(se_values_clean))
            filtered_se = se_values_clean[z_scores < 3.0]
            if len(filtered_se) > 0:
                se_values_clean = filtered_se

        statistics_rows.extend([
            {
                'Category': 'SpectralEntropy',
                'Metric': 'spectral_entropy_Mean',
                'Value': se_values_clean.mean(),
                'Unit': 'normalized',
                'DisplayName': 'Spectral Entropy平均 (集中度)',
            },
            {
                'Category': 'SpectralEntropy',
                'Metric': 'spectral_entropy_Median',
                'Value': se_values_clean.median(),
                'Unit': 'normalized',
                'DisplayName': 'Spectral Entropy中央値 (集中度)',
            },
            {
                'Category': 'SpectralEntropy',
                'Metric': 'spectral_entropy_Std',
                'Value': se_values_clean.std(),
                'Unit': 'normalized',
                'DisplayName': 'Spectral Entropy標準偏差 (集中度)',
            },
        ])

    # バンド比率統計
    ratio_configs = [
        ('alpha_beta', 'α/β比', 'ratio', 'リラックス度'),
        ('beta_theta', 'β/θ比', 'ratio', '覚醒度'),
        ('theta_alpha', 'θ/α比', 'ratio', '瞑想深度'),
        ('delta_beta', 'δ/β比', 'ratio', '睡眠傾向'),
        ('gamma_theta', 'γ/θ比', 'ratio', '認知負荷'),
        ('alpha_beta_bels', 'α/β比', 'Bels', 'リラックス度（対数）'),
        ('beta_theta_bels', 'β/θ比', 'Bels', '覚醒度（対数）'),
        ('theta_alpha_bels', 'θ/α比', 'Bels', '瞑想深度（対数）'),
    ]

    for metric_key, ratio_name, unit, description in ratio_configs:
        values = band_ratios_df[metric_key].dropna()
        if len(values) == 0:
            continue

        # Z-score外れ値除去（閾値3.0）
        if len(values) > 3:
            z_scores = np.abs(stats.zscore(values))
            filtered_values = values[z_scores < 3.0]
            if len(filtered_values) > 0:
                values = filtered_values

        statistics_rows.extend([
            {
                'Category': 'BandRatio',
                'Metric': f'{metric_key}_Mean',
                'Value': values.mean(),
                'Unit': unit,
                'DisplayName': f'{ratio_name}平均 ({unit}) - {description}',
            },
            {
                'Category': 'BandRatio',
                'Metric': f'{metric_key}_Median',
                'Value': values.median(),
                'Unit': unit,
                'DisplayName': f'{ratio_name}中央値 ({unit}) - {description}',
            },
            {
                'Category': 'BandRatio',
                'Metric': f'{metric_key}_Std',
                'Value': values.std(),
                'Unit': unit,
                'DisplayName': f'{ratio_name}標準偏差 ({unit}) - {description}',
            },
        ])

    statistics_df = pd.DataFrame(statistics_rows)

    return {
        'band_powers': band_powers_df,
        'band_ratios': band_ratios_df,
        'spectral_entropy': se_df,
        'statistics': statistics_df,
    }


def get_band_power_at_time(
    band_powers_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    band: str,
) -> float:
    """
    指定時間範囲のバンドパワー平均を取得する。

    Parameters
    ----------
    band_powers_df : pd.DataFrame
        create_statistical_dataframe()が返すband_powers DataFrame
    start_time : pd.Timestamp
        開始時刻
    end_time : pd.Timestamp
        終了時刻
    band : str
        バンド名（'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'）

    Returns
    -------
    float
        指定範囲の平均値（Bels）、データがない場合はnp.nan
    """
    if band not in band_powers_df.columns:
        return np.nan

    mask = (band_powers_df.index >= start_time) & (band_powers_df.index < end_time)
    values = band_powers_df.loc[mask, band]

    if len(values) == 0:
        return np.nan

    return values.mean()


def get_band_ratio_at_time(
    band_ratios_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    ratio: str,
) -> float:
    """
    指定時間範囲のバンド比率平均を取得する。

    Parameters
    ----------
    band_ratios_df : pd.DataFrame
        create_statistical_dataframe()が返すband_ratios DataFrame
    start_time : pd.Timestamp
        開始時刻
    end_time : pd.Timestamp
        終了時刻
    ratio : str
        比率名（'alpha_beta', 'beta_theta', 'theta_alpha', etc.）
        対数スケールの場合は'alpha_beta_bels'など

    Returns
    -------
    float
        指定範囲の平均値、データがない場合はnp.nan
    """
    if ratio not in band_ratios_df.columns:
        return np.nan

    mask = (band_ratios_df.index >= start_time) & (band_ratios_df.index < end_time)
    values = band_ratios_df.loc[mask, ratio]

    if len(values) == 0:
        return np.nan

    return values.mean()
