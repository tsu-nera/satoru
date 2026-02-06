"""
PSD-based peak frequency calculation - shared logic for PAF and ITF.

このモジュールは、PAF（Peak Alpha Frequency）とITF（Individual Theta Frequency）の
共通計算ロジックを提供します。
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


HEMISPHERE_CONFIG: Dict[str, List[str]] = {
    'Left': ['RAW_TP9', 'RAW_AF7'],
    'Right': ['RAW_AF8', 'RAW_TP10'],
}


def calculate_cog(freqs: np.ndarray, psd: np.ndarray) -> float:
    """
    Center of Gravity (CoG) を計算

    CoG = Σ(f × P(f)) / Σ(P(f))

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列
    psd : np.ndarray
        パワースペクトル密度

    Returns
    -------
    cog : float
        Center of Gravity (Hz)
    """
    return np.sum(freqs * psd) / np.sum(psd)


def calculate_peak_frequency(
    psd_dict: dict,
    freq_range: Tuple[float, float],
    use_hemisphere_average: bool = True
) -> dict:
    """
    汎用的な周波数帯域ピーク計算

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の戻り値
        {'freqs': ndarray, 'psds': ndarray, 'channels': list}
    freq_range : tuple
        対象周波数帯域の範囲（Hz）。例: (8.0, 12.0) for Alpha, (5.0, 7.0) for Theta
    use_hemisphere_average : bool
        Trueの場合、左右半球ごとにチャネルを平均化してからピークを計算
        - Left: (TP9 + AF7) / 2
        - Right: (AF8 + TP10) / 2
        これによりノイズが低減され、より安定したピークが得られる

    Returns
    -------
    result : dict
        {
            'peak_by_channel': dict,  # チャネル/半球別 {label: {'Peak': float, 'CoG': float, 'Power': float, 'PSD': array}}
            'individual_peak': float,  # 左右ピークの平均
            'individual_cog': float,   # 左右CoGの平均
            'individual_std': float,   # ピーク値の標準偏差
            'band_range': tuple,       # 周波数帯域範囲
            'band_freqs': ndarray,     # 帯域内の周波数配列
        }
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    freq_low, freq_high = freq_range
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    band_freqs = freqs[freq_mask]

    peak_results = {}

    if use_hemisphere_average:
        # 左右半球ごとにPSDを平均化してからピークを計算
        for hemi_name, hemi_channels in HEMISPHERE_CONFIG.items():
            # 該当チャネルのインデックスを取得
            hemi_indices = [i for i, ch in enumerate(channels) if ch in hemi_channels]

            if len(hemi_indices) >= 1:
                # PSDを平均化
                hemi_psds = np.mean([psds[i] for i in hemi_indices], axis=0)
                band_psd = hemi_psds[freq_mask]

                # Peak方式
                peak_idx = band_psd.argmax()
                peak_freq = band_freqs[peak_idx]
                peak_power = band_psd[peak_idx]

                # CoG方式
                cog = calculate_cog(band_freqs, band_psd)

                peak_results[hemi_name] = {
                    'Peak': peak_freq,
                    'CoG': cog,
                    'Power': peak_power,
                    'PSD': band_psd,
                }
    else:
        # 個別チャネル処理
        for i, ch_name in enumerate(channels):
            ch_label = ch_name.replace('RAW_', '')
            band_psd = psds[i][freq_mask]

            # Peak方式
            peak_idx = band_psd.argmax()
            peak_freq = band_freqs[peak_idx]
            peak_power = band_psd[peak_idx]

            # CoG方式
            cog = calculate_cog(band_freqs, band_psd)

            peak_results[ch_label] = {
                'Peak': peak_freq,
                'CoG': cog,
                'Power': peak_power,
                'PSD': band_psd,
            }

    # Individual frequency（全チャネル/半球の平均）
    all_peaks = [v['Peak'] for v in peak_results.values()]
    all_cogs = [v['CoG'] for v in peak_results.values()]

    individual_peak = np.mean(all_peaks)
    individual_cog = np.mean(all_cogs)
    individual_std = np.std(all_peaks)

    return {
        'peak_by_channel': peak_results,
        'individual_peak': individual_peak,
        'individual_cog': individual_cog,
        'individual_std': individual_std,
        'band_range': freq_range,
        'band_freqs': band_freqs,
    }
