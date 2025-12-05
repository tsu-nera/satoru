"""
Peak Alpha Frequency (PAF) / Individual Alpha Frequency (IAF) 解析モジュール
"""

import numpy as np
import pandas as pd


def _calculate_cog(freqs, psd):
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


def calculate_paf(psd_dict, alpha_range=(8.0, 12.0), use_hemisphere_average=True):
    """
    Peak Alpha Frequency（PAF）とCenter of Gravity（CoG）を計算

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の戻り値
    alpha_range : tuple
        Alpha帯域の範囲（Hz）。デフォルトは(8.0, 12.0)でMuse標準に準拠
    use_hemisphere_average : bool
        Trueの場合、左右半球ごとにチャネルを平均化してからPAFを計算
        - Left: (TP9 + AF7) / 2
        - Right: (AF8 + TP10) / 2
        これによりノイズが低減され、より安定したPAFが得られる

    Returns
    -------
    paf_dict : dict
        {
            'paf_by_channel': dict,  # チャネル/半球別 {channel: {'PAF': float, 'CoG': float, 'Power': float, 'PSD': array}}
            'iaf_peak': float,       # IAF Peak方式 (左右PAFの平均)
            'iaf_cog': float,        # IAF CoG方式 (左右CoGの平均)
            'iaf': float,            # IAF (iaf_peakのエイリアス、Muse app互換)
            'iaf_std': float,        # IAF Peak方式の標準偏差
            'alpha_range': tuple     # Alpha帯域範囲
        }
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    alpha_low, alpha_high = alpha_range
    alpha_mask = (freqs >= alpha_low) & (freqs <= alpha_high)
    alpha_freqs = freqs[alpha_mask]

    paf_results = {}

    if use_hemisphere_average:
        # 左右半球ごとにPSDを平均化してからPAFを計算
        # Left: TP9 + AF7, Right: AF8 + TP10
        hemisphere_config = {
            'Left': ['RAW_TP9', 'RAW_AF7'],
            'Right': ['RAW_AF8', 'RAW_TP10']
        }

        for hemi_name, hemi_channels in hemisphere_config.items():
            # 該当チャネルのインデックスを取得
            hemi_indices = [i for i, ch in enumerate(channels) if ch in hemi_channels]

            if len(hemi_indices) >= 1:
                # PSDを平均化
                hemi_psds = np.mean([psds[i] for i in hemi_indices], axis=0)
                alpha_psd = hemi_psds[alpha_mask]

                # Peak方式
                peak_idx = alpha_psd.argmax()
                paf = alpha_freqs[peak_idx]
                peak_power = alpha_psd[peak_idx]

                # CoG方式
                cog = _calculate_cog(alpha_freqs, alpha_psd)

                paf_results[hemi_name] = {
                    'PAF': paf,
                    'CoG': cog,
                    'Power': peak_power,
                    'PSD': alpha_psd,
                }
    else:
        # 従来の個別チャネル処理
        for i, ch_name in enumerate(channels):
            ch_label = ch_name.replace('RAW_', '')
            alpha_psd = psds[i][alpha_mask]

            # Peak方式
            peak_idx = alpha_psd.argmax()
            paf = alpha_freqs[peak_idx]
            peak_power = alpha_psd[peak_idx]

            # CoG方式
            cog = _calculate_cog(alpha_freqs, alpha_psd)

            paf_results[ch_label] = {
                'PAF': paf,
                'CoG': cog,
                'Power': peak_power,
                'PSD': alpha_psd,
            }

    # Individual Alpha Frequency（IAF）
    all_pafs = [v['PAF'] for v in paf_results.values()]
    all_cogs = [v['CoG'] for v in paf_results.values()]

    iaf_peak = np.mean(all_pafs)
    iaf_cog = np.mean(all_cogs)
    iaf_std = np.std(all_pafs)

    return {
        'paf_by_channel': paf_results,
        'iaf_peak': iaf_peak,
        'iaf_cog': iaf_cog,
        'iaf': iaf_peak,  # Muse app互換のためPeakを使用
        'iaf_std': iaf_std,
        'alpha_range': alpha_range,
        'alpha_freqs': alpha_freqs
    }


# エイリアス（将来の移行用）
calculate_iaf = calculate_paf
