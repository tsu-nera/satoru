"""
Individual Theta Frequency (ITF) 解析モジュール

瞑想中のシータ波ピーク周波数を計算します。

以前は窓内argmaxでピークを探していたが、PSDが1/fで単調減少する帯域では
argmaxが必ず探索窓の下限に張り付く（測定値ではなく窓設定の産物になる）
ことが判明した（Issue #31）。specparamによる非周期成分分離
（`aperiodic.fit_aperiodic`）で実際にピークとして検出された場合のみ値を返し、
検出されなければNaNを返す。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ._peak_frequency_base import HEMISPHERE_CONFIG
from .aperiodic import find_band_peak, fit_aperiodic


def calculate_itf(psd_dict, theta_range: Tuple[float, float] = (4.0, 8.0), use_hemisphere_average=True):
    """
    Individual Theta Frequency（ITF）を計算

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の戻り値
    theta_range : tuple
        Theta帯域の範囲（Hz）。デフォルトは(4.0, 8.0)（FREQ_BANDS['Theta']に合わせる）
    use_hemisphere_average : bool
        Trueの場合、左右半球ごとにチャネルを平均化してからITFを計算
        - Left: (TP9 + AF7) / 2
        - Right: (AF8 + TP10) / 2
        これによりノイズが低減され、より安定したITFが得られる

    Returns
    -------
    itf_dict : dict
        {
            # チャネル/半球別 {channel: {'PTF': float, 'CoG': float, 'Power': float, 'PSD': array}}
            'ptf_by_channel': dict,
            'itf_peak': float,       # ITF Peak方式（θピークが検出された半球/チャネルの平均、無ければNaN）
            'itf_cog': float,        # 常にNaN（ピーク検出ベースに変更したため概念的に意味を失った。
                                     # キーは呼び出し側との互換性のため残す）
            'itf': float,            # ITF (itf_peakのエイリアス)
            'itf_std': float,        # ITF Peak方式の標準偏差
            'theta_range': tuple,    # Theta帯域範囲
            'theta_freqs': ndarray,  # Theta帯域の周波数配列
        }
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    theta_low, theta_high = theta_range
    theta_mask = (freqs >= theta_low) & (freqs <= theta_high)
    theta_freqs = freqs[theta_mask]

    if use_hemisphere_average:
        groups = {}
        for hemi_name, hemi_channels in HEMISPHERE_CONFIG.items():
            hemi_indices = [i for i, ch in enumerate(channels) if ch in hemi_channels]
            if len(hemi_indices) >= 1:
                groups[hemi_name] = np.mean([psds[i] for i in hemi_indices], axis=0)
    else:
        groups = {ch.replace('RAW_', ''): psds[i] for i, ch in enumerate(channels)}

    ptf_by_channel = {}
    for label, psd in groups.items():
        band_psd = psd[theta_mask]
        result = fit_aperiodic(freqs, psd)
        peak = find_band_peak(result, theta_range) if result is not None else None

        if peak is not None:
            ptf = peak['center_hz']
            power_idx = int(np.abs(theta_freqs - ptf).argmin())
            power = float(band_psd[power_idx])
        else:
            ptf = np.nan
            power = np.nan

        ptf_by_channel[label] = {
            'PTF': ptf,
            'CoG': np.nan,
            'Power': power,
            'PSD': band_psd,
        }

    ptf_values = [v['PTF'] for v in ptf_by_channel.values()]
    valid_ptf = [v for v in ptf_values if not np.isnan(v)]

    itf_peak = float(np.mean(valid_ptf)) if valid_ptf else float('nan')
    itf_std = float(np.std(valid_ptf)) if len(valid_ptf) > 1 else float('nan')

    return {
        'ptf_by_channel': ptf_by_channel,
        'itf_peak': itf_peak,
        'itf_cog': float('nan'),
        'itf': itf_peak,
        'itf_std': itf_std,
        'theta_range': theta_range,
        'theta_freqs': theta_freqs,
    }
