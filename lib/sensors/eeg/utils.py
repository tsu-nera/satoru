"""
EEG解析ユーティリティ関数
"""

import pandas as pd

from .constants import FREQ_BANDS


def get_psd_peak_frequencies(psd_dict, bands=None):
    """
    各バンドのピーク周波数を取得

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の戻り値
    bands : dict, optional
        バンド定義辞書

    Returns
    -------
    peak_df : pd.DataFrame
        バンド別ピーク周波数テーブル
    """
    if bands is None:
        bands = FREQ_BANDS

    freqs = psd_dict['freqs']
    psds = psd_dict['psds']

    peak_info = []
    for band, (low, high, _) in bands.items():
        mask = (freqs >= low) & (freqs <= high)
        if mask.any():
            band_freqs = freqs[mask]
            band_psds = psds.mean(axis=0)[mask]
            peak_idx = band_psds.argmax()
            peak_info.append({
                'Band': band,
                'Peak Frequency (Hz)': band_freqs[peak_idx]
            })

    return pd.DataFrame(peak_info)


