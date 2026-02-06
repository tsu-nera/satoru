"""
Peak Alpha Frequency (PAF) / Individual Alpha Frequency (IAF) 解析モジュール
"""

from ._peak_frequency_base import calculate_peak_frequency


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
    # 共通ロジックで計算
    base = calculate_peak_frequency(
        psd_dict,
        freq_range=alpha_range,
        use_hemisphere_average=use_hemisphere_average
    )

    # PAF固有のキーにマッピング
    paf_by_channel = {}
    for label, values in base['peak_by_channel'].items():
        paf_by_channel[label] = {
            'PAF': values['Peak'],
            'CoG': values['CoG'],
            'Power': values['Power'],
            'PSD': values['PSD'],
        }

    return {
        'paf_by_channel': paf_by_channel,
        'iaf_peak': base['individual_peak'],
        'iaf_cog': base['individual_cog'],
        'iaf': base['individual_peak'],  # Muse app互換のためPeakを使用
        'iaf_std': base['individual_std'],
        'alpha_range': base['band_range'],
        'alpha_freqs': base['band_freqs'],
    }


# エイリアス（将来の移行用）
calculate_iaf = calculate_paf
