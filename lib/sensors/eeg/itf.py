"""
Individual Theta Frequency (ITF) 解析モジュール

瞑想中のシータ波ピーク周波数を計算します。
マルチセッション分析（issues/017_theta_peak_validation）により、
瞑想中のITFは約5.92 Hz（CV: 5.1%）で安定していることが確認されています。
"""

from ._peak_frequency_base import calculate_peak_frequency


def calculate_itf(psd_dict, theta_range=(5.0, 7.0), use_hemisphere_average=True):
    """
    Individual Theta Frequency（ITF）とCenter of Gravity（CoG）を計算

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の戻り値
    theta_range : tuple
        Theta帯域の範囲（Hz）。デフォルトは(5.0, 7.0)
        マルチセッション分析により ITF ≈ 5.92 Hz と確認済み
    use_hemisphere_average : bool
        Trueの場合、左右半球ごとにチャネルを平均化してからITFを計算
        - Left: (TP9 + AF7) / 2
        - Right: (AF8 + TP10) / 2
        これによりノイズが低減され、より安定したITFが得られる

    Returns
    -------
    itf_dict : dict
        {
            'ptf_by_channel': dict,  # チャネル/半球別 {channel: {'PTF': float, 'CoG': float, 'Power': float, 'PSD': array}}
            'itf_peak': float,       # ITF Peak方式 (左右PTFの平均)
            'itf_cog': float,        # ITF CoG方式 (左右CoGの平均)
            'itf': float,            # ITF (itf_peakのエイリアス)
            'itf_std': float,        # ITF Peak方式の標準偏差
            'theta_range': tuple,    # Theta帯域範囲
            'theta_freqs': ndarray,  # Theta帯域の周波数配列
        }
    """
    # 共通ロジックで計算
    base = calculate_peak_frequency(
        psd_dict,
        freq_range=theta_range,
        use_hemisphere_average=use_hemisphere_average
    )

    # ITF固有のキーにマッピング
    # PTF = Peak Theta Frequency (チャネル別のシータピーク)
    ptf_by_channel = {}
    for label, values in base['peak_by_channel'].items():
        ptf_by_channel[label] = {
            'PTF': values['Peak'],
            'CoG': values['CoG'],
            'Power': values['Power'],
            'PSD': values['PSD'],
        }

    return {
        'ptf_by_channel': ptf_by_channel,
        'itf_peak': base['individual_peak'],
        'itf_cog': base['individual_cog'],
        'itf': base['individual_peak'],  # itf_peakのエイリアス
        'itf_std': base['individual_std'],
        'theta_range': base['band_range'],
        'theta_freqs': base['band_freqs'],
    }
