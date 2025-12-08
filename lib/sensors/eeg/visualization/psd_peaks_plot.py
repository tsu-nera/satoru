"""
PSDピーク分析の可視化モジュール
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from ..psd_peaks import PsdPeaksResult, PeakType, DETAILED_FREQ_BANDS


def plot_psd_peaks(
    result: PsdPeaksResult,
    psd_dict: dict,
    img_path: Optional[Union[str, Path]] = None,
    freq_max: float = 35.0,
) -> plt.Figure:
    """
    PSDピーク分析結果をプロット

    Parameters
    ----------
    result : PsdPeaksResult
        analyze_psd_peaks() の戻り値
    psd_dict : dict
        calculate_psd() の戻り値
    img_path : str or Path, optional
        保存先パス
    freq_max : float
        表示する最大周波数

    Returns
    -------
    fig : matplotlib.figure.Figure
        生成された図オブジェクト
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    mask = (freqs >= 1) & (freqs <= freq_max)

    ax.plot(freqs[mask], psd_db[mask], 'b-', linewidth=1.5, alpha=0.7)

    # 周波数帯域を色分け
    band_colors = {
        'Delta': 'purple',
        'Theta': 'blue',
        'Low Alpha': 'green',
        'High Alpha': 'lightgreen',
        'SMR': 'cyan',
        'Low Beta': 'orange',
        'Mid Beta': 'coral',
        'High Beta': 'red',
        'Gamma': 'magenta',
    }

    for band_name, (low, high, _) in DETAILED_FREQ_BANDS.items():
        if low < freq_max:
            color = band_colors.get(band_name, 'gray')
            ax.axvspan(low, min(high, freq_max), alpha=0.1, color=color)

    # ピークをマーク（IAFは赤、SMRはシアン、4Hz倍数はグレー、その他は緑）
    for peak in result.peaks:
        if peak.frequency <= freq_max:
            # 4Hz倍数アーチファクトはグレーで表示
            if peak.is_4hz_harmonic:
                color = 'gray'
                marker = 'x'
                alpha = 0.5
            elif peak.peak_type == PeakType.FUNDAMENTAL:
                color = 'red'
                marker = 'o'
                alpha = 1.0
            elif peak.band_name == 'SMR':
                color = 'darkcyan'
                marker = 's'
                alpha = 1.0
            else:
                color = 'green'
                marker = 's'
                alpha = 1.0

            ax.scatter(
                [peak.frequency], [peak.power_db],
                color=color, s=100, marker=marker, zorder=5,
                edgecolors='white', linewidths=1, alpha=alpha
            )

            # ラベル（主要なピークのみ、4Hz倍数は除外）
            if not peak.is_4hz_harmonic:
                if peak.prominence > 1.0 or peak.peak_type == PeakType.FUNDAMENTAL or peak.band_name == 'SMR':
                    label = f'{peak.frequency:.1f}Hz\n({peak.band_name})'
                    ax.annotate(
                        label, (peak.frequency, peak.power_db),
                        textcoords='offset points', xytext=(0, 12),
                        ha='center', fontsize=8, color=color
                    )

    # 凡例用のダミープロット
    ax.scatter([], [], color='red', marker='o', s=80, label='IAF (Fundamental)')
    ax.scatter([], [], color='darkcyan', marker='s', s=80, label='SMR')
    ax.scatter([], [], color='green', marker='s', s=80, label='Other Peaks')
    ax.scatter([], [], color='gray', marker='x', s=80, alpha=0.5, label='4Hz Harmonic (Artifact)')

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power (dB)', fontsize=11)
    ax.set_title('PSD Peak Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, freq_max)

    plt.tight_layout()

    if img_path:
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


# 後方互換性のためのエイリアス
plot_harmonics = plot_psd_peaks
