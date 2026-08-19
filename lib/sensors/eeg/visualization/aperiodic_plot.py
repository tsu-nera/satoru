"""
非周期成分（1/f）分離の可視化モジュール
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_aperiodic_fit(freqs, psd, result, freq_range=None, img_path=None):
    """
    実測PSD・非周期成分（1/f）フィット・残差（振動成分）を重ねてプロットする。

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）。
    psd : np.ndarray
        パワースペクトル密度（線形スケール）。
    result : AperiodicResult
        aperiodic.fit_aperiodic() の戻り値。
    freq_range : tuple, optional
        表示する周波数範囲。Noneの場合は result.fit_range を使用。
    img_path : str or Path, optional
        保存先パス。

    Returns
    -------
    fig : matplotlib.figure.Figure
        生成された図オブジェクト
    """
    if freq_range is None:
        freq_range = result.fit_range

    low, high = freq_range
    mask = (freqs >= low) & (freqs <= high)
    plot_freqs = freqs[mask]
    plot_psd = psd[mask]

    aperiodic_fit = (10 ** result.offset) / (plot_freqs ** result.exponent)
    residual_db = 10 * np.log10(plot_psd / aperiodic_fit)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # 上段: log-logで実測PSDと非周期成分フィット
    ax1.loglog(plot_freqs, plot_psd, label='Measured PSD', color='#2196F3', linewidth=2)
    ax1.loglog(
        plot_freqs, aperiodic_fit,
        label=f'Aperiodic fit (offset={result.offset:.2f}, exponent={result.exponent:.2f})',
        color='#F44336', linewidth=2, linestyle='--',
    )

    if not result.peaks.empty:
        for _, peak in result.peaks.iterrows():
            ax1.axvline(peak['center_hz'], color='#4CAF50', alpha=0.4, linewidth=1.5)

    ax1.set_ylabel('PSD (μV²/Hz)', fontsize=11)
    ax1.set_title(
        f'Aperiodic (1/f) Fit — R²={result.r_squared:.3f}, n_peaks={result.n_peaks}',
        fontsize=13, fontweight='bold',
    )
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, which='both', alpha=0.3)

    # 下段: 残差（振動成分、dB）
    ax2.plot(plot_freqs, residual_db, color='#673AB7', linewidth=1.5)
    ax2.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    if not result.peaks.empty:
        for _, peak in result.peaks.iterrows():
            ax2.axvline(peak['center_hz'], color='#4CAF50', alpha=0.4, linewidth=1.5)
            ax2.annotate(
                f"{peak['center_hz']:.2f} Hz",
                (peak['center_hz'], residual_db[np.abs(plot_freqs - peak['center_hz']).argmin()]),
                fontsize=8, color='#2E7D32',
            )

    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Oscillatory residual (dB)', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
