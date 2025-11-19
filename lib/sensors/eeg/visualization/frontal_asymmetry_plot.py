"""
Frontal Alpha Asymmetry (FAA) 可視化モジュール
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt

from ..frontal_asymmetry import FrontalAsymmetryResult
from ....visualization.utils import format_time_axis


def plot_frontal_asymmetry(
    result: FrontalAsymmetryResult,
    img_path: Optional[str] = None,
    title: str = 'Frontal Alpha Asymmetry (FAA)',
) -> Tuple[object, object]:
    """
    FAA時系列と左右パワーをプロットする。

    Parameters
    ----------
    result : FrontalAsymmetryResult
        `calculate_frontal_asymmetry` の戻り値。
    img_path : str, optional
        画像保存パス。
    title : str
        グラフタイトル。

    Returns
    -------
    (fig, axes)
        Figureオブジェクトと軸。
    """
    faa_series = result.time_series
    left_power = result.left_power
    right_power = result.right_power

    if faa_series.empty:
        raise ValueError('FAA time series is empty.')

    elapsed_seconds = (faa_series.index - faa_series.index[0]).total_seconds()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # 上段: 左右アルファパワー
    ax1 = axes[0]
    left_elapsed = (left_power.index - left_power.index[0]).total_seconds()
    right_elapsed = (right_power.index - right_power.index[0]).total_seconds()

    ax1.plot(left_elapsed, left_power.values, color='#d62728', linewidth=2, label='Left Frontal (AF7)', alpha=0.8)
    ax1.plot(right_elapsed, right_power.values, color='#2ca02c', linewidth=2, label='Right Frontal (AF8)', alpha=0.8)
    ax1.set_ylabel('Alpha Power (μV²)', fontsize=12)
    ax1.set_title('Left and Right Frontal Alpha Power', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 下段: FAA (ln(右) - ln(左))
    ax2 = axes[1]
    ax2.plot(elapsed_seconds, faa_series.values, color='#1f77b4', linewidth=2.2, label='FAA')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Center (Balanced)')

    # FAA解釈表示
    interpretation = result.metadata.get('interpretation', '')
    mean_faa = result.metadata.get('first_half_mean', faa_series.mean())
    ax2.text(
        0.02,
        0.95,
        f'{interpretation}\nMean FAA: {mean_faa:.3f}',
        transform=ax2.transAxes,
        fontsize=11,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
        verticalalignment='top',
    )

    format_time_axis(ax2, elapsed_seconds, unit='minutes')
    ax2.set_ylabel('FAA [ln(Right) - ln(Left)]', fontsize=12)
    ax2.set_title('Frontal Alpha Asymmetry', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig, axes
