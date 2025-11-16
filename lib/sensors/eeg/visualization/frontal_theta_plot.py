"""
Frontal Midline Theta (Fmθ) 可視化モジュール
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..frontal_theta import FrontalThetaResult


def plot_frontal_theta(
    result: FrontalThetaResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[pd.Series, object]:
    """
    Fmθの時間推移をプロットする。

    Parameters
    ----------
    result : FrontalThetaResult
        `calculate_frontal_theta` の戻り値。
    img_path : str, optional
        画像保存パス。
    title : str
        グラフタイトル。

    Returns
    -------
    (series, fig)
        表示に使用した時系列とFigureオブジェクト。
    """
    series = result.time_series
    if series.empty:
        raise ValueError('Fmθ time series for plotting is empty.')

    elapsed_minutes = (series.index - series.index[0]).total_seconds() / 60.0

    metadata = result.metadata
    band_key = metadata.get('band_key', '')
    band_range = metadata.get('band', (None, None))
    label = f'Fmθ ({band_range[0]}-{band_range[1]} Hz)' if None not in band_range else 'Fmθ'
    default_title = f'Frontal Midline Theta - {band_key}' if band_key else 'Frontal Midline Theta'
    plot_title = title or default_title

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        elapsed_minutes,
        series.values,
        color='#1f77b4',
        linewidth=2.2,
        label=label,
    )
    midpoint = elapsed_minutes[len(elapsed_minutes) // 2] if len(elapsed_minutes) else 0
    if midpoint:
        ax.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, label='Session midpoint')

    increase = metadata.get('increase_rate_percent')
    if increase is not None and not np.isnan(increase):
        ax.text(
            0.02,
            0.95,
            f'Δ後半/前半: {increase:.1f}%',
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )
    ax.set_xlabel('Elapsed Time (min)', fontsize=12)
    ax.set_ylabel('Fmθ Power (μV²)', fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return series, fig
