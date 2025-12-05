"""
Frontal Midline Theta (Fmθ) 可視化モジュール
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..frontal_theta import FrontalThetaResult
from ....visualization.utils import format_time_axis


def plot_frontal_theta(
    result: FrontalThetaResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
    show_alpha: bool = True,
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
    show_alpha : bool
        Trueの場合、アルファ波も同時に表示する。

    Returns
    -------
    (series, fig)
        表示に使用した時系列とFigureオブジェクト。
    """
    series = result.time_series
    if series.empty:
        raise ValueError('Fmθ time series for plotting is empty.')

    elapsed_seconds = (series.index - series.index[0]).total_seconds()

    metadata = result.metadata
    band_key = metadata.get('band_key', '')
    band_range = metadata.get('band', (None, None))
    theta_label = f'Fmθ ({band_range[0]}-{band_range[1]} Hz)' if None not in band_range else 'Fmθ'

    # アルファ波の有無を確認
    has_alpha = show_alpha and result.alpha_series is not None and not result.alpha_series.empty

    if has_alpha:
        default_title = 'Frontal Theta & Alpha Power'
    else:
        default_title = f'Frontal Midline Theta - {band_key}' if band_key else 'Frontal Midline Theta'
    plot_title = title or default_title

    fig, ax = plt.subplots(figsize=(14, 6))

    # シータ波をプロット
    ax.plot(
        elapsed_seconds,
        series.values,
        color='#1f77b4',
        linewidth=2.5,
        label=theta_label,
        alpha=0.9,
    )

    # アルファ波をプロット（存在する場合）
    if has_alpha:
        alpha_series = result.alpha_series
        alpha_elapsed = (alpha_series.index - alpha_series.index[0]).total_seconds()
        alpha_band = metadata.get('alpha_band', (8.0, 12.0))
        alpha_label = f'Alpha ({alpha_band[0]}-{alpha_band[1]} Hz)'

        ax.plot(
            alpha_elapsed,
            alpha_series.values,
            color='#2ca02c',
            linewidth=2.5,
            label=alpha_label,
            alpha=0.9,
        )

    # セッション中点
    midpoint = elapsed_seconds[len(elapsed_seconds) // 2] if len(elapsed_seconds) else 0
    if midpoint:
        ax.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Session midpoint')

    # Fmθの増加率を表示
    increase = metadata.get('increase_rate_percent')
    if increase is not None and not np.isnan(increase):
        ax.text(
            0.02,
            0.95,
            f'Fmθ Δ後半/前半: {increase:.1f}%',
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
        )

    format_time_axis(ax, elapsed_seconds, unit='minutes')
    ax.set_ylabel('Power (dB)', fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return series, fig
