"""
SMR (Sensorimotor Rhythm) 可視化モジュール
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..smr import SMRResult
from ....visualization.utils import format_time_axis


def plot_smr(
    result: SMRResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[pd.Series, object]:
    """
    SMRの時間推移をプロットする。

    Parameters
    ----------
    result : SMRResult
        `calculate_smr` の戻り値。
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
        raise ValueError('SMR time series for plotting is empty.')

    elapsed_seconds = (series.index - series.index[0]).total_seconds()

    metadata = result.metadata
    band_key = metadata.get('band_key', '')
    band_range = metadata.get('band', (None, None))
    smr_label = f'SMR ({band_range[0]}-{band_range[1]} Hz)' if None not in band_range else 'SMR'

    default_title = f'SMR-band Power (AF) - {band_key}' if band_key else 'SMR-band Power (AF)'
    plot_title = title or default_title

    fig, ax = plt.subplots(figsize=(14, 6))

    # SMRをプロット（オレンジ系の色）
    ax.plot(
        elapsed_seconds,
        series.values,
        color='#ff7f0e',  # オレンジ
        linewidth=2.5,
        label=smr_label,
        alpha=0.9,
    )

    # セッション中点
    midpoint = elapsed_seconds[len(elapsed_seconds) // 2] if len(elapsed_seconds) else 0
    if midpoint:
        ax.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Session midpoint')

    # SMRの増加率を表示
    increase = metadata.get('increase_rate_percent')
    if increase is not None and not np.isnan(increase):
        ax.text(
            0.02,
            0.95,
            f'SMR Δ後半/前半: {increase:.1f}%',
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
