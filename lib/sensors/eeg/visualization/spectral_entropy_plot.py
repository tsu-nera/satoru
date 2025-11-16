"""
Spectral Entropy (SE) 可視化モジュール
"""

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..spectral_entropy import SpectralEntropyResult


def plot_spectral_entropy(
    result: SpectralEntropyResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Tuple[pd.DataFrame, object]:
    """
    Spectral Entropyの統計情報をプロット

    Parameters
    ----------
    result : SpectralEntropyResult
        calculate_spectral_entropy または calculate_spectral_entropy_time_series の戻り値
    img_path : str, optional
        画像保存パス
    title : str, optional
        グラフタイトル

    Returns
    -------
    (stats_df, fig)
        統計情報とFigureオブジェクト
    """
    stats_df = result.statistics
    metadata = result.metadata

    # プロットデータの準備
    labels = stats_df['Metric'].tolist()
    values = stats_df['Value'].tolist()

    # 変化率以外の値でプロット
    plot_labels = [l for l in labels if 'Change Rate' not in l]
    plot_values = [v for l, v in zip(labels, values) if 'Change Rate' not in l]

    default_title = 'Spectral Entropy (SE)'
    plot_title = title or default_title

    fig, ax = plt.subplots(figsize=(10, 6))

    # 棒グラフ
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax.bar(
        range(len(plot_values)),
        plot_values,
        color=colors[:len(plot_values)],
        alpha=0.7,
        edgecolor='black',
    )

    ax.set_xticks(range(len(plot_labels)))
    ax.set_xticklabels(plot_labels, rotation=15, ha='right')
    ax.set_ylabel('Spectral Entropy', fontsize=12)
    ax.set_title(plot_title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    # 変化率を表示
    change_percent = metadata.get('change_percent')
    if change_percent is not None and not np.isnan(change_percent):
        interpretation = "低下（注意集中）" if change_percent < 0 else "上昇（注意散漫）"
        ax.text(
            0.98,
            0.95,
            f'変化率: {change_percent:+.1f}%\n{interpretation}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
        )

    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return stats_df, fig
