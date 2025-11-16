"""
セグメント分析可視化モジュール
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..segment_analysis import SegmentAnalysisResult


def plot_segment_comparison(
    result: SegmentAnalysisResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
) -> 'matplotlib.figure.Figure':
    """
    セグメントごとの主要指標を可視化する。

    Parameters
    ----------
    result : SegmentAnalysisResult
        `calculate_segment_analysis` の戻り値。
    img_path : str, optional
        保存先パス（Noneなら保存しない）。
    title : str, optional
        グラフタイトル。

    Returns
    -------
    matplotlib.figure.Figure
        生成したFigureオブジェクト。
    """
    if result.normalized.empty:
        raise ValueError('プロット対象のセグメントデータが空です。')

    metric_labels = {
        'fmtheta_mean': 'Fmθ',
        'alpha_mean': 'Alpha',
        'beta_mean': 'Beta',
        'theta_alpha_ratio': 'θ/α',
    }
    colors = {
        'fmtheta_mean': '#1f77b4',
        'alpha_mean': '#2ca02c',
        'beta_mean': '#ff7f0e',
        'theta_alpha_ratio': '#9467bd',
    }

    segments = result.segments
    x_positions = np.arange(len(segments))
    xtick_labels = segments['label'].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))

    for metric in result.normalized.columns:
        series = result.normalized[metric]
        if series.isna().all():
            continue
        ax.plot(
            x_positions,
            series.values,
            marker='o',
            linewidth=2.2,
            label=metric_labels.get(metric, metric),
            color=colors.get(metric, None),
        )

    peak_idx = result.metadata.get('peak_segment_index')
    if peak_idx is not None:
        peak_pos = segments.index.get_loc(peak_idx)
        ax.axvspan(
            peak_pos - 0.35,
            peak_pos + 0.35,
            color='gold',
            alpha=0.18,
            label='Peak Segment' if 'Peak Segment' not in ax.get_legend_handles_labels()[1] else None,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right')
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel('Normalized Score (0-1)')
    ax.set_xlabel('Time Segment')
    ax.set_title(title or 'Key Metrics by Time Segment', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
