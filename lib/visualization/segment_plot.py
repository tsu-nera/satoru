"""
セグメント分析可視化モジュール
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..segment_analysis import SegmentAnalysisResult
from .utils import format_time_axis


def plot_segment_comparison(
    result: SegmentAnalysisResult,
    img_path: Optional[str] = None,
) -> 'matplotlib.figure.Figure':
    """
    セグメントごとの主要指標を時系列で可視化する。

    Parameters
    ----------
    result : SegmentAnalysisResult
        `calculate_segment_analysis` の戻り値。
    img_path : str, optional
        保存先パス（Noneなら保存しない）。

    Returns
    -------
    matplotlib.figure.Figure
        生成したFigureオブジェクト。
    """
    if result.normalized.empty:
        raise ValueError('プロット対象のセグメントデータが空です。')

    # プロット設定
    metrics_config = [
        ('fmtheta_mean', 'Fmθ (Normalized)', '#1f77b4'),
        ('theta_alpha_ratio', 'θ/α (Normalized)', '#9467bd'),
        ('alpha_mean', 'Alpha (Normalized)', '#2ca02c'),
        ('beta_mean', 'Beta (Normalized)', '#ff7f0e'),
    ]

    # 有効な指標のみフィルタ
    valid_metrics = [
        (col, label, color) for col, label, color in metrics_config
        if col in result.normalized.columns and not result.normalized[col].isna().all()
    ]

    if not valid_metrics:
        raise ValueError('プロット可能な指標がありません。')

    segments = result.segments
    segment_minutes = result.metadata.get('segment_minutes', 3)

    # 経過時間（分）を計算
    time_min = np.array([i * segment_minutes for i in range(len(segments))])
    # 各セグメントの中央時刻（秒）を計算
    time_sec = time_min * 60 + (segment_minutes * 60 / 2)

    n_plots = len(valid_metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3 * n_plots), sharex=True)

    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Segment Performance', fontsize=14, fontweight='bold')

    peak_idx = result.metadata.get('peak_segment_index')

    for i, (col, label, color) in enumerate(valid_metrics):
        ax = axes[i]
        values = result.normalized[col].values

        # エリアプロット
        ax.fill_between(time_sec, 0, values, alpha=0.4, color=color)
        ax.plot(time_sec, values, color=color, linewidth=2, marker='o', markersize=6)

        # ピーク区間をハイライト
        if peak_idx is not None:
            peak_pos = list(segments.index).index(peak_idx)
            peak_time = time_sec[peak_pos]
            half_seg = segment_minutes * 60 / 2
            ax.axvspan(
                peak_time - half_seg,
                peak_time + half_seg,
                alpha=0.2, color='gold'
            )

        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        # 値をラベル表示
        for t, v in zip(time_sec, values):
            if not np.isnan(v):
                ax.annotate(
                    f'{v:.2f}',
                    (t, v),
                    textcoords='offset points',
                    xytext=(0, 8),
                    ha='center',
                    fontsize=8,
                    alpha=0.8
                )

    # x軸フォーマット（最後のサブプロットのみ）
    format_time_axis(axes[-1], time_sec, unit='minutes')

    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig
