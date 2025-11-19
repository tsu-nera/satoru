"""
可視化用共通ユーティリティ
"""

import numpy as np


def format_time_axis(ax, times, unit='minutes'):
    """
    時間軸のフォーマットを統一的に設定

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        対象のAxes
    times : array-like
        時間データ（秒単位）
    unit : str, optional
        表示単位 ('minutes' or 'seconds')
    """
    if times is None or len(times) == 0:
        return

    start = float(times[0])
    end = float(times[-1])
    duration = max(end - start, np.finfo(float).eps)

    if unit == 'minutes':
        total_minutes = duration / 60.0
        if total_minutes <= 5:
            step = 1
        elif total_minutes <= 15:
            step = 2
        else:
            step = 5
        ticks_min = np.arange(0, total_minutes + step, step)
        ticks_sec = start + ticks_min * 60.0
        if len(ticks_sec) == 0:
            ticks_sec = np.array([start])
            ticks_min = np.array([0.0])
        ax.set_xticks(ticks_sec)
        tick_labels = [f'{tick:.0f}' if step >= 1 else f'{tick:.1f}' for tick in ticks_min]
        ax.set_xticklabels(tick_labels)
        ax.set_xlabel('Time (min)', fontsize=12)
    else:
        ax.set_xlabel('Time (seconds)', fontsize=12)

    ax.set_xlim(start, end)
