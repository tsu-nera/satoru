"""
fNIRS可視化モジュール
"""

import matplotlib.pyplot as plt
import numpy as np

from .utils import format_time_axis


def plot_fnirs(fnirs_results, figsize=(14, 10)):
    """
    fNIRS時系列データを4つのサブプロットで可視化

    Parameters
    ----------
    fnirs_results : dict
        analyze_fnirs()の戻り値
    figsize : tuple
        図のサイズ

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('fNIRS Analysis: HbO/HbR Concentration Changes', fontsize=16, fontweight='bold')

    time = fnirs_results['time']
    left_hbo = fnirs_results['left_hbo']
    left_hbr = fnirs_results['left_hbr']
    right_hbo = fnirs_results['right_hbo']
    right_hbr = fnirs_results['right_hbr']

    # Left HbO
    axes[0, 0].plot(time, left_hbo, color='red', linewidth=1.5)
    axes[0, 0].set_title('Left Hemisphere - HbO (Oxygenated)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Δ[HbO] (µM)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    format_time_axis(axes[0, 0], time, unit='minutes')

    # Left HbR
    axes[1, 0].plot(time, left_hbr, color='blue', linewidth=1.5)
    axes[1, 0].set_title('Left Hemisphere - HbR (Deoxygenated)', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Δ[HbR] (µM)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    format_time_axis(axes[1, 0], time, unit='minutes')

    # Right HbO
    axes[0, 1].plot(time, right_hbo, color='red', linewidth=1.5)
    axes[0, 1].set_title('Right Hemisphere - HbO (Oxygenated)', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Δ[HbO] (µM)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    format_time_axis(axes[0, 1], time, unit='minutes')

    # Right HbR
    axes[1, 1].plot(time, right_hbr, color='blue', linewidth=1.5)
    axes[1, 1].set_title('Right Hemisphere - HbR (Deoxygenated)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Δ[HbR] (µM)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    format_time_axis(axes[1, 1], time, unit='minutes')

    plt.tight_layout()
    return fig, axes


def plot_fnirs_muse_style(fnirs_results, figsize=(14, 8)):
    """
    fNIRS時系列データをMuse App風に可視化
    4つの折れ線を重ねて表示（Left HbR, Right HbR, Left HbO, Right HbO）hl

    Parameters
    ----------
    fnirs_results : dict
        analyze_fnirs()の戻り値
    figsize : tuple
        図のサイズ

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    time = fnirs_results['time']

    left_hbo = fnirs_results['left_hbo']
    left_hbr = fnirs_results['left_hbr']
    right_hbo = fnirs_results['right_hbo']
    right_hbr = fnirs_results['right_hbr']

    # Muse App風のカラー設定
    # HbO: 赤/オレンジ系、HbR: 青系
    ax.plot(time, left_hbr, color='#3B82F6', linewidth=1.2, label='Left HbR', alpha=0.9)
    ax.plot(time, right_hbr, color='#60A5FA', linewidth=1.2, label='Right HbR', alpha=0.9)
    ax.plot(time, left_hbo, color='#EF4444', linewidth=1.2, label='Left HbO', alpha=0.9)
    ax.plot(time, right_hbo, color='#F97316', linewidth=1.2, label='Right HbO', alpha=0.9)

    # 0ラインを追加
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    # グリッド
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)

    # ラベルとタイトル
    format_time_axis(ax, time, unit='minutes')
    ax.set_ylabel('Concentration Change (µM)', fontsize=12)
    ax.set_title('Brain Oxygenation (Muse App Style)', fontsize=14, fontweight='bold')

    # 凡例
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)

    # Y軸の範囲を適切に設定（NaNを無視）
    all_values = np.concatenate([left_hbo, left_hbr, right_hbo, right_hbr])
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size > 0:
        y_min = np.floor(np.min(finite_values))
        y_max = np.ceil(np.max(finite_values))
        if y_min == y_max:
            padding = max(abs(y_min) * 0.1, 1.0)
            y_min -= padding
            y_max += padding
        ax.set_ylim([y_min, y_max])

    plt.tight_layout()
    return fig, ax
