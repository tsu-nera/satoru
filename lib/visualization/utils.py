"""
可視化用共通ユーティリティ
"""

from typing import Dict, Tuple, Optional

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


def power_to_db(power: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    パワーをdBスケールに変換（EEG/HRV周波数解析で共通）

    Parameters
    ----------
    power : np.ndarray
        パワースペクトル密度
    epsilon : float, default=1e-10
        ゼロ除算を防ぐための微小値

    Returns
    -------
    np.ndarray
        dBスケールに変換されたパワー

    Examples
    --------
    >>> psd = np.array([0.1, 1.0, 10.0])
    >>> power_to_db(psd)
    array([-10.,   0.,  10.])
    """
    return 10 * np.log10(power + epsilon)


def apply_frequency_band_shading(
    ax,
    freq_bands: Dict[str, Tuple[float, float, str]],
    freq_max: float,
    alpha: float = 0.12,
    add_labels: bool = True
):
    """
    周波数プロットに帯域の背景色を適用（EEG/HRV共通スタイル）

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        対象のAxes
    freq_bands : dict
        周波数帯域定義 {band_name: (low, high, color)}
        例: {'Alpha': (8, 13, 'green'), 'Beta': (13, 30, 'orange')}
    freq_max : float
        表示する最大周波数
    alpha : float, default=0.12
        背景色の透明度
    add_labels : bool, default=True
        凡例ラベルを追加するか

    Examples
    --------
    >>> freq_bands = {'Alpha': (8, 13, 'green'), 'Beta': (13, 30, 'orange')}
    >>> apply_frequency_band_shading(ax, freq_bands, freq_max=35)
    """
    for band_name, (low, high, color) in freq_bands.items():
        if low < freq_max:
            label = f'{band_name} ({low}-{high}Hz)' if add_labels else None
            ax.axvspan(
                low,
                min(high, freq_max),
                alpha=alpha,
                color=color,
                label=label
            )


def style_frequency_plot(
    ax,
    freq_max: float,
    ylabel: str = 'Power Spectral Density (dB)',
    title: Optional[str] = None,
    xlabel: str = 'Frequency (Hz)',
    grid_alpha: float = 0.3
):
    """
    周波数プロットの軸スタイルを統一的に設定（EEG/HRV共通）

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        対象のAxes
    freq_max : float
        表示する最大周波数
    ylabel : str, default='Power Spectral Density (dB)'
        Y軸ラベル
    title : str, optional
        グラフタイトル
    xlabel : str, default='Frequency (Hz)'
        X軸ラベル
    grid_alpha : float, default=0.3
        グリッドの透明度

    Examples
    --------
    >>> style_frequency_plot(ax, freq_max=35, title='EEG Power Spectrum')
    """
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=grid_alpha)
    ax.set_xlim(0, freq_max)
