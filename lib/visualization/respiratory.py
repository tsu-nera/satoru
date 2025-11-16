"""
呼吸・心拍可視化モジュール
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_respiratory(hr_data, respiratory_results, figsize=(15, 12)):
    """
    心拍数・HRV・呼吸数の時系列データを可視化

    Parameters
    ----------
    hr_data : dict
        get_heart_rate_data()の戻り値
    respiratory_results : dict
        analyze_respiratory()の戻り値
    figsize : tuple
        図のサイズ

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    fig.suptitle('Respiratory Rate Estimation', fontsize=16, fontweight='bold')

    time = respiratory_results['time']
    heart_rate = hr_data['heart_rate']
    rr_intervals = respiratory_results['rr_intervals']
    resp_rates = respiratory_results['respiratory_rates_fft']
    resp_timestamps = respiratory_results['respiratory_timestamps']

    # プロット1: 心拍数
    axes[0].plot(time, heart_rate, 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Heart Rate (BPM)')
    axes[0].set_title('Heart Rate Over Time')
    axes[0].grid(True, alpha=0.3)

    # プロット2: RRインターバル（心拍変動）
    axes[1].plot(time, rr_intervals, 'g-', alpha=0.7, linewidth=0.5)
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('RR Interval (ms)')
    axes[1].set_title('Heart Rate Variability (RR Intervals)')
    axes[1].grid(True, alpha=0.3)

    # プロット3: 推定呼吸数
    if len(resp_rates) > 0:
        resp_time = time[resp_timestamps]
        axes[2].plot(resp_time, resp_rates, 'r-', marker='o', markersize=3, linewidth=1.5)
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Respiratory Rate (breaths/min)')
        axes[2].set_title(f'Estimated Respiratory Rate (Mean: {np.mean(resp_rates):.1f} ± {np.std(resp_rates):.1f} breaths/min)')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 30])

    plt.tight_layout()
    return fig, axes


def plot_frequency_spectrum(respiratory_results, figsize=(12, 6)):
    """
    周波数スペクトルを可視化

    Parameters
    ----------
    respiratory_results : dict
        analyze_respiratory()の戻り値
    figsize : tuple
        図のサイズ

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    freqs = respiratory_results['freqs']
    psd = respiratory_results['psd']

    ax.plot(freqs, psd, 'b-', linewidth=1.5)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density')
    ax.set_title('Heart Rate Variability - Frequency Spectrum')
    ax.grid(True, alpha=0.3)

    # 呼吸域をハイライト
    ax.axvspan(0.15, 0.4, alpha=0.3, color='red', label='Respiratory Range (0.15-0.4 Hz)')
    ax.legend()
    ax.set_xlim([0, 0.5])

    plt.tight_layout()
    return fig, ax
