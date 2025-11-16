"""
統合ダッシュボード可視化モジュール
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_integrated_dashboard(fnirs_results, hr_data, respiratory_results, figsize=(18, 12)):
    """
    fNIRS、心拍数、呼吸数を統合したダッシュボード

    Parameters
    ----------
    fnirs_results : dict
        analyze_fnirs()の戻り値
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
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Row 1: fNIRS HbO (Left & Right)
    ax_left_hbo = fig.add_subplot(gs[0, 0])
    ax_right_hbo = fig.add_subplot(gs[0, 1])

    time_fnirs = fnirs_results['time']
    ax_left_hbo.plot(time_fnirs, fnirs_results['left_hbo'], 'r-', linewidth=1.0)
    ax_left_hbo.set_title('Left Hemisphere - HbO', fontweight='bold')
    ax_left_hbo.set_ylabel('Δ[HbO] (µM)')
    ax_left_hbo.grid(True, alpha=0.3)
    ax_left_hbo.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax_right_hbo.plot(time_fnirs, fnirs_results['right_hbo'], 'r-', linewidth=1.0)
    ax_right_hbo.set_title('Right Hemisphere - HbO', fontweight='bold')
    ax_right_hbo.set_ylabel('Δ[HbO] (µM)')
    ax_right_hbo.grid(True, alpha=0.3)
    ax_right_hbo.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Row 2: fNIRS HbR (Left & Right)
    ax_left_hbr = fig.add_subplot(gs[1, 0])
    ax_right_hbr = fig.add_subplot(gs[1, 1])

    ax_left_hbr.plot(time_fnirs, fnirs_results['left_hbr'], 'b-', linewidth=1.0)
    ax_left_hbr.set_title('Left Hemisphere - HbR', fontweight='bold')
    ax_left_hbr.set_ylabel('Δ[HbR] (µM)')
    ax_left_hbr.grid(True, alpha=0.3)
    ax_left_hbr.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    ax_right_hbr.plot(time_fnirs, fnirs_results['right_hbr'], 'b-', linewidth=1.0)
    ax_right_hbr.set_title('Right Hemisphere - HbR', fontweight='bold')
    ax_right_hbr.set_ylabel('Δ[HbR] (µM)')
    ax_right_hbr.grid(True, alpha=0.3)
    ax_right_hbr.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    # Row 3: Heart Rate & Respiratory Rate
    ax_hr = fig.add_subplot(gs[2, 0])
    ax_resp = fig.add_subplot(gs[2, 1])

    time_hr = respiratory_results['time']
    ax_hr.plot(time_hr, hr_data['heart_rate'], 'g-', linewidth=0.8, alpha=0.7)
    ax_hr.set_title('Heart Rate', fontweight='bold')
    ax_hr.set_xlabel('Time (seconds)')
    ax_hr.set_ylabel('Heart Rate (BPM)')
    ax_hr.grid(True, alpha=0.3)

    resp_rates = respiratory_results['respiratory_rates_fft']
    resp_timestamps = respiratory_results['respiratory_timestamps']
    if len(resp_rates) > 0:
        resp_time = time_hr[resp_timestamps]
        ax_resp.plot(resp_time, resp_rates, 'r-', marker='o', markersize=2, linewidth=1.0)
        ax_resp.set_title(f'Respiratory Rate (Mean: {np.mean(resp_rates):.1f} bpm)', fontweight='bold')
        ax_resp.set_xlabel('Time (seconds)')
        ax_resp.set_ylabel('Respiratory Rate (breaths/min)')
        ax_resp.grid(True, alpha=0.3)
        ax_resp.set_ylim([0, 30])

    fig.suptitle('Integrated Physiological Monitoring Dashboard', fontsize=18, fontweight='bold', y=0.995)
    return fig, gs
