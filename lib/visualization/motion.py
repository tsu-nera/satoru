"""
動作検出可視化モジュール

加速度センサー・ジャイロスコープによる動作検出と心拍数の時系列を可視化します。
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import format_time_axis


def plot_motion_heart_rate(motion_result, hr_data=None, df=None, figsize=(15, 10)):
    """
    動作検出と心拍数の時系列データを可視化

    Parameters
    ----------
    motion_result : dict
        analyze_motion()の戻り値
    hr_data : dict, optional
        get_heart_rate_data()の戻り値
    df : pd.DataFrame, optional
        Mind Monitorデータフレーム（タイムスタンプ取得用）
    figsize : tuple
        図のサイズ

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray
    """
    motion_df = motion_result['motion_df']
    stats = motion_result['stats']
    motion_ratio = motion_result['motion_ratio']

    # セッション開始時刻を基準にした経過時間（秒）を計算
    start_time = motion_df['interval'].iloc[0]
    motion_df = motion_df.copy()
    motion_df['time_sec'] = (motion_df['interval'] - start_time).dt.total_seconds()

    n_plots = 3 if hr_data is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)

    # タイトル
    fig.suptitle(
        f'Motion Detection & Heart Rate (Motion Ratio: {motion_ratio*100:.1f}%)',
        fontsize=14, fontweight='bold'
    )

    # プロット1: 加速度（標準偏差）
    ax1 = axes[0]
    time_sec = motion_df['time_sec'].values
    acc_std = motion_df['acc_magnitude_std'].values

    ax1.fill_between(time_sec, 0, acc_std, alpha=0.5, color='steelblue')
    ax1.plot(time_sec, acc_std, 'b-', linewidth=1)

    # 動作検出閾値ライン
    ax1.axhline(y=0.02, color='red', linestyle='--', alpha=0.7, label='Threshold (0.02 g)')

    # 動作検出された区間をハイライト
    motion_mask = motion_df['is_motion'].values
    for i, is_motion in enumerate(motion_mask):
        if is_motion:
            ax1.axvspan(
                time_sec[i] - 5, time_sec[i] + 5,
                alpha=0.3, color='red'
            )

    ax1.set_ylabel('Accel Std (g)')
    ax1.set_title('Accelerometer (Linear Movement)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # プロット2: ジャイロスコープ（平均）
    ax2 = axes[1]
    gyro_mean = motion_df['gyro_magnitude_mean'].values

    ax2.fill_between(time_sec, 0, gyro_mean, alpha=0.5, color='darkorange')
    ax2.plot(time_sec, gyro_mean, color='darkorange', linewidth=1)

    # 動作検出閾値ライン
    ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Threshold (3.0 deg/s)')

    # 動作検出された区間をハイライト
    for i, is_motion in enumerate(motion_mask):
        if is_motion:
            ax2.axvspan(
                time_sec[i] - 5, time_sec[i] + 5,
                alpha=0.3, color='red'
            )

    ax2.set_ylabel('Gyro Mean (deg/s)')
    ax2.set_title('Gyroscope (Head Rotation)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # プロット3: 心拍数（オプション）
    if hr_data is not None:
        ax3 = axes[2]

        # 心拍数を10秒間隔で集計
        hr_time = hr_data['time']
        heart_rate = hr_data['heart_rate']
        timestamps = hr_data.get('timestamps')

        if timestamps is not None and len(timestamps) > 0:
            # タイムスタンプベースで10秒間隔に集計
            hr_df = pd.DataFrame({
                'timestamp': timestamps,
                'heart_rate': heart_rate
            })
            hr_df['interval'] = hr_df['timestamp'].dt.floor('10s')
            hr_grouped = hr_df.groupby('interval')['heart_rate'].mean().reset_index()
            hr_grouped['time_sec'] = (hr_grouped['interval'] - start_time).dt.total_seconds()

            hr_time_sec = hr_grouped['time_sec'].values
            hr_values = hr_grouped['heart_rate'].values
        else:
            # フォールバック: 時間ベースで集計
            hr_time_sec = hr_time
            hr_values = heart_rate

        ax3.plot(hr_time_sec, hr_values, 'g-', linewidth=1.5, alpha=0.8)
        ax3.fill_between(hr_time_sec, 0, hr_values, alpha=0.3, color='green')

        # 動作検出された区間をハイライト
        for i, is_motion in enumerate(motion_mask):
            if is_motion:
                ax3.axvspan(
                    time_sec[i] - 5, time_sec[i] + 5,
                    alpha=0.3, color='red'
                )

        hr_mean = np.nanmean(hr_values)
        hr_std = np.nanstd(hr_values)
        ax3.set_ylabel('Heart Rate (BPM)')
        ax3.set_title(f'Heart Rate (Mean: {hr_mean:.1f} ± {hr_std:.1f} BPM)')
        ax3.grid(True, alpha=0.3)

        # Y軸範囲を適切に設定
        valid_hr = hr_values[~np.isnan(hr_values)]
        if len(valid_hr) > 0:
            hr_min = max(40, np.percentile(valid_hr, 1) - 5)
            hr_max = min(150, np.percentile(valid_hr, 99) + 5)
            ax3.set_ylim(hr_min, hr_max)

    # 時間軸フォーマット
    format_time_axis(axes[-1], time_sec, unit='minutes')

    plt.tight_layout()
    return fig, axes


def create_motion_stats_table(motion_result, hr_data=None):
    """
    心拍数の統計情報をDataFrame化

    Parameters
    ----------
    motion_result : dict
        analyze_motion()の戻り値
    hr_data : dict, optional
        get_heart_rate_data()の戻り値

    Returns
    -------
    pd.DataFrame
        統計情報のDataFrame
    """
    if hr_data is None or len(hr_data.get('heart_rate', [])) == 0:
        return pd.DataFrame({'Metric': [], 'Value': []})

    heart_rate = hr_data['heart_rate']
    valid_hr = heart_rate[~np.isnan(heart_rate)]

    if len(valid_hr) == 0:
        return pd.DataFrame({'Metric': [], 'Value': []})

    data = {
        'Metric': [
            '心拍数（平均）',
            '心拍数（最小）',
            '心拍数（最大）',
        ],
        'Value': [
            f'{np.mean(valid_hr):.1f} BPM',
            f'{np.min(valid_hr):.1f} BPM',
            f'{np.max(valid_hr):.1f} BPM',
        ]
    }

    return pd.DataFrame(data)
