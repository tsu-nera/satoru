"""
心拍変動（HRV）可視化モジュール
"""

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..hrv import HRVResult
from ....visualization.utils import format_time_axis


def plot_hrv_time_series(
    result: HRVResult,
    img_path: Optional[str] = None,
    title: Optional[str] = None,
    hr_data: Optional[dict] = None,
) -> Tuple[Dict[str, pd.Series], object]:
    """
    HRV時系列（RMSSD、LF/HF ratio、心拍数）をプロット

    デザイン仕様:
    - 2段または3段プロット（上: RMSSD、中: LF/HF ratio、下: Heart Rate（オプション））
    - RMSSD色: 心臓の赤 #d62728
    - LF/HF色: 紫 #9467bd
    - Heart Rate色: 緑 #2ca02c
    - セッション中点: 縦破線（グレー）
    - タイトル: 14pt bold
    - 軸ラベル: 12pt
    - メタデータボックス: 左上に表示

    Parameters
    ----------
    result : HRVResult
        calculate_hrv_standard_set()の戻り値
    img_path : str, optional
        画像保存パス
    title : str, optional
        グラフタイトル（デフォルト: "HRV Time Series Analysis"）
    hr_data : dict, optional
        心拍数データ（get_heart_rate_data()の戻り値）

    Returns
    -------
    (time_series_dict, fig)
        表示に使用した時系列とFigureオブジェクト

    Examples
    --------
    >>> hrv_result = calculate_hrv_standard_set(hrv_data)
    >>> plot_hrv_time_series(
    ...     hrv_result,
    ...     img_path='output/hrv_time_series.png',
    ...     title='Meditation Session - HRV Analysis'
    ... )
    """
    rmssd_series = result.time_series['rmssd']
    lfhf_series = result.time_series['lfhf_ratio']
    metadata = result.metadata

    if rmssd_series.empty and lfhf_series.empty:
        raise ValueError('HRV time series for plotting is empty.')

    # 経過時間（秒）を計算
    rmssd_elapsed = (rmssd_series.index - rmssd_series.index[0]).total_seconds()
    lfhf_elapsed = (lfhf_series.index - lfhf_series.index[0]).total_seconds()

    # タイトル設定
    plot_title = title or 'HRV Time Series Analysis'

    # プロット数を決定（心拍数がある場合は3段、なければ2段）
    n_plots = 3 if hr_data is not None else 2
    fig_height = 12 if n_plots == 3 else 8

    # プロット作成
    fig, axes = plt.subplots(n_plots, 1, figsize=(14, fig_height), sharex=True)
    if n_plots == 2:
        ax1, ax2 = axes
    else:
        ax1, ax2, ax3 = axes

    # ========================================
    # 上段: RMSSD（副交感神経活動）
    # ========================================
    ax1.plot(
        rmssd_elapsed,
        rmssd_series.values,
        color='#d62728',  # 心臓の赤
        linewidth=2.5,
        label='RMSSD',
        alpha=0.9,
    )

    ax1.set_ylabel('RMSSD (ms)', fontsize=12)
    ax1.set_title('RMSSD - Parasympathetic Activity', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10)

    # セッション中点
    midpoint = rmssd_elapsed[len(rmssd_elapsed) // 2] if len(rmssd_elapsed) else 0
    if midpoint:
        ax1.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='Session midpoint')

    # ========================================
    # 下段: LF/HF ratio（自律神経バランス）
    # ========================================
    ax2.plot(
        lfhf_elapsed,
        lfhf_series.values,
        color='#9467bd',  # 紫
        linewidth=2.5,
        label='LF/HF Ratio',
        alpha=0.9,
    )

    # 参照線: y=1.0（副交感神経優位/交感神経優位の境界）
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.7, linewidth=1.5, label='Threshold (LF/HF=1.0)')

    ax2.set_ylabel('LF/HF Ratio', fontsize=12)
    ax2.set_title('LF/HF Ratio - Autonomic Balance', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10)

    # セッション中点
    if midpoint:
        ax2.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # ========================================
    # 下段: 心拍数（オプション）
    # ========================================
    if hr_data is not None:
        # セッション開始時刻を取得
        start_time = rmssd_series.index[0]

        # 心拍数データを処理
        hr_time = hr_data['time']
        heart_rate = hr_data['heart_rate']
        timestamps = hr_data.get('timestamps')

        if timestamps is not None and len(timestamps) > 0:
            # タイムスタンプベースで集計
            hr_df = pd.DataFrame({
                'timestamp': timestamps,
                'heart_rate': heart_rate
            })
            hr_df['time_sec'] = (hr_df['timestamp'] - start_time).dt.total_seconds()

            hr_time_sec = hr_df['time_sec'].values
            hr_values = hr_df['heart_rate'].values
        else:
            # フォールバック: 時間ベースで集計
            hr_time_sec = hr_time
            hr_values = heart_rate

        ax3.plot(hr_time_sec, hr_values, color='#2ca02c', linewidth=2.5, alpha=0.9, label='Heart Rate')
        ax3.fill_between(hr_time_sec, 0, hr_values, alpha=0.2, color='#2ca02c')

        hr_mean = np.nanmean(hr_values)
        hr_std = np.nanstd(hr_values)
        ax3.set_ylabel('Heart Rate (BPM)', fontsize=12)
        ax3.set_title(f'Heart Rate (Mean: {hr_mean:.1f} ± {hr_std:.1f} BPM)', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.legend(loc='upper right', fontsize=10)

        # Y軸範囲を適切に設定
        valid_hr = hr_values[~np.isnan(hr_values)]
        if len(valid_hr) > 0:
            hr_min = max(40, np.percentile(valid_hr, 1) - 5)
            hr_max = min(150, np.percentile(valid_hr, 99) + 5)
            ax3.set_ylim(hr_min, hr_max)

        # セッション中点
        if midpoint:
            ax3.axvline(midpoint, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)

    # X軸フォーマット（分単位）
    last_ax = axes[-1] if n_plots > 2 else ax2
    format_time_axis(last_ax, rmssd_elapsed, unit='minutes')

    # ========================================
    # メタデータボックス（左上）
    # ========================================
    window_sec = metadata.get('window_seconds', 180)
    mean_rmssd = metadata.get('mean_rmssd', np.nan)
    mean_lfhf = metadata.get('mean_lfhf', np.nan)

    metadata_text = f'Window: {window_sec:.0f}s\n'
    metadata_text += f'Mean RMSSD: {mean_rmssd:.1f} ms\n'
    metadata_text += f'Mean LF/HF: {mean_lfhf:.2f}'

    ax1.text(
        0.02,
        0.95,
        metadata_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7),
    )

    # 全体タイトル
    fig.suptitle(plot_title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if img_path:
        fig.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return result.time_series, fig
