"""
心拍変動（HRV）可視化モジュール
"""

from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
from scipy import signal

from ..hrv import HRVResult
from ....visualization.utils import (
    format_time_axis,
    power_to_db,
    apply_frequency_band_shading,
    style_frequency_plot
)


# HRV周波数帯域定義（NeuroKit2準拠）
HRV_FREQ_BANDS = {
    'VLF': (0.0, 0.04, 'purple'),   # Very Low Frequency
    'LF': (0.04, 0.15, 'blue'),     # Low Frequency（交感神経＋副交感神経）
    'HF': (0.15, 0.4, 'green'),     # High Frequency（副交感神経）
}


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


def plot_hrv_frequency(
    hrv_data: Dict,
    hrv_indices: Optional[pd.DataFrame] = None,
    img_path: Optional[Union[str, Path]] = None,
    freq_max: float = 0.5
) -> plt.Figure:
    """
    EEG PSDスタイルでHRV周波数解析をプロット

    デザイン仕様:
    - 既存のEEG PSD可視化と同じスタイル（14x6インチ、dBスケール）
    - 周波数帯域の色分け（VLF/LF/HF）
    - ピークマーカー（LF/HFピーク）
    - HRV指標をテキスト表示

    Parameters
    ----------
    hrv_data : dict
        HRVデータ（rr_intervals, sampling_rate）
        get_hrv_data()の戻り値
    hrv_indices : pd.DataFrame, optional
        NeuroKit2のHRV解析結果（analyze_hrv()の戻り値）
        指定されない場合は内部で計算
    img_path : str or Path, optional
        保存先パス
    freq_max : float, default=0.5
        表示する最大周波数（Hz）

    Returns
    -------
    fig : matplotlib.figure.Figure
        生成された図

    Examples
    --------
    >>> from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
    >>> from lib.sensors.ecg.analysis import analyze_hrv
    >>> df = load_selfloops_csv('data.csv')
    >>> hrv_data = get_hrv_data(df)
    >>> hrv_indices = analyze_hrv(hrv_data)
    >>> plot_hrv_frequency(hrv_data, hrv_indices, 'hrv_frequency.png')
    """
    # HRV指標が指定されていない場合は計算
    if hrv_indices is None:
        from ..analysis import analyze_hrv
        hrv_indices = analyze_hrv(hrv_data, show=False)

    # RR間隔の時系列を等間隔に補間
    rr_ms = hrv_data['rr_intervals']
    rr_times = np.cumsum(rr_ms) / 1000.0  # msからsに変換
    rr_times = np.insert(rr_times, 0, 0)
    rr_values = np.append(rr_ms, rr_ms[-1])

    # 4Hzで補間（NeuroKit2デフォルト）
    sampling_rate = 4.0
    interpolated_times = np.arange(0, rr_times[-1], 1.0 / sampling_rate)
    interpolated_rr = np.interp(interpolated_times, rr_times, rr_values)

    # Welch法でPSD計算
    freqs, psd = signal.welch(
        interpolated_rr,
        fs=sampling_rate,
        nperseg=min(len(interpolated_rr), 256),
        window='hann'
    )

    # dBスケールに変換（共通関数を使用）
    psd_db = power_to_db(psd)

    # プロット（EEG PSDスタイル）
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # 周波数範囲をマスク
    mask = (freqs >= 0.0) & (freqs <= freq_max)

    # メインのPSDライン
    ax.plot(freqs[mask], psd_db[mask], 'b-', linewidth=2.0, alpha=0.9, label='HRV PSD')

    # 周波数帯域を色分け（共通関数を使用）
    apply_frequency_band_shading(ax, HRV_FREQ_BANDS, freq_max, alpha=0.12)

    # HRV指標から値を取得
    lf_power = hrv_indices['HRV_LF'].values[0]
    hf_power = hrv_indices['HRV_HF'].values[0]
    lf_hf_ratio = hrv_indices['HRV_LFHF'].values[0]

    # ピーク周波数を推定してマーク
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)

    if np.any(lf_mask):
        lf_peak_idx = np.argmax(psd[lf_mask])
        lf_peak_freq = freqs[lf_mask][lf_peak_idx]
        lf_peak_power = psd_db[lf_mask][lf_peak_idx]
        ax.scatter([lf_peak_freq], [lf_peak_power], color='blue', s=150,
                  marker='o', zorder=5, edgecolors='white', linewidths=2, label='LF Peak')

    if np.any(hf_mask):
        hf_peak_idx = np.argmax(psd[hf_mask])
        hf_peak_freq = freqs[hf_mask][hf_peak_idx]
        hf_peak_power = psd_db[hf_mask][hf_peak_idx]
        ax.scatter([hf_peak_freq], [hf_peak_power], color='green', s=150,
                  marker='o', zorder=5, edgecolors='white', linewidths=2, label='HF Peak')

    # 軸スタイル設定（共通関数を使用）
    style_frequency_plot(ax, freq_max, title='HRV Frequency Analysis')

    # テキスト注釈（HRV指標）
    text_str = f'LF Power: {lf_power:.2f} ms²\nHF Power: {hf_power:.2f} ms²\nLF/HF: {lf_hf_ratio:.2f}'
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    if img_path:
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def plot_hrv_nonlinear(
    hrv_data: Dict,
    result: Optional[HRVResult] = None,
    img_path: Optional[Union[str, Path]] = None
) -> plt.Figure:
    """
    HRV非線形解析をプロット（Poincaré + DFA）

    左: Poincaréプロット（SD1/SD2の可視化）
    右: DFAプロット（スケーリング指数の可視化）

    Parameters
    ----------
    hrv_data : dict
        HRVデータ（rr_intervals_clean等）
        get_hrv_data()の戻り値
    result : HRVResult, optional
        HRV解析結果（統計値の表示用）
    img_path : str or Path, optional
        保存先パス

    Returns
    -------
    fig : matplotlib.figure.Figure
        生成された図

    Examples
    --------
    >>> from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
    >>> df = load_selfloops_csv('data.csv')
    >>> hrv_data = get_hrv_data(df)
    >>> plot_hrv_nonlinear(hrv_data, img_path='hrv_nonlinear.png')
    """
    import neurokit2 as nk

    rr_intervals = hrv_data['rr_intervals_clean']
    sampling_rate = hrv_data.get('sampling_rate', 1000)

    # NeuroKit2でPeaksに変換
    peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=sampling_rate)

    # 非線形解析
    hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate, show=False)

    # 統計値を取得
    sd1 = hrv_nonlinear['HRV_SD1'].values[0]
    sd2 = hrv_nonlinear['HRV_SD2'].values[0]
    sd1_sd2 = hrv_nonlinear['HRV_SD1SD2'].values[0]
    dfa_alpha1 = hrv_nonlinear['HRV_DFA_alpha1'].values[0]
    dfa_alpha2 = hrv_nonlinear['HRV_DFA_alpha2'].values[0]

    # 図を作成（2列）
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ========================================
    # 左：Poincaréプロット
    # ========================================
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]

    ax1.scatter(rr_n, rr_n1, alpha=0.5, s=20, c='#1f77b4', edgecolors='none')

    # SD1/SD2楕円を描画
    mean_rr = np.mean(rr_intervals)
    angle = 45  # 45度回転

    # SD1楕円（短軸）
    ellipse_sd1 = Ellipse(
        (mean_rr, mean_rr),
        width=2 * sd1 * np.sqrt(2),
        height=2 * sd1 * np.sqrt(2),
        angle=angle,
        facecolor='none',
        edgecolor='red',
        linewidth=2,
        linestyle='--',
        label=f'SD1={sd1:.1f} ms'
    )
    ax1.add_patch(ellipse_sd1)

    # SD2楕円（長軸）
    ellipse_sd2 = Ellipse(
        (mean_rr, mean_rr),
        width=2 * sd2 * np.sqrt(2),
        height=2 * sd2 * np.sqrt(2),
        angle=angle,
        facecolor='none',
        edgecolor='green',
        linewidth=2,
        linestyle='--',
        label=f'SD2={sd2:.1f} ms'
    )
    ax1.add_patch(ellipse_sd2)

    # 対角線（y=x）
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),
        np.max([ax1.get_xlim(), ax1.get_ylim()]),
    ]
    ax1.plot(lims, lims, 'k--', alpha=0.3, zorder=0)

    ax1.set_xlabel('RR(n) [ms]', fontsize=12)
    ax1.set_ylabel('RR(n+1) [ms]', fontsize=12)
    ax1.set_title(f'Poincaré Plot (SD1/SD2={sd1_sd2:.3f})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.set_aspect('equal')

    # ========================================
    # 右：DFAプロット（スケール vs ゆらぎ）
    # ========================================
    # DFA計算を実行してスケールとゆらぎのデータを取得
    scales, fluctuations = _compute_dfa(rr_intervals)

    if len(scales) > 0 and len(fluctuations) > 0:
        # ログログプロット
        ax2.loglog(scales, fluctuations, 'o-', color='#1f77b4',
                   markersize=6, linewidth=2, alpha=0.7, label='DFA')

        # α1とα2の領域を分けて線形フィッティング
        # α1: 短期スケール（4-11拍）
        # α2: 長期スケール（>11拍）
        short_mask = (scales >= 4) & (scales <= 11)
        long_mask = scales > 11

        if np.sum(short_mask) >= 2:
            # α1のフィッティング線
            log_scales_short = np.log10(scales[short_mask])
            log_fluct_short = np.log10(fluctuations[short_mask])
            z1 = np.polyfit(log_scales_short, log_fluct_short, 1)
            p1 = np.poly1d(z1)
            ax2.loglog(scales[short_mask], 10**p1(log_scales_short),
                      'r--', linewidth=2.5, label=f'α1={dfa_alpha1:.3f}')

        if np.sum(long_mask) >= 2:
            # α2のフィッティング線
            log_scales_long = np.log10(scales[long_mask])
            log_fluct_long = np.log10(fluctuations[long_mask])
            z2 = np.polyfit(log_scales_long, log_fluct_long, 1)
            p2 = np.poly1d(z2)
            ax2.loglog(scales[long_mask], 10**p2(log_scales_long),
                      'g--', linewidth=2.5, label=f'α2={dfa_alpha2:.3f}')

        # 領域境界線
        ax2.axvline(11, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

        ax2.set_xlabel('Scale (beats)', fontsize=12)
        ax2.set_ylabel('Fluctuation (ms)', fontsize=12)
        ax2.set_title('DFA (Detrended Fluctuation Analysis)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, which='both', linestyle='--')
        ax2.legend(loc='lower right', fontsize=10)

        # 解釈テキスト
        if dfa_alpha1 < 0.5:
            interpretation = 'Anti-correlated'
            color = 'red'
        elif dfa_alpha1 < 0.6:
            interpretation = 'Random walk\n(Experienced meditator)'
            color = 'green'
        elif dfa_alpha1 < 0.9:
            interpretation = 'Moderate correlation'
            color = 'blue'
        else:
            interpretation = 'Strong correlation'
            color = 'orange'

        ax2.text(0.05, 0.95, interpretation, transform=ax2.transAxes,
                fontsize=10, verticalalignment='top', color=color,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 全体タイトル
    fig.suptitle('HRV Nonlinear Analysis', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if img_path:
        plt.savefig(img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    return fig


def _compute_dfa(rr_intervals, min_scale=4, max_scale=None, num_scales=20):
    """
    DFA（Detrended Fluctuation Analysis）を計算

    Parameters
    ----------
    rr_intervals : np.ndarray
        RR間隔（ms）
    min_scale : int
        最小スケール（拍数）
    max_scale : int, optional
        最大スケール（拍数）。Noneの場合はデータ長の1/4
    num_scales : int
        スケール数

    Returns
    -------
    scales : np.ndarray
        スケール配列
    fluctuations : np.ndarray
        各スケールでのゆらぎ
    """
    # RR間隔の累積和（積分）
    y = np.cumsum(rr_intervals - np.mean(rr_intervals))

    # 最大スケールの設定
    if max_scale is None:
        max_scale = len(rr_intervals) // 4

    # スケール配列（対数スケールで均等）
    scales = np.unique(np.logspace(
        np.log10(min_scale),
        np.log10(max_scale),
        num_scales
    ).astype(int))

    fluctuations = []

    for scale in scales:
        # データをウィンドウに分割
        n_windows = len(y) // scale
        if n_windows < 1:
            continue

        # 各ウィンドウでトレンド除去
        residuals = []
        for i in range(n_windows):
            start = i * scale
            end = (i + 1) * scale
            segment = y[start:end]

            # 線形トレンドをフィット
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)

            # デトレンド後の残差
            detrended = segment - trend
            residuals.extend(detrended)

        # ゆらぎ（RMS）を計算
        if len(residuals) > 0:
            fluctuation = np.sqrt(np.mean(np.array(residuals)**2))
            fluctuations.append(fluctuation)
        else:
            fluctuations.append(np.nan)

    # 有効な値のみ返す
    valid_mask = ~np.isnan(fluctuations)
    scales = scales[valid_mask]
    fluctuations = np.array(fluctuations)[valid_mask]

    return scales, fluctuations
