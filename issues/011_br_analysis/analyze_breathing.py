#!/usr/bin/env python3
"""
ECGデータから呼吸系指標を計算する分析スクリプト

NeuroKit2の組み込み機能（ECG-Derived Respiration）を使用して、
R-R間隔データから呼吸数や呼吸パターンを推定します。

Usage:
    python analyze_breathing.py

Functions:
    - nk.ecg_rsp(): ECG-Derived Respiration (EDR) 抽出
    - nk.rsp_clean(): 呼吸信号のクリーニング
    - nk.rsp_findpeaks(): 呼吸ピーク検出
    - nk.rsp_rate(): 呼吸数計算
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data


def calculate_windowed_metrics(hrv_data, hr_resampled, rsp_rate, time_resampled, target_fs, window_minutes=3):
    """
    3分ごとのウィンドウでHR, RMSSD, LF/HF, BRを計算

    Parameters
    ----------
    hrv_data : dict
        HRVデータ
    hr_resampled : np.ndarray
        リサンプリング済み心拍数
    rsp_rate : np.ndarray
        瞬時呼吸数
    time_resampled : np.ndarray
        リサンプリング済み時間軸（秒）
    target_fs : float
        サンプリング周波数
    window_minutes : int
        ウィンドウ幅（分）

    Returns
    -------
    pd.DataFrame
        時系列メトリクステーブル
    """
    window_sec = window_minutes * 60
    total_duration = time_resampled[-1]

    results = []

    # R-R間隔の元データ
    rr_intervals = hrv_data['rr_intervals_clean']
    rr_time = hrv_data['time']

    # ウィンドウごとに処理
    for start_time in np.arange(0, total_duration, window_sec):
        end_time = min(start_time + window_sec, total_duration)

        # タイムスタンプ（ウィンドウの中央）
        timestamp_min = (start_time + end_time) / 2 / 60

        # 1. 心拍数（リサンプリング済みデータから）
        hr_mask = (time_resampled >= start_time) & (time_resampled < end_time)
        hr_window = hr_resampled[hr_mask]
        hr_mean = np.mean(hr_window) if len(hr_window) > 0 else np.nan

        # 2. 呼吸数（リサンプリング済みデータから）
        br_window = rsp_rate[hr_mask]
        br_mean = np.nanmean(br_window) if len(br_window) > 0 else np.nan

        # 3. R-R間隔のウィンドウを取得（元データから）
        rr_mask = (rr_time >= start_time) & (rr_time < end_time)
        rr_window = rr_intervals[rr_mask]

        # RMSSD, LF/HFを計算（NeuroKit2使用）
        if len(rr_window) > 10:  # 最低限のデータ数
            try:
                # R-R間隔をピーク形式に変換
                peaks = nk.intervals_to_peaks(rr_window, sampling_rate=1000)

                # 時間領域HRV（RMSSD）
                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                rmssd = hrv_time['HRV_RMSSD'].values[0]

                # 周波数領域HRV（LF/HF, LF Power, HF Power）
                # normalize=Falseで実際のパワー値（ms^2）を取得
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False, normalize=False)
                lf_hf = hrv_freq['HRV_LFHF'].values[0]
                lf_power = hrv_freq['HRV_LF'].values[0]
                hf_power = hrv_freq['HRV_HF'].values[0]
            except Exception as e:
                # データが不足している場合はNaN
                rmssd = np.nan
                lf_hf = np.nan
                lf_power = np.nan
                hf_power = np.nan
        else:
            rmssd = np.nan
            lf_hf = np.nan
            lf_power = np.nan
            hf_power = np.nan

        results.append({
            'Time (min)': timestamp_min,
            'HR (bpm)': hr_mean,
            'RMSSD (ms)': rmssd,
            'LF/HF': lf_hf,
            'LF Power (ms^2)': lf_power,
            'HF Power (ms^2)': hf_power,
            'BR (bpm)': br_mean
        })

    return pd.DataFrame(results)


def analyze_breathing_hrv_correlation(metrics_df):
    """
    呼吸数とHRV振幅の相関を分析し、最適呼吸数範囲を推定

    Parameters
    ----------
    metrics_df : pd.DataFrame
        時系列メトリクステーブル（BR, RMSSD, LF Power含む）

    Returns
    -------
    dict
        分析結果（ビン統計、最適範囲など）
    """
    # NaNを除外
    df_clean = metrics_df.dropna(subset=['BR (bpm)', 'RMSSD (ms)', 'LF Power (ms^2)'])

    if len(df_clean) == 0:
        return None

    # 呼吸数をビン分け（0.5 bpm刻み）
    br_min = max(3.0, np.floor(df_clean['BR (bpm)'].min() * 2) / 2)
    br_max = min(10.0, np.ceil(df_clean['BR (bpm)'].max() * 2) / 2)
    br_bins = np.arange(br_min, br_max + 0.5, 0.5)

    # 各ビンでHRV指標の平均を計算
    bin_stats = []
    for i in range(len(br_bins) - 1):
        bin_start = br_bins[i]
        bin_end = br_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        mask = (df_clean['BR (bpm)'] >= bin_start) & (df_clean['BR (bpm)'] < bin_end)
        bin_data = df_clean[mask]

        if len(bin_data) > 0:
            bin_stats.append({
                'BR Range': f'{bin_start:.1f}-{bin_end:.1f}',
                'BR Center (bpm)': bin_center,
                'Count': len(bin_data),
                'RMSSD Mean (ms)': bin_data['RMSSD (ms)'].mean(),
                'RMSSD Std (ms)': bin_data['RMSSD (ms)'].std(),
                'LF Power Mean (ms^2)': bin_data['LF Power (ms^2)'].mean(),
                'LF Power Std (ms^2)': bin_data['LF Power (ms^2)'].std(),
            })

    bin_stats_df = pd.DataFrame(bin_stats)

    if len(bin_stats_df) == 0:
        return None

    # 最適呼吸数範囲を特定（RMSSD最大、LF Power最大）
    optimal_rmssd_idx = bin_stats_df['RMSSD Mean (ms)'].idxmax()
    optimal_lf_idx = bin_stats_df['LF Power Mean (ms^2)'].idxmax()

    optimal_rmssd_range = bin_stats_df.loc[optimal_rmssd_idx, 'BR Range']
    optimal_rmssd_value = bin_stats_df.loc[optimal_rmssd_idx, 'RMSSD Mean (ms)']

    optimal_lf_range = bin_stats_df.loc[optimal_lf_idx, 'BR Range']
    optimal_lf_value = bin_stats_df.loc[optimal_lf_idx, 'LF Power Mean (ms^2)']

    return {
        'bin_stats': bin_stats_df,
        'raw_data': df_clean,
        'optimal_rmssd': {
            'range': optimal_rmssd_range,
            'value': optimal_rmssd_value,
            'center': bin_stats_df.loc[optimal_rmssd_idx, 'BR Center (bpm)']
        },
        'optimal_lf': {
            'range': optimal_lf_range,
            'value': optimal_lf_value,
            'center': bin_stats_df.loc[optimal_lf_idx, 'BR Center (bpm)']
        }
    }


def plot_breathing_hrv_correlation(correlation_results, output_dir):
    """
    呼吸数とHRV振幅の相関を可視化

    Parameters
    ----------
    correlation_results : dict
        analyze_breathing_hrv_correlation()の結果
    output_dir : Path
        出力ディレクトリ
    """
    if correlation_results is None:
        print('相関分析データが不足しています')
        return

    bin_stats = correlation_results['bin_stats']
    raw_data = correlation_results['raw_data']
    optimal_rmssd = correlation_results['optimal_rmssd']
    optimal_lf = correlation_results['optimal_lf']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 散布図: BR vs RMSSD
    ax = axes[0, 0]
    ax.scatter(raw_data['BR (bpm)'], raw_data['RMSSD (ms)'],
              alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
    ax.axvline(optimal_rmssd['center'], color='red', linestyle='--', linewidth=2,
              label=f'Optimal: {optimal_rmssd["center"]:.1f} bpm')
    ax.set_xlabel('Breathing Rate (bpm)', fontsize=12)
    ax.set_ylabel('RMSSD (ms)', fontsize=12)
    ax.set_title('Breathing Rate vs RMSSD (Scatter)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (2) 散布図: BR vs LF Power
    ax = axes[0, 1]
    ax.scatter(raw_data['BR (bpm)'], raw_data['LF Power (ms^2)'],
              alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
    ax.axvline(optimal_lf['center'], color='red', linestyle='--', linewidth=2,
              label=f'Optimal: {optimal_lf["center"]:.1f} bpm')
    ax.set_xlabel('Breathing Rate (bpm)', fontsize=12)
    ax.set_ylabel('LF Power (ms²)', fontsize=12)
    ax.set_title('Breathing Rate vs LF Power (Scatter)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (3) ビン分析: BR vs RMSSD
    ax = axes[1, 0]
    x_pos = np.arange(len(bin_stats))
    ax.bar(x_pos, bin_stats['RMSSD Mean (ms)'],
          yerr=bin_stats['RMSSD Std (ms)'],
          alpha=0.7, color='blue', edgecolor='black', linewidth=1.2,
          capsize=5, error_kw={'linewidth': 1.5})

    # 最適範囲をハイライト
    optimal_idx = bin_stats['BR Range'] == optimal_rmssd['range']
    ax.bar(x_pos[optimal_idx], bin_stats.loc[optimal_idx, 'RMSSD Mean (ms)'],
          alpha=0.9, color='red', edgecolor='darkred', linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats['BR Range'], rotation=45, ha='right')
    ax.set_xlabel('Breathing Rate Range (bpm)', fontsize=12)
    ax.set_ylabel('RMSSD Mean (ms)', fontsize=12)
    ax.set_title('RMSSD by Breathing Rate Bins', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # (4) ビン分析: BR vs LF Power
    ax = axes[1, 1]
    ax.bar(x_pos, bin_stats['LF Power Mean (ms^2)'],
          yerr=bin_stats['LF Power Std (ms^2)'],
          alpha=0.7, color='green', edgecolor='black', linewidth=1.2,
          capsize=5, error_kw={'linewidth': 1.5})

    # 最適範囲をハイライト
    optimal_idx = bin_stats['BR Range'] == optimal_lf['range']
    ax.bar(x_pos[optimal_idx], bin_stats.loc[optimal_idx, 'LF Power Mean (ms^2)'],
          alpha=0.9, color='red', edgecolor='darkred', linewidth=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats['BR Range'], rotation=45, ha='right')
    ax.set_xlabel('Breathing Rate Range (bpm)', fontsize=12)
    ax.set_ylabel('LF Power Mean (ms²)', fontsize=12)
    ax.set_title('LF Power by Breathing Rate Bins', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'breathing_hrv_correlation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ 相関プロット保存: {output_path}')


def analyze_breathing_neurokit2(csv_path, output_dir):
    """
    NeuroKit2を使用した呼吸分析

    Parameters
    ----------
    csv_path : str
        SelfLoops HRV CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    """
    print('='*60)
    print('ECG呼吸分析（NeuroKit2使用）')
    print('='*60)
    print()

    # 1. データ読み込み
    print(f'Loading: {Path(csv_path).name}')
    df = load_selfloops_csv(csv_path, warmup_seconds=0)
    hrv_data = get_hrv_data(df, use_device_hr=False, clean_artifacts=True)
    print(f'測定時間: {hrv_data["time"][-1]/60:.2f} 分')
    print()

    # 2. R-R間隔を等間隔にリサンプリング
    print('R-R間隔をリサンプリング中...')
    rr_intervals = hrv_data['rr_intervals_clean']

    # R-R間隔から心拍数を計算
    hr_signal = 60000.0 / rr_intervals  # bpm

    # 等間隔にリサンプリング（8Hzに - NeuroKit2のフィルタに対応）
    from scipy import interpolate
    time_original = np.cumsum(rr_intervals) / 1000  # 秒
    target_fs = 8.0
    time_resampled = np.arange(0, time_original[-1], 1.0/target_fs)

    # 線形補間
    f = interpolate.interp1d(time_original, hr_signal, kind='cubic', fill_value='extrapolate')
    hr_resampled = f(time_resampled)

    print(f'リサンプリング後の信号長: {len(hr_resampled)} サンプル ({len(hr_resampled)/target_fs/60:.2f} 分)')
    print()

    # 3. ECG-Derived Respiration (EDR)を抽出
    print('EDR（ECG由来呼吸信号）を抽出中...')
    edr_signal = nk.ecg_rsp(hr_resampled, sampling_rate=int(target_fs), method='vangent2019')
    print(f'EDR信号長: {len(edr_signal)} サンプル')
    print()

    # 4. 呼吸信号のクリーニングとピーク検出
    print('呼吸ピークを検出中...')
    # EDR信号を呼吸信号として扱う
    rsp_cleaned = nk.rsp_clean(edr_signal, sampling_rate=target_fs)

    # 深い瞑想呼吸（4-6 bpm = 10-15秒/呼吸）に対応するため、
    # scipyメソッドでpeak_distanceを長く設定
    rsp_peaks_dict = nk.rsp_findpeaks(
        rsp_cleaned,
        sampling_rate=target_fs,
        method='scipy',        # scipyメソッドを使用
        peak_distance=8.0      # 8秒以上離れたピークのみ検出（7.5 bpm以下に対応）
    )

    # ピークとトラフ（谷）を取得
    peaks_idx = rsp_peaks_dict['RSP_Peaks']
    troughs_idx = rsp_peaks_dict['RSP_Troughs']

    print(f'検出されたピーク数: {len(peaks_idx)}')
    print(f'検出されたトラフ数: {len(troughs_idx)}')
    print()

    # 5. 呼吸数を計算
    print('呼吸数を計算中...')
    rsp_rate = nk.rsp_rate(
        rsp_cleaned,
        troughs=rsp_peaks_dict,
        sampling_rate=target_fs,
        method='trough'
    )

    # 呼吸数の統計
    mean_rate = np.nanmean(rsp_rate)
    std_rate = np.nanstd(rsp_rate)

    print(f'平均呼吸数: {mean_rate:.1f} ± {std_rate:.1f} bpm')
    print()

    # 6. 周波数解析
    print('周波数解析中...')
    # Welch法でパワースペクトル密度を計算
    from scipy import signal as sp_signal
    freqs, psd = sp_signal.welch(
        edr_signal,
        fs=target_fs,
        nperseg=min(256, len(edr_signal)),
        scaling='density'
    )

    # 呼吸帯域（0.15-0.4 Hz = 9-24 bpm）でピーク検出
    br_mask = (freqs >= 0.15) & (freqs <= 0.4)
    br_freqs = freqs[br_mask]
    br_psd = psd[br_mask]

    if len(br_psd) > 0:
        peak_idx = np.argmax(br_psd)
        peak_freq = br_freqs[peak_idx]
        breathing_rate_spectral = peak_freq * 60
        print(f'スペクトル法による呼吸数: {breathing_rate_spectral:.1f} bpm')
    print()

    # 6.5. 時系列メトリクス（3分ごと）
    print('時系列メトリクスを計算中（3分ごと）...')
    metrics_df = calculate_windowed_metrics(
        hrv_data,
        hr_resampled,
        rsp_rate,
        time_resampled,
        target_fs,
        window_minutes=3
    )
    print(f'計算されたウィンドウ数: {len(metrics_df)}')
    print()

    # 6.6. 呼吸数とHRV振幅の相関分析
    print('呼吸数とHRV振幅の相関を分析中...')
    correlation_results = analyze_breathing_hrv_correlation(metrics_df)

    if correlation_results is not None:
        print(f'最適呼吸数範囲（RMSSD最大）: {correlation_results["optimal_rmssd"]["range"]} bpm')
        print(f'最適呼吸数範囲（LF Power最大）: {correlation_results["optimal_lf"]["range"]} bpm')
    else:
        print('相関分析に十分なデータがありません')
    print()

    # 6.7. 相関プロット生成
    if correlation_results is not None:
        plot_breathing_hrv_correlation(correlation_results, output_dir)
        print()

    # 7. 可視化
    print('プロットを生成中...')
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    # 時間軸（分）
    time_min = time_resampled / 60

    # (1) 心拍数
    axes[0].plot(time_min, hr_resampled, linewidth=0.8, color='red')
    axes[0].set_ylabel('Heart Rate (bpm)')
    axes[0].set_title('Heart Rate Signal')
    axes[0].grid(True, alpha=0.3)

    # (2) EDR信号（呼吸成分）
    axes[1].plot(time_min, edr_signal, linewidth=0.8, color='blue')
    axes[1].plot(time_min[peaks_idx], edr_signal[peaks_idx],
                'ro', markersize=3, label=f'Peaks (n={len(peaks_idx)})')
    axes[1].plot(time_min[troughs_idx], edr_signal[troughs_idx],
                'go', markersize=3, label=f'Troughs (n={len(troughs_idx)})')
    axes[1].set_ylabel('EDR Signal (a.u.)')
    axes[1].set_title('ECG-Derived Respiration (EDR) - NeuroKit2')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # (3) 瞬時呼吸数
    axes[2].plot(time_min, rsp_rate, linewidth=0.8, color='green')
    axes[2].axhline(mean_rate, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_rate:.1f} bpm')
    axes[2].fill_between(time_min, mean_rate - std_rate, mean_rate + std_rate,
                        alpha=0.2, color='red', label=f'±1 SD: {std_rate:.1f} bpm')
    axes[2].set_ylabel('Breathing Rate (bpm)')
    axes[2].set_title('Instantaneous Breathing Rate')
    axes[2].set_ylim([0, 30])
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    # (4) パワースペクトル
    axes[3].plot(freqs, psd, linewidth=1.5, color='navy')
    axes[3].axvline(peak_freq, color='red', linestyle='--', linewidth=2,
                   label=f'Peak: {peak_freq:.3f} Hz ({breathing_rate_spectral:.1f} bpm)')
    axes[3].axvspan(0.15, 0.4, alpha=0.2, color='cyan', label='Breathing Range (0.15-0.4 Hz)')
    axes[3].set_xlabel('Frequency (Hz)')
    axes[3].set_ylabel('Power Spectral Density')
    axes[3].set_title('Respiratory Power Spectrum')
    axes[3].set_xlim([0, 0.5])
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'breathing_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ プロット保存: {output_path}')
    print()

    # 8. レポート生成
    # マークダウンテーブルを生成（手動）
    def df_to_markdown(df):
        """DataFrameをマークダウンテーブルに変換"""
        lines = []

        # ヘッダー
        header = '| ' + ' | '.join(df.columns) + ' |'
        lines.append(header)

        # セパレータ
        separator = '|' + '|'.join([':-----:' for _ in df.columns]) + '|'
        lines.append(separator)

        # データ行
        for _, row in df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('-')
                elif isinstance(val, float):
                    values.append(f'{val:.1f}')
                else:
                    values.append(str(val))
            line = '| ' + ' | '.join(values) + ' |'
            lines.append(line)

        return '\n'.join(lines)

    metrics_table = df_to_markdown(metrics_df)

    # 推奨呼吸数セクションの生成
    if correlation_results is not None:
        optimal_section = f"""
## 推奨呼吸数（共鳴周波数推定）

このセクションでは、30分の瞑想データから呼吸数とHRV振幅の関係を分析し、
HRV振幅が最大化される呼吸数範囲を推定しています。

### RMSSD基準（副交感神経活動）
- **推奨範囲**: {correlation_results['optimal_rmssd']['range']} bpm
- **中心値**: {correlation_results['optimal_rmssd']['center']:.1f} bpm
- **平均RMSSD**: {correlation_results['optimal_rmssd']['value']:.1f} ms

### LF Power基準（心臓血管系共鳴）
- **推奨範囲**: {correlation_results['optimal_lf']['range']} bpm
- **中心値**: {correlation_results['optimal_lf']['center']:.1f} bpm
- **平均LF Power**: {correlation_results['optimal_lf']['value']:.1f} ms²

**注意事項:**
- これらは自然な瞑想データからの探索的推定値です
- より正確な共鳴周波数を知るには、4.5-6.5 bpmの範囲で制御された呼吸測定が必要です
- 個人の共鳴周波数は日によって変動する可能性があります（平均心拍数との関連）

### 相関分析

![呼吸数とHRV振幅の相関](breathing_hrv_correlation.png)
"""
    else:
        optimal_section = """
## 推奨呼吸数（共鳴周波数推定）

相関分析に十分なデータがありません。
"""

    report = f"""# ECG呼吸分析レポート

## 概要

NeuroKit2を使用してECGのR-R間隔データから呼吸パターンを推定しました。
ECG-Derived Respiration (EDR) 法による分析です。

## 主要指標

| 指標 | 値 |
|:-----|---:|
| **測定時間** | {hrv_data['time'][-1]/60:.2f} 分 |
| **平均呼吸数（瞬時）** | {mean_rate:.1f} ± {std_rate:.1f} bpm |
| **スペクトル法による呼吸数** | {breathing_rate_spectral:.1f} bpm |
| **検出されたピーク数** | {len(peaks_idx)} |
| **検出されたトラフ数** | {len(troughs_idx)} |

## 時系列変化（3分ごと）

{metrics_table}

**指標の説明:**
- **Time (min)**: ウィンドウの中央時刻（分）
- **HR (bpm)**: 心拍数（beats per minute）
- **RMSSD (ms)**: 連続R-R間隔差の二乗平均平方根（副交感神経活動の指標）
- **LF/HF**: 低周波/高周波比（自律神経バランスの指標）
- **LF Power (ms²)**: 低周波パワー（0.04-0.15 Hz、交感神経・副交感神経活動）
- **HF Power (ms²)**: 高周波パワー（0.15-0.4 Hz、副交感神経活動・呼吸の影響）
- **BR (bpm)**: 呼吸数（breaths per minute）

{optimal_section}

## 使用した関数

- `nk.ecg_rsp()`: ECG-Derived Respiration（EDR）抽出
- `nk.rsp_clean()`: 呼吸信号のクリーニング
- `nk.rsp_findpeaks()`: 呼吸ピーク検出
- `nk.rsp_rate()`: 呼吸数計算

## 可視化

![呼吸分析結果](breathing_analysis.png)

---

生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report_path = output_dir / 'BREATHING_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ レポート保存: {report_path}')
    print()

    print('='*60)
    print('分析完了!')
    print('='*60)


if __name__ == '__main__':
    csv_path = 'data/selfloops/selfloops_2026-01-12--06-21-05.csv'
    output_dir = Path(__file__).parent

    analyze_breathing_neurokit2(csv_path, output_dir)
