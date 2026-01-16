#!/usr/bin/env python3
"""
呼吸数（BR）の妥当性検証スクリプト

問題:
- 現在のSpectral BR (9.4 bpm) は実際の呼吸ではない可能性
- 固定帯域 0.15-0.4 Hz (9-24 bpm) が深い瞑想呼吸に適していない

検証:
1. 全周波数帯域でEDRパワースペクトルを分析
2. 異なる周波数範囲でのSpectral BR計算
3. ピーク検出法による呼吸数との比較
4. 実際の呼吸周期（15-30秒/回）との整合性確認
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import neurokit2 as nk

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data


def calculate_spectral_br_multiple_ranges(freqs, psd):
    """
    複数の周波数範囲でSpectral BRを計算

    Returns
    -------
    dict
        各範囲名をキーとした辞書
    """
    ranges = {
        'VLF-LF (1-10 bpm)': (1/60, 10/60),    # 0.017-0.167 Hz
        'LF帯域 (2-9 bpm)': (2/60, 9/60),      # 0.033-0.150 Hz
        '深呼吸 (2-6 bpm)': (2/60, 6/60),      # 0.033-0.100 Hz
        '通常呼吸 (9-24 bpm)': (9/60, 24/60),  # 0.150-0.400 Hz ← 現在の実装
        'HF帯域 (9-24 bpm)': (0.15, 0.4),      # 0.150-0.400 Hz
    }

    results = {}

    for name, (f_min, f_max) in ranges.items():
        mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(mask):
            range_freqs = freqs[mask]
            range_psd = psd[mask]

            if len(range_psd) > 0:
                peak_idx = np.argmax(range_psd)
                peak_freq = range_freqs[peak_idx]
                peak_power = range_psd[peak_idx]
                peak_bpm = peak_freq * 60
                period_sec = 1 / peak_freq

                results[name] = {
                    'freq_hz': peak_freq,
                    'bpm': peak_bpm,
                    'period_sec': period_sec,
                    'power': peak_power,
                    'range': (f_min, f_max)
                }

    return results


def analyze_breathing_validation(csv_path, output_dir):
    """
    呼吸数の妥当性を検証
    """
    print("=" * 70)
    print("呼吸数（BR）妥当性検証")
    print("=" * 70)
    print()

    # データ読み込み
    print(f"データ読み込み: {csv_path}")
    sl_df = load_selfloops_csv(csv_path, warmup_seconds=60.0)

    print(f"データ形状: {sl_df.shape}")
    print(f"測定時間: {sl_df['TimeStamp'].iloc[-1] - sl_df['TimeStamp'].iloc[0]}")
    print()

    # HRVデータ取得
    hrv_data = get_hrv_data(sl_df, clean_artifacts=True)
    rr_intervals = hrv_data['rr_intervals_clean']

    print(f"R-R間隔数: {len(rr_intervals)}")
    print(f"平均R-R間隔: {np.mean(rr_intervals):.1f} ms")
    print()

    # ========================================
    # 1. EDR抽出（NeuroKit2）
    # ========================================
    print("-" * 70)
    print("1. EDR（ECG-Derived Respiration）抽出")
    print("-" * 70)

    # 心拍数時系列を作成
    rr_times = np.cumsum(rr_intervals) / 1000.0
    rr_times = np.insert(rr_times, 0, 0)
    rr_values = np.append(rr_intervals, rr_intervals[-1])

    # リサンプリング
    target_fs = 8.0  # 8 Hz
    time_resampled = np.arange(0, rr_times[-1], 1.0 / target_fs)
    rr_interp = np.interp(time_resampled, rr_times, rr_values)
    hr_resampled = 60000 / rr_interp  # ms -> bpm

    print(f"リサンプリング周波数: {target_fs} Hz")
    print(f"リサンプリング後のサンプル数: {len(hr_resampled)}")

    # EDR抽出
    edr_signal = nk.ecg_rsp(hr_resampled, sampling_rate=target_fs)
    print(f"EDR信号長: {len(edr_signal)}")
    print()

    # ========================================
    # 2. ピーク検出法による呼吸数
    # ========================================
    print("-" * 70)
    print("2. ピーク検出法による呼吸数")
    print("-" * 70)

    # 呼吸信号のクリーニングとピーク検出
    rsp_cleaned = nk.rsp_clean(edr_signal, sampling_rate=target_fs)
    rsp_peaks = nk.rsp_findpeaks(rsp_cleaned, sampling_rate=target_fs)

    peaks_idx = rsp_peaks['RSP_Peaks']
    troughs_idx = rsp_peaks['RSP_Troughs']

    # 瞬時呼吸数計算（troughメソッド使用 - より正確）
    rsp_rate = nk.rsp_rate(
        rsp_cleaned,
        troughs=rsp_peaks,
        sampling_rate=target_fs,
        method='trough'
    )

    # 統計
    valid_rate = rsp_rate[~np.isnan(rsp_rate)]
    mean_rate_peak = np.mean(valid_rate)
    std_rate_peak = np.std(valid_rate)

    print(f"検出されたピーク数: {len(peaks_idx)}")
    print(f"検出されたトラフ数: {len(troughs_idx)}")
    print(f"平均呼吸数: {mean_rate_peak:.1f} ± {std_rate_peak:.1f} bpm")

    # 周期に換算
    period_sec = 60 / mean_rate_peak
    print(f"平均呼吸周期: {period_sec:.1f} 秒/回")
    print()

    # ========================================
    # 3. スペクトル法による呼吸数（複数範囲）
    # ========================================
    print("-" * 70)
    print("3. スペクトル法による呼吸数（複数範囲で検証）")
    print("-" * 70)

    # EDRのパワースペクトル計算
    freqs, psd = signal.welch(
        edr_signal,
        fs=target_fs,
        nperseg=min(256, len(edr_signal)),
        scaling='density'
    )

    # 複数範囲で計算
    spectral_results = calculate_spectral_br_multiple_ranges(freqs, psd)

    print("\n各周波数範囲でのSpectral BR:")
    print("-" * 70)
    for name, result in spectral_results.items():
        print(f"\n{name}:")
        print(f"  ピーク周波数: {result['freq_hz']:.4f} Hz")
        print(f"  呼吸数:      {result['bpm']:.1f} bpm")
        print(f"  呼吸周期:     {result['period_sec']:.1f} 秒/回")
        print(f"  パワー:      {result['power']:.2e}")

    print()

    # ========================================
    # 4. 妥当性評価
    # ========================================
    print("-" * 70)
    print("4. 妥当性評価")
    print("-" * 70)

    print("\n【ユーザー証言との比較】")
    print("実際の呼吸周期: 15-30秒/回 (2-4 bpm)")
    print()

    print("各手法の評価:")
    print(f"  ピーク検出法: {mean_rate_peak:.1f} bpm ({period_sec:.1f}秒/回)")

    if 2 <= mean_rate_peak <= 6:
        print(f"    → ✓ 妥当な範囲（深い瞑想呼吸）")
    else:
        print(f"    → ✗ 範囲外")

    for name, result in spectral_results.items():
        print(f"\n  {name}: {result['bpm']:.1f} bpm ({result['period_sec']:.1f}秒/回)")

        # 妥当性判定
        if 2 <= result['bpm'] <= 6:
            validity = "✓ 妥当（深い瞑想呼吸）"
        elif 6 <= result['bpm'] <= 10:
            validity = "△ やや速い（浅い呼吸？）"
        else:
            validity = "✗ 速すぎる（呼吸ではない可能性）"

        print(f"    → {validity}")

    print()

    # ========================================
    # 5. 可視化
    # ========================================
    print("-" * 70)
    print("5. 可視化")
    print("-" * 70)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # ----- プロット1: EDR信号とピーク -----
    ax1 = axes[0]
    time_min = time_resampled / 60
    ax1.plot(time_min, rsp_cleaned, 'b-', linewidth=1.5, alpha=0.7, label='EDR Signal')

    # ピークとトラフをマーク
    if len(peaks_idx) > 0:
        ax1.scatter(time_min[peaks_idx], rsp_cleaned[peaks_idx],
                   color='red', s=50, zorder=5, label=f'Peaks (n={len(peaks_idx)})')
    if len(troughs_idx) > 0:
        ax1.scatter(time_min[troughs_idx], rsp_cleaned[troughs_idx],
                   color='green', s=50, zorder=5, label=f'Troughs (n={len(troughs_idx)})')

    ax1.set_xlabel('Time (min)', fontsize=12)
    ax1.set_ylabel('EDR Signal', fontsize=12)
    ax1.set_title(f'EDR Signal with Peaks (Mean BR: {mean_rate_peak:.1f} bpm)',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ----- プロット2: パワースペクトル（全体） -----
    ax2 = axes[1]
    ax2.plot(freqs * 60, psd, 'b-', linewidth=2, alpha=0.8)
    ax2.set_xlim(0, 30)
    ax2.set_xlabel('Frequency (bpm)', fontsize=12)
    ax2.set_ylabel('Power Spectral Density', fontsize=12)
    ax2.set_title('EDR Power Spectrum (0-30 bpm)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 各範囲のピークをマーク
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    for (name, result), color in zip(spectral_results.items(), colors):
        ax2.axvline(result['bpm'], color=color, linestyle='--',
                   linewidth=1.5, alpha=0.7, label=f'{name}: {result["bpm"]:.1f} bpm')

    # ピーク検出法の結果もマーク
    ax2.axvline(mean_rate_peak, color='black', linestyle=':',
               linewidth=2, alpha=0.9, label=f'Peak method: {mean_rate_peak:.1f} bpm')

    ax2.legend(loc='upper right', fontsize=9)

    # ----- プロット3: パワースペクトル（低周波域拡大） -----
    ax3 = axes[2]
    low_freq_mask = freqs * 60 <= 12
    ax3.plot(freqs[low_freq_mask] * 60, psd[low_freq_mask], 'b-',
            linewidth=2, alpha=0.8)
    ax3.set_xlim(0, 12)
    ax3.set_xlabel('Frequency (bpm)', fontsize=12)
    ax3.set_ylabel('Power Spectral Density', fontsize=12)
    ax3.set_title('EDR Power Spectrum - Low Frequency Zoom (0-12 bpm)',
                  fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 低周波範囲のピークをマーク
    for name, result in spectral_results.items():
        if result['bpm'] <= 12:
            ax3.axvline(result['bpm'], color='red', linestyle='--',
                       linewidth=1.5, alpha=0.7)
            ax3.scatter([result['bpm']], [result['power']],
                       color='red', s=100, zorder=5, edgecolors='white', linewidths=2)
            ax3.text(result['bpm'], result['power']*1.1,
                    f'{result["bpm"]:.1f}\nbpm',
                    ha='center', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # ピーク検出法もマーク
    if mean_rate_peak <= 12:
        ax3.axvline(mean_rate_peak, color='black', linestyle=':',
                   linewidth=2, alpha=0.9, label=f'Peak method: {mean_rate_peak:.1f} bpm')
        ax3.legend(loc='upper right')

    plt.tight_layout()

    output_path = output_dir / 'breathing_rate_validation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"図を保存: {output_path}")
    plt.close()

    # ========================================
    # 6. レポート生成
    # ========================================
    print()
    print("-" * 70)
    print("6. レポート生成")
    print("-" * 70)

    report_lines = []
    report_lines.append("# 呼吸数（BR）妥当性検証レポート\n")
    report_lines.append("## 概要\n")
    report_lines.append("深い瞑想状態における呼吸数推定の妥当性を検証しました。\n")
    report_lines.append("## データ情報\n")
    report_lines.append(f"- **測定時間**: {hrv_data['time'][-1]/60:.1f} 分\n")
    report_lines.append(f"- **R-R間隔数**: {len(rr_intervals)}\n")
    report_lines.append(f"- **平均心拍数**: {60000/np.mean(rr_intervals):.1f} bpm\n\n")

    report_lines.append("## ユーザー証言\n")
    report_lines.append("- **実際の呼吸周期**: 15-30秒/回\n")
    report_lines.append("- **推定呼吸数**: 2-4 bpm\n\n")

    report_lines.append("## 検証結果\n\n")
    report_lines.append("### 1. ピーク検出法\n\n")
    report_lines.append(f"- **平均呼吸数**: {mean_rate_peak:.1f} ± {std_rate_peak:.1f} bpm\n")
    report_lines.append(f"- **平均呼吸周期**: {period_sec:.1f} 秒/回\n")
    report_lines.append(f"- **検出ピーク数**: {len(peaks_idx)}\n")

    if 2 <= mean_rate_peak <= 6:
        report_lines.append(f"- **評価**: ✓ 妥当（深い瞑想呼吸）\n\n")
    else:
        report_lines.append(f"- **評価**: 要確認\n\n")

    report_lines.append("### 2. スペクトル法（複数範囲）\n\n")
    report_lines.append("| 周波数範囲 | 呼吸数 (bpm) | 呼吸周期 (秒) | 評価 |\n")
    report_lines.append("|:-----------|-------------:|--------------:|:-----|\n")

    for name, result in spectral_results.items():
        if 2 <= result['bpm'] <= 6:
            validity = "✓ 妥当"
        elif 6 <= result['bpm'] <= 10:
            validity = "△ やや速い"
        else:
            validity = "✗ 速すぎる"

        report_lines.append(f"| {name} | {result['bpm']:.1f} | {result['period_sec']:.1f} | {validity} |\n")

    report_lines.append("\n## 結論\n\n")

    # 最も妥当な範囲を選択
    valid_ranges = []
    for name, result in spectral_results.items():
        if 2 <= result['bpm'] <= 6:
            valid_ranges.append((name, result))

    if valid_ranges:
        report_lines.append("**推奨される呼吸数推定方法:**\n\n")
        for name, result in valid_ranges:
            report_lines.append(f"- **{name}**: {result['bpm']:.1f} bpm ({result['period_sec']:.1f}秒/回)\n")

    report_lines.append("\n**問題点:**\n\n")
    report_lines.append("- 現在の実装で使用している「通常呼吸 (9-24 bpm)」範囲は、深い瞑想状態には適していません\n")
    report_lines.append("- この範囲で検出される9.4 bpmは、実際の呼吸ではなく別の生理現象（Mayer波など）の可能性があります\n\n")

    report_lines.append("**推奨事項:**\n\n")
    report_lines.append("- 深い瞑想データには「深呼吸 (2-6 bpm)」または「LF帯域 (2-9 bpm)」範囲を使用すべき\n")
    report_lines.append("- ピーク検出法の結果も参考にして、適応的に範囲を選択する実装が望ましい\n\n")

    report_lines.append("## 可視化\n\n")
    report_lines.append("![呼吸数検証結果](breathing_rate_validation.png)\n\n")
    report_lines.append(f"---\n\n生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report_path = output_dir / 'BREATHING_VALIDATION_REPORT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"レポート生成: {report_path}")

    print()
    print("=" * 70)
    print("検証完了")
    print("=" * 70)


if __name__ == '__main__':
    import sys

    # データファイルパス（コマンドライン引数またはデフォルト）
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        # デフォルト: 以前の分析で使用したデータ
        csv_path = project_root / 'data/selfloops/selfloops_2026-01-12--06-21-05.csv'

    output_dir = Path(__file__).parent

    print(f"使用データファイル: {csv_path}")
    print()

    analyze_breathing_validation(csv_path, output_dir)
