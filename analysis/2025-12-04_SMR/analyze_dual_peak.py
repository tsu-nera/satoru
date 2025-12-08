#!/usr/bin/env python3
"""
二峰性Alpha分析スクリプト

10-15Hz帯域の異常をより詳しく分析し、
2つのピーク（8-9Hz と 12-13Hz）が存在するかを確認する。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_psd,
    calculate_paf,
    FREQ_BANDS,
)

# 設定
DATA_PATH = project_root / 'data' / 'mindMonitor_2025-12-04--07-39-03_7794313749178367799.csv'
OUTPUT_DIR = Path(__file__).parent / 'img'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']


def find_alpha_peaks(freqs, psd, freq_range=(6, 15)):
    """
    Alpha帯域内のピークを検出
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freq_subset = freqs[mask]
    psd_subset = psd[mask]

    # ピーク検出
    peaks, properties = find_peaks(psd_subset, prominence=0.1, distance=5)

    peak_freqs = freq_subset[peaks]
    peak_powers = psd_subset[peaks]

    # パワー順にソート
    sorted_idx = np.argsort(peak_powers)[::-1]
    peak_freqs = peak_freqs[sorted_idx]
    peak_powers = peak_powers[sorted_idx]

    return peak_freqs, peak_powers


def analyze_segment_peaks(raw, segment_minutes=3, warmup_minutes=1):
    """
    各セグメントでAlpha帯域のピークを検出
    """
    sfreq = raw.info['sfreq']
    warmup_samples = int(warmup_minutes * 60 * sfreq)
    segment_samples = int(segment_minutes * 60 * sfreq)

    results = []
    start_sample = warmup_samples
    segment_num = 1

    while start_sample + segment_samples <= raw.n_times:
        end_sample = start_sample + segment_samples

        raw_segment = raw.copy().crop(
            tmin=start_sample / sfreq,
            tmax=end_sample / sfreq
        )

        psd_dict = calculate_psd(raw_segment)
        freqs = psd_dict['freqs']
        psd_avg = np.mean(psd_dict['psds'], axis=0)

        # ピーク検出
        peak_freqs, peak_powers = find_alpha_peaks(freqs, psd_avg)

        results.append({
            'segment': segment_num,
            'start_min': (start_sample / sfreq) / 60,
            'end_min': (end_sample / sfreq) / 60,
            'freqs': freqs,
            'psd_avg': psd_avg,
            'peak_freqs': peak_freqs,
            'peak_powers': peak_powers,
            'num_peaks': len(peak_freqs),
        })

        start_sample = end_sample
        segment_num += 1

    return results


def plot_dual_peak_analysis(segment_results, output_path):
    """
    二峰性ピークの詳細プロット
    """
    n_segments = len(segment_results)
    n_cols = 3
    n_rows = (n_segments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, seg in enumerate(segment_results):
        ax = axes[i]

        freqs = seg['freqs']
        psd_avg = seg['psd_avg']

        # 6-15Hz範囲でプロット
        mask = (freqs >= 6) & (freqs <= 15)
        ax.plot(freqs[mask], psd_avg[mask], 'b-', linewidth=2)

        # ピークをマーク
        for pf, pp in zip(seg['peak_freqs'][:3], seg['peak_powers'][:3]):  # 上位3ピーク
            ax.scatter([pf], [pp], color='red', s=100, zorder=5)
            ax.annotate(f'{pf:.1f}Hz', (pf, pp), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=9, color='red')

        # 帯域境界を表示
        ax.axvline(x=8, color='green', linestyle='--', alpha=0.5, label='Alpha low')
        ax.axvline(x=12, color='green', linestyle='--', alpha=0.5, label='Alpha high')
        ax.axvline(x=10, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(x=15, color='purple', linestyle=':', alpha=0.5)

        ax.set_xlabel('周波数 (Hz)')
        ax.set_ylabel('PSD (μV²/Hz)')
        ax.set_title(f"Seg {seg['segment']} ({seg['start_min']:.0f}-{seg['end_min']:.0f}min)\n"
                     f"Peaks: {len(seg['peak_freqs'])}")
        ax.grid(True, alpha=0.3)

    # 余ったaxesを非表示
    for i in range(len(segment_results), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle('セグメント別Alpha帯域ピーク分析（6-15Hz）', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def plot_peak_evolution(segment_results, output_path):
    """
    ピーク周波数の時間変化
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    segments = [seg['segment'] for seg in segment_results]
    times = [(seg['start_min'] + seg['end_min']) / 2 for seg in segment_results]

    # 上: ピーク周波数の時間変化
    ax1 = axes[0]

    # 各ピークを個別にプロット
    max_peaks = max(seg['num_peaks'] for seg in segment_results)
    colors = plt.cm.tab10(np.linspace(0, 1, max_peaks))

    for peak_idx in range(min(max_peaks, 3)):  # 上位3ピークまで
        peak_freqs = []
        peak_powers = []
        for seg in segment_results:
            if peak_idx < len(seg['peak_freqs']):
                peak_freqs.append(seg['peak_freqs'][peak_idx])
                peak_powers.append(seg['peak_powers'][peak_idx])
            else:
                peak_freqs.append(np.nan)
                peak_powers.append(np.nan)

        ax1.plot(times, peak_freqs, 'o-', color=colors[peak_idx],
                 label=f'Peak {peak_idx + 1}', markersize=8, linewidth=2)

    ax1.axhline(y=8.5, color='green', linestyle='--', alpha=0.5, label='Typical Alpha')
    ax1.axhline(y=12, color='red', linestyle='--', alpha=0.5, label='SMR boundary')
    ax1.set_xlabel('経過時間 (分)')
    ax1.set_ylabel('ピーク周波数 (Hz)')
    ax1.set_title('ピーク周波数の時間変化')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(6, 16)

    # 下: 各周波数帯のパワー時系列
    ax2 = axes[1]

    # 8-9Hz と 11-13Hz のパワーを比較
    low_alpha_power = []
    high_alpha_power = []

    for seg in segment_results:
        freqs = seg['freqs']
        psd = seg['psd_avg']

        # 8-9Hz パワー
        mask_low = (freqs >= 8) & (freqs <= 9)
        low_alpha_power.append(np.mean(psd[mask_low]))

        # 11-13Hz パワー
        mask_high = (freqs >= 11) & (freqs <= 13)
        high_alpha_power.append(np.mean(psd[mask_high]))

    ax2.plot(times, low_alpha_power, 'go-', label='Low Alpha (8-9Hz)', markersize=8, linewidth=2)
    ax2.plot(times, high_alpha_power, 'ro-', label='High Alpha/SMR (11-13Hz)', markersize=8, linewidth=2)

    ax2.set_xlabel('経過時間 (分)')
    ax2.set_ylabel('平均パワー (μV²/Hz)')
    ax2.set_title('帯域別パワーの時間変化')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def generate_detailed_report(segment_results, output_path):
    """
    詳細レポート生成
    """
    report = """# 二峰性Alpha分析レポート

## 分析目的

10-15Hz帯域の異常パターンについて、Alpha帯域内に複数のピークが存在するかを確認し、
その原因を特定する。

---

## セグメント別ピーク検出結果

| Seg | 時間 | Peak 1 (Hz) | Peak 2 (Hz) | Peak 3 (Hz) | ピーク数 |
|:----|:-----|:------------|:------------|:------------|:---------|
"""

    for seg in segment_results:
        peaks = seg['peak_freqs']
        peak1 = f"{peaks[0]:.2f}" if len(peaks) > 0 else "N/A"
        peak2 = f"{peaks[1]:.2f}" if len(peaks) > 1 else "N/A"
        peak3 = f"{peaks[2]:.2f}" if len(peaks) > 2 else "N/A"

        report += f"| {seg['segment']} | {seg['start_min']:.0f}-{seg['end_min']:.0f}分 | "
        report += f"{peak1} | {peak2} | {peak3} | {seg['num_peaks']} |\n"

    # 分析サマリー
    all_primary_peaks = [seg['peak_freqs'][0] for seg in segment_results if len(seg['peak_freqs']) > 0]
    all_secondary_peaks = [seg['peak_freqs'][1] for seg in segment_results if len(seg['peak_freqs']) > 1]

    primary_mean = np.mean(all_primary_peaks) if all_primary_peaks else 0
    secondary_mean = np.mean(all_secondary_peaks) if all_secondary_peaks else 0

    report += f"""
---

## 統計サマリー

- **主要ピーク平均周波数**: {primary_mean:.2f} Hz
- **二次ピーク平均周波数**: {secondary_mean:.2f} Hz（検出されたセグメントのみ）

## 解釈

"""

    # 二峰性の判定
    has_dual_peak = any(len(seg['peak_freqs']) >= 2 for seg in segment_results)

    if has_dual_peak:
        report += """
### 二峰性ピークの検出

一部のセグメントで**2つ以上のピーク**が検出されました。これは以下の可能性を示唆します：

1. **真の二峰性Alpha活動**
   - 一部の人は8-9Hzと11-12Hzに2つの異なるAlphaジェネレータを持つ
   - これは必ずしも異常ではなく、脳の個人差

2. **SMR (Sensorimotor Rhythm) との混在**
   - 12-15HzはSMR（感覚運動リズム）帯域
   - 瞑想中の「動かない」努力がSMRを増加させた可能性

3. **Alphaピークの動的シフト**
   - セッション中にAlphaピークが低周波から高周波へシフト
   - 覚醒レベルの変化に伴う現象
"""
    else:
        report += """
### 単一ピーク

ほとんどのセグメントで単一の優位なピークが検出されました。
IAFの変動は、セッション中の脳状態の変化を反映している可能性があります。
"""

    report += """
---

## 結論

**接触不良の可能性**: HSI品質100%のため、**ハードウェア問題ではない**と考えられます。

**最も可能性の高い原因**:
"""

    # セグメント後半でIAFが高くなっているかチェック
    late_segments = segment_results[6:]  # 後半3セグメント
    early_segments = segment_results[:3]  # 前半3セグメント

    late_mean = np.mean([seg['peak_freqs'][0] for seg in late_segments if len(seg['peak_freqs']) > 0])
    early_mean = np.mean([seg['peak_freqs'][0] for seg in early_segments if len(seg['peak_freqs']) > 0])

    if late_mean > early_mean + 1:
        report += f"""
- セッション後半でIAFが上昇（前半 {early_mean:.1f}Hz → 後半 {late_mean:.1f}Hz）
- これは**覚醒レベルの変化**または**SMRの増加**を示唆
- 瞑想セッション終盤で意識が浮上してきた可能性
"""
    else:
        report += """
- IAFは比較的安定しており、大きな異常は見られない
- 一部のセグメントで見られる高周波シフトは正常な変動範囲内
"""

    report += """
---

## 生成画像

- `dual_peak_analysis.png`: セグメント別ピーク分析
- `peak_evolution.png`: ピーク周波数の時間変化

---

*分析日時: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'保存: {output_path}')


def main():
    print('='*60)
    print('二峰性Alpha分析')
    print('='*60)
    print()

    print(f'Loading: {DATA_PATH}')
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)
    print(f'データ形状: {df.shape}')

    print('\n準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)

    if not mne_dict:
        print('エラー: MNE RAWデータの準備に失敗しました')
        return 1

    raw = mne_dict['raw']

    print('\n分析中: セグメント別ピーク検出...')
    segment_results = analyze_segment_peaks(raw, segment_minutes=3, warmup_minutes=1)

    print('\nセグメント別ピーク:')
    for seg in segment_results:
        peaks_str = ", ".join([f"{p:.1f}Hz" for p in seg['peak_freqs'][:3]])
        print(f"  Seg {seg['segment']}: {peaks_str}")

    print('\nプロット中: 二峰性ピーク分析...')
    plot_dual_peak_analysis(segment_results, OUTPUT_DIR / 'dual_peak_analysis.png')

    print('プロット中: ピーク時間変化...')
    plot_peak_evolution(segment_results, OUTPUT_DIR / 'peak_evolution.png')

    print('\n生成中: 詳細レポート...')
    generate_detailed_report(segment_results, OUTPUT_DIR.parent / 'ANALYSIS_DUAL_PEAK.md')

    print()
    print('='*60)
    print('分析完了!')
    print('='*60)

    return 0


if __name__ == '__main__':
    exit(main())
