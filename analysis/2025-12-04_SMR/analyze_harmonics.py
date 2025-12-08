#!/usr/bin/env python3
"""
ハーモニクス（高調波）分析

PSDの複数ピークが基本周波数のハーモニクスかどうかを分析
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv, prepare_mne_raw, calculate_psd

DATA_PATH = project_root / 'data' / 'mindMonitor_2025-12-04--07-39-03_7794313749178367799.csv'
OUTPUT_DIR = Path(__file__).parent / 'img'

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']


def find_all_peaks(freqs, psd, freq_range=(1, 35), prominence=0.5):
    """
    指定範囲内の全ピークを検出
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freq_subset = freqs[mask]
    psd_subset = psd[mask]

    # dBスケールで検出
    psd_db = 10 * np.log10(psd_subset + 1e-10)

    peaks, properties = find_peaks(psd_db, prominence=prominence, distance=5)

    peak_freqs = freq_subset[peaks]
    peak_powers_db = psd_db[peaks]
    prominences = properties['prominences']

    return peak_freqs, peak_powers_db, prominences


def check_harmonics(peak_freqs, fundamental_candidates=(7, 8, 8.5, 9, 10)):
    """
    ピークが特定の基本周波数のハーモニクスかどうかをチェック
    """
    results = []

    for fund in fundamental_candidates:
        harmonics = [fund * n for n in range(1, 6)]  # 1-5倍音

        matches = []
        for h_num, h_freq in enumerate(harmonics, 1):
            # 最も近いピークを探す
            if len(peak_freqs) > 0:
                distances = np.abs(peak_freqs - h_freq)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]

                if min_dist < 1.0:  # 1Hz以内なら一致とみなす
                    matches.append({
                        'harmonic': h_num,
                        'expected': h_freq,
                        'actual': peak_freqs[min_idx],
                        'error': min_dist,
                    })

        results.append({
            'fundamental': fund,
            'matches': matches,
            'match_count': len(matches),
        })

    return results


def analyze_peak_names():
    """
    EEGの各周波数帯域と名称
    """
    bands = {
        'Delta': (0.5, 4, '深い睡眠'),
        'Theta': (4, 8, '眠気、瞑想、記憶'),
        'Alpha': (8, 13, 'リラックス、閉眼'),
        'SMR': (12, 15, '感覚運動リズム、集中'),
        'Beta': (13, 30, '覚醒、思考、集中'),
        'Low Beta': (13, 15, '軽い集中'),
        'Mid Beta': (15, 20, '活発な思考'),
        'High Beta': (20, 30, '不安、興奮'),
        'Gamma': (30, 50, '認知処理、意識'),
    }
    return bands


def plot_harmonics_analysis(freqs, psd_avg, peak_freqs, peak_powers, harmonic_results, output_path):
    """
    ハーモニクス分析プロット
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 上: PSDと検出されたピーク
    ax1 = axes[0]
    mask = (freqs >= 1) & (freqs <= 35)
    psd_db = 10 * np.log10(psd_avg[mask] + 1e-10)

    ax1.plot(freqs[mask], psd_db, 'b-', linewidth=1.5)

    # ピークをマーク
    for pf, pp in zip(peak_freqs, peak_powers):
        ax1.scatter([pf], [pp], color='red', s=100, zorder=5)
        ax1.annotate(f'{pf:.1f}Hz', (pf, pp), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9, color='red')

    # 帯域を色分け
    bands = analyze_peak_names()
    colors = {'Delta': 'purple', 'Theta': 'blue', 'Alpha': 'green',
              'Beta': 'orange', 'Gamma': 'red'}

    ax1.axvspan(0.5, 4, alpha=0.1, color='purple', label='Delta')
    ax1.axvspan(4, 8, alpha=0.1, color='blue', label='Theta')
    ax1.axvspan(8, 13, alpha=0.1, color='green', label='Alpha')
    ax1.axvspan(13, 30, alpha=0.1, color='orange', label='Beta')
    ax1.axvspan(30, 35, alpha=0.1, color='red', label='Gamma')

    ax1.set_xlabel('周波数 (Hz)')
    ax1.set_ylabel('パワー (dB)')
    ax1.set_title('PSDピーク検出と周波数帯域')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 下: ハーモニクス分析
    ax2 = axes[1]

    # 最もマッチした基本周波数を表示
    best_match = max(harmonic_results, key=lambda x: x['match_count'])
    fund = best_match['fundamental']

    ax2.plot(freqs[mask], psd_db, 'b-', linewidth=1.5, alpha=0.5)

    # 期待されるハーモニクス位置を表示
    for n in range(1, 6):
        h_freq = fund * n
        if h_freq <= 35:
            ax2.axvline(x=h_freq, color='green', linestyle='--', alpha=0.7)
            ax2.annotate(f'{n}倍音\n({h_freq:.1f}Hz)', (h_freq, ax2.get_ylim()[1] - 2),
                        ha='center', fontsize=9, color='green')

    # 実際のピークをマーク
    for pf, pp in zip(peak_freqs, peak_powers):
        ax2.scatter([pf], [pp], color='red', s=100, zorder=5)

    ax2.set_xlabel('周波数 (Hz)')
    ax2.set_ylabel('パワー (dB)')
    ax2.set_title(f'ハーモニクス分析（基本周波数 {fund} Hz 仮定）')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def generate_harmonics_report(peak_freqs, peak_powers, harmonic_results, output_path):
    """
    ハーモニクス分析レポート
    """
    report = """# PSDピーク分析 - ハーモニクスと周波数帯域

## 検出されたピーク

| 周波数 (Hz) | パワー (dB) | 推定される名称 | 解釈 |
|:------------|:------------|:---------------|:-----|
"""

    # ピークに名前をつける
    for pf, pp in sorted(zip(peak_freqs, peak_powers)):
        if pf < 4:
            name = "Delta"
            interp = "深い睡眠、徐波"
        elif pf < 8:
            name = "Theta"
            interp = "眠気、瞑想、内省"
        elif pf < 10:
            name = "Low Alpha"
            interp = "リラックス、閉眼安静"
        elif pf < 13:
            name = "High Alpha / Mu"
            interp = "リラックス、運動抑制"
        elif pf < 15:
            name = "SMR / Low Beta"
            interp = "集中、感覚運動リズム"
        elif pf < 20:
            name = "Mid Beta"
            interp = "活発な思考、注意"
        elif pf < 30:
            name = "High Beta"
            interp = "興奮、不安、過覚醒"
        else:
            name = "Gamma"
            interp = "認知処理、意識統合"

        report += f"| {pf:.1f} | {pp:.1f} | **{name}** | {interp} |\n"

    report += """
---

## ハーモニクス（高調波）分析

脳波は完全な正弦波ではないため、基本周波数の整数倍に**ハーモニクス**が現れることがあります。

### ハーモニクスとは

```
基本周波数 f₀ がある場合：
├─ 1倍音（基本）: f₀
├─ 2倍音: 2 × f₀
├─ 3倍音: 3 × f₀
└─ ...
```

### 基本周波数候補との一致度

"""

    for result in sorted(harmonic_results, key=lambda x: -x['match_count']):
        fund = result['fundamental']
        matches = result['matches']
        count = result['match_count']

        report += f"#### 基本周波数 {fund} Hz の場合（一致: {count}個）\n\n"

        if matches:
            report += "| 倍音 | 期待値 (Hz) | 実測値 (Hz) | 誤差 (Hz) |\n"
            report += "|:-----|:------------|:------------|:----------|\n"
            for m in matches:
                report += f"| {m['harmonic']}倍音 | {m['expected']:.1f} | {m['actual']:.1f} | {m['error']:.2f} |\n"
        else:
            report += "（一致するピークなし）\n"

        report += "\n"

    report += """
---

## 解釈

### ハーモニクスの原因

1. **非正弦波形**
   - Alpha波などが鋭いピークを持つ波形の場合、ハーモニクスが発生
   - これは正常な現象

2. **独立した脳活動**
   - 一部のピークは独立した脳リズム（SMR、Betaなど）
   - ハーモニクスではなく、異なる神経回路からの信号

3. **アーティファクト**
   - 電源ノイズ（50/60Hz）やその分周
   - 筋電図（EMG）の混入

### 今回のデータの特徴

- **8.5Hz付近**: 主要なAlphaピーク（基本周波数）
- **12-13Hz付近**: SMR/Mu（独立した脳活動 or Alphaの影響）
- **14Hz付近**: Low Beta or 8.5Hzの準ハーモニクス
- **17Hz付近**: 8.5Hzの2倍音の可能性
- **20-21Hz付近**: Mid Beta（独立 or ハーモニクス）
- **25-26Hz付近**: 8.5Hzの3倍音の可能性

---

## 結論

PSDに見られる複数のピークは：

1. **一部はハーモニクス**（基本周波数の整数倍）
2. **一部は独立した脳リズム**（SMR、Beta等）
3. **区別には波形解析が必要**

通常の瞑想解析では、Alpha帯域（8-13Hz）とTheta帯域（4-8Hz）に注目すれば十分です。

---

*分析日時: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'保存: {output_path}')


def main():
    print('='*60)
    print('ハーモニクス分析')
    print('='*60)

    print(f'\nLoading: {DATA_PATH}')
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)

    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)
    raw = mne_dict['raw']

    print('計算中: PSD...')
    psd_dict = calculate_psd(raw)
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)

    print('検出中: ピーク...')
    peak_freqs, peak_powers, prominences = find_all_peaks(freqs, psd_avg)

    print(f'\n検出されたピーク:')
    for pf, pp, prom in zip(peak_freqs, peak_powers, prominences):
        print(f'  {pf:.1f} Hz (Power: {pp:.1f} dB, Prominence: {prom:.1f})')

    print('\n分析中: ハーモニクス...')
    harmonic_results = check_harmonics(peak_freqs)

    for result in harmonic_results:
        print(f"  基本周波数 {result['fundamental']}Hz: {result['match_count']}個一致")

    print('\nプロット中...')
    plot_harmonics_analysis(freqs, psd_avg, peak_freqs, peak_powers,
                           harmonic_results, OUTPUT_DIR / 'harmonics_analysis.png')

    print('レポート生成中...')
    generate_harmonics_report(peak_freqs, peak_powers, harmonic_results,
                             OUTPUT_DIR.parent / 'HARMONICS_ANALYSIS.md')

    print('\n' + '='*60)
    print('分析完了!')
    print('='*60)

    return 0


if __name__ == '__main__':
    exit(main())
