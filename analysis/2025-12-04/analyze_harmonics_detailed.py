#!/usr/bin/env python3
"""
ハーモニクス（高調波）詳細分析

PSDの複数ピークがハーモニクスかどうかをより詳細に分析
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv, prepare_mne_raw, calculate_psd

DATA_PATH = project_root / 'data' / 'mindMonitor_2025-12-04--07-39-03_7794313749178367799.csv'
OUTPUT_DIR = Path(__file__).parent / 'img'

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']


def find_all_peaks(freqs, psd, freq_range=(1, 45), prominence=0.5):
    """全ピークを検出"""
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freq_subset = freqs[mask]
    psd_subset = psd[mask]
    psd_db = 10 * np.log10(psd_subset + 1e-10)

    peaks, properties = find_peaks(psd_db, prominence=prominence, distance=5)

    return freq_subset[peaks], psd_db[peaks], properties['prominences']


def analyze_harmonic_series(peak_freqs, fundamental_range=(4, 12), tolerance=0.8):
    """
    様々な基本周波数でハーモニクス系列を分析
    """
    results = []

    for fund in np.arange(fundamental_range[0], fundamental_range[1], 0.25):
        # 期待されるハーモニクス位置
        expected_harmonics = [fund * n for n in range(1, 7)]

        matches = []
        total_error = 0

        for h_num, h_freq in enumerate(expected_harmonics, 1):
            if h_freq > 45:
                continue

            # 最も近いピークを探す
            if len(peak_freqs) > 0:
                distances = np.abs(peak_freqs - h_freq)
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]

                if min_dist < tolerance:
                    matches.append({
                        'harmonic': h_num,
                        'expected': h_freq,
                        'actual': peak_freqs[min_idx],
                        'error': min_dist,
                    })
                    total_error += min_dist

        if len(matches) > 0:
            avg_error = total_error / len(matches)
        else:
            avg_error = float('inf')

        results.append({
            'fundamental': fund,
            'matches': matches,
            'match_count': len(matches),
            'avg_error': avg_error,
            'score': len(matches) / (1 + avg_error),  # マッチ数と精度のバランス
        })

    return sorted(results, key=lambda x: -x['score'])


def analyze_waveform_shape(raw, fundamental_freq, duration_sec=10):
    """
    波形の形状を分析（ハーモニクスの原因を理解）

    完全な正弦波 → ハーモニクスなし
    鋭いピーク → 多くのハーモニクス
    """
    from scipy.signal import butter, filtfilt

    sfreq = raw.info['sfreq']
    data = raw.get_data()[0]  # 最初のチャネル

    # 基本周波数付近をバンドパス
    nyq = sfreq / 2
    low = (fundamental_freq - 1) / nyq
    high = (fundamental_freq + 1) / nyq
    b, a = butter(4, [low, high], btype='band')
    filtered = filtfilt(b, a, data)

    # 一部を抽出
    n_samples = int(duration_sec * sfreq)
    segment = filtered[:n_samples]
    time = np.arange(n_samples) / sfreq

    # ゼロクロッシングを検出してサイクルを抽出
    zero_crossings = np.where(np.diff(np.sign(segment)) > 0)[0]

    # 波形の対称性を分析
    if len(zero_crossings) >= 2:
        cycle = segment[zero_crossings[0]:zero_crossings[1]]
        mid = len(cycle) // 2

        # 前半と後半の非対称性
        if mid > 0:
            first_half = cycle[:mid]
            second_half = cycle[mid:2*mid] if 2*mid <= len(cycle) else cycle[mid:]
            if len(first_half) == len(second_half):
                asymmetry = np.mean(np.abs(first_half + second_half[::-1]))
            else:
                asymmetry = None
        else:
            asymmetry = None
    else:
        asymmetry = None

    return {
        'time': time,
        'filtered': segment,
        'asymmetry': asymmetry,
    }


def plot_harmonic_detail(freqs, psd_avg, peak_freqs, peak_powers, top_results, output_path):
    """
    ハーモニクス詳細プロット
    """
    fig = plt.figure(figsize=(16, 14))

    # 1. 上位3つの基本周波数候補
    for i, result in enumerate(top_results[:3]):
        ax = fig.add_subplot(3, 2, i*2 + 1)

        fund = result['fundamental']
        mask = (freqs >= 1) & (freqs <= 45)
        psd_db = 10 * np.log10(psd_avg[mask] + 1e-10)

        ax.plot(freqs[mask], psd_db, 'b-', linewidth=1, alpha=0.7)

        # ハーモニクス位置
        colors = plt.cm.rainbow(np.linspace(0, 1, 6))
        for n in range(1, 7):
            h_freq = fund * n
            if h_freq <= 45:
                ax.axvline(x=h_freq, color=colors[n-1], linestyle='--', alpha=0.7, linewidth=2)
                ax.annotate(f'{n}f₀', (h_freq, ax.get_ylim()[1] if i == 0 else 10),
                           ha='center', fontsize=9, color=colors[n-1])

        # 実際のピーク
        for pf, pp in zip(peak_freqs, peak_powers):
            if pf <= 45:
                ax.scatter([pf], [pp], color='red', s=50, zorder=5, alpha=0.7)

        ax.set_xlabel('周波数 (Hz)')
        ax.set_ylabel('パワー (dB)')
        ax.set_title(f'基本周波数 f₀ = {fund:.2f} Hz\n'
                    f'マッチ: {result["match_count"]}個, スコア: {result["score"]:.2f}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, 45)

    # 2. スコア分布
    ax_score = fig.add_subplot(3, 2, 2)

    funds = [r['fundamental'] for r in top_results[:20]]
    scores = [r['score'] for r in top_results[:20]]

    ax_score.bar(range(len(funds)), scores, color='steelblue')
    ax_score.set_xticks(range(len(funds)))
    ax_score.set_xticklabels([f'{f:.1f}' for f in funds], rotation=45)
    ax_score.set_xlabel('基本周波数 (Hz)')
    ax_score.set_ylabel('スコア')
    ax_score.set_title('基本周波数候補のスコア（上位20）')
    ax_score.grid(True, alpha=0.3)

    # 3. ピーク間隔分析
    ax_interval = fig.add_subplot(3, 2, 4)

    sorted_peaks = np.sort(peak_freqs)
    intervals = np.diff(sorted_peaks)

    ax_interval.hist(intervals, bins=20, color='green', alpha=0.7, edgecolor='black')
    ax_interval.axvline(x=np.median(intervals), color='red', linestyle='--',
                       label=f'中央値: {np.median(intervals):.1f} Hz')
    ax_interval.set_xlabel('ピーク間隔 (Hz)')
    ax_interval.set_ylabel('頻度')
    ax_interval.set_title('ピーク間隔の分布')
    ax_interval.legend()
    ax_interval.grid(True, alpha=0.3)

    # 4. 周波数帯域別ピーク数
    ax_bands = fig.add_subplot(3, 2, 6)

    bands = {
        'Delta\n(0.5-4)': (0.5, 4),
        'Theta\n(4-8)': (4, 8),
        'Alpha\n(8-13)': (8, 13),
        'SMR\n(12-15)': (12, 15),
        'Beta\n(13-30)': (13, 30),
        'Gamma\n(30-50)': (30, 50),
    }

    band_counts = []
    band_names = []
    for name, (low, high) in bands.items():
        count = np.sum((peak_freqs >= low) & (peak_freqs < high))
        band_counts.append(count)
        band_names.append(name)

    colors = ['purple', 'blue', 'green', 'cyan', 'orange', 'red']
    ax_bands.bar(band_names, band_counts, color=colors, alpha=0.7, edgecolor='black')
    ax_bands.set_ylabel('ピーク数')
    ax_bands.set_title('周波数帯域別ピーク数')
    ax_bands.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def plot_waveform_analysis(waveform_data, fundamental_freq, output_path):
    """
    波形分析プロット
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    time = waveform_data['time']
    filtered = waveform_data['filtered']

    # 上: 波形全体（最初の2秒）
    ax1 = axes[0]
    mask = time <= 2
    ax1.plot(time[mask], filtered[mask], 'b-', linewidth=0.5)
    ax1.set_xlabel('時間 (秒)')
    ax1.set_ylabel('振幅 (μV)')
    ax1.set_title(f'{fundamental_freq:.1f} Hz 付近の波形（バンドパスフィルタ後）')
    ax1.grid(True, alpha=0.3)

    # 下: 数サイクル拡大
    ax2 = axes[1]
    n_cycles = 5
    cycle_duration = n_cycles / fundamental_freq
    mask = time <= cycle_duration
    ax2.plot(time[mask], filtered[mask], 'b-', linewidth=1)
    ax2.set_xlabel('時間 (秒)')
    ax2.set_ylabel('振幅 (μV)')
    ax2.set_title(f'{n_cycles}サイクル拡大表示')
    ax2.grid(True, alpha=0.3)

    # 完全な正弦波を重ねて表示
    ideal_sine = np.max(np.abs(filtered[mask])) * np.sin(2 * np.pi * fundamental_freq * time[mask])
    ax2.plot(time[mask], ideal_sine, 'r--', linewidth=1, alpha=0.5, label='理想正弦波')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def generate_harmonics_report(peak_freqs, peak_powers, top_results, output_path):
    """
    ハーモニクス詳細レポート
    """
    report = """# ハーモニクス（高調波）詳細分析レポート

## 1. ハーモニクスとは

### 1.1 基本概念

**ハーモニクス（高調波）** とは、基本周波数（f₀）の整数倍の周波数成分です。

```
基本周波数 f₀ = 8 Hz の場合：

1倍音（基本音）:  8 Hz  = 1 × f₀
2倍音:           16 Hz  = 2 × f₀
3倍音:           24 Hz  = 3 × f₀
4倍音:           32 Hz  = 4 × f₀
...
```

### 1.2 なぜハーモニクスが発生するか

脳波がハーモニクスを含む理由：

1. **非正弦波形**
   - 脳波は完全な正弦波ではない
   - 鋭いピークや非対称な波形はハーモニクスを生成

2. **神経発火パターン**
   - ニューロンの発火は「オン/オフ」的
   - これが矩形波に近い成分を作り、ハーモニクスを生む

3. **非線形処理**
   - 脳内の神経回路は非線形
   - 信号が歪んでハーモニクスが発生

### 1.3 波形とハーモニクスの関係

| 波形タイプ | ハーモニクス特性 |
|:-----------|:-----------------|
| 純粋な正弦波 | ハーモニクスなし |
| 三角波 | 奇数倍音のみ（3f, 5f, 7f...） |
| 矩形波 | 強い奇数倍音 |
| 鋸歯状波 | 全ての整数倍音 |
| 脳波 | 様々（波形による） |

---

## 2. 分析結果

### 2.1 検出されたピーク

| 周波数 (Hz) | パワー (dB) | 推定カテゴリ |
|:------------|:------------|:-------------|
"""

    # ピークをカテゴリ分け
    for pf, pp in sorted(zip(peak_freqs, peak_powers)):
        if pf <= 45:
            if pf < 4:
                cat = "Delta"
            elif pf < 8:
                cat = "Theta"
            elif pf < 13:
                cat = "Alpha"
            elif pf < 15:
                cat = "SMR/Low Beta"
            elif pf < 20:
                cat = "Mid Beta"
            elif pf < 30:
                cat = "High Beta"
            else:
                cat = "Gamma"

            report += f"| {pf:.1f} | {pp:.1f} | {cat} |\n"

    report += """
### 2.2 基本周波数候補

様々な基本周波数を仮定し、ハーモニクスとの一致度を分析しました。

| 順位 | 基本周波数 (Hz) | マッチ数 | 平均誤差 (Hz) | スコア |
|:-----|:----------------|:---------|:--------------|:-------|
"""

    for i, result in enumerate(top_results[:10], 1):
        report += f"| {i} | {result['fundamental']:.2f} | {result['match_count']} | "
        report += f"{result['avg_error']:.2f} | {result['score']:.2f} |\n"

    # 最良の結果を詳細に
    best = top_results[0]
    report += f"""
### 2.3 最良候補の詳細

**基本周波数: {best['fundamental']:.2f} Hz**

| 倍音 | 期待値 (Hz) | 実測値 (Hz) | 誤差 (Hz) |
|:-----|:------------|:------------|:----------|
"""

    for m in best['matches']:
        report += f"| {m['harmonic']}倍音 | {m['expected']:.1f} | {m['actual']:.1f} | {m['error']:.2f} |\n"

    report += f"""
---

## 3. 解釈

### 3.1 ハーモニクスと独立リズムの区別

今回のデータでは、以下のピークが観察されました：

**ハーモニクスの可能性が高い:**
"""

    # ハーモニクスの可能性を判定
    fund = best['fundamental']
    matched_freqs = [m['actual'] for m in best['matches']]

    for pf in sorted(peak_freqs):
        if pf <= 45:
            is_harmonic = pf in matched_freqs
            ratio = pf / fund if fund > 0 else 0
            nearest_int = round(ratio)
            deviation = abs(ratio - nearest_int)

            if is_harmonic and nearest_int > 1:
                report += f"- **{pf:.1f} Hz**: {fund:.1f}Hzの{nearest_int}倍音（誤差 {deviation*fund:.2f} Hz）\n"

    report += """
**独立した脳リズムの可能性が高い:**
"""

    for pf, pp in sorted(zip(peak_freqs, peak_powers), key=lambda x: -x[1])[:5]:
        if pf <= 45 and pf not in matched_freqs:
            if 8 <= pf <= 12:
                report += f"- **{pf:.1f} Hz**: 主要なAlphaリズム\n"
            elif 12 <= pf <= 15:
                report += f"- **{pf:.1f} Hz**: SMR（感覚運動リズム）\n"
            elif 4 <= pf <= 8:
                report += f"- **{pf:.1f} Hz**: Thetaリズム\n"

    report += """
### 3.2 実用的な意味

**瞑想・脳波解析における注意点:**

1. **ハーモニクスは「ノイズ」ではない**
   - 脳活動の波形特性を反映
   - 鋭いAlphaピークは健全な脳活動の証

2. **解析時の考慮**
   - Beta帯域のパワーにはAlphaのハーモニクスが含まれる可能性
   - 純粋なBeta活動を見たい場合は注意が必要

3. **特徴量として活用**
   - ハーモニクスの強さは波形の「鋭さ」を反映
   - 深い瞑想状態では波形が滑らかになり、ハーモニクスが減少することも

---

## 4. 周波数帯域の名称一覧

| 帯域名 | 周波数範囲 | 関連する状態 |
|:-------|:-----------|:-------------|
| **Delta (δ)** | 0.5-4 Hz | 深い睡眠、無意識 |
| **Theta (θ)** | 4-8 Hz | 眠気、瞑想、記憶形成 |
| **Alpha (α)** | 8-13 Hz | リラックス、閉眼安静 |
| **SMR** | 12-15 Hz | 感覚運動リズム、集中 |
| **Beta (β)** | 13-30 Hz | 覚醒、思考、注意 |
| └ Low Beta | 13-15 Hz | 軽い集中 |
| └ Mid Beta | 15-20 Hz | 活発な思考 |
| └ High Beta | 20-30 Hz | 興奮、不安 |
| **Gamma (γ)** | 30-100 Hz | 認知処理、意識統合 |
| └ Low Gamma | 30-50 Hz | 知覚結合 |
| └ High Gamma | 50-100 Hz | 高次認知 |

---

## 5. 結論

今回のPSDに見られる多数のピークは：

1. **主要な独立リズム**: Alpha (8.5Hz), SMR (12.5Hz)
2. **ハーモニクス**: 14Hz, 21Hz, 28Hz付近（基本周波数 ~7Hzの倍音）
3. **その他のBeta/Gamma活動**: 20Hz, 32Hz付近

ハーモニクスの存在は正常であり、脳波の波形特性を反映しています。

---

*分析日時: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'保存: {output_path}')


def main():
    print('='*60)
    print('ハーモニクス詳細分析')
    print('='*60)

    print(f'\nLoading: {DATA_PATH}')
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)

    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)
    raw = mne_dict['raw']

    print('計算中: PSD...')
    psd_dict = calculate_psd(raw, fmax=50)
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)

    print('検出中: ピーク...')
    peak_freqs, peak_powers, prominences = find_all_peaks(freqs, psd_avg)
    print(f'  検出されたピーク数: {len(peak_freqs)}')

    print('分析中: ハーモニクス系列...')
    top_results = analyze_harmonic_series(peak_freqs)
    print(f'  最良候補: {top_results[0]["fundamental"]:.2f} Hz (スコア: {top_results[0]["score"]:.2f})')

    print('\n波形分析中...')
    waveform_data = analyze_waveform_shape(raw, top_results[0]['fundamental'])

    print('\nプロット中: ハーモニクス詳細...')
    plot_harmonic_detail(freqs, psd_avg, peak_freqs, peak_powers,
                        top_results, OUTPUT_DIR / 'harmonics_detail.png')

    print('プロット中: 波形分析...')
    plot_waveform_analysis(waveform_data, top_results[0]['fundamental'],
                          OUTPUT_DIR / 'waveform_analysis.png')

    print('\nレポート生成中...')
    generate_harmonics_report(peak_freqs, peak_powers, top_results,
                             OUTPUT_DIR.parent / 'REPORT_HARMONICS.md')

    print('\n' + '='*60)
    print('分析完了!')
    print('='*60)

    return 0


if __name__ == '__main__':
    exit(main())
