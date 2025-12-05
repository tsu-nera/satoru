#!/usr/bin/env python3
"""
アーティファクト vs 生体信号の判別分析

10-15Hz帯域の異常信号が人工的なノイズか生体信号かを調査
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import kurtosis

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv, prepare_mne_raw, calculate_psd

DATA_PATH = project_root / 'data' / 'mindMonitor_2025-12-04--07-39-03_7794313749178367799.csv'
OUTPUT_DIR = Path(__file__).parent / 'img'

plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans', 'sans-serif']


def analyze_peak_characteristics(psd_dict):
    """
    ピークの特性を分析（幅、形状など）

    人工ノイズ: 非常に狭いピーク（スパイク状）
    生体信号: 広めのピーク（滑らかな山型）
    """
    freqs = psd_dict['freqs']

    results = {}

    for i, ch in enumerate(psd_dict['channels']):
        psd = psd_dict['psds'][i]

        # 10-15Hz範囲でピークを探す
        mask = (freqs >= 10) & (freqs <= 15)
        freq_subset = freqs[mask]
        psd_subset = psd[mask]

        if len(psd_subset) == 0:
            continue

        # ピーク位置
        peak_idx = np.argmax(psd_subset)
        peak_freq = freq_subset[peak_idx]
        peak_power = psd_subset[peak_idx]

        # 半値幅（FWHM）を計算
        half_max = peak_power / 2
        above_half = psd_subset >= half_max

        # FWHMの近似計算
        if np.any(above_half):
            indices = np.where(above_half)[0]
            fwhm = freq_subset[indices[-1]] - freq_subset[indices[0]]
        else:
            fwhm = 0

        # Q factor (ピークのシャープさ)
        q_factor = peak_freq / fwhm if fwhm > 0 else float('inf')

        results[ch] = {
            'peak_freq': peak_freq,
            'peak_power': peak_power,
            'fwhm': fwhm,
            'q_factor': q_factor,
        }

    return results


def analyze_signal_variability(raw, freq_band=(10, 15)):
    """
    信号の変動性を分析

    人工ノイズ: 振幅が一定
    生体信号: 振幅が変動する
    """
    sfreq = raw.info['sfreq']

    # バンドパスフィルタ
    raw_filtered = raw.copy().filter(
        l_freq=freq_band[0],
        h_freq=freq_band[1],
        verbose=False
    )

    data = raw_filtered.get_data()

    results = {}
    for i, ch in enumerate(raw.ch_names):
        ch_data = data[i]

        # 1秒ごとのRMSを計算
        window_samples = int(sfreq)
        n_windows = len(ch_data) // window_samples

        rms_values = []
        for w in range(n_windows):
            start = w * window_samples
            end = start + window_samples
            rms = np.sqrt(np.mean(ch_data[start:end] ** 2))
            rms_values.append(rms)

        rms_values = np.array(rms_values)

        # 変動係数（CV）- 高いほど生体信号らしい
        cv = np.std(rms_values) / np.mean(rms_values) if np.mean(rms_values) > 0 else 0

        # 尖度 - 正常分布からの逸脱
        kurt = kurtosis(ch_data)

        results[ch] = {
            'rms_mean': np.mean(rms_values),
            'rms_std': np.std(rms_values),
            'cv': cv,
            'kurtosis': kurt,
            'rms_values': rms_values,
        }

    return results


def analyze_channel_correlation(raw, freq_band=(10, 15)):
    """
    チャネル間の相関を分析

    人工ノイズ: 全チャネルで高相関（同じ外部源）
    生体信号: チャネル間で相関が低い（異なる脳領域）
    """
    raw_filtered = raw.copy().filter(
        l_freq=freq_band[0],
        h_freq=freq_band[1],
        verbose=False
    )

    data = raw_filtered.get_data()

    # 相関行列
    corr_matrix = np.corrcoef(data)

    return {
        'correlation_matrix': corr_matrix,
        'channel_names': raw.ch_names,
        'mean_correlation': np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]),
    }


def plot_artifact_analysis(peak_chars, variability, correlation, output_path):
    """
    アーティファクト分析結果をプロット
    """
    fig = plt.figure(figsize=(16, 12))

    # 1. ピーク特性
    ax1 = fig.add_subplot(2, 2, 1)
    channels = list(peak_chars.keys())
    fwhm_values = [peak_chars[ch]['fwhm'] for ch in channels]
    q_values = [min(peak_chars[ch]['q_factor'], 50) for ch in channels]  # 上限を設定

    x = np.arange(len(channels))
    width = 0.35
    ax1.bar(x - width/2, fwhm_values, width, label='FWHM (Hz)', color='blue')
    ax1.bar(x + width/2, q_values, width, label='Q Factor', color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels([ch.replace('RAW_', '') for ch in channels])
    ax1.set_ylabel('値')
    ax1.set_title('ピーク特性（10-15Hz帯域）\n狭いピーク(高Q)=ノイズの可能性、広いピーク(低Q)=生体信号')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 信号変動性（時系列）
    ax2 = fig.add_subplot(2, 2, 2)
    for ch in channels:
        rms_values = variability[ch]['rms_values']
        time_min = np.arange(len(rms_values)) / 60  # 分に変換
        ax2.plot(time_min, rms_values, label=ch.replace('RAW_', ''), alpha=0.7)

    ax2.set_xlabel('経過時間 (分)')
    ax2.set_ylabel('RMS (μV)')
    ax2.set_title('10-15Hz帯域のRMS時系列\n一定=ノイズ、変動=生体信号')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 変動係数
    ax3 = fig.add_subplot(2, 2, 3)
    cv_values = [variability[ch]['cv'] for ch in channels]
    colors = ['green' if cv > 0.3 else 'red' for cv in cv_values]
    ax3.bar([ch.replace('RAW_', '') for ch in channels], cv_values, color=colors)
    ax3.axhline(y=0.3, color='orange', linestyle='--', label='閾値 (0.3)')
    ax3.set_ylabel('変動係数 (CV)')
    ax3.set_title('信号変動係数\n高い(緑)=生体信号、低い(赤)=ノイズの可能性')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. チャネル間相関
    ax4 = fig.add_subplot(2, 2, 4)
    corr_matrix = correlation['correlation_matrix']
    ch_labels = [ch.replace('RAW_', '') for ch in correlation['channel_names']]

    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax4.set_xticks(range(len(ch_labels)))
    ax4.set_yticks(range(len(ch_labels)))
    ax4.set_xticklabels(ch_labels)
    ax4.set_yticklabels(ch_labels)
    plt.colorbar(im, ax=ax4, label='相関係数')
    ax4.set_title(f'チャネル間相関（平均: {correlation["mean_correlation"]:.2f}）\n高相関=共通ノイズ、低相関=独立した脳活動')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'保存: {output_path}')


def generate_artifact_report(peak_chars, variability, correlation, output_path):
    """
    アーティファクト分析レポート生成
    """
    report = """# アーティファクト vs 生体信号 判別分析

## 分析目的

10-15Hz帯域の異常信号が**人工的なノイズ**か**生体信号**かを判別する。

---

## 判別基準

| 特徴 | 人工ノイズ | 生体信号 |
|:-----|:-----------|:---------|
| ピーク幅 (FWHM) | 狭い (<0.5Hz) | 広い (>1Hz) |
| Q Factor | 高い (>20) | 低い (<10) |
| 振幅変動 (CV) | 低い (<0.2) | 高い (>0.3) |
| チャネル間相関 | 高い (>0.8) | 低い (<0.5) |

---

## 分析結果

### 1. ピーク特性

| チャネル | ピーク周波数 | FWHM (Hz) | Q Factor | 判定 |
|:---------|:-------------|:----------|:---------|:-----|
"""

    for ch, data in peak_chars.items():
        ch_label = ch.replace('RAW_', '')
        q = data['q_factor']
        fwhm = data['fwhm']

        if q > 20 or fwhm < 0.5:
            judgment = "⚠️ ノイズ疑い"
        elif q < 10 and fwhm > 1:
            judgment = "✓ 生体信号"
        else:
            judgment = "？ 不明確"

        report += f"| {ch_label} | {data['peak_freq']:.2f} Hz | {fwhm:.2f} | {q:.1f} | {judgment} |\n"

    report += """
### 2. 信号変動性

| チャネル | RMS平均 (μV) | 変動係数 (CV) | 判定 |
|:---------|:-------------|:--------------|:-----|
"""

    for ch, data in variability.items():
        ch_label = ch.replace('RAW_', '')
        cv = data['cv']

        if cv > 0.3:
            judgment = "✓ 生体信号"
        elif cv < 0.2:
            judgment = "⚠️ ノイズ疑い"
        else:
            judgment = "？ 不明確"

        report += f"| {ch_label} | {data['rms_mean']:.3f} | {cv:.3f} | {judgment} |\n"

    mean_corr = correlation['mean_correlation']

    report += f"""
### 3. チャネル間相関

**平均相関係数**: {mean_corr:.3f}

"""

    if mean_corr > 0.8:
        report += "**判定**: ⚠️ 高相関 → 共通の外部ノイズ源の可能性\n"
    elif mean_corr < 0.5:
        report += "**判定**: ✓ 低相関 → 独立した脳活動の可能性\n"
    else:
        report += "**判定**: ？ 中程度の相関 → 判断が難しい\n"

    report += """
---

## 総合判定

"""

    # 総合判定ロジック
    noise_indicators = 0
    bio_indicators = 0

    for ch, data in peak_chars.items():
        if data['q_factor'] > 20:
            noise_indicators += 1
        elif data['q_factor'] < 10:
            bio_indicators += 1

    for ch, data in variability.items():
        if data['cv'] > 0.3:
            bio_indicators += 1
        elif data['cv'] < 0.2:
            noise_indicators += 1

    if mean_corr > 0.8:
        noise_indicators += 2
    elif mean_corr < 0.5:
        bio_indicators += 2

    if noise_indicators > bio_indicators:
        report += """
### ⚠️ 人工ノイズの可能性が高い

考えられる原因：
- 電磁干渉（EMI）：近くの電子機器
- 電源ノイズ：充電器、ACアダプタ
- 無線干渉：Bluetooth、WiFi機器
- 照明：LED照明、蛍光灯のフリッカー

**推奨**: 次回セッションでは周囲の電子機器を確認・移動してみてください。
"""
    elif bio_indicators > noise_indicators:
        report += """
### ✓ 生体信号の可能性が高い

考えられる原因：
- その日の脳状態の変化
- 睡眠状態の影響
- 無意識の筋緊張（眼筋、前頭筋）
- 自律神経状態の変化

**推奨**: 次回セッションで再現するか観察してください。
"""
    else:
        report += """
### ？ 判定困難

ノイズと生体信号の両方の特徴が混在しています。
追加データ（他のセッションとの比較）が必要です。
"""

    report += """
---

## 生成画像

- `artifact_analysis.png`: アーティファクト分析結果

---

*分析日時: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*\n"

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'保存: {output_path}')


def main():
    print('='*60)
    print('アーティファクト vs 生体信号 判別分析')
    print('='*60)

    print(f'\nLoading: {DATA_PATH}')
    df = load_mind_monitor_csv(DATA_PATH, filter_headband=False, warmup_seconds=60)

    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df)
    raw = mne_dict['raw']

    print('\n分析中: ピーク特性...')
    psd_dict = calculate_psd(raw)
    peak_chars = analyze_peak_characteristics(psd_dict)

    for ch, data in peak_chars.items():
        print(f"  {ch}: Peak={data['peak_freq']:.2f}Hz, FWHM={data['fwhm']:.2f}Hz, Q={data['q_factor']:.1f}")

    print('\n分析中: 信号変動性...')
    variability = analyze_signal_variability(raw)

    for ch, data in variability.items():
        print(f"  {ch}: CV={data['cv']:.3f}")

    print('\n分析中: チャネル間相関...')
    correlation = analyze_channel_correlation(raw)
    print(f"  平均相関: {correlation['mean_correlation']:.3f}")

    print('\nプロット中...')
    plot_artifact_analysis(peak_chars, variability, correlation,
                          OUTPUT_DIR / 'artifact_analysis.png')

    print('\nレポート生成中...')
    generate_artifact_report(peak_chars, variability, correlation,
                            OUTPUT_DIR.parent / 'ANALYSIS_ARTIFACT.md')

    print('\n' + '='*60)
    print('分析完了!')
    print('='*60)

    return 0


if __name__ == '__main__':
    exit(main())
