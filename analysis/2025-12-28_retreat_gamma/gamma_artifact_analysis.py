"""
Gamma Band Artifact Analysis Script

2025-12-28の瞑想データにおけるガンマ波領域のピークがアーチファクトか脳波由来かを判定

特に注目するピーク周波数:
- 16, 20, 24, 28, 32 Hz（4Hzの倍数）
- その他のガンマ帯域のピーク

アーチファクトの判定基準:
1. チャネル間相関: アーチファクトは全チャネルで同時に出現しやすい
2. 等間隔性: 4Hz間隔は電源ノイズの高調波を示唆
3. 時間的安定性: アーチファクトは時間的に安定、脳波は変動
4. パワー分布: アーチファクトは鋭いピーク、脳波は幅広い分布
5. スペクトル形状: Q値（ピークの鋭さ）の解析
"""

import sys
sys.path.insert(0, '/home/tsu-nera/repo/satoru')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks, welch
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

from lib.loaders.mind_monitor import load_mind_monitor_csv
from lib.sensors.eeg import prepare_mne_raw, calculate_psd

# 出力ディレクトリ
OUTPUT_DIR = Path('/home/tsu-nera/repo/satoru/analysis/2025-12-28_retreat_gamma')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# データパス
DATA_PATH = '/home/tsu-nera/repo/satoru/data/mindMonitor_2025-12-28--19-54-40_5999051304967425957.csv'

# 対象ピーク周波数（4Hzの倍数）
TARGET_PEAKS = [16, 20, 24, 28, 32]


def analyze_channel_correlation(psd_dict, target_freqs, tolerance=1.0):
    """
    チャネル間のピークパワー相関を分析

    アーチファクトは全チャネルで同時に現れる傾向がある
    脳波は電極位置により異なるパターンを示す
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']  # shape: (n_channels, n_freqs)
    channels = psd_dict['channels']

    results = []

    for target_freq in target_freqs:
        # 対象周波数付近のインデックスを取得
        mask = np.abs(freqs - target_freq) < tolerance
        if not mask.any():
            continue

        # 各チャネルの対象周波数帯のパワー
        powers = psds[:, mask].mean(axis=1)  # チャネルごとの平均パワー

        # チャネル間の相関を計算
        correlations = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                # 周波数帯全体での相関
                corr, _ = pearsonr(psds[i, mask], psds[j, mask])
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else np.nan
        power_cv = np.std(powers) / np.mean(powers) if np.mean(powers) > 0 else np.nan

        results.append({
            'frequency': target_freq,
            'avg_channel_correlation': avg_correlation,
            'power_cv': power_cv,  # 変動係数: 小さいほど均一（アーチファクト的）
            'channel_powers_db': 10 * np.log10(powers + 1e-10),
        })

    return results


def analyze_peak_sharpness(psd_dict, target_freqs, bandwidth=2.0):
    """
    ピークの鋭さ（Q値）を分析

    Q値 = 中心周波数 / 半値幅
    高いQ値: 鋭いピーク（アーチファクト的）
    低いQ値: 幅広いピーク（生理的）
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    results = []

    for target_freq in target_freqs:
        # 対象周波数周辺を抽出
        mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
        if not mask.any():
            continue

        local_freqs = freqs[mask]
        local_psd = psd_db[mask]

        # ピーク位置
        peak_idx = np.argmax(local_psd)
        peak_freq = local_freqs[peak_idx]
        peak_power = local_psd[peak_idx]

        # 半値幅を計算（-3dB幅）
        half_power = peak_power - 3
        above_half = local_psd >= half_power

        if above_half.sum() > 1:
            half_width = local_freqs[above_half].max() - local_freqs[above_half].min()
            q_value = peak_freq / half_width if half_width > 0 else np.inf
        else:
            half_width = np.nan
            q_value = np.nan

        results.append({
            'target_freq': target_freq,
            'actual_peak_freq': peak_freq,
            'peak_power_db': peak_power,
            'half_width_hz': half_width,
            'q_value': q_value,
        })

    return results


def analyze_temporal_stability(raw, target_freqs, n_segments=10, tolerance=1.0):
    """
    時間セグメント間でのピークパワーの安定性を分析

    アーチファクトは時間的に安定している傾向
    脳波は状態変化により変動する
    """
    from scipy.signal import welch

    data = raw.get_data()  # (n_channels, n_samples)
    sfreq = raw.info['sfreq']
    n_samples = data.shape[1]
    segment_size = n_samples // n_segments

    segment_powers = {freq: [] for freq in target_freqs}

    for i in range(n_segments):
        start = i * segment_size
        end = start + segment_size
        segment_data = data[:, start:end]

        # 各チャネルの平均PSD
        freqs_welch, psd_welch = welch(segment_data.mean(axis=0), fs=sfreq, nperseg=min(2048, segment_size))

        for target_freq in target_freqs:
            mask = np.abs(freqs_welch - target_freq) < tolerance
            if mask.any():
                power = np.mean(psd_welch[mask])
                segment_powers[target_freq].append(10 * np.log10(power + 1e-10))

    results = []
    for freq in target_freqs:
        powers = segment_powers[freq]
        if powers:
            results.append({
                'frequency': freq,
                'mean_power_db': np.mean(powers),
                'std_power_db': np.std(powers),
                'cv': np.std(powers) / np.abs(np.mean(powers)) if np.mean(powers) != 0 else np.nan,
                'range_db': np.max(powers) - np.min(powers),
            })

    return results


def analyze_harmonic_structure(psd_dict, base_freq=4.0, max_harmonic=10):
    """
    4Hz基準のハーモニクス構造を分析

    16, 20, 24, 28, 32 Hz = 4 x (4, 5, 6, 7, 8)
    全てが4Hzの倍数であることがアーチファクトを示唆
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    harmonic_powers = []
    expected_freqs = [base_freq * i for i in range(1, max_harmonic + 1)]

    for expected_freq in expected_freqs:
        if expected_freq > freqs.max():
            break

        mask = np.abs(freqs - expected_freq) < 0.5
        if mask.any():
            power = np.max(psd_db[mask])
            harmonic_powers.append({
                'harmonic_number': int(expected_freq / base_freq),
                'expected_freq': expected_freq,
                'power_db': power,
            })

    return harmonic_powers


def analyze_spectral_flatness(psd_dict, target_freqs, bandwidth=3.0):
    """
    スペクトル平坦度（Spectral Flatness）を分析

    ピーク周辺のスペクトル平坦度が低い = 鋭いピーク（アーチファクト的）
    スペクトル平坦度が高い = 平坦（ノイズ的 or 幅広い活動）
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)

    results = []

    for target_freq in target_freqs:
        mask = (freqs >= target_freq - bandwidth) & (freqs <= target_freq + bandwidth)
        if not mask.any():
            continue

        local_psd = psd_avg[mask]

        # スペクトル平坦度 = 幾何平均 / 算術平均
        geometric_mean = np.exp(np.mean(np.log(local_psd + 1e-10)))
        arithmetic_mean = np.mean(local_psd)
        flatness = geometric_mean / arithmetic_mean if arithmetic_mean > 0 else np.nan

        results.append({
            'frequency': target_freq,
            'spectral_flatness': flatness,
            'interpretation': 'narrow peak' if flatness < 0.3 else 'broad activity' if flatness > 0.7 else 'moderate'
        })

    return results


def plot_comprehensive_analysis(psd_dict, raw, target_freqs, output_path):
    """包括的な分析結果をプロット"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    # 1. 全体PSDとターゲットピーク
    ax1 = axes[0, 0]
    ax1.plot(freqs, psd_db, 'b-', linewidth=0.8)
    for freq in target_freqs:
        ax1.axvline(freq, color='r', linestyle='--', alpha=0.7, label=f'{freq} Hz')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('PSD with Target Peaks (16, 20, 24, 28, 32 Hz)')
    ax1.set_xlim(0, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. チャネル別PSD
    ax2 = axes[0, 1]
    for i, ch in enumerate(psd_dict['channels']):
        ch_psd_db = 10 * np.log10(psd_dict['psds'][i] + 1e-10)
        ax2.plot(freqs, ch_psd_db, label=ch, alpha=0.7)
    for freq in target_freqs:
        ax2.axvline(freq, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Power (dB)')
    ax2.set_title('Channel-wise PSD')
    ax2.set_xlim(10, 40)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 4Hzハーモニクス構造
    ax3 = axes[1, 0]
    harmonic_results = analyze_harmonic_structure(psd_dict)
    if harmonic_results:
        harmonics = [r['harmonic_number'] for r in harmonic_results]
        powers = [r['power_db'] for r in harmonic_results]
        colors = ['red' if r['expected_freq'] in target_freqs else 'blue' for r in harmonic_results]
        ax3.bar(harmonics, powers, color=colors, alpha=0.7)
        ax3.set_xlabel('Harmonic Number (of 4 Hz)')
        ax3.set_ylabel('Power (dB)')
        ax3.set_title('4 Hz Harmonic Structure (Red = Target Peaks)')
        ax3.grid(True, alpha=0.3)

    # 4. ピーク周辺の詳細
    ax4 = axes[1, 1]
    for i, target_freq in enumerate(target_freqs):
        mask = (freqs >= target_freq - 3) & (freqs <= target_freq + 3)
        if mask.any():
            ax4.plot(freqs[mask], psd_db[mask], label=f'{target_freq} Hz',
                    color=plt.cm.viridis(i / len(target_freqs)))
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power (dB)')
    ax4.set_title('Detailed View Around Target Peaks')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 時間安定性
    ax5 = axes[2, 0]
    stability_results = analyze_temporal_stability(raw, target_freqs)
    if stability_results:
        freqs_plot = [r['frequency'] for r in stability_results]
        cvs = [r['cv'] for r in stability_results]
        ax5.bar(freqs_plot, cvs, color='purple', alpha=0.7, width=2)
        ax5.axhline(0.1, color='green', linestyle='--', label='Artifact threshold (CV<0.1)')
        ax5.set_xlabel('Frequency (Hz)')
        ax5.set_ylabel('Coefficient of Variation')
        ax5.set_title('Temporal Stability (Lower CV = More Stable = Artifact-like)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # 6. スペクトル平坦度
    ax6 = axes[2, 1]
    flatness_results = analyze_spectral_flatness(psd_dict, target_freqs)
    if flatness_results:
        freqs_plot = [r['frequency'] for r in flatness_results]
        flatness = [r['spectral_flatness'] for r in flatness_results]
        ax6.bar(freqs_plot, flatness, color='orange', alpha=0.7, width=2)
        ax6.axhline(0.3, color='red', linestyle='--', label='Narrow peak threshold')
        ax6.axhline(0.7, color='green', linestyle='--', label='Broad activity threshold')
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Spectral Flatness')
        ax6.set_title('Spectral Flatness (Lower = Sharper Peak = Artifact-like)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_report(all_results, output_path):
    """分析レポートを生成"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Gamma Band Artifact Analysis Report\n\n")
        f.write("## セッション情報\n")
        f.write("- **日付**: 2025-12-28\n")
        f.write("- **セッション**: 瞑想リトリート\n")
        f.write(f"- **データファイル**: {DATA_PATH}\n\n")

        f.write("## 分析対象\n")
        f.write(f"- ターゲット周波数: {TARGET_PEAKS} Hz\n")
        f.write("- 特徴: 全て4Hzの倍数（4×4, 4×5, 4×6, 4×7, 4×8）\n\n")

        f.write("## 判定基準\n")
        f.write("| 指標 | アーチファクト | 脳波由来 |\n")
        f.write("|------|--------------|----------|\n")
        f.write("| チャネル間相関 | 高い (>0.8) | 低い〜中程度 |\n")
        f.write("| Q値（鋭さ） | 高い (>5) | 低い (<3) |\n")
        f.write("| 時間安定性(CV) | 低い (<0.1) | 高い (>0.2) |\n")
        f.write("| スペクトル平坦度 | 低い (<0.3) | 高い (>0.5) |\n\n")

        # チャネル相関結果
        f.write("## 1. チャネル間相関分析\n")
        if 'channel_correlation' in all_results:
            f.write("| 周波数 (Hz) | 平均相関 | パワーCV | 判定 |\n")
            f.write("|------------|---------|---------|------|\n")
            for r in all_results['channel_correlation']:
                corr = r['avg_channel_correlation']
                judgment = "アーチファクト的" if corr > 0.8 else "脳波的" if corr < 0.5 else "不明確"
                f.write(f"| {r['frequency']} | {corr:.3f} | {r['power_cv']:.3f} | {judgment} |\n")
        f.write("\n")

        # ピークの鋭さ
        f.write("## 2. ピークの鋭さ（Q値）分析\n")
        if 'peak_sharpness' in all_results:
            f.write("| 周波数 (Hz) | 実ピーク | パワー(dB) | 半値幅 | Q値 | 判定 |\n")
            f.write("|------------|---------|----------|-------|-----|------|\n")
            for r in all_results['peak_sharpness']:
                q = r['q_value']
                judgment = "アーチファクト的" if q > 5 else "脳波的" if q < 3 else "不明確"
                half_width = f"{r['half_width_hz']:.2f}" if not np.isnan(r['half_width_hz']) else "N/A"
                q_str = f"{q:.1f}" if not np.isnan(q) and not np.isinf(q) else "N/A"
                f.write(f"| {r['target_freq']} | {r['actual_peak_freq']:.1f} | {r['peak_power_db']:.1f} | {half_width} | {q_str} | {judgment} |\n")
        f.write("\n")

        # 時間安定性
        f.write("## 3. 時間安定性分析\n")
        if 'temporal_stability' in all_results:
            f.write("| 周波数 (Hz) | 平均パワー(dB) | 標準偏差 | CV | 判定 |\n")
            f.write("|------------|--------------|---------|-----|------|\n")
            for r in all_results['temporal_stability']:
                cv = r['cv']
                judgment = "アーチファクト的" if cv < 0.1 else "脳波的" if cv > 0.2 else "不明確"
                cv_str = f"{cv:.3f}" if not np.isnan(cv) else "N/A"
                f.write(f"| {r['frequency']} | {r['mean_power_db']:.1f} | {r['std_power_db']:.2f} | {cv_str} | {judgment} |\n")
        f.write("\n")

        # スペクトル平坦度
        f.write("## 4. スペクトル平坦度分析\n")
        if 'spectral_flatness' in all_results:
            f.write("| 周波数 (Hz) | 平坦度 | 解釈 | 判定 |\n")
            f.write("|------------|-------|------|------|\n")
            for r in all_results['spectral_flatness']:
                flat = r['spectral_flatness']
                judgment = "アーチファクト的" if flat < 0.3 else "脳波的" if flat > 0.5 else "不明確"
                f.write(f"| {r['frequency']} | {flat:.3f} | {r['interpretation']} | {judgment} |\n")
        f.write("\n")

        # 4Hzハーモニクス構造
        f.write("## 5. 4Hzハーモニクス構造\n")
        if 'harmonic_structure' in all_results:
            f.write("| 倍音番号 | 周波数 (Hz) | パワー (dB) | ターゲット |\n")
            f.write("|---------|------------|------------|----------|\n")
            for r in all_results['harmonic_structure']:
                is_target = "✓" if r['expected_freq'] in TARGET_PEAKS else ""
                f.write(f"| {r['harmonic_number']} | {r['expected_freq']:.0f} | {r['power_db']:.1f} | {is_target} |\n")
        f.write("\n")

        # 総合判定
        f.write("## 6. 総合判定\n\n")

        # 判定ロジック
        artifact_votes = 0
        brain_votes = 0

        # 各指標から投票を集計
        if 'channel_correlation' in all_results:
            for r in all_results['channel_correlation']:
                if r['avg_channel_correlation'] > 0.8:
                    artifact_votes += 1
                elif r['avg_channel_correlation'] < 0.5:
                    brain_votes += 1

        if 'peak_sharpness' in all_results:
            for r in all_results['peak_sharpness']:
                if not np.isnan(r['q_value']) and not np.isinf(r['q_value']):
                    if r['q_value'] > 5:
                        artifact_votes += 1
                    elif r['q_value'] < 3:
                        brain_votes += 1

        if 'temporal_stability' in all_results:
            for r in all_results['temporal_stability']:
                if not np.isnan(r['cv']):
                    if r['cv'] < 0.1:
                        artifact_votes += 1
                    elif r['cv'] > 0.2:
                        brain_votes += 1

        if 'spectral_flatness' in all_results:
            for r in all_results['spectral_flatness']:
                if r['spectral_flatness'] < 0.3:
                    artifact_votes += 1
                elif r['spectral_flatness'] > 0.5:
                    brain_votes += 1

        f.write(f"### 投票集計\n")
        f.write(f"- アーチファクト判定: {artifact_votes} 票\n")
        f.write(f"- 脳波由来判定: {brain_votes} 票\n\n")

        if artifact_votes > brain_votes:
            f.write("### 結論: **アーチファクトの可能性が高い**\n\n")
            f.write("理由:\n")
            f.write("- 16, 20, 24, 28, 32 Hz は全て4Hzの倍数\n")
            f.write("- この等間隔パターンは、電気的ノイズ源の高調波を示唆\n")
            f.write("- 各指標がアーチファクト特性を示している\n\n")
            f.write("考えられる原因:\n")
            f.write("1. **Bluetooth通信ノイズ**: Museデバイスの無線通信に起因\n")
            f.write("2. **内部クロックノイズ**: デバイス内部のタイミング回路由来\n")
            f.write("3. **電源回路ノイズ**: 電池やレギュレータからの漏れ\n")
            f.write("4. **サンプリング関連アーチファクト**: 256Hzサンプリングとの相互作用\n")
        elif brain_votes > artifact_votes:
            f.write("### 結論: **脳波由来の可能性が高い**\n\n")
            f.write("理由:\n")
            f.write("- チャネル間で異なるパターンを示している\n")
            f.write("- 時間的な変動がある\n")
            f.write("- ピークが比較的幅広い\n\n")
            f.write("**解釈**: 瞑想中の集中状態や認知処理の活性化を示唆する可能性があります。\n")
        else:
            f.write("### 結論: **判定困難**\n\n")
            f.write("アーチファクトと脳波の両方の特性が混在しています。\n")
            f.write("追加の分析（異なるセッションでの比較、眼球運動時の変化など）が推奨されます。\n")

        f.write("\n## 7. 推奨対策\n\n")
        f.write("もしアーチファクトの場合:\n")
        f.write("1. **ノッチフィルタの追加**: 4Hz基本周波数とその倍音を除去\n")
        f.write("2. **ICA分析**: 独立成分分析でアーチファクト成分を分離\n")
        f.write("3. **周波数帯域の除外**: 分析から16-32Hz帯を慎重に扱う\n")
        f.write("4. **他のセッションとの比較**: 再現性を確認\n\n")

        f.write("もし脳波由来の場合:\n")
        f.write("1. **ガンマ波の機能的意義**: 瞑想深度、注意集中、認知処理との関連を探る\n")
        f.write("2. **時系列分析**: セッション中のガンマ波パワーの変化を追跡\n")
        f.write("3. **他の指標との相関**: Fmθ、IAF、HbOなどとの関係を調査\n")
        f.write("4. **文献照合**: 瞑想とガンマ波に関する先行研究との比較\n")

    return output_path


def main():
    print("="*60)
    print("Gamma Band Artifact Analysis")
    print("="*60)
    print()

    print("Loading data...")
    df = load_mind_monitor_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    print()

    print("Preparing MNE raw...")
    # ノッチフィルタを無効化して生データに近い状態で分析
    mne_dict = prepare_mne_raw(df, apply_notch=False)
    raw = mne_dict['raw']
    print(f"Sampling rate: {raw.info['sfreq']:.2f} Hz")
    print(f"Channels: {mne_dict['channels']}")
    print()

    print("Calculating PSD...")
    psd_dict = calculate_psd(raw)
    print()

    print("Analyzing target peaks...")
    all_results = {}

    # 各分析を実行
    print("  - Channel correlation analysis...")
    all_results['channel_correlation'] = analyze_channel_correlation(psd_dict, TARGET_PEAKS)

    print("  - Peak sharpness analysis...")
    all_results['peak_sharpness'] = analyze_peak_sharpness(psd_dict, TARGET_PEAKS)

    print("  - Temporal stability analysis...")
    all_results['temporal_stability'] = analyze_temporal_stability(raw, TARGET_PEAKS)

    print("  - Spectral flatness analysis...")
    all_results['spectral_flatness'] = analyze_spectral_flatness(psd_dict, TARGET_PEAKS)

    print("  - Harmonic structure analysis...")
    all_results['harmonic_structure'] = analyze_harmonic_structure(psd_dict)
    print()

    print("Generating plots...")
    plot_path = OUTPUT_DIR / 'gamma_artifact_analysis.png'
    plot_comprehensive_analysis(psd_dict, raw, TARGET_PEAKS, plot_path)
    print(f"  Plot saved: {plot_path}")
    print()

    print("Generating report...")
    report_path = OUTPUT_DIR / 'gamma_artifact_analysis_report.md'
    generate_report(all_results, report_path)
    print(f"  Report saved: {report_path}")
    print()

    print("="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")

    return all_results


if __name__ == '__main__':
    results = main()
