"""
瞑想者のガンマ波パターン比較分析

研究文献との比較:
- Lutz et al. (2004): チベット僧の瞑想中、25-42Hzの広帯域ガンマ波パワー増加
- Davidson et al.: 長期瞑想者で40Hz付近のガンマ波同期

あなたのパターンは:
- 飛び飛びのピーク（13.8, 17.6, 21.8, 26.2, 30.2, 34.2, 38.2, 42.2 Hz）
- それとも広帯域のパワー増加？
"""

import sys
sys.path.insert(0, '/home/tsu-nera/repo/satoru')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

from lib.loaders.mind_monitor import load_mind_monitor_csv
from lib.sensors.eeg import prepare_mne_raw, calculate_psd

# 出力ディレクトリ
OUTPUT_DIR = Path('/home/tsu-nera/repo/satoru/analysis/2025-12-28_retreat_gamma')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# データパス
DATA_PATH = '/home/tsu-nera/repo/satoru/data/mindMonitor_2025-12-28--19-54-40_5999051304967425957.csv'


def analyze_gamma_band_power(psd_dict, gamma_ranges=None):
    """
    ガンマ帯域のパワーを分析

    複数のガンマ帯域定義で解析:
    - Low Gamma: 25-35 Hz
    - Mid Gamma: 35-45 Hz
    - High Gamma: 45-55 Hz
    - Broad Gamma (Lutz et al.): 25-42 Hz
    - 40Hz Band: 38-42 Hz (Davidson et al.)
    """
    if gamma_ranges is None:
        gamma_ranges = {
            'Low Gamma (25-35 Hz)': (25, 35),
            'Mid Gamma (35-45 Hz)': (35, 45),
            'High Gamma (45-55 Hz)': (45, 55),
            'Broad Gamma (25-42 Hz, Lutz)': (25, 42),
            '40Hz Band (38-42 Hz, Davidson)': (38, 42),
            'Extended Gamma (13-50 Hz)': (13, 50),
        }

    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    results = {}

    for band_name, (fmin, fmax) in gamma_ranges.items():
        mask = (freqs >= fmin) & (freqs <= fmax)

        # 各チャネルの平均パワー
        channel_powers = {}
        for i, ch in enumerate(channels):
            power_linear = np.mean(psds[i, mask])
            power_db = 10 * np.log10(power_linear + 1e-10)
            channel_powers[ch] = {
                'power_db': power_db,
                'power_linear': power_linear
            }

        # 全チャネル平均
        avg_power_linear = np.mean([ch['power_linear'] for ch in channel_powers.values()])
        avg_power_db = 10 * np.log10(avg_power_linear + 1e-10)

        results[band_name] = {
            'freq_range': (fmin, fmax),
            'channel_powers': channel_powers,
            'avg_power_db': avg_power_db,
            'avg_power_linear': avg_power_linear,
        }

    return results


def analyze_gamma_spectral_shape(psd_dict, freq_range=(13, 50)):
    """
    ガンマ帯域のスペクトル形状を分析

    - 飛び飛びのピーク型 vs 広帯域パワー増加型
    - スペクトル平坦度で判定
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    gamma_freqs = freqs[mask]
    gamma_psd = psd_avg[mask]

    # スペクトル平坦度（幾何平均/算術平均）
    geometric_mean = np.exp(np.mean(np.log(gamma_psd + 1e-10)))
    arithmetic_mean = np.mean(gamma_psd)
    spectral_flatness = geometric_mean / arithmetic_mean

    # ピーク性の評価（ピーク数と鋭さ）
    from scipy.signal import find_peaks

    gamma_psd_db = 10 * np.log10(gamma_psd + 1e-10)
    peaks, properties = find_peaks(gamma_psd_db, prominence=1.0, distance=3)

    num_peaks = len(peaks)
    peak_freqs = gamma_freqs[peaks]
    peak_powers = gamma_psd_db[peaks]

    # パワーの変動係数
    power_cv = np.std(gamma_psd_db) / np.abs(np.mean(gamma_psd_db))

    # 判定
    if spectral_flatness > 0.7 and power_cv < 0.15:
        pattern_type = "Broad-band power increase (広帯域パワー増加型)"
        interpretation = "瞑想熟達者パターンに近い"
    elif num_peaks > 5 and power_cv > 0.2:
        pattern_type = "Discrete peaks (飛び飛びピーク型)"
        interpretation = "複数の独立した周波数成分"
    else:
        pattern_type = "Mixed pattern (混合型)"
        interpretation = "広帯域成分とピーク成分が混在"

    return {
        'spectral_flatness': spectral_flatness,
        'num_peaks': num_peaks,
        'peak_freqs': peak_freqs,
        'peak_powers': peak_powers,
        'power_cv': power_cv,
        'pattern_type': pattern_type,
        'interpretation': interpretation,
    }


def analyze_gamma_synchrony(raw, freq_range=(25, 42)):
    """
    ガンマ波の同期性を分析

    瞑想熟達者では、チャネル間のガンマ波同期が増加
    """
    from mne.connectivity import spectral_connectivity_epochs
    from mne import make_fixed_length_epochs

    # 2秒エポックに分割
    epochs = make_fixed_length_epochs(raw, duration=2.0, preload=True)

    # 周波数帯域を定義
    fmin, fmax = freq_range

    # スペクトル接続性を計算（PLV: Phase Locking Value）
    con = spectral_connectivity_epochs(
        epochs,
        method='plv',
        mode='multitaper',
        sfreq=raw.info['sfreq'],
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=1
    )

    # 平均同期性
    avg_synchrony = np.mean(con.get_data())

    return {
        'avg_synchrony': avg_synchrony,
        'freq_range': (fmin, fmax),
        'connectivity_matrix': con.get_data(),
    }


def compare_with_literature(your_results):
    """
    あなたの結果と文献値を比較
    """
    literature = {
        'Lutz et al. (2004) - Tibetan Monks': {
            'gamma_range': (25, 42),
            'pattern': 'Broad-band gamma increase',
            'synchrony': 'High (PLV > 0.6)',
            'description': '慈悲の瞑想中、25-42Hzの広帯域ガンマ波パワーが増加し、チャネル間同期も増加',
        },
        'Davidson et al. - Long-term meditators': {
            'gamma_range': (38, 42),
            'pattern': '40Hz centered increase',
            'synchrony': 'Increased inter-hemispheric',
            'description': '長期瞑想者で40Hz付近のガンマ波活動が増強',
        },
        'Beginner meditators': {
            'gamma_range': None,
            'pattern': 'Minimal or no gamma increase',
            'synchrony': 'Low',
            'description': '初心者ではガンマ波の顕著な変化は見られないことが多い',
        },
    }

    # あなたのパターン
    your_pattern = {
        'gamma_range': 'Multiple discrete peaks (13-50 Hz)',
        'pattern': your_results['spectral_shape']['pattern_type'],
        'broad_gamma_power': your_results['band_powers']['Broad Gamma (25-42 Hz, Lutz)']['avg_power_db'],
        '40hz_power': your_results['band_powers']['40Hz Band (38-42 Hz, Davidson)']['avg_power_db'],
    }

    return {
        'literature': literature,
        'your_pattern': your_pattern,
    }


def plot_gamma_analysis(psd_dict, band_powers, spectral_shape, output_path):
    """ガンマ波分析の可視化"""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    # 1. 全体PSD（ガンマ帯域を強調）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(freqs, psd_db, 'b-', linewidth=1.5, label='Average PSD')

    # ガンマ帯域を塗りつぶし
    gamma_ranges = {
        'Lutz et al. (25-42 Hz)': ((25, 42), 'red', 0.2),
        '40Hz Band (38-42 Hz)': ((38, 42), 'orange', 0.3),
    }

    for label, ((fmin, fmax), color, alpha) in gamma_ranges.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        ax1.fill_between(freqs[mask], -150, psd_db[mask], alpha=alpha, color=color, label=label)

    # ピークをマーク
    for peak_freq, peak_power in zip(spectral_shape['peak_freqs'], spectral_shape['peak_powers']):
        ax1.plot(peak_freq, peak_power, 'ro', markersize=8)
        ax1.annotate(f'{peak_freq:.1f}', (peak_freq, peak_power),
                    textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power (dB)', fontsize=12)
    ax1.set_title('PSD with Gamma Band Regions (Meditation Research)', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 60)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # 2. ガンマ帯域詳細（13-50 Hz）
    ax2 = fig.add_subplot(gs[1, 0])
    gamma_mask = (freqs >= 13) & (freqs <= 50)
    ax2.plot(freqs[gamma_mask], psd_db[gamma_mask], 'b-', linewidth=2)

    # 研究で使われる帯域を縦線で示す
    ax2.axvline(25, color='red', linestyle='--', alpha=0.5, label='Lutz range start')
    ax2.axvline(42, color='red', linestyle='--', alpha=0.5, label='Lutz range end')
    ax2.axvline(40, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Davidson 40Hz')

    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Power (dB)', fontsize=12)
    ax2.set_title('Gamma Band Detail (13-50 Hz)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # 3. チャネル別ガンマ帯域パワー比較
    ax3 = fig.add_subplot(gs[1, 1])

    bands_to_plot = [
        'Broad Gamma (25-42 Hz, Lutz)',
        '40Hz Band (38-42 Hz, Davidson)',
    ]

    x_positions = np.arange(len(bands_to_plot))
    width = 0.2

    for i, ch in enumerate(psd_dict['channels']):
        powers = [band_powers[band]['channel_powers'][ch]['power_db']
                 for band in bands_to_plot]
        ax3.bar(x_positions + i*width, powers, width, label=ch, alpha=0.7)

    ax3.set_xlabel('Gamma Band', fontsize=12)
    ax3.set_ylabel('Power (dB)', fontsize=12)
    ax3.set_title('Channel-wise Gamma Power', fontsize=12, fontweight='bold')
    ax3.set_xticks(x_positions + width * 1.5)
    ax3.set_xticklabels(bands_to_plot, rotation=15, ha='right', fontsize=9)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. パターン判定結果
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    pattern_text = f"""
YOUR GAMMA PATTERN

Pattern Type: {spectral_shape['pattern_type']}
Interpretation: {spectral_shape['interpretation']}

Metrics:
- Spectral Flatness: {spectral_shape['spectral_flatness']:.3f}
- Number of Peaks: {spectral_shape['num_peaks']}
- Power CV: {spectral_shape['power_cv']:.3f}

Broad Gamma (25-42 Hz): {band_powers['Broad Gamma (25-42 Hz, Lutz)']['avg_power_db']:.1f} dB
40Hz Band (38-42 Hz): {band_powers['40Hz Band (38-42 Hz, Davidson)']['avg_power_db']:.1f} dB
"""

    ax4.text(0.05, 0.95, pattern_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 5. 文献比較
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    literature_text = """
MEDITATION RESEARCH FINDINGS

Lutz et al. (2004) - Tibetan Monks:
- Broad gamma (25-42 Hz) increase
- High inter-channel synchrony
- During compassion meditation

Davidson et al. - Long-term meditators:
- 40Hz centered activity
- Increased hemispheric synchrony
- Correlated with practice duration

Beginners:
- Minimal gamma changes
- Low synchrony
"""

    ax5.text(0.05, 0.95, literature_text, transform=ax5.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_comparison_report(band_powers, spectral_shape, comparison, output_path):
    """比較レポートを生成"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 瞑想者のガンマ波パターン比較レポート\n\n")

        f.write("## 質問\n\n")
        f.write("瞑想熟達者のガンマ波は：\n")
        f.write("1. **飛び飛びのピーク**が現れるのか？\n")
        f.write("2. それとも**全体的にバンドパワーが強まる**のか？\n\n")

        f.write("## 研究文献からの知見\n\n")

        for study_name, study_info in comparison['literature'].items():
            f.write(f"### {study_name}\n\n")
            f.write(f"{study_info['description']}\n\n")
            f.write(f"- **ガンマ帯域**: {study_info['gamma_range']}\n")
            f.write(f"- **パターン**: {study_info['pattern']}\n")
            f.write(f"- **同期性**: {study_info['synchrony']}\n\n")

        f.write("## あなたのガンマ波パターン\n\n")

        f.write("### スペクトル形状分析\n\n")
        f.write(f"**判定結果**: {spectral_shape['pattern_type']}\n\n")
        f.write(f"**解釈**: {spectral_shape['interpretation']}\n\n")

        f.write("### 詳細指標\n\n")
        f.write(f"- **スペクトル平坦度**: {spectral_shape['spectral_flatness']:.3f}\n")
        f.write(f"  - 値が高い（>0.7）: 広帯域パワー増加型\n")
        f.write(f"  - 値が低い（<0.5）: 飛び飛びピーク型\n\n")

        f.write(f"- **検出ピーク数**: {spectral_shape['num_peaks']} 個\n")
        f.write(f"  - 多い（>5）: 複数の独立成分\n")
        f.write(f"  - 少ない（<3）: 広帯域活動\n\n")

        f.write(f"- **パワー変動係数**: {spectral_shape['power_cv']:.3f}\n")
        f.write(f"  - 高い（>0.2）: 不均一なスペクトル\n")
        f.write(f"  - 低い（<0.15）: 平坦なスペクトル\n\n")

        f.write("### 文献定義による帯域パワー\n\n")

        f.write("| 帯域定義 | 周波数範囲 | パワー (dB) |\n")
        f.write("|----------|-----------|-------------|\n")

        for band_name, band_data in band_powers.items():
            fmin, fmax = band_data['freq_range']
            power = band_data['avg_power_db']
            f.write(f"| {band_name} | {fmin}-{fmax} Hz | {power:.1f} |\n")

        f.write("\n")

        f.write("## 答え：どちらのパターンなのか？\n\n")

        # 判定ロジック
        flatness = spectral_shape['spectral_flatness']
        num_peaks = spectral_shape['num_peaks']
        cv = spectral_shape['power_cv']

        if flatness > 0.7 and cv < 0.15:
            f.write("### あなたのパターン：**広帯域パワー増加型**\n\n")
            f.write("瞑想熟達者の典型的なパターンに**近い**特徴を示しています。\n\n")
            f.write("**特徴**:\n")
            f.write("- スペクトルが比較的平坦（広帯域活動）\n")
            f.write("- 特定のピークに集中せず、広い範囲でパワーが増加\n")
            f.write("- 瞑想中の統合的な脳活動を示唆\n\n")

            f.write("**文献との比較**:\n")
            f.write("- Lutz et al.の研究と類似したパターン\n")
            f.write("- ただし、熟達者ほどの強いパワーではない可能性\n\n")

        elif num_peaks > 5 and cv > 0.2:
            f.write("### あなたのパターン：**飛び飛びピーク型**\n\n")
            f.write("瞑想熟達者の典型的なパターンとは**異なる**特徴を示しています。\n\n")
            f.write("**特徴**:\n")
            f.write("- 複数の独立した周波数成分（ピーク）\n")
            f.write("- 各ピークは異なる認知プロセスを反映している可能性\n")
            f.write("- まだ統合的なガンマ波活動には至っていない可能性\n\n")

            f.write("**解釈の可能性**:\n")
            f.write("1. **瞑想の発達段階**: まだ初期〜中期段階\n")
            f.write("2. **複数の認知プロセス**: 異なる周波数帯域で独立した処理\n")
            f.write("3. **個人差**: 瞑想スタイルや体質による違い\n\n")

        else:
            f.write("### あなたのパターン：**混合型**\n\n")
            f.write("広帯域成分と飛び飛びピークが**混在**しています。\n\n")
            f.write("**特徴**:\n")
            f.write("- 部分的に広帯域パワー増加\n")
            f.write("- 特定の周波数でのピークも存在\n")
            f.write("- 瞑想の過渡期を示唆する可能性\n\n")

            f.write("**解釈**:\n")
            f.write("熟達者パターンへの移行段階、または個人特有のパターンの可能性があります。\n\n")

        f.write("## 重要な注意事項\n\n")
        f.write("1. **瞑想歴との関係**\n")
        f.write("   - 研究では数千〜数万時間の瞑想経験者が対象\n")
        f.write("   - パターンは瞑想の継続で変化する可能性\n\n")

        f.write("2. **個人差**\n")
        f.write("   - 瞑想スタイル（マインドフルネス、慈悲、集中など）により異なる\n")
        f.write("   - 個人の脳の特性による違い\n\n")

        f.write("3. **測定条件**\n")
        f.write("   - Museは研究用EEGと異なる\n")
        f.write("   - チャネル数と位置の違い\n")
        f.write("   - アーチファクトの影響\n\n")

        f.write("## 推奨される次のステップ\n\n")
        f.write("1. **継続的な測定**: 複数セッションでパターンの変化を追跡\n")
        f.write("2. **瞑想深度との相関**: Fmθ、HbOなどとの関係を調査\n")
        f.write("3. **時系列分析**: セッション中のガンマ波変化を詳細に分析\n")
        f.write("4. **瞑想スタイルの記録**: どの瞑想法でどのパターンが出るか\n")

    return output_path


def main():
    print("="*60)
    print("Gamma Wave Pattern Comparison with Meditation Research")
    print("="*60)
    print()

    print("Loading data...")
    df = load_mind_monitor_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    print()

    print("Preparing MNE raw...")
    mne_dict = prepare_mne_raw(df, apply_notch=False)
    raw = mne_dict['raw']
    print()

    print("Calculating PSD...")
    psd_dict = calculate_psd(raw)
    print()

    print("Analyzing gamma band power (multiple definitions)...")
    band_powers = analyze_gamma_band_power(psd_dict)

    print("\nGamma Band Powers:")
    for band_name, band_data in band_powers.items():
        print(f"  {band_name}: {band_data['avg_power_db']:.1f} dB")
    print()

    print("Analyzing gamma spectral shape...")
    spectral_shape = analyze_gamma_spectral_shape(psd_dict, freq_range=(13, 50))
    print(f"  Pattern Type: {spectral_shape['pattern_type']}")
    print(f"  Interpretation: {spectral_shape['interpretation']}")
    print(f"  Spectral Flatness: {spectral_shape['spectral_flatness']:.3f}")
    print(f"  Number of Peaks: {spectral_shape['num_peaks']}")
    print(f"  Power CV: {spectral_shape['power_cv']:.3f}")
    print()

    # print("Analyzing gamma synchrony...")
    # synchrony = analyze_gamma_synchrony(raw, freq_range=(25, 42))
    # print(f"  Average Synchrony (25-42 Hz): {synchrony['avg_synchrony']:.3f}")
    # print()

    print("Comparing with literature...")
    your_results = {
        'band_powers': band_powers,
        'spectral_shape': spectral_shape,
    }
    comparison = compare_with_literature(your_results)
    print()

    print("Generating plots...")
    plot_path = OUTPUT_DIR / 'gamma_pattern_comparison.png'
    plot_gamma_analysis(psd_dict, band_powers, spectral_shape, plot_path)
    print(f"  Plot saved: {plot_path}")
    print()

    print("Generating comparison report...")
    report_path = OUTPUT_DIR / 'gamma_pattern_comparison_report.md'
    generate_comparison_report(band_powers, spectral_shape, comparison, report_path)
    print(f"  Report saved: {report_path}")
    print()

    print("="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")

    return {
        'band_powers': band_powers,
        'spectral_shape': spectral_shape,
        'comparison': comparison,
    }


if __name__ == '__main__':
    results = main()
