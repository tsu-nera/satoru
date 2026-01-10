"""
Non-4Hz Multiple Peaks and High Gamma Pattern Analysis

分析対象:
1. 4の倍数でないピーク（13.8, 17.6, 21.8, 26.2, 30.2, 34.2 Hz など）
2. AF8のスペクトログラムで見られる高次ガンマ帯（30-50Hz）の模様

これらがアーティファクトか脳波由来かを判定
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
from lib.sensors.eeg import prepare_mne_raw, calculate_psd, calculate_spectrogram

# 出力ディレクトリ
OUTPUT_DIR = Path('/home/tsu-nera/repo/satoru/analysis/2025-12-28_retreat_gamma')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# データパス
DATA_PATH = '/home/tsu-nera/repo/satoru/data/mindMonitor_2025-12-28--19-54-40_5999051304967425957.csv'


def is_4hz_multiple(freq, tolerance=0.5):
    """周波数が4Hzの倍数かどうかを判定"""
    remainder = freq % 4
    return remainder < tolerance or remainder > (4 - tolerance)


def detect_non_4hz_peaks(psd_dict, freq_range=(10, 50), prominence=1.0):
    """
    4Hzの倍数でないピークを検出

    Parameters
    ----------
    psd_dict : dict
        PSD計算結果
    freq_range : tuple
        検出する周波数範囲 (min, max) Hz
    prominence : float
        ピーク検出の閾値（dB）
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    # 周波数範囲でマスク
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_range = freqs[mask]
    psd_range = psd_db[mask]

    # ピーク検出
    peaks, properties = find_peaks(psd_range, prominence=prominence, distance=3)

    peak_freqs = freqs_range[peaks]
    peak_powers = psd_range[peaks]

    # 4Hzの倍数でないピークをフィルタ
    non_4hz_peaks = []
    for freq, power in zip(peak_freqs, peak_powers):
        if not is_4hz_multiple(freq):
            non_4hz_peaks.append({
                'frequency': freq,
                'power_db': power,
                'is_4hz_multiple': False
            })

    # すべてのピーク（比較用）
    all_peaks = []
    for freq, power in zip(peak_freqs, peak_powers):
        all_peaks.append({
            'frequency': freq,
            'power_db': power,
            'is_4hz_multiple': is_4hz_multiple(freq)
        })

    return non_4hz_peaks, all_peaks


def analyze_non_4hz_peak_characteristics(psd_dict, peak_freqs, tolerance=1.0):
    """
    非4Hz倍数ピークの特徴を分析
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    results = []

    for peak_freq in peak_freqs:
        # 対象周波数付近のインデックス
        mask = np.abs(freqs - peak_freq) < tolerance
        if not mask.any():
            continue

        # 各チャネルのパワー
        powers = psds[:, mask].mean(axis=1)

        # チャネル間相関
        correlations = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                corr, _ = pearsonr(psds[i, mask], psds[j, mask])
                correlations.append(corr)

        avg_correlation = np.mean(correlations) if correlations else np.nan
        power_cv = np.std(powers) / np.mean(powers) if np.mean(powers) > 0 else np.nan

        # Q値計算
        psd_avg = np.mean(psds, axis=0)
        psd_db = 10 * np.log10(psd_avg + 1e-10)

        local_mask = (freqs >= peak_freq - 2) & (freqs <= peak_freq + 2)
        local_freqs = freqs[local_mask]
        local_psd = psd_db[local_mask]

        peak_idx = np.argmax(local_psd)
        peak_power = local_psd[peak_idx]

        half_power = peak_power - 3
        above_half = local_psd >= half_power

        if above_half.sum() > 1:
            half_width = local_freqs[above_half].max() - local_freqs[above_half].min()
            q_value = peak_freq / half_width if half_width > 0 else np.inf
        else:
            half_width = np.nan
            q_value = np.nan

        results.append({
            'frequency': peak_freq,
            'power_db': peak_power,
            'avg_correlation': avg_correlation,
            'power_cv': power_cv,
            'q_value': q_value,
            'half_width': half_width,
        })

    return results


def analyze_high_gamma_pattern(raw, channel='RAW_AF8', freq_range=(30, 50)):
    """
    高次ガンマ帯（30-50Hz）の時間変動パターンを分析

    特にAF8チャネルのスペクトログラムで見られる模様を調査
    """
    # 指定チャネルのデータを取得
    ch_idx = raw.ch_names.index(channel)

    # スペクトログラム計算
    from scipy import signal

    data = raw.get_data()[ch_idx]
    sfreq = raw.info['sfreq']

    # スペクトログラム（短時間フーリエ変換）
    nperseg = int(4 * sfreq)  # 4秒窓
    noverlap = int(3.5 * sfreq)  # 0.5秒ステップ

    freqs, times, Sxx = signal.spectrogram(
        data,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling='density'
    )

    # 高次ガンマ帯域を抽出
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    gamma_freqs = freqs[freq_mask]
    gamma_Sxx = Sxx[freq_mask, :]

    # dB変換
    gamma_Sxx_db = 10 * np.log10(gamma_Sxx + 1e-10)

    # 時間平均パワー
    time_avg_power = np.mean(gamma_Sxx_db, axis=0)

    # 周波数平均パワー（どの周波数が強いか）
    freq_avg_power = np.mean(gamma_Sxx_db, axis=1)

    # パターンの周期性を検出（オートコリレーション）
    from scipy.signal import correlate

    # 各周波数での時間変動の自己相関
    autocorrs = []
    for i, freq in enumerate(gamma_freqs):
        signal_power = gamma_Sxx_db[i, :]
        signal_normalized = signal_power - np.mean(signal_power)
        autocorr = correlate(signal_normalized, signal_normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # 正のラグのみ
        autocorrs.append(autocorr[:min(50, len(autocorr))])  # 最初の50ポイント

    autocorrs = np.array(autocorrs)

    # 時間変動の標準偏差（変動の大きさ）
    temporal_std = np.std(gamma_Sxx_db, axis=1)

    # 周波数間の相関（縞模様があるか）
    freq_correlations = []
    for i in range(len(gamma_freqs)):
        for j in range(i + 1, len(gamma_freqs)):
            corr, _ = pearsonr(gamma_Sxx_db[i, :], gamma_Sxx_db[j, :])
            freq_correlations.append(corr)

    avg_freq_correlation = np.mean(freq_correlations) if freq_correlations else np.nan

    return {
        'freqs': gamma_freqs,
        'times': times,
        'spectrogram_db': gamma_Sxx_db,
        'time_avg_power': time_avg_power,
        'freq_avg_power': freq_avg_power,
        'temporal_std': temporal_std,
        'autocorrs': autocorrs,
        'avg_freq_correlation': avg_freq_correlation,
        'channel': channel,
    }


def plot_comprehensive_analysis(psd_dict, raw, non_4hz_peaks, all_peaks, gamma_analysis, output_path):
    """包括的な分析結果をプロット"""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)
    psd_db = 10 * np.log10(psd_avg + 1e-10)

    # 1. PSD全体図（4Hz倍数 vs 非4Hz倍数のピーク）
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(freqs, psd_db, 'b-', linewidth=0.8, label='Average PSD')

    # 全ピークをマーク
    for peak in all_peaks:
        freq = peak['frequency']
        power = peak['power_db']
        if peak['is_4hz_multiple']:
            ax1.plot(freq, power, 'x', color='gray', markersize=8, alpha=0.5, label='4Hz multiple' if freq == all_peaks[0]['frequency'] else '')
        else:
            ax1.plot(freq, power, 'o', color='red', markersize=8, label='Non-4Hz peak' if freq == non_4hz_peaks[0]['frequency'] else '')
            ax1.annotate(f'{freq:.1f}', (freq, power), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power (dB)')
    ax1.set_title('PSD: 4Hz Multiple (×) vs Non-4Hz Peaks (●)')
    ax1.set_xlim(0, 50)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    # 2. 非4Hz倍数ピークの詳細
    ax2 = fig.add_subplot(gs[1, 0])
    if non_4hz_peaks:
        peak_freqs = [p['frequency'] for p in non_4hz_peaks]
        peak_powers = [p['power_db'] for p in non_4hz_peaks]
        colors = plt.cm.viridis(np.linspace(0, 1, len(peak_freqs)))

        ax2.bar(range(len(peak_freqs)), peak_powers, color=colors, alpha=0.7)
        ax2.set_xticks(range(len(peak_freqs)))
        ax2.set_xticklabels([f'{f:.1f}' for f in peak_freqs], rotation=45)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (dB)')
        ax2.set_title('Non-4Hz Peaks Power Distribution')
        ax2.grid(True, alpha=0.3)

    # 3. チャネル別PSD（非4Hz倍数ピークを強調）
    ax3 = fig.add_subplot(gs[1, 1])
    for i, ch in enumerate(psd_dict['channels']):
        ch_psd_db = 10 * np.log10(psd_dict['psds'][i] + 1e-10)
        ax3.plot(freqs, ch_psd_db, label=ch, alpha=0.7)

    for peak in non_4hz_peaks:
        ax3.axvline(peak['frequency'], color='red', linestyle=':', alpha=0.5)

    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Power (dB)')
    ax3.set_title('Channel-wise PSD (Non-4Hz Peaks Marked)')
    ax3.set_xlim(10, 50)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 高次ガンマ帯スペクトログラム（AF8）
    ax4 = fig.add_subplot(gs[2, :])
    im = ax4.pcolormesh(
        gamma_analysis['times'],
        gamma_analysis['freqs'],
        gamma_analysis['spectrogram_db'],
        shading='gouraud',
        cmap='viridis'
    )
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title(f'High Gamma Spectrogram ({gamma_analysis["channel"]}, 30-50 Hz)')
    plt.colorbar(im, ax=ax4, label='Power (dB)')

    # 5. 高次ガンマ帯の周波数別パワー
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.plot(gamma_analysis['freqs'], gamma_analysis['freq_avg_power'], 'b-', linewidth=2)
    ax5.fill_between(
        gamma_analysis['freqs'],
        gamma_analysis['freq_avg_power'] - gamma_analysis['temporal_std'],
        gamma_analysis['freq_avg_power'] + gamma_analysis['temporal_std'],
        alpha=0.3
    )
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Average Power (dB)')
    ax5.set_title('High Gamma: Frequency Profile (Mean ± SD)')
    ax5.grid(True, alpha=0.3)

    # 6. 高次ガンマ帯の時間変動
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.plot(gamma_analysis['times'], gamma_analysis['time_avg_power'], 'g-', linewidth=2)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Average Power (dB)')
    ax6.set_title('High Gamma: Temporal Profile (30-50 Hz Average)')
    ax6.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_report(non_4hz_peaks, peak_characteristics, gamma_analysis, output_path):
    """分析レポートを生成"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 非4Hz倍数ピークと高次ガンマ帯分析レポート\n\n")
        f.write("## セッション情報\n")
        f.write("- **日付**: 2025-12-28\n")
        f.write("- **セッション**: 瞑想リトリート\n\n")

        # 検出されたピーク
        f.write("## 1. 検出された非4Hz倍数ピーク\n\n")
        if non_4hz_peaks:
            f.write(f"検出数: {len(non_4hz_peaks)} 個\n\n")
            f.write("| 周波数 (Hz) | パワー (dB) |\n")
            f.write("|-------------|-------------|\n")
            for peak in sorted(non_4hz_peaks, key=lambda x: x['frequency']):
                f.write(f"| {peak['frequency']:.1f} | {peak['power_db']:.1f} |\n")
        else:
            f.write("非4Hz倍数のピークは検出されませんでした。\n")
        f.write("\n")

        # ピーク特徴分析
        f.write("## 2. ピーク特徴分析\n\n")
        f.write("### 判定基準\n")
        f.write("| 指標 | アーチファクト | 脳波由来 |\n")
        f.write("|------|--------------|----------|\n")
        f.write("| チャネル間相関 | 高い (>0.8) | 低い〜中程度 (<0.5) |\n")
        f.write("| Q値（鋭さ） | 高い (>5) | 低い (<3) |\n")
        f.write("| パワーCV | 低い (<0.3) | 高い (>0.5) |\n\n")

        if peak_characteristics:
            f.write("### 詳細分析\n\n")
            f.write("| 周波数 (Hz) | パワー (dB) | 相関 | CV | Q値 | 判定 |\n")
            f.write("|-------------|-------------|------|-----|-----|------|\n")

            for char in peak_characteristics:
                freq = char['frequency']
                power = char['power_db']
                corr = char['avg_correlation']
                cv = char['power_cv']
                q = char['q_value']

                # 判定ロジック
                votes = {'artifact': 0, 'brain': 0}

                if not np.isnan(corr):
                    if corr > 0.8:
                        votes['artifact'] += 1
                    elif corr < 0.5:
                        votes['brain'] += 1

                if not np.isnan(q) and not np.isinf(q):
                    if q > 5:
                        votes['artifact'] += 1
                    elif q < 3:
                        votes['brain'] += 1

                if not np.isnan(cv):
                    if cv < 0.3:
                        votes['artifact'] += 1
                    elif cv > 0.5:
                        votes['brain'] += 1

                if votes['artifact'] > votes['brain']:
                    judgment = "アーチファクト的"
                elif votes['brain'] > votes['artifact']:
                    judgment = "脳波的"
                else:
                    judgment = "不明確"

                corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
                cv_str = f"{cv:.3f}" if not np.isnan(cv) else "N/A"
                q_str = f"{q:.1f}" if not np.isnan(q) and not np.isinf(q) else "N/A"

                f.write(f"| {freq:.1f} | {power:.1f} | {corr_str} | {cv_str} | {q_str} | {judgment} |\n")
        f.write("\n")

        # 高次ガンマ帯分析
        f.write("## 3. 高次ガンマ帯分析 (30-50 Hz)\n\n")
        f.write(f"### 分析チャネル: {gamma_analysis['channel']}\n\n")

        f.write("### 基本統計\n\n")
        f.write(f"- **平均パワー**: {np.mean(gamma_analysis['freq_avg_power']):.2f} dB\n")
        f.write(f"- **時間変動**: {np.std(gamma_analysis['time_avg_power']):.2f} dB\n")
        f.write(f"- **周波数間相関**: {gamma_analysis['avg_freq_correlation']:.3f}\n\n")

        # 周波数間相関の解釈
        freq_corr = gamma_analysis['avg_freq_correlation']
        f.write("### 周波数間相関の解釈\n\n")
        if freq_corr > 0.7:
            f.write(f"**高い相関 ({freq_corr:.3f})**\n\n")
            f.write("- 全周波数帯域で同期的な変動\n")
            f.write("- **アーチファクトの可能性が高い**\n")
            f.write("- 考えられる原因:\n")
            f.write("  - 電気的ノイズ（筋電図、眼電図）\n")
            f.write("  - デバイス由来のノイズ\n")
            f.write("  - 広帯域のアーチファクト\n")
        elif freq_corr < 0.3:
            f.write(f"**低い相関 ({freq_corr:.3f})**\n\n")
            f.write("- 各周波数が独立して変動\n")
            f.write("- **脳波由来の可能性が高い**\n")
            f.write("- 考えられる解釈:\n")
            f.write("  - 認知処理の活性化\n")
            f.write("  - 注意集中の変化\n")
            f.write("  - 瞑想状態の深化\n")
        else:
            f.write(f"**中程度の相関 ({freq_corr:.3f})**\n\n")
            f.write("- 混合信号の可能性\n")
            f.write("- アーチファクトと脳波が混在\n")
        f.write("\n")

        # 時間変動パターン
        time_std = np.std(gamma_analysis['time_avg_power'])
        f.write("### 時間変動パターン\n\n")
        if time_std < 1.0:
            f.write(f"**安定 (SD={time_std:.2f} dB)**\n\n")
            f.write("- 時間的に一定のパワー\n")
            f.write("- **アーチファクト的な特徴**\n")
        elif time_std > 3.0:
            f.write(f"**変動大 (SD={time_std:.2f} dB)**\n\n")
            f.write("- 時間的に大きく変動\n")
            f.write("- **脳波的な特徴**\n")
            f.write("- セッション中の状態変化を反映\n")
        else:
            f.write(f"**中程度の変動 (SD={time_std:.2f} dB)**\n\n")
            f.write("- 適度な時間変動\n")
        f.write("\n")

        # 総合判定
        f.write("## 4. 総合判定\n\n")

        # 非4Hz倍数ピークの判定集計
        if peak_characteristics:
            artifact_count = 0
            brain_count = 0

            for char in peak_characteristics:
                corr = char['avg_correlation']
                cv = char['power_cv']
                q = char['q_value']

                votes = 0
                if not np.isnan(corr):
                    votes += 1 if corr > 0.8 else (-1 if corr < 0.5 else 0)
                if not np.isnan(q) and not np.isinf(q):
                    votes += 1 if q > 5 else (-1 if q < 3 else 0)
                if not np.isnan(cv):
                    votes += 1 if cv < 0.3 else (-1 if cv > 0.5 else 0)

                if votes > 0:
                    artifact_count += 1
                elif votes < 0:
                    brain_count += 1

            f.write("### 非4Hz倍数ピーク\n\n")
            f.write(f"- アーチファクト的: {artifact_count} 個\n")
            f.write(f"- 脳波的: {brain_count} 個\n\n")

            if artifact_count > brain_count:
                f.write("**結論**: 多くのピークがアーチファクトの可能性\n\n")
            elif brain_count > artifact_count:
                f.write("**結論**: 多くのピークが脳波由来の可能性\n\n")
            else:
                f.write("**結論**: 判定困難（混合信号）\n\n")

        # 高次ガンマ帯の判定
        f.write("### 高次ガンマ帯（30-50 Hz）\n\n")

        gamma_artifact_score = 0
        if freq_corr > 0.7:
            gamma_artifact_score += 1
        if time_std < 1.0:
            gamma_artifact_score += 1

        if gamma_artifact_score >= 2:
            f.write("**結論**: アーチファクトの可能性が高い\n\n")
            f.write("**推奨対策**:\n")
            f.write("- 筋電図（EMG）のチェック\n")
            f.write("- 眼球運動のチェック\n")
            f.write("- ICA による成分分離\n")
        elif gamma_artifact_score == 0:
            f.write("**結論**: 脳波由来の可能性が高い\n\n")
            f.write("**解釈の可能性**:\n")
            f.write("- 高次認知処理の活性化\n")
            f.write("- 瞑想中の注意集中\n")
            f.write("- ガンマ波バースト（瞑想関連）\n")
        else:
            f.write("**結論**: 判定困難（混合信号）\n\n")

        f.write("\n## 5. 推奨される追加分析\n\n")
        f.write("1. **EMG（筋電図）チェック**: 顔面・頭部の筋緊張の影響を確認\n")
        f.write("2. **EOG（眼電図）チェック**: 眼球運動のアーチファクトを確認\n")
        f.write("3. **ICA分析**: 独立成分分析でアーチファクト成分を分離\n")
        f.write("4. **他のセッションとの比較**: 再現性の確認\n")
        f.write("5. **安静時データとの比較**: 瞑想特有の変化かを判定\n")

    return output_path


def main():
    print("="*60)
    print("Non-4Hz Multiple Peaks and High Gamma Pattern Analysis")
    print("="*60)
    print()

    print("Loading data...")
    df = load_mind_monitor_csv(DATA_PATH)
    print(f"Loaded {len(df)} records")
    print()

    print("Preparing MNE raw...")
    mne_dict = prepare_mne_raw(df, apply_notch=False)
    raw = mne_dict['raw']
    print(f"Sampling rate: {raw.info['sfreq']:.2f} Hz")
    print(f"Channels: {mne_dict['channels']}")
    print()

    print("Calculating PSD...")
    psd_dict = calculate_psd(raw)
    print()

    print("Detecting non-4Hz multiple peaks...")
    non_4hz_peaks, all_peaks = detect_non_4hz_peaks(psd_dict, freq_range=(10, 50), prominence=1.0)
    print(f"  Detected {len(all_peaks)} total peaks")
    print(f"  Non-4Hz peaks: {len(non_4hz_peaks)}")

    if non_4hz_peaks:
        print("  Non-4Hz peak frequencies:")
        for peak in sorted(non_4hz_peaks, key=lambda x: x['frequency']):
            print(f"    {peak['frequency']:.1f} Hz: {peak['power_db']:.1f} dB")
    print()

    print("Analyzing non-4Hz peak characteristics...")
    peak_freqs = [p['frequency'] for p in non_4hz_peaks]
    peak_characteristics = analyze_non_4hz_peak_characteristics(psd_dict, peak_freqs)
    print()

    print("Analyzing high gamma pattern (AF8, 30-50 Hz)...")
    gamma_analysis = analyze_high_gamma_pattern(raw, channel='RAW_AF8', freq_range=(30, 50))
    print(f"  Frequency correlation: {gamma_analysis['avg_freq_correlation']:.3f}")
    print(f"  Temporal variation (SD): {np.std(gamma_analysis['time_avg_power']):.2f} dB")
    print()

    print("Generating plots...")
    plot_path = OUTPUT_DIR / 'non_4hz_peaks_analysis.png'
    plot_comprehensive_analysis(psd_dict, raw, non_4hz_peaks, all_peaks, gamma_analysis, plot_path)
    print(f"  Plot saved: {plot_path}")
    print()

    print("Generating report...")
    report_path = OUTPUT_DIR / 'non_4hz_peaks_analysis_report.md'
    generate_report(non_4hz_peaks, peak_characteristics, gamma_analysis, report_path)
    print(f"  Report saved: {report_path}")
    print()

    print("="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"Report: {report_path}")
    print(f"Plot: {plot_path}")

    return {
        'non_4hz_peaks': non_4hz_peaks,
        'all_peaks': all_peaks,
        'peak_characteristics': peak_characteristics,
        'gamma_analysis': gamma_analysis,
    }


if __name__ == '__main__':
    results = main()
