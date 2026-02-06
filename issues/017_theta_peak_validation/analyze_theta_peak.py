#!/usr/bin/env python3
"""
シータ波ピーク(5.5-6.0 Hz)の妥当性検証
- 接続品質との相関
- 空間分布分析
- 時間的安定性
- 瞑想指標との相関
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, stats
from typing import Dict, Tuple
import sys

# プロジェクトルートのlibをインポートパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))
from lib.loaders.mind_monitor import load_mind_monitor_csv, get_eeg_data

# 出力ディレクトリ
OUTPUT_DIR = Path(__file__).parent / "theta_peak_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# スタイル設定
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def compute_psd_with_peaks(
    data: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float] = (1, 30)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """PSD計算とピーク検出"""
    freqs, psd = signal.welch(
        data,
        fs=fs,
        nperseg=min(len(data), 4*fs),
        noverlap=min(len(data), 2*fs)
    )

    # 周波数範囲でフィルタ
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs = freqs[mask]
    psd = psd[mask]

    # ピーク検出 (prominence=パワーの5%以上)
    psd_db = 10 * np.log10(psd + 1e-12)
    peaks, properties = signal.find_peaks(
        psd_db,
        prominence=np.ptp(psd_db) * 0.05,
        distance=int(0.5 / (freqs[1] - freqs[0]))  # 0.5Hz以上離れている
    )

    return freqs, psd_db, peaks, properties


def analyze_theta_peak_by_quality(
    csv_path: str,
    theta_range: Tuple[float, float] = (5.0, 7.0)
) -> pd.DataFrame:
    """接続品質別のシータピーク分析"""

    results = []

    # データロード
    df = load_mind_monitor_csv(csv_path, filter_headband=True, warmup_seconds=60.0)

    # EEGデータ取得
    eeg_dict = get_eeg_data(df)
    if eeg_dict is None:
        raise ValueError("EEG data not found in CSV")

    # HSIデータ取得
    hsi_columns = ['HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10']
    if not all(col in df.columns for col in hsi_columns):
        raise ValueError("HSI data not found in CSV")

    # 30秒ウィンドウで分析
    window_size = 30  # seconds
    fs = 256
    window_samples = window_size * fs

    n_windows = len(eeg_dict['TP9']) // window_samples

    for i in range(n_windows):
        start_idx = i * window_samples
        end_idx = start_idx + window_samples

        # HSIデータをインデックスベースで取得
        window_hsi = df[hsi_columns].iloc[start_idx:end_idx]

        # 平均接続品質
        avg_quality = window_hsi.mean().mean()  # 全チャンネル平均
        quality_category = 'Good' if avg_quality < 1.5 else 'Medium' if avg_quality < 3.0 else 'Bad'

        # 各チャンネルでPSD計算
        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            window_data = eeg_dict[ch][start_idx:end_idx]

            freqs, psd_db, peaks, props = compute_psd_with_peaks(
                window_data,
                fs=fs,
                freq_range=(1, 30)
            )

            # シータ帯域のピーク検出
            theta_mask = (freqs[peaks] >= theta_range[0]) & (freqs[peaks] <= theta_range[1])
            theta_peaks = peaks[theta_mask]

            if len(theta_peaks) > 0:
                # 最も強いピークを選択
                max_peak_idx = theta_peaks[np.argmax(psd_db[theta_peaks])]
                peak_freq = freqs[max_peak_idx]
                peak_power = psd_db[max_peak_idx]
                peak_prominence = props['prominences'][np.where(peaks == max_peak_idx)[0][0]]
            else:
                peak_freq = np.nan
                peak_power = np.nan
                peak_prominence = np.nan

            # シータ帯域の平均パワー
            theta_band_mask = (freqs >= 4) & (freqs <= 8)
            theta_avg_power = np.mean(psd_db[theta_band_mask])

            results.append({
                'window': i,
                'time_min': i * window_size / 60,
                'channel': ch,
                'avg_quality': avg_quality,
                'quality_category': quality_category,
                'theta_peak_freq': peak_freq,
                'theta_peak_power': peak_power,
                'theta_peak_prominence': peak_prominence,
                'theta_avg_power': theta_avg_power,
                'has_peak': not np.isnan(peak_freq)
            })

    return pd.DataFrame(results)


def analyze_spatial_distribution(
    csv_path: str,
    theta_range: Tuple[float, float] = (5.0, 7.0)
) -> Dict:
    """シータピークの空間分布分析"""

    # データロード
    df = load_mind_monitor_csv(csv_path, filter_headband=True, warmup_seconds=60.0)

    # EEGデータ取得
    eeg_dict = get_eeg_data(df)
    if eeg_dict is None:
        raise ValueError("EEG data not found in CSV")

    fs = 256
    channel_peaks = {}

    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        freqs, psd_db, peaks, props = compute_psd_with_peaks(
            eeg_dict[ch],
            fs=fs,
            freq_range=(1, 30)
        )

        # シータ帯域のピーク
        theta_mask = (freqs[peaks] >= theta_range[0]) & (freqs[peaks] <= theta_range[1])
        theta_peaks = peaks[theta_mask]

        if len(theta_peaks) > 0:
            max_peak_idx = theta_peaks[np.argmax(psd_db[theta_peaks])]
            channel_peaks[ch] = {
                'freq': freqs[max_peak_idx],
                'power': psd_db[max_peak_idx],
                'prominence': props['prominences'][np.where(peaks == max_peak_idx)[0][0]]
            }
        else:
            channel_peaks[ch] = {'freq': np.nan, 'power': np.nan, 'prominence': np.nan}

    return channel_peaks


def plot_quality_vs_theta_peak(df: pd.DataFrame, output_dir: Path):
    """接続品質とシータピーク出現率の関係"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 品質カテゴリ別のピーク出現率
    ax = axes[0, 0]
    peak_rate = df.groupby('quality_category')['has_peak'].mean() * 100
    peak_rate.plot(kind='bar', ax=ax, color=['green', 'orange', 'red'])
    ax.set_ylabel('Peak Detection Rate (%)')
    ax.set_xlabel('Connection Quality')
    ax.set_title('Theta Peak Detection Rate by Connection Quality')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # 2. 品質スコアとピーク周波数の散布図
    ax = axes[0, 1]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ch_data = df[df['channel'] == ch]
        valid_data = ch_data[ch_data['has_peak']]
        ax.scatter(valid_data['avg_quality'], valid_data['theta_peak_freq'],
                  alpha=0.6, label=ch, s=30)
    ax.set_xlabel('Average HSI Quality Score')
    ax.set_ylabel('Theta Peak Frequency (Hz)')
    ax.set_title('Connection Quality vs Theta Peak Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=5.5, color='gray', linestyle='--', alpha=0.5, label='5.5 Hz')
    ax.axhline(y=6.0, color='gray', linestyle='--', alpha=0.5, label='6.0 Hz')

    # 3. 品質スコアとピークパワーの散布図
    ax = axes[1, 0]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ch_data = df[df['channel'] == ch]
        valid_data = ch_data[ch_data['has_peak']]
        ax.scatter(valid_data['avg_quality'], valid_data['theta_peak_power'],
                  alpha=0.6, label=ch, s=30)
    ax.set_xlabel('Average HSI Quality Score')
    ax.set_ylabel('Theta Peak Power (dB)')
    ax.set_title('Connection Quality vs Theta Peak Power')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 品質スコアとピーク顕著性の散布図
    ax = axes[1, 1]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ch_data = df[df['channel'] == ch]
        valid_data = ch_data[ch_data['has_peak']]
        ax.scatter(valid_data['avg_quality'], valid_data['theta_peak_prominence'],
                  alpha=0.6, label=ch, s=30)
    ax.set_xlabel('Average HSI Quality Score')
    ax.set_ylabel('Peak Prominence (dB)')
    ax.set_title('Connection Quality vs Peak Prominence')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'quality_vs_theta_peak.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_spatial_distribution(channel_peaks: Dict, output_dir: Path):
    """チャンネル別のシータピーク分布"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # 1. ピーク周波数
    ax = axes[0]
    freqs = [channel_peaks[ch]['freq'] for ch in channels]
    colors = ['blue' if not np.isnan(f) else 'gray' for f in freqs]
    ax.bar(channels, freqs, color=colors, alpha=0.7)
    ax.set_ylabel('Peak Frequency (Hz)')
    ax.set_title('Theta Peak Frequency by Channel')
    ax.axhline(y=5.5, color='red', linestyle='--', alpha=0.5, label='5.5 Hz')
    ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.5, label='6.0 Hz')
    ax.set_ylim(4, 8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ピークパワー
    ax = axes[1]
    powers = [channel_peaks[ch]['power'] for ch in channels]
    colors = ['green' if not np.isnan(p) else 'gray' for p in powers]
    ax.bar(channels, powers, color=colors, alpha=0.7)
    ax.set_ylabel('Peak Power (dB)')
    ax.set_title('Theta Peak Power by Channel')
    ax.grid(True, alpha=0.3)

    # 3. ピーク顕著性
    ax = axes[2]
    proms = [channel_peaks[ch]['prominence'] for ch in channels]
    colors = ['orange' if not np.isnan(p) else 'gray' for p in proms]
    ax.bar(channels, proms, color=colors, alpha=0.7)
    ax.set_ylabel('Peak Prominence (dB)')
    ax.set_title('Theta Peak Prominence by Channel')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_temporal_stability(df: pd.DataFrame, output_dir: Path):
    """時間的安定性の分析"""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. ピーク周波数の時系列
    ax = axes[0]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ch_data = df[df['channel'] == ch]
        ax.plot(ch_data['time_min'], ch_data['theta_peak_freq'],
               marker='o', label=ch, alpha=0.7, markersize=4)
    ax.set_ylabel('Peak Frequency (Hz)')
    ax.set_title('Theta Peak Frequency Over Time')
    ax.axhline(y=5.5, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.5)
    ax.set_ylim(4, 8)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. ピークパワーの時系列
    ax = axes[1]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ch_data = df[df['channel'] == ch]
        ax.plot(ch_data['time_min'], ch_data['theta_peak_power'],
               marker='o', label=ch, alpha=0.7, markersize=4)
    ax.set_ylabel('Peak Power (dB)')
    ax.set_title('Theta Peak Power Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 接続品質の時系列
    ax = axes[2]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ch_data = df[df['channel'] == ch]
        ax.plot(ch_data['time_min'], ch_data['avg_quality'],
               marker='o', label=ch, alpha=0.7, markersize=4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('HSI Quality Score')
    ax.set_title('Connection Quality Over Time')
    ax.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='Good/Medium')
    ax.axhline(y=3.0, color='orange', linestyle='--', alpha=0.5, label='Medium/Bad')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'temporal_stability.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(
    df: pd.DataFrame,
    channel_peaks: Dict,
    data_file: str,
    output_dir: Path
):
    """分析レポート生成"""

    report = f"""# Theta Peak Validation Analysis

**Data File**: `{data_file}`
**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Theta Range**: 5.0 - 7.0 Hz

---

## Summary Statistics

### Overall Peak Detection Rate

"""

    # ピーク検出率
    total_windows = len(df) // 4  # 4チャンネル
    peak_detected = df.groupby('window')['has_peak'].any().sum()
    detection_rate = (peak_detected / total_windows) * 100

    report += f"- **Total Windows**: {total_windows}\n"
    report += f"- **Windows with Peak**: {peak_detected}\n"
    report += f"- **Detection Rate**: {detection_rate:.1f}%\n\n"

    # 品質別の検出率
    report += "### Detection Rate by Connection Quality\n\n"
    quality_stats = df.groupby('quality_category').agg({
        'has_peak': ['sum', 'count', 'mean']
    }).round(3)
    quality_stats.columns = ['Peaks Detected', 'Total', 'Detection Rate']
    quality_stats['Detection Rate'] = quality_stats['Detection Rate'] * 100
    report += quality_stats.to_markdown() + "\n\n"

    # チャンネル別統計
    report += "### Channel-wise Statistics\n\n"

    valid_peaks = df[df['has_peak']]

    if len(valid_peaks) > 0:
        channel_stats = valid_peaks.groupby('channel').agg({
            'theta_peak_freq': ['mean', 'std', 'min', 'max'],
            'theta_peak_power': ['mean', 'std'],
            'theta_peak_prominence': ['mean', 'std']
        }).round(2)

        channel_stats.columns = ['_'.join(col) for col in channel_stats.columns]
        report += channel_stats.to_markdown() + "\n\n"

        # 相関分析
        report += "## Correlation Analysis\n\n"
        report += "### Peak Frequency vs Connection Quality\n\n"

        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            ch_data = valid_peaks[valid_peaks['channel'] == ch]
            if len(ch_data) > 3:
                corr, p_value = stats.pearsonr(ch_data['avg_quality'], ch_data['theta_peak_freq'])
                sig = "**Significant**" if p_value < 0.05 else "Not significant"
                report += f"- **{ch}**: r = {corr:.3f}, p = {p_value:.3f} ({sig})\n"

        report += "\n### Peak Power vs Connection Quality\n\n"

        for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
            ch_data = valid_peaks[valid_peaks['channel'] == ch]
            if len(ch_data) > 3:
                corr, p_value = stats.pearsonr(ch_data['avg_quality'], ch_data['theta_peak_power'])
                sig = "**Significant**" if p_value < 0.05 else "Not significant"
                report += f"- **{ch}**: r = {corr:.3f}, p = {p_value:.3f} ({sig})\n"

    # 空間分布
    report += "\n\n## Spatial Distribution (Full Session)\n\n"
    report += "| Channel | Peak Freq (Hz) | Peak Power (dB) | Prominence (dB) |\n"
    report += "|---------|----------------|-----------------|------------------|\n"

    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        freq = channel_peaks[ch]['freq']
        power = channel_peaks[ch]['power']
        prom = channel_peaks[ch]['prominence']

        freq_str = f"{freq:.2f}" if not np.isnan(freq) else "No peak"
        power_str = f"{power:.2f}" if not np.isnan(power) else "-"
        prom_str = f"{prom:.2f}" if not np.isnan(prom) else "-"

        report += f"| {ch} | {freq_str} | {power_str} | {prom_str} |\n"

    # 結論
    report += "\n\n## Interpretation\n\n"

    # 前頭部優位性チェック
    frontal_has_peak = not (np.isnan(channel_peaks['AF7']['freq']) and np.isnan(channel_peaks['AF8']['freq']))
    temporal_has_peak = not (np.isnan(channel_peaks['TP9']['freq']) and np.isnan(channel_peaks['TP10']['freq']))

    if frontal_has_peak:
        report += "### ✅ Frontal Theta Peak Detected\n\n"
        report += "前頭部(AF7/AF8)でシータピークが検出されました。これは**Frontal Midline Theta (FMT)**の特徴と一致し、瞑想状態における生理学的な現象を示唆します。\n\n"

    # 品質との相関チェック
    good_quality_rate = df[df['quality_category'] == 'Good']['has_peak'].mean() * 100
    bad_quality_rate = df[df['quality_category'] == 'Bad']['has_peak'].mean() * 100

    if good_quality_rate > bad_quality_rate * 1.5:
        report += "### ⚠️ Quality-Dependent Detection\n\n"
        report += f"接続品質が良い時の検出率({good_quality_rate:.1f}%)が悪い時({bad_quality_rate:.1f}%)より明らかに高いです。これは一部がアーチファクトである可能性を示唆します。\n\n"
    else:
        report += "### ✅ Quality-Independent Detection\n\n"
        report += "接続品質に関わらず比較的安定してピークが検出されています。これは生理学的な現象である可能性を支持します。\n\n"

    # 周波数範囲チェック
    if len(valid_peaks) > 0:
        mean_freq = valid_peaks['theta_peak_freq'].mean()
        std_freq = valid_peaks['theta_peak_freq'].std()

        if 5.0 <= mean_freq <= 7.0:
            report += f"### ✅ Physiologically Valid Frequency Range\n\n"
            report += f"平均ピーク周波数 {mean_freq:.2f} ± {std_freq:.2f} Hz は、瞑想研究で報告されているシータ帯域範囲内です。\n\n"

    report += "---\n\n"
    report += "## Figures\n\n"
    report += "1. [Connection Quality vs Theta Peak](quality_vs_theta_peak.png)\n"
    report += "2. [Spatial Distribution](spatial_distribution.png)\n"
    report += "3. [Temporal Stability](temporal_stability.png)\n"

    # レポート保存
    with open(output_dir / 'THETA_PEAK_VALIDATION.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)


def main():
    """メイン処理"""

    # 最新のデータファイルを使用
    data_dir = Path(__file__).parent.parent.parent / "data" / "muse"
    csv_files = sorted(data_dir.glob("mindMonitor_*.csv"))

    if not csv_files:
        print("No data files found!")
        return

    # 最新ファイル
    latest_file = csv_files[-1]
    print(f"Analyzing: {latest_file.name}")

    # 分析1: 接続品質別のシータピーク分析
    print("\n1. Analyzing theta peaks by connection quality...")
    df_quality = analyze_theta_peak_by_quality(str(latest_file))
    df_quality.to_csv(OUTPUT_DIR / 'theta_peak_by_quality.csv', index=False)

    # 分析2: 空間分布分析
    print("2. Analyzing spatial distribution...")
    channel_peaks = analyze_spatial_distribution(str(latest_file))

    # プロット生成
    print("3. Generating plots...")
    plot_quality_vs_theta_peak(df_quality, OUTPUT_DIR)
    plot_spatial_distribution(channel_peaks, OUTPUT_DIR)
    plot_temporal_stability(df_quality, OUTPUT_DIR)

    # レポート生成
    print("4. Generating report...")
    generate_report(df_quality, channel_peaks, latest_file.name, OUTPUT_DIR)

    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
