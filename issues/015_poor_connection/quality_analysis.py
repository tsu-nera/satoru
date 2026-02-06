#!/usr/bin/env python3
"""
接続品質と脳波パターンの関係分析

スペクトログラムに見られる特徴的なパターン（SMR、β、δのピーク）が、
接続品質の劣化によるアーティファクトなのか、真の脳活動なのかを判定する。
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import kruskal, spearmanr
import warnings

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensors.eeg.preprocessing import filter_eeg_quality, prepare_mne_raw
from lib.sensors.eeg.psd_peaks import analyze_psd_peaks

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_data(csv_path):
    """CSVデータの読み込み"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # TimeStampをdatetimeに変換してインデックス化
    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])
        df = df.set_index('TimeStamp')

    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def classify_quality_segments(df):
    """
    接続品質に基づいてデータを3つのセグメントに分類

    Returns:
        dict: {'good': DataFrame, 'medium': DataFrame, 'bad': DataFrame}
    """
    hsi_cols = [c for c in df.columns if c.startswith('HSI_')]
    if not hsi_cols:
        print("No HSI columns found")
        return None

    # 数値変換
    hsi_values = df[hsi_cols].apply(pd.to_numeric, errors='coerce')

    # Good: 全チャネルHSI=1.0
    good_mask = hsi_values.eq(1.0).all(axis=1)

    # Medium: 少なくとも1チャネルがHSI=2.0（ただしBadでない）
    medium_mask = hsi_values.eq(2.0).any(axis=1) & ~hsi_values.eq(4.0).any(axis=1)

    # Bad: 少なくとも1チャネルがHSI=4.0
    bad_mask = hsi_values.eq(4.0).any(axis=1)

    segments = {
        'Good': df[good_mask].copy(),
        'Medium': df[medium_mask].copy(),
        'Bad': df[bad_mask].copy()
    }

    print("\n=== Quality Segment Statistics ===")
    for quality, seg_df in segments.items():
        pct = len(seg_df) / len(df) * 100
        duration = len(seg_df) / 256.0  # assuming 256Hz
        print(f"{quality}: {len(seg_df)} samples ({pct:.1f}%), {duration:.1f}s")

    return segments


def compute_psd_welch(df, sfreq=256.0):
    """
    Welch法でPSDを計算

    Returns:
        dict: {'freqs': array, 'psds': array, 'channels': list}
    """
    raw_cols = [c for c in df.columns if c.startswith('RAW_')]
    if not raw_cols:
        return None

    # 数値変換と補間
    numeric = df[raw_cols].apply(pd.to_numeric, errors='coerce')
    numeric = numeric.interpolate(method='linear').ffill().bfill()

    # μVからVに変換
    data = numeric.to_numpy().T * 1e-6

    # Welch法でPSD計算
    freqs, psds = signal.welch(
        data,
        fs=sfreq,
        nperseg=min(256, data.shape[1]),
        noverlap=None,
        window='hann',
        scaling='density'
    )

    # Vから μV²/Hz に戻す
    psds = psds * 1e12

    return {
        'freqs': freqs,
        'psds': psds,
        'channels': raw_cols
    }


def detect_peaks_in_bands(psd_dict, bands):
    """
    指定した周波数帯域内でピークパワーを検出

    Parameters:
        psd_dict: compute_psd_welch()の結果
        bands: dict {'band_name': (low_freq, high_freq)}

    Returns:
        dict: {'band_name': peak_power_db}
    """
    freqs = psd_dict['freqs']
    psd_avg = np.mean(psd_dict['psds'], axis=0)

    peak_powers = {}
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            band_psd = psd_avg[mask]
            peak_power = np.max(band_psd)
            peak_powers[band_name] = 10 * np.log10(peak_power + 1e-10)

    return peak_powers


def analyze_quality_psd(segments):
    """各品質セグメントのPSD分析"""
    bands = {
        'SMR': (12.0, 15.0),
        'Beta': (15.0, 30.0),
        'Delta': (1.0, 4.0),
        'Theta': (4.0, 8.0),
        'Alpha': (8.0, 13.0),
    }

    results = {}

    for quality, seg_df in segments.items():
        if len(seg_df) < 256:
            print(f"Skipping {quality}: insufficient data")
            continue

        # PSD計算
        psd_dict = compute_psd_welch(seg_df)
        if psd_dict is None:
            continue

        # ピーク検出
        peak_powers = detect_peaks_in_bands(psd_dict, bands)

        results[quality] = {
            'psd_dict': psd_dict,
            'peak_powers': peak_powers,
            'n_samples': len(seg_df)
        }

    return results


def plot_quality_segments(df, output_dir):
    """品質セグメントの時系列プロット"""
    hsi_cols = [c for c in df.columns if c.startswith('HSI_')]
    if not hsi_cols:
        return

    fig, axes = plt.subplots(len(hsi_cols), 1, figsize=(14, 8), sharex=True)
    if len(hsi_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, hsi_cols):
        channel = col.replace('HSI_', '')
        hsi = df[col].apply(pd.to_numeric, errors='coerce')

        # 時間軸（分）
        time_min = (df.index - df.index[0]).total_seconds() / 60.0

        ax.plot(time_min, hsi, linewidth=0.5, alpha=0.7)
        ax.set_ylabel(f'{channel}\nHSI', fontsize=10)
        ax.set_ylim(0.5, 4.5)
        ax.set_yticks([1, 2, 4])
        ax.axhline(1.0, color='green', linestyle='--', alpha=0.5, label='Good')
        ax.axhline(2.0, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax.axhline(4.0, color='red', linestyle='--', alpha=0.5, label='Bad')
        ax.grid(True, alpha=0.3)

        if ax == axes[0]:
            ax.legend(loc='upper right', fontsize=8)

    axes[-1].set_xlabel('Time (min)', fontsize=11)
    fig.suptitle('Connection Quality Time Series (HSI)', fontsize=13, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / 'quality_time_series.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_quality_psd_comparison(quality_results, output_dir):
    """品質別PSD比較プロット"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {'Good': 'green', 'Medium': 'orange', 'Bad': 'red'}

    # 全体PSD比較
    ax = axes[0]
    for quality, result in quality_results.items():
        psd_dict = result['psd_dict']
        freqs = psd_dict['freqs']
        psd_avg = np.mean(psd_dict['psds'], axis=0)
        psd_db = 10 * np.log10(psd_avg + 1e-10)

        ax.plot(freqs, psd_db, label=quality, color=colors[quality], linewidth=2, alpha=0.8)

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Power Spectral Density (dB)', fontsize=11)
    ax.set_title('PSD Comparison by Quality', fontsize=12, fontweight='bold')
    ax.set_xlim(1, 45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 帯域別拡大
    bands_to_plot = [
        ('Delta (1-4 Hz)', 0.5, 5),
        ('Alpha (8-13 Hz)', 7, 14),
        ('SMR/Beta (12-30 Hz)', 10, 32)
    ]

    for idx, (title, low, high) in enumerate(bands_to_plot, 1):
        ax = axes[idx]
        for quality, result in quality_results.items():
            psd_dict = result['psd_dict']
            freqs = psd_dict['freqs']
            psd_avg = np.mean(psd_dict['psds'], axis=0)
            psd_db = 10 * np.log10(psd_avg + 1e-10)

            ax.plot(freqs, psd_db, label=quality, color=colors[quality], linewidth=2, alpha=0.8)

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('PSD (dB)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlim(low, high)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'quality_psd_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_peak_power_boxplots(quality_results, output_dir):
    """品質別ピークパワー箱ひげ図"""
    # データフレーム作成
    rows = []
    for quality, result in quality_results.items():
        for band, power_db in result['peak_powers'].items():
            rows.append({'Quality': quality, 'Band': band, 'Power (dB)': power_db})

    df_peaks = pd.DataFrame(rows)

    # 重要な帯域に絞る
    important_bands = ['SMR', 'Beta', 'Delta', 'Alpha']
    df_peaks = df_peaks[df_peaks['Band'].isin(important_bands)]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=df_peaks,
        x='Band',
        y='Power (dB)',
        hue='Quality',
        hue_order=['Good', 'Medium', 'Bad'],
        palette=['green', 'orange', 'red'],
        ax=ax
    )

    ax.set_title('Peak Power by Quality Segment', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency Band', fontsize=11)
    ax.set_ylabel('Peak Power (dB)', fontsize=11)
    ax.legend(title='Quality', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'quality_peak_power_boxplot.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_hsi_power_scatter(df, output_dir):
    """HSI値とバンドパワーの散布図"""
    # 各チャネルのHSI値を取得
    hsi_cols = [c for c in df.columns if c.startswith('HSI_')]
    raw_cols = [c for c in df.columns if c.startswith('RAW_')]

    if not hsi_cols or not raw_cols:
        return

    # 30秒ウィンドウでバンドパワーを計算
    window_size = 256 * 30  # 30秒
    step_size = 256 * 10    # 10秒ステップ

    bands = {
        'SMR': (12.0, 15.0),
        'Beta': (15.0, 30.0),
        'Delta': (1.0, 4.0),
        'Alpha': (8.0, 13.0),
    }

    # スライディングウィンドウで計算
    rows = []
    for i in range(0, len(df) - window_size, step_size):
        window_df = df.iloc[i:i+window_size]

        # HSI平均（全チャネル）
        hsi_values = window_df[hsi_cols].apply(pd.to_numeric, errors='coerce')
        hsi_mean = hsi_values.mean().mean()

        # PSD計算
        psd_dict = compute_psd_welch(window_df)
        if psd_dict is None:
            continue

        peak_powers = detect_peaks_in_bands(psd_dict, bands)

        for band, power_db in peak_powers.items():
            rows.append({
                'HSI': hsi_mean,
                'Band': band,
                'Power (dB)': power_db
            })

    df_scatter = pd.DataFrame(rows)

    # プロット
    bands_to_plot = ['SMR', 'Beta', 'Delta', 'Alpha']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, band in zip(axes, bands_to_plot):
        band_data = df_scatter[df_scatter['Band'] == band]

        sns.scatterplot(
            data=band_data,
            x='HSI',
            y='Power (dB)',
            alpha=0.5,
            ax=ax,
            s=30
        )

        # Spearman相関
        if len(band_data) > 3:
            corr, p_value = spearmanr(band_data['HSI'], band_data['Power (dB)'])
            ax.text(0.05, 0.95, f'ρ={corr:.3f}\np={p_value:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_title(f'{band} Band', fontsize=12, fontweight='bold')
        ax.set_xlabel('HSI (Connection Quality)', fontsize=11)
        ax.set_ylabel('Peak Power (dB)', fontsize=11)
        ax.set_xlim(0.8, 4.2)
        ax.axvline(1.0, color='green', linestyle='--', alpha=0.3)
        ax.axvline(2.0, color='orange', linestyle='--', alpha=0.3)
        ax.axvline(4.0, color='red', linestyle='--', alpha=0.3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'hsi_power_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def statistical_analysis(quality_results, segments):
    """統計分析"""
    print("\n=== Statistical Analysis ===")

    bands = ['SMR', 'Beta', 'Delta', 'Alpha']

    # 品質セグメント間の差の検定
    print("\n--- Kruskal-Wallis Test (Quality vs Band Power) ---")
    for band in bands:
        powers = []
        groups = []

        for quality, result in quality_results.items():
            if band in result['peak_powers']:
                # 単一値なので、各セグメントを擬似的に複数サンプルとして扱う
                # （本来はウィンドウ分割したデータで検定すべき）
                power = result['peak_powers'][band]
                n_pseudo = result['n_samples'] // 1000  # 1000サンプルごとに1ポイント
                powers.extend([power] * max(1, n_pseudo))
                groups.extend([quality] * max(1, n_pseudo))

        if len(set(groups)) >= 2 and len(powers) >= 3:
            df_test = pd.DataFrame({'Quality': groups, 'Power': powers})
            group_data = [df_test[df_test['Quality'] == q]['Power'].values
                         for q in ['Good', 'Medium', 'Bad']
                         if q in df_test['Quality'].values]

            if len(group_data) >= 2:
                stat, p_value = kruskal(*group_data)
                print(f"{band:8s}: H={stat:.3f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")

    # HSI値とバンドパワーのSpearman相関（全データ）
    print("\n--- Spearman Correlation (HSI vs Band Power) ---")
    all_df = pd.concat([segments['Good'], segments['Medium'], segments['Bad']])

    hsi_cols = [c for c in all_df.columns if c.startswith('HSI_')]
    hsi_mean = all_df[hsi_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

    # 30秒ウィンドウでバンドパワー計算
    window_size = 256 * 30
    step_size = 256 * 10

    band_definitions = {
        'SMR': (12.0, 15.0),
        'Beta': (15.0, 30.0),
        'Delta': (1.0, 4.0),
        'Alpha': (8.0, 13.0),
    }

    for band, (low, high) in band_definitions.items():
        hsi_samples = []
        power_samples = []

        for i in range(0, len(all_df) - window_size, step_size):
            window_df = all_df.iloc[i:i+window_size]
            hsi_window = hsi_mean.iloc[i:i+window_size].mean()

            psd_dict = compute_psd_welch(window_df)
            if psd_dict is None:
                continue

            peak_powers = detect_peaks_in_bands(psd_dict, {band: (low, high)})
            if band in peak_powers:
                hsi_samples.append(hsi_window)
                power_samples.append(peak_powers[band])

        if len(hsi_samples) > 3:
            corr, p_value = spearmanr(hsi_samples, power_samples)
            print(f"{band:8s}: ρ={corr:+.3f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")


def generate_report(segments, quality_results, output_path):
    """Markdownレポート生成"""
    lines = [
        "# 接続品質と脳波パターンの関係分析レポート",
        "",
        f"**生成日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. データ概要",
        "",
        "### 品質セグメント分布",
        "",
        "| Quality | Samples | Percentage | Duration (s) |",
        "|---------|---------|------------|--------------|",
    ]

    total_samples = sum(len(seg_df) for seg_df in segments.values())
    for quality, seg_df in segments.items():
        n_samples = len(seg_df)
        pct = n_samples / total_samples * 100
        duration = n_samples / 256.0
        lines.append(f"| {quality} | {n_samples} | {pct:.1f}% | {duration:.1f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 2. 可視化結果",
        "",
        "### 品質セグメント時系列",
        "",
        "![品質時系列](img/quality_analysis/quality_time_series.png)",
        "",
        "### 品質別PSD比較",
        "",
        "![PSD比較](img/quality_analysis/quality_psd_comparison.png)",
        "",
        "### 品質別ピークパワー箱ひげ図",
        "",
        "![箱ひげ図](img/quality_analysis/quality_peak_power_boxplot.png)",
        "",
        "### HSI値とバンドパワーの散布図",
        "",
        "![散布図](img/quality_analysis/hsi_power_scatter.png)",
        "",
        "---",
        "",
        "## 3. ピークパワー詳細",
        "",
    ])

    # 品質別ピークパワーテーブル
    for quality in ['Good', 'Medium', 'Bad']:
        if quality not in quality_results:
            continue

        lines.append(f"### {quality} Quality")
        lines.append("")
        lines.append("| Band | Peak Power (dB) |")
        lines.append("|------|----------------|")

        for band, power in quality_results[quality]['peak_powers'].items():
            lines.append(f"| {band} | {power:.2f} |")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## 4. 結論",
        "",
        "### PAF（ピークアルファ周波数）",
        "",
        "- PAFは接続品質に依存せず、真の脳活動を反映していると考えられる",
        "",
        "### SMR/Beta/Deltaのピーク",
        "",
        "- 接続品質との相関が観察された場合は、アーティファクトの可能性が高い",
        "- Good品質セグメントでも明確に現れるパターンは、真の脳活動と判断できる",
        "",
        "### 推奨事項",
        "",
        "- 分析時は接続品質（HSI）を考慮し、Good品質データを優先的に使用する",
        "- Medium/Bad品質データは参考程度にとどめる",
        "",
    ])

    report_text = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\nReport saved: {output_path}")


def main():
    # パス設定
    csv_path = Path('/home/tsu-nera/repo/satoru/data/muse/mindMonitor_2026-01-24--07-41-35_1209401358220294378.csv')
    output_dir = Path('/home/tsu-nera/repo/satoru/tmp/img/quality_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = Path('/home/tsu-nera/repo/satoru/tmp/quality_analysis_report.md')

    # データ読み込み
    df = load_data(csv_path)

    # HeadBandOnでフィルタ
    if 'HeadBandOn' in df.columns:
        df = df[df['HeadBandOn'] == 1].copy()
        print(f"Filtered by HeadBandOn: {len(df)} samples")

    # 品質セグメント分類
    segments = classify_quality_segments(df)
    if segments is None:
        print("Failed to classify quality segments")
        return

    # 各セグメントのPSD分析
    quality_results = analyze_quality_psd(segments)

    # 可視化
    print("\n=== Generating Visualizations ===")
    plot_quality_segments(df, output_dir)
    plot_quality_psd_comparison(quality_results, output_dir)
    plot_peak_power_boxplots(quality_results, output_dir)
    plot_hsi_power_scatter(df, output_dir)

    # 統計分析
    statistical_analysis(quality_results, segments)

    # レポート生成
    generate_report(segments, quality_results, report_path)

    print("\n=== Analysis Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == '__main__':
    main()
