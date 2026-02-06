#!/usr/bin/env python3
"""
複数セッションのシータ波ピーク分析
- セッション間の再現性検証
- 個人ITF（Individual Theta Frequency）の安定性評価
- 接続品質との相関の一貫性確認
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, stats
from typing import Dict, Tuple, List
import sys

# プロジェクトルートのlibをインポートパスに追加
sys.path.append(str(Path(__file__).parent.parent.parent))
from lib.loaders.mind_monitor import load_mind_monitor_csv, get_eeg_data

# 出力ディレクトリ
OUTPUT_DIR = Path(__file__).parent / "theta_peak_multi_output"
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

    # ピーク検出
    psd_db = 10 * np.log10(psd + 1e-12)
    peaks, properties = signal.find_peaks(
        psd_db,
        prominence=np.ptp(psd_db) * 0.05,
        distance=int(0.5 / (freqs[1] - freqs[0]))
    )

    return freqs, psd_db, peaks, properties


def analyze_single_session(
    csv_path: str,
    theta_range: Tuple[float, float] = (5.0, 7.0)
) -> Dict:
    """単一セッションの分析"""

    session_name = Path(csv_path).stem

    # データロード
    try:
        df = load_mind_monitor_csv(csv_path, filter_headband=True, warmup_seconds=60.0)
        eeg_dict = get_eeg_data(df)

        if eeg_dict is None:
            return {'session': session_name, 'error': 'No EEG data'}

        # HSIデータ
        hsi_columns = ['HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10']
        if not all(col in df.columns for col in hsi_columns):
            return {'session': session_name, 'error': 'No HSI data'}

        avg_quality = df[hsi_columns].mean().mean()

    except Exception as e:
        return {'session': session_name, 'error': str(e)}

    # 全セッションのPSD分析
    fs = 256
    channel_results = {}

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
            peak_freq = freqs[max_peak_idx]
            peak_power = psd_db[max_peak_idx]
            peak_prominence = props['prominences'][np.where(peaks == max_peak_idx)[0][0]]
        else:
            peak_freq = np.nan
            peak_power = np.nan
            peak_prominence = np.nan

        # シータ帯域平均パワー
        theta_band_mask = (freqs >= 4) & (freqs <= 8)
        theta_avg_power = np.mean(psd_db[theta_band_mask])

        channel_results[ch] = {
            'peak_freq': peak_freq,
            'peak_power': peak_power,
            'peak_prominence': peak_prominence,
            'theta_avg_power': theta_avg_power,
            'has_peak': not np.isnan(peak_freq)
        }

    # セッション統計
    all_freqs = [channel_results[ch]['peak_freq'] for ch in ['TP9', 'AF7', 'AF8', 'TP10']]
    valid_freqs = [f for f in all_freqs if not np.isnan(f)]

    result = {
        'session': session_name,
        'date': session_name.split('_')[1] if '_' in session_name else 'unknown',
        'avg_quality': avg_quality,
        'duration_min': len(eeg_dict['TP9']) / fs / 60,
        'channels': channel_results,
        'n_channels_with_peak': len(valid_freqs),
        'mean_peak_freq': np.mean(valid_freqs) if valid_freqs else np.nan,
        'std_peak_freq': np.std(valid_freqs) if len(valid_freqs) > 1 else np.nan,
        'frontal_has_peak': not (np.isnan(channel_results['AF7']['peak_freq']) and
                                 np.isnan(channel_results['AF8']['peak_freq'])),
        'error': None
    }

    return result


def analyze_multiple_sessions(
    csv_files: List[Path],
    theta_range: Tuple[float, float] = (5.0, 7.0)
) -> pd.DataFrame:
    """複数セッションの分析"""

    results = []

    for i, csv_file in enumerate(csv_files):
        print(f"\nAnalyzing {i+1}/{len(csv_files)}: {csv_file.name}")
        result = analyze_single_session(str(csv_file), theta_range)

        if result['error'] is None:
            # フラット化
            row = {
                'session': result['session'],
                'date': result['date'],
                'duration_min': result['duration_min'],
                'avg_quality': result['avg_quality'],
                'n_channels_with_peak': result['n_channels_with_peak'],
                'mean_peak_freq': result['mean_peak_freq'],
                'std_peak_freq': result['std_peak_freq'],
                'frontal_has_peak': result['frontal_has_peak']
            }

            # チャンネル別データ
            for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
                ch_data = result['channels'][ch]
                row[f'{ch}_peak_freq'] = ch_data['peak_freq']
                row[f'{ch}_peak_power'] = ch_data['peak_power']
                row[f'{ch}_peak_prominence'] = ch_data['peak_prominence']
                row[f'{ch}_theta_avg'] = ch_data['theta_avg_power']
                row[f'{ch}_has_peak'] = ch_data['has_peak']

            results.append(row)
        else:
            print(f"  ❌ Error: {result['error']}")

    return pd.DataFrame(results)


def plot_session_comparison(df: pd.DataFrame, output_dir: Path):
    """セッション間比較プロット"""

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))

    # 日付でソート
    df_sorted = df.sort_values('date')
    x_labels = [f"{row['date']}\n({row['avg_quality']:.2f})" for _, row in df_sorted.iterrows()]

    # 1. チャンネル別ピーク周波数
    ax = axes[0, 0]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        ax.plot(range(len(df_sorted)), df_sorted[f'{ch}_peak_freq'],
               marker='o', label=ch, linewidth=2, markersize=6)
    ax.axhline(y=5.5, color='red', linestyle='--', alpha=0.3, label='5.5 Hz')
    ax.axhline(y=6.0, color='red', linestyle='--', alpha=0.3, label='6.0 Hz')
    ax.set_ylabel('Peak Frequency (Hz)', fontsize=11)
    ax.set_title('Theta Peak Frequency Across Sessions', fontsize=12, fontweight='bold')
    ax.set_ylim(4, 8)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

    # 2. セッション平均周波数
    ax = axes[0, 1]
    ax.errorbar(range(len(df_sorted)), df_sorted['mean_peak_freq'],
               yerr=df_sorted['std_peak_freq'], marker='o', linewidth=2,
               markersize=8, capsize=5, color='darkblue')
    ax.axhline(y=df_sorted['mean_peak_freq'].mean(), color='red',
              linestyle='--', alpha=0.5, label=f'Overall Mean: {df_sorted["mean_peak_freq"].mean():.2f} Hz')
    ax.set_ylabel('Mean Peak Frequency (Hz)', fontsize=11)
    ax.set_title('Session Mean Theta Frequency (±SD)', fontsize=12, fontweight='bold')
    ax.set_ylim(4, 8)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

    # 3. ピーク検出率
    ax = axes[1, 0]
    detection_rates = []
    for _, row in df_sorted.iterrows():
        rate = sum([row[f'{ch}_has_peak'] for ch in ['TP9', 'AF7', 'AF8', 'TP10']]) / 4 * 100
        detection_rates.append(rate)

    bars = ax.bar(range(len(df_sorted)), detection_rates, color='green', alpha=0.7)
    ax.set_ylabel('Detection Rate (%)', fontsize=11)
    ax.set_title('Peak Detection Rate by Session', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

    # 4. 接続品質とピーク周波数の相関
    ax = axes[1, 1]
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        valid_data = df_sorted[df_sorted[f'{ch}_has_peak']]
        if len(valid_data) > 0:
            ax.scatter(valid_data['avg_quality'], valid_data[f'{ch}_peak_freq'],
                      label=ch, s=80, alpha=0.7)
    ax.set_xlabel('Average Connection Quality', fontsize=11)
    ax.set_ylabel('Peak Frequency (Hz)', fontsize=11)
    ax.set_title('Quality vs Frequency (All Sessions)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(4, 8)

    # 5. 前頭部ピーク出現率
    ax = axes[2, 0]
    frontal_rate = df_sorted['frontal_has_peak'].sum() / len(df_sorted) * 100
    ax.bar(['Frontal Peak\nDetected', 'No Frontal\nPeak'],
          [frontal_rate, 100-frontal_rate],
          color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title(f'Frontal (AF7/AF8) Peak Detection: {frontal_rate:.1f}%',
                fontsize=12, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # 6. チャンネル別パワー分布
    ax = axes[2, 1]
    power_data = []
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        valid_power = df_sorted[df_sorted[f'{ch}_has_peak']][f'{ch}_peak_power'].dropna()
        if len(valid_power) > 0:
            power_data.append(valid_power)
        else:
            power_data.append([])

    bp = ax.boxplot(power_data, labels=['TP9', 'AF7', 'AF8', 'TP10'],
                    patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Peak Power (dB)', fontsize=11)
    ax.set_title('Theta Peak Power Distribution by Channel', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'session_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_itf_stability(df: pd.DataFrame, output_dir: Path):
    """個人シータ周波数(ITF)の安定性分析"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    df_sorted = df.sort_values('date')

    # 1. ITFの時系列変化
    ax = axes[0, 0]
    ax.errorbar(range(len(df_sorted)), df_sorted['mean_peak_freq'],
               yerr=df_sorted['std_peak_freq'], marker='o', linewidth=2,
               markersize=10, capsize=5, color='darkblue', elinewidth=2)

    overall_mean = df_sorted['mean_peak_freq'].mean()
    overall_std = df_sorted['mean_peak_freq'].std()

    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {overall_mean:.2f} Hz')
    ax.fill_between(range(len(df_sorted)),
                    overall_mean - overall_std,
                    overall_mean + overall_std,
                    alpha=0.2, color='red', label=f'±1 SD: {overall_std:.2f} Hz')

    ax.set_xlabel('Session', fontsize=11)
    ax.set_ylabel('Individual Theta Frequency (Hz)', fontsize=11)
    ax.set_title('ITF Temporal Stability', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(len(df_sorted)))
    ax.set_xticklabels([row['date'] for _, row in df_sorted.iterrows()],
                       rotation=45, ha='right')

    # 2. ITF分布（ヒストグラム）
    ax = axes[0, 1]
    ax.hist(df_sorted['mean_peak_freq'].dropna(), bins=15, color='skyblue',
           edgecolor='black', alpha=0.7)
    ax.axvline(x=overall_mean, color='red', linestyle='--', linewidth=2,
              label=f'Mean: {overall_mean:.2f} Hz')
    ax.set_xlabel('Theta Peak Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Frequency Count', fontsize=11)
    ax.set_title('ITF Distribution Across Sessions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 3. 変動係数（CV）の計算
    ax = axes[1, 0]

    cv_data = []
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        valid_freqs = df_sorted[df_sorted[f'{ch}_has_peak']][f'{ch}_peak_freq'].dropna()
        if len(valid_freqs) > 1:
            cv = (valid_freqs.std() / valid_freqs.mean()) * 100
            cv_data.append({'Channel': ch, 'CV (%)': cv, 'Mean': valid_freqs.mean()})

    if cv_data:
        cv_df = pd.DataFrame(cv_data)
        bars = ax.bar(cv_df['Channel'], cv_df['CV (%)'], color='orange', alpha=0.7)
        ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
        ax.set_title('ITF Stability by Channel (Lower = More Stable)',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # 値をバーの上に表示
        for i, (bar, row) in enumerate(zip(bars, cv_df.itertuples())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%\n({row.Mean:.2f} Hz)',
                   ha='center', va='bottom', fontsize=9)

    # 4. セッション間相関マトリックス
    ax = axes[1, 1]

    # 各セッションのチャンネル平均を取得
    session_vectors = []
    valid_sessions = []
    for idx, row in df_sorted.iterrows():
        freqs = [row[f'{ch}_peak_freq'] for ch in ['TP9', 'AF7', 'AF8', 'TP10']]
        if not any(np.isnan(freqs)):  # 全チャンネルに有効なデータがある場合のみ
            session_vectors.append(freqs)
            valid_sessions.append(row['date'][:10])

    # 相関マトリックス計算
    if len(session_vectors) > 1:
        corr_matrix = np.corrcoef(session_vectors)

        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels(valid_sessions, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(valid_sessions, fontsize=8)
        ax.set_title('Session-to-Session Frequency Pattern Correlation',
                    fontsize=12, fontweight='bold')

        # カラーバー
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', fontsize=10)

        # 相関値をセルに表示
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis',
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Session-to-Session Frequency Pattern Correlation',
                    fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'itf_stability.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_multi_session_report(df: pd.DataFrame, output_dir: Path):
    """複数セッション分析レポート生成"""

    report = f"""# Multi-Session Theta Peak Analysis

**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Number of Sessions**: {len(df)}
**Date Range**: {df['date'].min()} ~ {df['date'].max()}
**Theta Range**: 5.0 - 7.0 Hz

---

## Overall Statistics

### Individual Theta Frequency (ITF)

"""

    valid_itf = df['mean_peak_freq'].dropna()

    if len(valid_itf) > 0:
        overall_mean = valid_itf.mean()
        overall_std = valid_itf.std()
        cv = (overall_std / overall_mean) * 100

        report += f"- **Mean ITF**: {overall_mean:.2f} ± {overall_std:.2f} Hz\n"
        report += f"- **Coefficient of Variation (CV)**: {cv:.1f}%\n"
        report += f"- **Range**: {valid_itf.min():.2f} - {valid_itf.max():.2f} Hz\n"
        report += f"- **Median**: {valid_itf.median():.2f} Hz\n\n"

        # 安定性評価
        if cv < 5:
            stability = "**非常に安定** (CV < 5%)"
        elif cv < 10:
            stability = "**安定** (CV < 10%)"
        elif cv < 15:
            stability = "**やや変動あり** (CV < 15%)"
        else:
            stability = "**変動が大きい** (CV ≥ 15%)"

        report += f"**安定性評価**: {stability}\n\n"

    # 検出率
    report += "### Peak Detection Statistics\n\n"

    frontal_rate = df['frontal_has_peak'].sum() / len(df) * 100
    report += f"- **Frontal (AF7/AF8) Detection Rate**: {frontal_rate:.1f}%\n"

    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        rate = df[f'{ch}_has_peak'].sum() / len(df) * 100
        report += f"- **{ch} Detection Rate**: {rate:.1f}%\n"

    report += "\n### Channel-wise ITF Statistics\n\n"

    ch_stats = []
    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        valid_data = df[df[f'{ch}_has_peak']][f'{ch}_peak_freq'].dropna()
        if len(valid_data) > 0:
            ch_stats.append({
                'Channel': ch,
                'Mean (Hz)': valid_data.mean(),
                'Std (Hz)': valid_data.std(),
                'CV (%)': (valid_data.std() / valid_data.mean()) * 100,
                'N Sessions': len(valid_data)
            })

    if ch_stats:
        ch_df = pd.DataFrame(ch_stats)
        report += ch_df.to_markdown(index=False, floatfmt='.2f') + "\n\n"

    # 品質との相関
    report += "## Connection Quality Analysis\n\n"
    report += "### Correlation: Peak Frequency vs Quality\n\n"

    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        valid_data = df[df[f'{ch}_has_peak']]
        if len(valid_data) > 3:
            corr, p_value = stats.pearsonr(valid_data['avg_quality'],
                                          valid_data[f'{ch}_peak_freq'])
            sig = "**Significant**" if p_value < 0.05 else "Not significant"
            report += f"- **{ch}**: r = {corr:.3f}, p = {p_value:.3f} ({sig})\n"

    report += "\n### Correlation: Peak Power vs Quality\n\n"

    for ch in ['TP9', 'AF7', 'AF8', 'TP10']:
        valid_data = df[df[f'{ch}_has_peak']]
        if len(valid_data) > 3:
            corr, p_value = stats.pearsonr(valid_data['avg_quality'],
                                          valid_data[f'{ch}_peak_power'])
            sig = "**Significant**" if p_value < 0.05 else "Not significant"
            report += f"- **{ch}**: r = {corr:.3f}, p = {p_value:.3f} ({sig})\n"

    # セッション別詳細
    report += "\n\n## Session Details\n\n"

    session_table = df[['date', 'duration_min', 'avg_quality', 'n_channels_with_peak',
                       'mean_peak_freq', 'std_peak_freq', 'frontal_has_peak']].copy()
    session_table.columns = ['Date', 'Duration (min)', 'Avg Quality',
                            'Channels w/ Peak', 'Mean Freq (Hz)', 'Std Freq (Hz)',
                            'Frontal Peak']

    report += session_table.to_markdown(index=False, floatfmt='.2f') + "\n\n"

    # 結論
    report += "## Interpretation\n\n"

    # 1. 再現性チェック
    if frontal_rate >= 80:
        report += "### ✅ Highly Reproducible Frontal Theta\n\n"
        report += f"前頭部シータピークが{frontal_rate:.0f}%のセッションで検出されました。これは**非常に高い再現性**を示し、アーチファクトではなく生理学的現象である強い証拠です。\n\n"
    elif frontal_rate >= 50:
        report += "### ⚠️ Moderately Reproducible\n\n"
        report += f"前頭部シータピークが{frontal_rate:.0f}%のセッションで検出されました。ある程度の再現性はありますが、接続品質やセッション状態に依存する可能性があります。\n\n"
    else:
        report += "### ❌ Low Reproducibility\n\n"
        report += f"前頭部シータピークの検出率が{frontal_rate:.0f}%と低いです。アーチファクトまたは不安定な測定条件の可能性があります。\n\n"

    # 2. ITF安定性
    if len(valid_itf) > 0 and cv < 10:
        report += "### ✅ Stable Individual Theta Frequency\n\n"
        report += f"あなたの個人シータ周波数(ITF)は **{overall_mean:.2f} ± {overall_std:.2f} Hz** で、変動係数{cv:.1f}%と非常に安定しています。これは**個人特性として信頼できる指標**です。\n\n"

    # 3. 品質独立性
    report += "### Quality Independence Assessment\n\n"
    report += "複数セッションの分析から、シータピーク周波数は接続品質に大きく依存していないことが確認できれば、アーチファクトではないと結論できます。上記の相関分析を参照してください。\n\n"

    report += "---\n\n"
    report += "## Figures\n\n"
    report += "1. [Session Comparison](session_comparison.png)\n"
    report += "2. [ITF Stability Analysis](itf_stability.png)\n"

    # レポート保存
    with open(output_dir / 'MULTI_SESSION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n" + "="*80)
    print(report)
    print("="*80)


def main():
    """メイン処理"""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-session theta peak analysis')
    parser.add_argument('--n-sessions', type=int, default=10,
                       help='Number of latest sessions to analyze (default: 10)')
    parser.add_argument('--all', action='store_true',
                       help='Analyze all sessions')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD)')

    args = parser.parse_args()

    # データディレクトリ
    data_dir = Path(__file__).parent.parent.parent / "data" / "muse"
    csv_files = sorted(data_dir.glob("mindMonitor_*.csv"))

    if not csv_files:
        print("No data files found!")
        return

    print(f"Found {len(csv_files)} CSV files")

    # ファイル選択
    if args.all:
        selected_files = csv_files
        print("Mode: All sessions")
    elif args.start_date and args.end_date:
        selected_files = [
            f for f in csv_files
            if args.start_date <= f.stem.split('_')[1] <= args.end_date
        ]
        print(f"Mode: Date range {args.start_date} to {args.end_date}")
    else:
        selected_files = csv_files[-args.n_sessions:]
        print(f"Mode: Latest {args.n_sessions} sessions")

    print(f"\nAnalyzing {len(selected_files)} sessions...")

    # 分析実行
    df_results = analyze_multiple_sessions(selected_files)

    if len(df_results) == 0:
        print("No valid sessions found!")
        return

    # 結果保存
    df_results.to_csv(OUTPUT_DIR / 'multi_session_results.csv', index=False)

    # プロット生成
    print("\nGenerating plots...")
    plot_session_comparison(df_results, OUTPUT_DIR)
    plot_itf_stability(df_results, OUTPUT_DIR)

    # レポート生成
    print("\nGenerating report...")
    generate_multi_session_report(df_results, OUTPUT_DIR)

    print(f"\n✅ Analysis complete! Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
