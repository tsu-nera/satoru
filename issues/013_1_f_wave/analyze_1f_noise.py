#!/usr/bin/env python3
"""
1/f ゆらぎ解析スクリプト

EEG（脳波）データにおける1/f^β型のパワースペクトル特性を解析します。

先行研究によると：
- 覚醒時: β ≈ 1.99（ピンクノイズに近い）
- REM睡眠: β ≈ 3.08
- NonREM睡眠: β ≈ 2.58

Usage:
    python analyze_1f_noise.py --data <CSV_PATH> [--output <OUTPUT_DIR>]
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_psd,
)


def fit_1f_noise(freqs, psd, freq_range=(1.0, 40.0)):
    """
    1/f^β モデルをPSDにフィッティング

    Parameters
    ----------
    freqs : array-like
        周波数配列 [Hz]
    psd : array-like
        パワースペクトル密度 [μV²/Hz]
    freq_range : tuple, default=(1.0, 40.0)
        フィッティングに使用する周波数範囲 [Hz]

    Returns
    -------
    dict
        - beta: 1/f^β の指数β
        - amplitude: 振幅A（1/f^β = A/f^β）
        - r_squared: 決定係数（フィッティングの良さ）
        - fitted_psd: フィッティング結果のPSD
        - freq_mask: 使用した周波数のマスク
    """
    # 周波数範囲でフィルタ
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    f = freqs[freq_mask]
    p = psd[freq_mask]

    # 0除算を避けるため、0に近い値を除外
    valid_mask = (f > 0) & (p > 0)
    f = f[valid_mask]
    p = p[valid_mask]

    if len(f) < 10:
        raise ValueError(f"有効な周波数点が不足しています（{len(f)}点）")

    # 対数空間で線形回帰
    # log(PSD) = log(A) - β * log(f)
    log_f = np.log10(f)
    log_p = np.log10(p)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_p)

    # β = -slope (負の傾きが1/f^βのβに相当)
    beta = -slope
    amplitude = 10 ** intercept
    r_squared = r_value ** 2

    # フィッティング曲線を全周波数範囲で計算
    fitted_psd = amplitude / (freqs ** beta)

    return {
        'beta': beta,
        'amplitude': amplitude,
        'r_squared': r_squared,
        'fitted_psd': fitted_psd,
        'freq_mask': freq_mask,
        'slope': slope,
        'intercept': intercept,
    }


def analyze_1f_noise_per_channel(psd_dict, freq_range=(1.0, 40.0)):
    """
    各チャネルの1/fゆらぎ解析

    Parameters
    ----------
    psd_dict : dict
        calculate_psd()の出力
    freq_range : tuple
        解析する周波数範囲 [Hz]

    Returns
    -------
    dict
        チャネル別の解析結果
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']  # (n_channels, n_freqs)
    channels = psd_dict['channels']
    results = {}

    for ch_idx, ch_label in enumerate(channels):
        try:
            psd = psds[ch_idx]
            fit_result = fit_1f_noise(freqs, psd, freq_range=freq_range)
            results[ch_label] = fit_result
        except Exception as e:
            print(f"警告: {ch_label} のフィッティングに失敗しました ({e})")
            results[ch_label] = None

    return results


def plot_1f_noise(psd_dict, fit_results, img_path, freq_range=(1.0, 40.0)):
    """
    1/fゆらぎのプロット

    Parameters
    ----------
    psd_dict : dict
        PSD計算結果
    fit_results : dict
        フィッティング結果
    img_path : str or Path
        保存先画像パス
    freq_range : tuple
        表示する周波数範囲
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']
    n_channels = len([r for r in fit_results.values() if r is not None])

    if n_channels == 0:
        print("警告: プロット可能なチャネルがありません")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    channel_labels = [ch for ch, r in fit_results.items() if r is not None]

    for idx, ch_label in enumerate(channel_labels[:4]):
        ax = axes[idx]
        fit_result = fit_results[ch_label]

        # チャネルインデックスを取得
        ch_idx = channels.index(ch_label)
        psd = psds[ch_idx]

        # 対数プロット
        ax.loglog(freqs, psd, 'b-', alpha=0.7, linewidth=1.5, label='Observed PSD')
        ax.loglog(freqs, fit_result['fitted_psd'], 'r--', linewidth=2,
                 label=f'Fitted: 1/f^{fit_result["beta"]:.2f}')

        # フィッティング範囲を強調
        freq_mask = fit_result['freq_mask']
        ax.loglog(freqs[freq_mask], psd[freq_mask], 'g.', markersize=4, alpha=0.5,
                 label=f'Fit range: {freq_range[0]}-{freq_range[1]} Hz')

        ax.set_xlabel('Frequency (Hz)', fontsize=11)
        ax.set_ylabel('PSD (μV²/Hz)', fontsize=11)
        ax.set_title(f'{ch_label}\nβ = {fit_result["beta"]:.3f}, R² = {fit_result["r_squared"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(0.5, 50)

    # 未使用のサブプロットを非表示
    for idx in range(n_channels, 4):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ 1/f noise plot saved: {img_path}')


def create_summary_table(fit_results):
    """
    サマリーテーブルを作成

    Parameters
    ----------
    fit_results : dict
        フィッティング結果

    Returns
    -------
    pd.DataFrame
        サマリーテーブル
    """
    rows = []
    for ch_label, result in fit_results.items():
        if result is not None:
            rows.append({
                'Channel': ch_label,
                'β (Exponent)': result['beta'],
                'Amplitude': result['amplitude'],
                'R²': result['r_squared'],
            })

    df = pd.DataFrame(rows)
    return df


def interpret_beta(beta_mean, beta_std):
    """
    β値の解釈

    Parameters
    ----------
    beta_mean : float
        β値の平均
    beta_std : float
        β値の標準偏差

    Returns
    -------
    str
        解釈テキスト
    """
    interpretations = []

    # 覚醒状態との比較（β ≈ 1.99）
    if 1.8 <= beta_mean <= 2.2:
        interpretations.append("覚醒時の典型的な1/fゆらぎ特性（β ≈ 2.0）を示しています。")
    elif beta_mean > 2.2:
        interpretations.append(f"β値が覚醒時の標準値（2.0）より高く（β = {beta_mean:.2f}）、睡眠状態やより深いリラックス状態に近い特性を示しています。")
    else:
        interpretations.append(f"β値が覚醒時の標準値（2.0）より低く（β = {beta_mean:.2f}）、より活発な脳活動の可能性があります。")

    # ピンクノイズとの比較（β ≈ 1.0）
    if 0.8 <= beta_mean <= 1.2:
        interpretations.append("ピンクノイズ（β ≈ 1.0）に近い特性を持っています。")

    # β値の安定性
    if beta_std < 0.1:
        interpretations.append(f"チャネル間でβ値が安定しています（SD = {beta_std:.3f}）。")
    elif beta_std > 0.3:
        interpretations.append(f"チャネル間でβ値にばらつきがあります（SD = {beta_std:.3f}）。")

    return " ".join(interpretations)


def generate_markdown_report(output_dir, results):
    """
    マークダウンレポート生成

    Parameters
    ----------
    output_dir : Path
        出力ディレクトリ
    results : dict
        解析結果
    """
    report_path = output_dir / 'REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 1/f Noise Analysis Report\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write("This report analyzes the 1/f^β power spectral characteristics of EEG data.\n\n")
        f.write("According to prior research:\n")
        f.write("- Waking state: β ≈ 1.99 (similar to pink noise)\n")
        f.write("- REM sleep: β ≈ 3.08\n")
        f.write("- NonREM sleep: β ≈ 2.58\n\n")

        # Summary Statistics
        f.write("## Summary Statistics\n\n")
        summary_table = results['summary_table']
        f.write(summary_table.to_markdown(index=False, floatfmt='.3f'))
        f.write("\n\n")

        # Mean β value
        beta_values = summary_table['β (Exponent)'].values
        beta_mean = beta_values.mean()
        beta_std = beta_values.std()

        f.write(f"**Mean β across channels**: {beta_mean:.3f} ± {beta_std:.3f}\n\n")

        # Interpretation
        f.write("## Interpretation\n\n")
        interpretation = interpret_beta(beta_mean, beta_std)
        f.write(f"{interpretation}\n\n")

        # Visualization
        f.write("## 1/f Noise Fitting\n\n")
        f.write("![1/f Noise](img/1f_noise.png)\n\n")

        # Technical Details
        f.write("## Technical Details\n\n")
        f.write("### Fitting Method\n\n")
        f.write("- Model: PSD(f) = A / f^β\n")
        f.write("- Method: Linear regression in log-log space\n")
        f.write(f"- Frequency range: {results['freq_range'][0]} - {results['freq_range'][1]} Hz\n")
        f.write("- R²: Coefficient of determination (goodness of fit)\n\n")

        # Data Info
        f.write("### Data Information\n\n")
        data_info = results['data_info']
        f.write(f"- Data shape: {data_info['shape'][0]} rows × {data_info['shape'][1]} columns\n")
        f.write(f"- Duration: {data_info['duration_sec'] / 60:.1f} minutes\n")
        f.write(f"- Start time: {data_info['start_time']}\n")
        f.write(f"- End time: {data_info['end_time']}\n\n")

    print(f'✓ Report generated: {report_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='EEG 1/f noise analysis'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent,
        help='Output directory (default: script directory)'
    )
    parser.add_argument(
        '--freq-min',
        type=float,
        default=1.0,
        help='Minimum frequency for fitting (Hz, default: 1.0)'
    )
    parser.add_argument(
        '--freq-max',
        type=float,
        default=40.0,
        help='Maximum frequency for fitting (Hz, default: 40.0)'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=1.0,
        help='Warmup exclusion time (minutes, default: 1.0)'
    )

    args = parser.parse_args()

    # パス検証
    if not args.data.exists():
        print(f'Error: Data file not found: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    img_dir = args.output / 'img'
    img_dir.mkdir(exist_ok=True)

    freq_range = (args.freq_min, args.freq_max)

    print('=' * 60)
    print('1/f Noise Analysis')
    print('=' * 60)
    print()

    # データ読み込み
    print(f'Loading: {args.data}')
    df = load_mind_monitor_csv(args.data, filter_headband=False, warmup_seconds=args.warmup * 60)

    data_info = {
        'shape': df.shape,
        'start_time': df['TimeStamp'].min(),
        'end_time': df['TimeStamp'].max(),
        'duration_sec': (df['TimeStamp'].max() - df['TimeStamp'].min()).total_seconds()
    }

    print(f'Data shape: {df.shape[0]} rows × {df.shape[1]} columns')
    print(f'Duration: {data_info["duration_sec"] / 60:.1f} minutes\n')

    # MNE RAW準備
    print('Preparing MNE RAW data...')
    mne_dict = prepare_mne_raw(df)

    if not mne_dict:
        print('Error: Failed to prepare MNE RAW data')
        return 1

    raw = mne_dict['raw']
    print(f'Channels: {mne_dict["channels"]}')
    print(f'Sampling rate: {mne_dict["sfreq"]:.2f} Hz\n')

    # PSD計算
    print('Calculating power spectral density...')
    psd_dict = calculate_psd(raw)
    print(f'Frequency resolution: {psd_dict["freqs"][1] - psd_dict["freqs"][0]:.3f} Hz\n')

    # 1/f解析
    print(f'Analyzing 1/f noise (freq range: {freq_range[0]}-{freq_range[1]} Hz)...')
    fit_results = analyze_1f_noise_per_channel(psd_dict, freq_range=freq_range)

    # サマリーテーブル
    summary_table = create_summary_table(fit_results)
    print('\nSummary:')
    print(summary_table.to_string(index=False))
    print()

    # プロット生成
    print('Generating plots...')
    plot_1f_noise(
        psd_dict,
        fit_results,
        img_path=img_dir / '1f_noise.png',
        freq_range=freq_range
    )

    # 結果を保存
    results = {
        'data_info': data_info,
        'freq_range': freq_range,
        'summary_table': summary_table,
        'fit_results': fit_results,
    }

    # レポート生成
    print('Generating report...')
    generate_markdown_report(args.output, results)

    # CSVエクスポート
    csv_path = args.output / 'summary.csv'
    summary_table.to_csv(csv_path, index=False)
    print(f'✓ Summary CSV saved: {csv_path}')

    print()
    print('=' * 60)
    print('Analysis complete!')
    print('=' * 60)
    print(f'Report: {args.output / "REPORT.md"}')
    print(f'Images: {img_dir}/')

    return 0


if __name__ == '__main__':
    exit(main())
