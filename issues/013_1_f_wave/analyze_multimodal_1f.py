#!/usr/bin/env python3
"""
マルチモーダル 1/f ゆらぎ解析スクリプト

Museの全センサーデータ（EEG、HRV、fNIRS、IMU）における
1/f^β型のパワースペクトル特性を網羅的に解析します。

分析対象：
- EEG (4ch): RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10
- IMU (6ch): AccelX/Y/Z, GyroX/Y/Z
- fNIRS (2-4ch): HbO/HbR (Left/Right)
- HRV (1ch): RR intervals (Selfloopsデータが必要)

Usage:
    python analyze_multimodal_1f.py --data <CSV_PATH> [--selfloops <SELFLOOPS_PATH>] [--output <OUTPUT_DIR>]
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
from scipy.signal import welch

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_psd,
    get_optics_data,
    analyze_fnirs,
)


def fit_1f_noise(freqs, psd, freq_range=(0.01, 1.0)):
    """
    1/f^β モデルをPSDにフィッティング

    Parameters
    ----------
    freqs : array-like
        周波数配列 [Hz]
    psd : array-like
        パワースペクトル密度
    freq_range : tuple
        フィッティングに使用する周波数範囲 [Hz]

    Returns
    -------
    dict or None
        成功時: beta, amplitude, r_squared, fitted_psd, freq_mask
        失敗時: None
    """
    try:
        # 周波数範囲でフィルタ
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        f = freqs[freq_mask]
        p = psd[freq_mask]

        # 0除算を避けるため、0に近い値を除外
        valid_mask = (f > 0) & (p > 0)
        f = f[valid_mask]
        p = p[valid_mask]

        if len(f) < 10:
            return None

        # 対数空間で線形回帰
        log_f = np.log10(f)
        log_p = np.log10(p)

        slope, intercept, r_value, p_value, std_err = stats.linregress(log_f, log_p)

        beta = -slope
        amplitude = 10 ** intercept
        r_squared = r_value ** 2

        # フィッティング曲線
        fitted_psd = amplitude / (freqs ** beta)

        return {
            'beta': beta,
            'amplitude': amplitude,
            'r_squared': r_squared,
            'fitted_psd': fitted_psd,
            'freq_mask': freq_mask,
        }
    except Exception:
        return None


def calculate_psd_from_timeseries(data, fs, nperseg=None):
    """
    時系列データからパワースペクトル密度を計算

    Parameters
    ----------
    data : array-like
        時系列データ
    fs : float
        サンプリングレート [Hz]
    nperseg : int, optional
        Welch法のセグメント長

    Returns
    -------
    freqs, psd : tuple
        周波数配列とPSD
    """
    if nperseg is None:
        nperseg = min(len(data), int(fs * 60))  # 最大60秒

    freqs, psd = welch(data, fs=fs, nperseg=nperseg, scaling='density')
    return freqs, psd


def analyze_eeg_1f(df):
    """
    EEG 1/f解析

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitor CSV

    Returns
    -------
    dict
        チャネル別のβ値と統計
    """
    print('Analyzing EEG 1/f...')
    mne_dict = prepare_mne_raw(df)
    if not mne_dict:
        return None

    raw = mne_dict['raw']
    psd_dict = calculate_psd(raw)
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']
    channels = psd_dict['channels']

    results = []
    for ch_idx, ch_label in enumerate(channels):
        psd = psds[ch_idx]
        fit_result = fit_1f_noise(freqs, psd, freq_range=(1.0, 40.0))
        if fit_result:
            results.append({
                'Sensor': 'EEG',
                'Channel': ch_label,
                'β': fit_result['beta'],
                'R²': fit_result['r_squared'],
                'Freq Range': '1-40 Hz',
            })

    return pd.DataFrame(results) if results else None


def analyze_imu_1f(df):
    """
    IMU 1/f解析（加速度・ジャイロ）

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitor CSV

    Returns
    -------
    pd.DataFrame
        チャネル別のβ値と統計
    """
    print('Analyzing IMU 1/f...')

    # 加速度とジャイロのカラム
    accel_cols = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
    gyro_cols = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']

    results = []
    fs = 52.0  # Muse IMUのサンプリングレート

    # 加速度解析
    for col in accel_cols:
        if col in df.columns:
            data = df[col].dropna().values
            if len(data) > 100:
                freqs, psd = calculate_psd_from_timeseries(data, fs)
                fit_result = fit_1f_noise(freqs, psd, freq_range=(0.1, 10.0))
                if fit_result:
                    results.append({
                        'Sensor': 'IMU (Accel)',
                        'Channel': col.replace('Accelerometer_', 'Accel_'),
                        'β': fit_result['beta'],
                        'R²': fit_result['r_squared'],
                        'Freq Range': '0.1-10 Hz',
                    })

    # ジャイロ解析
    for col in gyro_cols:
        if col in df.columns:
            data = df[col].dropna().values
            if len(data) > 100:
                freqs, psd = calculate_psd_from_timeseries(data, fs)
                fit_result = fit_1f_noise(freqs, psd, freq_range=(0.1, 10.0))
                if fit_result:
                    results.append({
                        'Sensor': 'IMU (Gyro)',
                        'Channel': col.replace('Gyro_', 'Gyro_'),
                        'β': fit_result['beta'],
                        'R²': fit_result['r_squared'],
                        'Freq Range': '0.1-10 Hz',
                    })

    return pd.DataFrame(results) if results else None


def analyze_fnirs_1f(df):
    """
    Optics (fNIRS) 1/f解析

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitor CSV

    Returns
    -------
    pd.DataFrame
        チャネル別のβ値と統計
    """
    print('Analyzing Optics (fNIRS) 1/f...')

    try:
        optics_data = get_optics_data(df)
        if not optics_data or len(optics_data['time']) == 0:
            return None

        # HbO/HbR を計算
        fnirs_results = analyze_fnirs(optics_data)

    except Exception as e:
        print(f'  fNIRS analysis failed: {e}')
        return None

    results = []

    # サンプリングレート推定
    time_diff = np.diff(fnirs_results['time'])
    fs = 1.0 / np.median(time_diff) if len(time_diff) > 0 else 10.0

    # HbO/HbR 左右
    channels = {
        'HbO_Left': fnirs_results.get('left_hbo'),
        'HbO_Right': fnirs_results.get('right_hbo'),
        'HbR_Left': fnirs_results.get('left_hbr'),
        'HbR_Right': fnirs_results.get('right_hbr'),
    }

    for ch_name, data in channels.items():
        if data is not None and len(data) > 100:
            # NaN値を除外
            valid_data = data[np.isfinite(data)]
            if len(valid_data) < 100:
                print(f'  {ch_name}: insufficient valid data')
                continue

            freqs, psd = calculate_psd_from_timeseries(valid_data, fs)
            fit_result = fit_1f_noise(freqs, psd, freq_range=(0.01, 1.0))
            if fit_result:
                results.append({
                    'Sensor': 'Optics (fNIRS)',
                    'Channel': ch_name,
                    'β': fit_result['beta'],
                    'R²': fit_result['r_squared'],
                    'Freq Range': '0.01-1 Hz',
                })

    return pd.DataFrame(results) if results else None


def analyze_hrv_1f(selfloops_path):
    """
    ECG (Selfloops) 1/f解析

    Parameters
    ----------
    selfloops_path : Path
        Selfloops CSVファイルパス

    Returns
    -------
    pd.DataFrame
        β値と統計
    """
    print('Analyzing ECG (Selfloops) 1/f...')

    if not selfloops_path or not selfloops_path.exists():
        print('  Selfloops data not found, skipping HRV analysis.')
        return None

    try:
        from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
        from lib.sensors.ecg.hrv import calculate_power_spectrum, calculate_1f_slope

        sl_df = load_selfloops_csv(str(selfloops_path), warmup_seconds=60.0)
        hrv_data = get_hrv_data(sl_df, clean_artifacts=True)

        rr_intervals = hrv_data['rr_intervals_clean']

        if len(rr_intervals) < 100:
            print('  Insufficient RR intervals for HRV analysis.')
            return None

        # パワースペクトル計算
        freqs, power = calculate_power_spectrum(rr_intervals, fs=4.0)

        # 1/f解析
        slope_result = calculate_1f_slope(freqs, power, freq_range=(0.01, 0.4))

        results = [{
            'Sensor': 'ECG (Selfloops)',
            'Channel': 'RR Intervals',
            'β': slope_result['beta'],
            'R²': slope_result['r_value'] ** 2,
            'Freq Range': '0.01-0.4 Hz',
        }]

        return pd.DataFrame(results)

    except Exception as e:
        print(f'  ECG analysis failed: {e}')
        return None


def plot_multimodal_1f(summary_table, img_path):
    """
    マルチモーダル1/fゆらぎのプロット

    Parameters
    ----------
    summary_table : pd.DataFrame
        全センサーのサマリーテーブル
    img_path : Path
        保存先画像パス
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # センサータイプ別にグループ化
    sensors = summary_table['Sensor'].unique()

    for idx, sensor in enumerate(sensors[:4]):
        ax = axes[idx]
        sensor_data = summary_table[summary_table['Sensor'] == sensor]

        channels = sensor_data['Channel'].values
        betas = sensor_data['β'].values
        r2s = sensor_data['R²'].values

        # バープロット
        x_pos = np.arange(len(channels))
        bars = ax.bar(x_pos, betas, alpha=0.7, edgecolor='black')

        # 色付け（R²に基づく）
        for i, (bar, r2) in enumerate(zip(bars, r2s)):
            if r2 > 0.8:
                bar.set_color('green')
            elif r2 > 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # ピンクノイズ基準線
        ax.axhline(y=1.0, color='blue', linestyle='--', linewidth=2,
                   label='Pink Noise (β=1)', alpha=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('β (Exponent)', fontsize=11)
        ax.set_title(f'{sensor}\n(Mean β = {betas.mean():.2f})',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(3.0, betas.max() * 1.2))

    # 未使用のサブプロットを非表示
    for idx in range(len(sensors), 4):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✓ Multimodal 1/f plot saved: {img_path}')


def generate_markdown_report(output_dir, summary_table):
    """
    マークダウンレポート生成

    Parameters
    ----------
    output_dir : Path
        出力ディレクトリ
    summary_table : pd.DataFrame
        全センサーのサマリーテーブル
    """
    report_path = output_dir / 'MULTIMODAL_REPORT.md'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Multimodal 1/f Noise Analysis Report\n\n")

        # Overview
        f.write("## Overview\n\n")
        f.write("This report analyzes the 1/f^β power spectral characteristics across all Muse sensors:\n\n")
        f.write("- **EEG** (4ch): Brain electrical activity\n")
        f.write("- **IMU** (6ch): Body sway (Accelerometer, Gyroscope)\n")
        f.write("- **Optics (fNIRS)** (4ch): Cerebral blood flow (HbO/HbR)\n")
        f.write("- **ECG (Selfloops)** (1ch): Heart rate variability (RR intervals)\n\n")

        # Summary Table
        f.write("## All Sensors Summary\n\n")
        f.write(summary_table.to_markdown(index=False, floatfmt='.3f'))
        f.write("\n\n")

        # Sensor-wise statistics
        f.write("## Sensor-wise Statistics\n\n")
        for sensor in summary_table['Sensor'].unique():
            sensor_data = summary_table[summary_table['Sensor'] == sensor]
            beta_mean = sensor_data['β'].mean()
            beta_std = sensor_data['β'].std()

            f.write(f"### {sensor}\n\n")
            f.write(f"- **Mean β**: {beta_mean:.3f} ± {beta_std:.3f}\n")
            f.write(f"- **Channels**: {len(sensor_data)}\n")

            # 解釈
            if 0.8 <= beta_mean <= 1.2:
                f.write(f"- **Interpretation**: ピンクノイズ（1/f）に近い特性を示しています。\n")
            elif beta_mean < 0.8:
                f.write(f"- **Interpretation**: ホワイトノイズに近く、ランダムな変動が支配的です。\n")
            elif 1.5 <= beta_mean <= 2.5:
                f.write(f"- **Interpretation**: ブラウンノイズ（1/f²）に近く、より強い自己相関を持ちます。\n")
            else:
                f.write(f"- **Interpretation**: 標準的な1/f^β範囲から外れています。\n")

            f.write("\n")

        # Visualization
        f.write("## Visualization\n\n")
        f.write("![Multimodal 1/f Analysis](img/multimodal_1f.png)\n\n")

        # Color coding explanation
        f.write("**Color coding**: Green (R² > 0.8), Orange (R² > 0.6), Red (R² ≤ 0.6)\n\n")

        # Interpretation
        f.write("## Overall Interpretation\n\n")

        # センサー間の比較
        sensor_means = summary_table.groupby('Sensor')['β'].mean().sort_values()

        f.write("### β値の比較（センサー別平均）\n\n")
        for sensor, beta_mean in sensor_means.items():
            f.write(f"- **{sensor}**: β = {beta_mean:.3f}\n")

        f.write("\n### 考察\n\n")
        f.write("1. **EEG（脳波）**: 高周波領域（1-40Hz）での1/f特性。覚醒時はβ≈2.0が標準。\n")
        f.write("2. **IMU（体軸）**: 身体動揺の周波数特性。姿勢制御の安定性を反映。\n")
        f.write("3. **Optics (fNIRS)（脳血流）**: 血流変動の周波数特性。心拍成分も含む生理的ゆらぎ。\n")
        f.write("4. **ECG（心拍変動）**: 自律神経系のバランス。ピンクノイズ（β≈1）が健康的。\n\n")

    print(f'✓ Report generated: {report_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Multimodal 1/f noise analysis for all Muse sensors'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Input Mind Monitor CSV file path'
    )
    parser.add_argument(
        '--selfloops',
        type=Path,
        default=None,
        help='Selfloops HRV CSV file path (optional, for HRV analysis)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent,
        help='Output directory (default: script directory)'
    )

    args = parser.parse_args()

    # パス検証
    if not args.data.exists():
        print(f'Error: Data file not found: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    img_dir = args.output / 'img'
    img_dir.mkdir(exist_ok=True)

    print('=' * 60)
    print('Multimodal 1/f Noise Analysis')
    print('=' * 60)
    print()

    # データ読み込み
    print(f'Loading: {args.data}')
    df = load_mind_monitor_csv(args.data, filter_headband=False, warmup_seconds=60.0)
    print(f'Data shape: {df.shape[0]} rows × {df.shape[1]} columns\n')

    # 各センサーの1/f解析
    all_results = []

    # 1. EEG
    eeg_results = analyze_eeg_1f(df)
    if eeg_results is not None:
        all_results.append(eeg_results)

    # 2. IMU
    imu_results = analyze_imu_1f(df)
    if imu_results is not None:
        all_results.append(imu_results)

    # 3. fNIRS
    fnirs_results = analyze_fnirs_1f(df)
    if fnirs_results is not None:
        all_results.append(fnirs_results)

    # 4. HRV
    hrv_results = analyze_hrv_1f(args.selfloops)
    if hrv_results is not None:
        all_results.append(hrv_results)

    # 統合テーブル
    if not all_results:
        print('Error: No sensor data could be analyzed.')
        return 1

    summary_table = pd.concat(all_results, ignore_index=True)

    print('\n' + '=' * 60)
    print('Summary Table')
    print('=' * 60)
    print(summary_table.to_string(index=False))
    print()

    # プロット生成
    print('Generating plots...')
    plot_multimodal_1f(summary_table, img_path=img_dir / 'multimodal_1f.png')

    # レポート生成
    print('Generating report...')
    generate_markdown_report(args.output, summary_table)

    # CSVエクスポート
    csv_path = args.output / 'multimodal_summary.csv'
    summary_table.to_csv(csv_path, index=False)
    print(f'✓ Summary CSV saved: {csv_path}')

    print()
    print('=' * 60)
    print('Analysis complete!')
    print('=' * 60)
    print(f'Report: {args.output / "MULTIMODAL_REPORT.md"}')
    print(f'Images: {img_dir}/')

    return 0


if __name__ == '__main__':
    exit(main())
