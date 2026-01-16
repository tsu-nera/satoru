#!/usr/bin/env python3
"""
1/fゆらぎ検証スクリプト

HRVデータのパワースペクトル解析を行い、1/fゆらぎ（ピンクノイズ）特性を検証します。

Usage:
    # デフォルト出力（issues/013_1_f_wave/）
    python generate_1f_analysis.py --data <SELFLOOPS_CSV_PATH>

    # 出力先指定
    python generate_1f_analysis.py --data <SELFLOOPS_CSV_PATH> --output <OUTPUT_DIR>
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import signal
from scipy.stats import linregress

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data


def calculate_power_spectrum(rr_intervals, fs=4.0):
    """
    R-R間隔からパワースペクトルを計算

    Parameters
    ----------
    rr_intervals : array-like
        R-R間隔データ（ms）
    fs : float, default=4.0
        リサンプリング周波数（Hz）

    Returns
    -------
    freqs : np.ndarray
        周波数配列（Hz）
    power : np.ndarray
        パワースペクトル密度
    """
    # R-R間隔を秒に変換
    rr_sec = np.array(rr_intervals) / 1000.0

    # 時間軸を生成
    time = np.cumsum(rr_sec)
    time = time - time[0]

    # 等間隔にリサンプリング
    time_regular = np.arange(0, time[-1], 1/fs)
    rr_interp = np.interp(time_regular, time, rr_sec)

    # デトレンド（線形トレンド除去）
    rr_detrend = signal.detrend(rr_interp)

    # ハニング窓を適用
    window = signal.windows.hann(len(rr_detrend))
    rr_windowed = rr_detrend * window

    # Welchメソッドでパワースペクトル密度を計算
    freqs, power = signal.welch(
        rr_windowed,
        fs=fs,
        nperseg=min(256, len(rr_windowed)),
        scaling='density'
    )

    return freqs, power


def calculate_1f_slope(freqs, power, freq_range=(0.01, 0.4)):
    """
    パワースペクトルから1/f傾きを計算

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）
    power : np.ndarray
        パワースペクトル密度
    freq_range : tuple, default=(0.01, 0.4)
        解析周波数範囲（Hz）

    Returns
    -------
    dict
        - slope: 傾き（1/fなら-1に近い）
        - intercept: 切片
        - r_value: 相関係数
        - p_value: p値
        - std_err: 標準誤差
    """
    # 周波数範囲でフィルタ
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1]) & (freqs > 0) & (power > 0)
    freqs_filtered = freqs[mask]
    power_filtered = power[mask]

    # 対数変換
    log_freq = np.log10(freqs_filtered)
    log_power = np.log10(power_filtered)

    # 線形回帰
    slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_power)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'freq_range': freq_range
    }


def classify_noise_type(slope):
    """
    傾きからノイズタイプを分類

    Parameters
    ----------
    slope : float
        パワースペクトルの傾き

    Returns
    -------
    str
        ノイズタイプ分類
    """
    if slope > -0.5:
        return "White Noise (無相関)"
    elif -1.5 < slope <= -0.5:
        return "Pink Noise (1/fゆらぎ)"
    elif -2.5 < slope <= -1.5:
        return "Brown Noise (強相関)"
    else:
        return "Black Noise (非常に強い相関)"


def plot_power_spectrum(freqs, power, slope_result, output_path):
    """
    パワースペクトルと1/f傾きをプロット

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）
    power : np.ndarray
        パワースペクトル密度
    slope_result : dict
        calculate_1f_slope()の結果
    output_path : str
        出力画像パス
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 1. 線形スケール
    ax1.plot(freqs, power, 'b-', linewidth=1, alpha=0.7)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Power Spectral Density', fontsize=12)
    ax1.set_title('HRV Power Spectrum (Linear Scale)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # VLF, LF, HF帯域を色分け
    ax1.axvspan(0.003, 0.04, alpha=0.2, color='purple', label='VLF (0.003-0.04 Hz)')
    ax1.axvspan(0.04, 0.15, alpha=0.2, color='green', label='LF (0.04-0.15 Hz)')
    ax1.axvspan(0.15, 0.4, alpha=0.2, color='red', label='HF (0.15-0.4 Hz)')
    ax1.legend(loc='upper right')

    # 2. 対数スケール
    # 0を除外
    mask = (freqs > 0) & (power > 0)
    freqs_plot = freqs[mask]
    power_plot = power[mask]

    ax2.loglog(freqs_plot, power_plot, 'b-', linewidth=1, alpha=0.7, label='Observed')

    # フィッティング直線を追加
    freq_range = slope_result['freq_range']
    freq_fit = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 100)
    power_fit = 10**(slope_result['intercept'] + slope_result['slope'] * np.log10(freq_fit))

    noise_type = classify_noise_type(slope_result['slope'])
    label = f'Fit: slope={slope_result["slope"]:.2f}\n{noise_type}\n$R^2$={slope_result["r_value"]**2:.3f}'
    ax2.loglog(freq_fit, power_fit, 'r--', linewidth=2, label=label)

    # 参考線（1/f, 1/f^2）
    ax2.loglog(freq_fit, power_fit[0] * (freq_fit[0]/freq_fit)**1,
               'g:', linewidth=1, alpha=0.5, label='1/f (Pink, slope=-1)')
    ax2.loglog(freq_fit, power_fit[0] * (freq_fit[0]/freq_fit)**2,
               'm:', linewidth=1, alpha=0.5, label='1/f² (Brown, slope=-2)')

    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Power Spectral Density', fontsize=12)
    ax2.set_title('HRV Power Spectrum (Log-Log Scale)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(data_path, output_dir, slope_result, rr_stats):
    """
    解析レポートを生成

    Parameters
    ----------
    data_path : Path
        入力CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    slope_result : dict
        1/f傾き解析結果
    rr_stats : dict
        R-R間隔統計情報
    """
    report_path = output_dir / '1F_ANALYSIS_REPORT.md'

    noise_type = classify_noise_type(slope_result['slope'])

    # 解釈文
    if -1.5 < slope_result['slope'] <= -0.5:
        interpretation = """
**結果**: あなたのHRVは **1/fゆらぎ（ピンクノイズ）** の特性を示しています。

1/fゆらぎは自然界に広く見られる理想的なゆらぎパターンで、以下のような特徴があります：
- 心拍変動が適度に予測可能で、かつ適度にランダム
- 自律神経系が柔軟に機能している証拠
- ストレス対応能力が高い状態
"""
    elif slope_result['slope'] > -0.5:
        interpretation = """
**結果**: あなたのHRVは **ホワイトノイズ** に近い特性を示しています。

これは心拍変動が非常にランダムで予測困難な状態を意味します：
- 自律神経系の調整が不安定な可能性
- 過度なストレスや疲労の影響
- リラクゼーションや休息が推奨されます
"""
    elif -2.5 < slope_result['slope'] <= -1.5:
        interpretation = """
**結果**: あなたのHRVは **ブラウンノイズ** に近い特性を示しています。

これは心拍変動が強い相関を持つ状態を意味します：
- 心拍パターンが過度に規則的
- 自律神経の柔軟性が低下している可能性
- リラクゼーションや呼吸法の実践が推奨されます
"""
    else:
        interpretation = """
**結果**: あなたのHRVは **ブラックノイズ** の特性を示しています。

これは非常に強い相関を持つ異常な状態です：
- 心拍変動が極端に規則的または異常
- 健康状態の確認が推奨されます
"""

    content = f"""# 1/fゆらぎ検証レポート

**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**データファイル**: {data_path.name}

---

## 解析結果サマリー

{interpretation}

---

## パワースペクトル解析

### 傾き（Slope）

| Metric | Value | Unit |
|--------|-------|------|
| Slope | {slope_result['slope']:.3f} | - |
| Correlation (R) | {slope_result['r_value']:.3f} | - |
| R² | {slope_result['r_value']**2:.3f} | - |
| P-value | {slope_result['p_value']:.2e} | - |
| Frequency Range | {slope_result['freq_range'][0]}-{slope_result['freq_range'][1]} | Hz |

### ノイズ分類

**{noise_type}**

---

## HRVデータ統計

| Metric | Value | Unit |
|--------|-------|------|
| Data Points | {rr_stats['count']} | samples |
| Mean RR | {rr_stats['mean']:.1f} | ms |
| Std RR | {rr_stats['std']:.1f} | ms |
| Min RR | {rr_stats['min']:.1f} | ms |
| Max RR | {rr_stats['max']:.1f} | ms |
| Duration | {rr_stats['duration']:.1f} | sec |

---

## パワースペクトル図

![Power Spectrum](img/power_spectrum.png)

---

## 参考: ノイズタイプ分類

| Noise Type | Slope Range | Characteristics |
|------------|-------------|-----------------|
| White Noise | > -0.5 | 無相関、完全ランダム |
| **Pink Noise (1/f)** | **-1.5 to -0.5** | **理想的なゆらぎ** |
| Brown Noise | -2.5 to -1.5 | 強相関、規則的 |
| Black Noise | < -2.5 | 非常に強い相関 |

---

## 備考

- パワースペクトルはWelch法で計算されています
- 解析周波数範囲: {slope_result['freq_range'][0]}-{slope_result['freq_range'][1]} Hz
- リサンプリング周波数: 4.0 Hz
"""

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f'\n✓ レポート生成完了: {report_path}')


def analyze_1f_fluctuation(data_path, output_dir, warmup_seconds=60.0):
    """
    1/fゆらぎ解析を実行

    Parameters
    ----------
    data_path : Path
        Selfloops CSVファイルパス
    output_dir : Path
        出力ディレクトリ
    warmup_seconds : float, default=60.0
        ウォームアップ除外時間（秒）
    """
    print('='*60)
    print('1/fゆらぎ検証解析')
    print('='*60)
    print()

    # 画像出力ディレクトリ
    img_dir = output_dir / 'img'
    img_dir.mkdir(exist_ok=True, parents=True)

    # データ読み込み
    print(f'Loading: {data_path}')
    sl_df = load_selfloops_csv(str(data_path), warmup_seconds=warmup_seconds)

    print(f'データ形状: {sl_df.shape[0]} 行 × {sl_df.shape[1]} 列')

    # HRVデータ取得
    print('計算中: HRVデータ抽出...')
    hrv_data = get_hrv_data(sl_df, clean_artifacts=True)

    rr_intervals = hrv_data['rr_intervals_clean']
    print(f'R-R間隔データ数: {len(rr_intervals)} サンプル')

    # 統計情報
    rr_stats = {
        'count': len(rr_intervals),
        'mean': np.mean(rr_intervals),
        'std': np.std(rr_intervals),
        'min': np.min(rr_intervals),
        'max': np.max(rr_intervals),
        'duration': np.sum(rr_intervals) / 1000.0  # 秒
    }

    print(f'\nR-R間隔統計:')
    print(f'  平均: {rr_stats["mean"]:.1f} ms')
    print(f'  標準偏差: {rr_stats["std"]:.1f} ms')
    print(f'  計測時間: {rr_stats["duration"]:.1f} 秒')

    # パワースペクトル計算
    print('\n計算中: パワースペクトル解析...')
    freqs, power = calculate_power_spectrum(rr_intervals, fs=4.0)

    # 1/f傾き計算
    print('計算中: 1/f傾き解析...')
    slope_result = calculate_1f_slope(freqs, power, freq_range=(0.01, 0.4))

    print(f'\n1/f傾き解析結果:')
    print(f'  傾き (Slope): {slope_result["slope"]:.3f}')
    print(f'  相関係数 (R): {slope_result["r_value"]:.3f}')
    print(f'  決定係数 (R²): {slope_result["r_value"]**2:.3f}')
    print(f'  P値: {slope_result["p_value"]:.2e}')
    print(f'  分類: {classify_noise_type(slope_result["slope"])}')

    # プロット
    print('\nプロット中: パワースペクトル...')
    plot_path = img_dir / 'power_spectrum.png'
    plot_power_spectrum(freqs, power, slope_result, str(plot_path))
    print(f'✓ 保存: {plot_path}')

    # レポート生成
    generate_report(data_path, output_dir, slope_result, rr_stats)

    print()
    print('='*60)
    print('1/fゆらぎ解析完了!')
    print('='*60)
    print(f'出力先: {output_dir}/')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='HRVデータの1/fゆらぎ検証解析'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='Selfloops CSVファイルパス'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=project_root / 'issues' / '013_1_f_wave',
        help='出力ディレクトリ（デフォルト: issues/013_1_f_wave/）'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=60.0,
        help='ウォームアップ除外時間（秒）（デフォルト: 60.0）'
    )

    args = parser.parse_args()

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # 1/fゆらぎ解析実行
    analyze_1f_fluctuation(
        args.data,
        args.output,
        warmup_seconds=args.warmup
    )

    return 0


if __name__ == '__main__':
    exit(main())
