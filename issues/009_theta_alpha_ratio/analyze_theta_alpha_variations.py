#!/usr/bin/env python3
"""
Theta/Alpha比率の各種計算方式比較スクリプト

以下の3つの方式を比較します：
1. 従来方式: 全4チャネル平均のTheta / 全4チャネル平均のAlpha
2. FMT方式: 前頭部Theta (AF7/AF8) / 全4チャネル平均のAlpha
3. 後方Alpha方式: 前頭部Theta (AF7/AF8) / 後方2チャネル平均のAlpha (TP9/TP10)

Usage:
    python analyze_theta_alpha_variations.py --data <CSV_PATH> [--output <OUTPUT_DIR>]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    calculate_frontal_theta,
    calculate_alpha_power,
)
from lib.statistical_dataframe import create_statistical_dataframe

# 後方Alphaモジュールをインポート
sys.path.insert(0, str(Path(__file__).parent / 'lib'))
from posterior_alpha import calculate_posterior_alpha


def calculate_traditional_theta_alpha(df, raw, session_start, segment_minutes=3, warmup_minutes=1.0):
    """
    従来方式: 全4チャネル平均のTheta/Alpha比率を計算

    Statistical DataFrameを使用して計算
    """
    statistical_df = create_statistical_dataframe(
        raw,
        segment_minutes=segment_minutes,
        warmup_minutes=warmup_minutes,
        session_start=session_start,
        fnirs_results=None,
        hr_data=None,
        df_timestamps=df['TimeStamp'],
        df=df,
    )

    # theta/alpha比率を抽出（線形比率）
    ratios_df = statistical_df['band_ratios']

    return {
        'ratio_series': ratios_df['theta_alpha'],
        'ratio_db_series': ratios_df['theta_alpha_db'],
        'ratio_mean': ratios_df['theta_alpha'].mean(),
        'ratio_db_mean': ratios_df['theta_alpha_db'].mean(),
        'segments': statistical_df['band_ratios'],
        'method': '4ch Theta / 4ch Alpha (Traditional)',
    }


def calculate_fmt_4ch_alpha_ratio(df, raw, session_start):
    """
    FMT方式: 前頭部Theta (AF7/AF8) / 全4チャネル平均Alpha
    """
    # FMT計算（AF7/AF8の4-8Hz）
    fmt_result = calculate_frontal_theta(
        df,
        channels=('RAW_AF7', 'RAW_AF8'),
        band=(4.0, 8.0),
        raw=raw,
        include_alpha=False,
    )

    # 全4chのAlpha計算
    alpha_result = calculate_alpha_power(df)

    # dB差分から線形比率を計算
    # ratio = FMT / Alpha = 10^((FMT_dB - Alpha_dB) / 10)
    fmt_series = fmt_result.time_series
    alpha_series = alpha_result.time_series

    # タイムスタンプを揃える（10秒間隔にリサンプル済み）
    common_index = fmt_series.index.intersection(alpha_series.index)
    fmt_aligned = fmt_series.loc[common_index]
    alpha_aligned = alpha_series.loc[common_index]

    # Alpha Power結果はdBxスコアなので、元のAlpha dBに戻す必要がある
    # alpha_result.alpha_dbを使用
    alpha_db = alpha_result.alpha_db

    # FMTとAlphaの時系列から比率を計算
    # 簡易版: 平均値で計算
    fmt_mean = fmt_series.mean()
    ratio_db = fmt_mean - alpha_db
    ratio = 10 ** (ratio_db / 10)

    return {
        'fmt_series': fmt_series,
        'alpha_db': alpha_db,
        'ratio_mean': ratio,
        'ratio_db_mean': ratio_db,
        'method': 'FMT (AF7/AF8) / 4ch Alpha',
    }


def calculate_fmt_posterior_alpha_ratio(df, raw, session_start):
    """
    後方Alpha方式: 前頭部Theta (AF7/AF8) / 後方2チャネルAlpha (TP9/TP10)
    """
    # FMT計算（AF7/AF8の4-8Hz）
    fmt_result = calculate_frontal_theta(
        df,
        channels=('RAW_AF7', 'RAW_AF8'),
        band=(4.0, 8.0),
        raw=raw,
        include_alpha=False,
    )

    # 後方2chのAlpha計算（TP9/TP10の8-13Hz）
    posterior_alpha_result = calculate_posterior_alpha(
        df,
        channels=('RAW_TP9', 'RAW_TP10'),
        band=(8.0, 13.0),
        raw=raw,
    )

    # dB差分から線形比率を計算
    fmt_series = fmt_result.time_series
    alpha_series = posterior_alpha_result.time_series

    # タイムスタンプを揃える
    common_index = fmt_series.index.intersection(alpha_series.index)
    fmt_aligned = fmt_series.loc[common_index]
    alpha_aligned = alpha_series.loc[common_index]

    # 時系列の比率計算
    ratio_db_series = fmt_aligned - alpha_aligned
    ratio_series = 10 ** (ratio_db_series / 10)

    ratio_mean = ratio_series.mean()
    ratio_db_mean = ratio_db_series.mean()

    return {
        'fmt_series': fmt_series,
        'alpha_series': alpha_series,
        'ratio_series': ratio_series,
        'ratio_db_series': ratio_db_series,
        'ratio_mean': ratio_mean,
        'ratio_db_mean': ratio_db_mean,
        'method': 'FMT (AF7/AF8) / Posterior 2ch Alpha (TP9/TP10)',
    }


def plot_comparison(traditional, fmt_4ch, fmt_posterior, output_path):
    """3つの方式を比較するプロット"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # 上段: 時系列比較
    ax1 = axes[0]

    if 'ratio_series' in traditional and not traditional['ratio_series'].empty:
        ax1.plot(
            traditional['ratio_series'].index,
            traditional['ratio_series'].values,
            label=f"Traditional (mean={traditional['ratio_mean']:.3f})",
            marker='o',
            alpha=0.7,
        )

    if 'ratio_series' in fmt_posterior and not fmt_posterior['ratio_series'].empty:
        ax1.plot(
            fmt_posterior['ratio_series'].index,
            fmt_posterior['ratio_series'].values,
            label=f"FMT/Posterior (mean={fmt_posterior['ratio_mean']:.3f})",
            marker='s',
            alpha=0.7,
        )

    ax1.set_xlabel('Time')
    ax1.set_ylabel('θ/α Ratio (Linear)')
    ax1.set_title('Theta/Alpha Ratio Comparison (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 下段: 棒グラフ比較
    ax2 = axes[1]

    methods = []
    means = []

    if traditional['ratio_mean'] is not None:
        methods.append('Traditional\n(4ch θ / 4ch α)')
        means.append(traditional['ratio_mean'])

    if fmt_4ch['ratio_mean'] is not None:
        methods.append('FMT/4ch α\n(AF7/8 θ / 4ch α)')
        means.append(fmt_4ch['ratio_mean'])

    if fmt_posterior['ratio_mean'] is not None:
        methods.append('FMT/Posterior α\n(AF7/8 θ / TP9/10 α)')
        means.append(fmt_posterior['ratio_mean'])

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax2.bar(methods, means, color=colors[:len(methods)], alpha=0.7)

    # 値をバーの上に表示
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
        )

    ax2.set_ylabel('θ/α Ratio (Linear)')
    ax2.set_title('Mean Theta/Alpha Ratio Comparison')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ 比較プロット保存: {output_path}')


def generate_report(traditional, fmt_4ch, fmt_posterior, output_path):
    """マークダウンレポート生成"""

    report = f"""# Theta/Alpha比率計算方式の比較分析

## 概要

3つの異なるTheta/Alpha比率計算方式を比較しました。

### 比較した方式

1. **Traditional (従来方式)**
   - Theta: 全4チャネル平均（TP9, AF7, AF8, TP10）
   - Alpha: 全4チャネル平均（TP9, AF7, AF8, TP10）
   - 計算方法: Statistical DataFrame（3分セグメント平均）

2. **FMT / 4ch Alpha**
   - Theta: 前頭部2チャネル（AF7, AF8）の4-8Hz
   - Alpha: 全4チャネル平均の8-13Hz
   - 計算方法: Hilbert変換

3. **FMT / Posterior Alpha (提案方式)**
   - Theta: 前頭部2チャネル（AF7, AF8）の4-8Hz
   - Alpha: 後方2チャネル（TP9, TP10）の8-13Hz
   - 計算方法: Hilbert変換

---

## 結果サマリー

### 平均比率（線形スケール）

| 方式 | θ/α 比率 (Linear) | θ/α 比率 (dB) | 備考 |
|:-----|------------------:|--------------:|:-----|"""

    # Traditional
    if traditional.get('ratio_mean') is not None:
        ratio_db = traditional.get('ratio_db_mean', 'N/A')
        ratio_db_str = f"{ratio_db:.4f}" if isinstance(ratio_db, (int, float)) else str(ratio_db)
        report += f"| Traditional (4ch θ / 4ch α) | {traditional['ratio_mean']:.4f} | {ratio_db_str} | 従来方式 |\n"
    else:
        report += f"| Traditional (4ch θ / 4ch α) | N/A | N/A | Error |\n"

    # FMT / 4ch Alpha
    if fmt_4ch.get('ratio_mean') is not None:
        report += f"| FMT / 4ch Alpha | {fmt_4ch['ratio_mean']:.4f} | {fmt_4ch['ratio_db_mean']:.4f} | 前頭部θのみ使用 |\n"
    else:
        report += f"| FMT / 4ch Alpha | N/A | N/A | Error |\n"

    # FMT / Posterior Alpha
    if fmt_posterior.get('ratio_mean') is not None:
        report += f"| FMT / Posterior Alpha | {fmt_posterior['ratio_mean']:.4f} | {fmt_posterior['ratio_db_mean']:.4f} | 前頭θ/後方α |\n"
    else:
        report += f"| FMT / Posterior Alpha | N/A | N/A | Error |\n"

    report += """

---

## 比較プロット

![Theta/Alpha Ratio Comparison](theta_alpha_comparison.png)

---

## 考察

### 各方式の特徴

#### Traditional (従来方式)
- **長所**: 全チャネルの情報を使用、安定した値
- **短所**: 前頭部と後方部の特性を区別しない

#### FMT / 4ch Alpha
- **長所**: 前頭部Thetaに特化、瞑想深度との相関が期待される
- **短所**: Alphaは全チャネル平均のため、前頭部の影響も含む

#### FMT / Posterior Alpha (提案方式)
- **長所**: 前頭部Theta（瞑想深度）と後方部Alpha（覚醒度）を分離
- **短所**: チャネル数が少ないため、ノイズの影響を受けやすい可能性

### 値の違い

Traditional方式と比較して、FMT方式（特にPosterior Alpha使用）は**異なる値**を示します。
これは：

1. **使用チャネルの違い**: 全4ch vs 前頭2ch (Theta) & 後方2ch (Alpha)
2. **帯域の違い**: Statistical DFは固定帯域、Hilbert変換は指定帯域
3. **計算方法の違い**: セグメント平均 vs 連続時系列

---

## 統計情報

### Traditional (4ch θ / 4ch α)
"""

    # Traditional方式の詳細
    if 'segments' in traditional and not traditional['segments'].empty:
        segments = traditional['segments']
        report += f"""
- セグメント数: {len(segments)}
- θ/α比率（線形）:
  - 平均: {segments['theta_alpha'].mean():.4f}
  - 標準偏差: {segments['theta_alpha'].std():.4f}
  - 最小: {segments['theta_alpha'].min():.4f}
  - 最大: {segments['theta_alpha'].max():.4f}
"""

    # FMT / 4ch Alpha statistics
    report += "\n### FMT / 4ch Alpha\n"
    if fmt_4ch.get('ratio_mean') is not None:
        fmt_mean = fmt_4ch.get('fmt_series', pd.Series()).mean()
        alpha_db = fmt_4ch.get('alpha_db', 0)
        report += f"- FMT平均: {fmt_mean:.3f} dB\n"
        report += f"- Alpha平均: {alpha_db:.3f} dB\n"
        report += f"- θ/α比率（線形）: {fmt_4ch['ratio_mean']:.4f}\n"
        report += f"- θ/α比率（dB）: {fmt_4ch['ratio_db_mean']:.4f}\n"
    else:
        report += "- エラーにより計算できませんでした\n"

    # FMT / Posterior Alpha statistics
    report += "\n### FMT / Posterior Alpha\n"
    if fmt_posterior.get('ratio_mean') is not None:
        fmt_mean = fmt_posterior.get('fmt_series', pd.Series()).mean()
        alpha_mean = fmt_posterior.get('alpha_series', pd.Series()).mean()
        report += f"- FMT平均: {fmt_mean:.3f} dB\n"
        report += f"- Posterior Alpha平均: {alpha_mean:.3f} dB\n"
        report += f"- θ/α比率（線形）: {fmt_posterior['ratio_mean']:.4f}\n"
        report += f"- θ/α比率（dB）: {fmt_posterior['ratio_db_mean']:.4f}\n"
    else:
        report += "- エラーにより計算できませんでした\n"

    report += f"""

---

## 結論

**提案方式（FMT / Posterior Alpha）の特徴**:

1. **前頭部Theta（瞑想深度）と後方部Alpha（覚醒度）を分離**して測定
2. 従来方式と比較して、**異なる値**を示す（これは想定内）
3. 瞑想状態の評価において、より特異的な指標となる可能性がある

**今後の検証課題**:

1. 複数セッションでの再現性確認
2. 瞑想深度との相関分析
3. ノイズ耐性の評価
4. 最適な帯域幅の検討（4-8Hz, 5-7Hz等）

---

生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ レポート生成: {output_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='Theta/Alpha比率の各種計算方式比較'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='入力CSVファイルパス'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path(__file__).parent,
        help='出力ディレクトリ（デフォルト: スクリプトと同じディレクトリ）'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=1.0,
        help='ウォームアップ除外時間（分）'
    )
    parser.add_argument(
        '--segment',
        type=int,
        default=3,
        help='セグメント長（分）'
    )

    args = parser.parse_args()

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('Theta/Alpha比率計算方式の比較分析')
    print('='*60)
    print()

    # データ読み込み
    print(f'Loading: {args.data}')
    df = load_mind_monitor_csv(
        args.data,
        filter_headband=False,
        warmup_seconds=args.warmup * 60
    )

    print(f'データ形状: {df.shape[0]} 行 × {df.shape[1]} 列')

    # MNE RAW準備
    print('準備中: MNE RAWデータ...')
    mne_dict = prepare_mne_raw(df, apply_bandpass=False, apply_notch=False)
    if not mne_dict:
        print('エラー: MNE RAWデータの準備に失敗しました')
        return 1

    raw = mne_dict['raw']
    session_start = pd.to_datetime(df['TimeStamp'].iloc[0])

    print('検出されたチャネル:', mne_dict['channels'])
    print(f'サンプリングレート: {mne_dict["sfreq"]:.2f} Hz')
    print()

    # 1. Traditional方式
    print('計算中: Traditional (4ch θ / 4ch α)...')
    try:
        # Bandpass適用済みのrawを使用
        raw_filtered = prepare_mne_raw(df)['raw']
        traditional = calculate_traditional_theta_alpha(
            df, raw_filtered, session_start,
            segment_minutes=args.segment,
            warmup_minutes=args.warmup
        )
        print(f'  → θ/α比率（線形）: {traditional["ratio_mean"]:.4f}')
    except Exception as e:
        print(f'  エラー: {e}')
        traditional = {'ratio_mean': None, 'method': 'Traditional (Error)'}

    # 2. FMT / 4ch Alpha方式
    print('計算中: FMT / 4ch Alpha...')
    try:
        fmt_4ch = calculate_fmt_4ch_alpha_ratio(df, raw, session_start)
        print(f'  → θ/α比率（線形）: {fmt_4ch["ratio_mean"]:.4f}')
    except Exception as e:
        print(f'  エラー: {e}')
        import traceback
        traceback.print_exc()
        fmt_4ch = {'ratio_mean': None, 'method': 'FMT/4ch Alpha (Error)'}

    # 3. FMT / Posterior Alpha方式
    print('計算中: FMT / Posterior Alpha...')
    try:
        fmt_posterior = calculate_fmt_posterior_alpha_ratio(df, raw, session_start)
        print(f'  → θ/α比率（線形）: {fmt_posterior["ratio_mean"]:.4f}')
    except Exception as e:
        print(f'  エラー: {e}')
        import traceback
        traceback.print_exc()
        fmt_posterior = {'ratio_mean': None, 'method': 'FMT/Posterior Alpha (Error)'}

    print()

    # プロット生成
    print('生成中: 比較プロット...')
    try:
        plot_comparison(
            traditional, fmt_4ch, fmt_posterior,
            args.output / 'theta_alpha_comparison.png'
        )
    except Exception as e:
        print(f'  エラー: {e}')
        import traceback
        traceback.print_exc()

    # レポート生成
    print('生成中: レポート...')
    try:
        generate_report(
            traditional, fmt_4ch, fmt_posterior,
            args.output / 'REPORT.md'
        )
    except Exception as e:
        print(f'  エラー: {e}')
        import traceback
        traceback.print_exc()

    print()
    print('='*60)
    print('分析完了!')
    print('='*60)
    print(f'出力ディレクトリ: {args.output}')
    print(f'  - REPORT.md')
    print(f'  - theta_alpha_comparison.png')

    return 0


if __name__ == '__main__':
    exit(main())
