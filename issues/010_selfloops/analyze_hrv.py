#!/usr/bin/env python3
"""
HRV (Heart Rate Variability) 分析スクリプト - NeuroKit2版

SelfLoops HRV dataからNeuroKit2を使用してHRV指標を計算し、可視化とレポートを生成します。

Usage:
    python analyze_hrv.py --data <TXT_PATH> [--output <OUTPUT_DIR>] [--warmup <SECONDS>]
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.analysis import (
    analyze_hrv,
    analyze_hrv_time_domain,
    analyze_hrv_frequency_domain,
    analyze_hrv_nonlinear
)


def plot_hrv_analysis(df, hrv_indices, output_path):
    """
    HRV分析結果を可視化

    Args:
        df: SelfLoopsデータフレーム
        hrv_indices: NeuroKit2のHRV解析結果
        output_path: 出力画像パス
    """
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. R-R間隔の時系列
    ax1 = fig.add_subplot(gs[0, :])
    time_min = df['Time_sec'].values / 60
    ax1.plot(time_min, df['R-R (ms)'].values, alpha=0.7, linewidth=0.8)
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('R-R Interval (ms)')
    ax1.set_title('R-R Interval Time Series')
    ax1.grid(True, alpha=0.3)

    # 統計情報をテキストで追加
    mean_rr = hrv_indices['HRV_MeanNN'].values[0]
    sdnn = hrv_indices['HRV_SDNN'].values[0]
    rmssd = hrv_indices['HRV_RMSSD'].values[0]
    textstr = f'Mean: {mean_rr:.1f} ms\nSDNN: {sdnn:.1f} ms\nRMSSD: {rmssd:.1f} ms'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. 心拍数の時系列
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_min, df['HR (bpm)'].values, color='orangered', alpha=0.7, linewidth=0.8)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Heart Rate (bpm)')
    ax2.set_title('Heart Rate Time Series')
    ax2.grid(True, alpha=0.3)

    # 3. R-R間隔のヒストグラム
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(df['R-R (ms)'].values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax3.axvline(mean_rr, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_rr:.1f} ms')
    ax3.set_xlabel('R-R Interval (ms)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('R-R Interval Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. パワースペクトル密度（NeuroKit2から再計算）
    ax4 = fig.add_subplot(gs[2, 0])
    # 簡易的にLF/HFを棒グラフで表示
    lf_power = hrv_indices['HRV_LF'].values[0]
    hf_power = hrv_indices['HRV_HF'].values[0]
    vlf_power = hrv_indices['HRV_VLF'].values[0]

    bands = ['VLF\n(0.003-0.04 Hz)', 'LF\n(0.04-0.15 Hz)', 'HF\n(0.15-0.4 Hz)']
    powers = [vlf_power, lf_power, hf_power]
    colors = ['#9467bd', '#ffbb78', '#98df8a']

    bars = ax4.bar(bands, powers, color=colors, alpha=0.7)
    ax4.set_ylabel('Power (ms²)')
    ax4.set_title('Frequency Domain Power')
    ax4.grid(True, alpha=0.3, axis='y')

    # 値をバーの上に表示
    for bar, value in zip(bars, powers):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.0f}',
                ha='center', va='bottom', fontsize=9)

    # 5. HRV指標の棒グラフ
    ax5 = fig.add_subplot(gs[2, 1])
    pnn50 = hrv_indices['HRV_pNN50'].values[0]
    lf_hf_ratio = hrv_indices['HRV_LFHF'].values[0]

    metrics = ['SDNN\n(ms)', 'RMSSD\n(ms)', 'pNN50\n(%)', 'LF/HF\nRatio']
    values = [sdnn, rmssd, pnn50, lf_hf_ratio]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars = ax5.bar(metrics, values, color=colors, alpha=0.7)

    # 値をバーの上に表示
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=9)

    ax5.set_ylabel('Value')
    ax5.set_title('HRV Metrics Summary')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ 可視化プロット保存: {output_path}')


def generate_report(df, hrv_indices, output_path):
    """
    マークダウンレポートを生成

    Args:
        df: SelfLoopsデータフレーム
        hrv_indices: NeuroKit2のHRV解析結果
        output_path: 出力レポートパス
    """
    # 測定時間を計算
    duration_min = df['Time_sec'].iloc[-1] / 60

    # NeuroKit2の指標を取得
    mean_rr = hrv_indices['HRV_MeanNN'].values[0]
    sdnn = hrv_indices['HRV_SDNN'].values[0]
    rmssd = hrv_indices['HRV_RMSSD'].values[0]
    pnn50 = hrv_indices['HRV_pNN50'].values[0]

    vlf_power = hrv_indices['HRV_VLF'].values[0]
    lf_power = hrv_indices['HRV_LF'].values[0]
    hf_power = hrv_indices['HRV_HF'].values[0]
    lf_hf_ratio = hrv_indices['HRV_LFHF'].values[0]
    lf_nu = hrv_indices['HRV_LFn'].values[0] if 'HRV_LFn' in hrv_indices.columns else 0
    hf_nu = hrv_indices['HRV_HFn'].values[0] if 'HRV_HFn' in hrv_indices.columns else 0

    mean_hr = 60000 / mean_rr

    # セッション開始時刻
    session_start = df.attrs.get('session_start', 'N/A')

    report = f"""# HRV (Heart Rate Variability) 分析レポート

## 概要

SelfLoops HRV dataから心拍変動(HRV)を分析しました（NeuroKit2使用）。

### 測定情報

- **測定時間**: {duration_min:.2f} 分
- **データポイント数**: {len(df)} 点
- **測定開始日時**: {session_start}

---

## 基本統計

### 心拍数 (HR)

| 指標 | 値 |
|:-----|---:|
| 平均 | {df['HR (bpm)'].mean():.2f} bpm |
| 最小 | {df['HR (bpm)'].min()} bpm |
| 最大 | {df['HR (bpm)'].max()} bpm |
| 標準偏差 | {df['HR (bpm)'].std():.2f} bpm |

### R-R間隔

| 指標 | 値 |
|:-----|---:|
| 平均 | {df['R-R (ms)'].mean():.2f} ms |
| 最小 | {df['R-R (ms)'].min()} ms |
| 最大 | {df['R-R (ms)'].max()} ms |
| 標準偏差 | {df['R-R (ms)'].std():.2f} ms |

---

## HRV指標

### 時間領域指標

| 指標 | 値 | 説明 |
|:-----|---:|:-----|
| **SDNN** | {sdnn:.2f} ms | R-R間隔の標準偏差。全体的な心拍変動の指標 |
| **RMSSD** | {rmssd:.2f} ms | 連続R-R間隔差の二乗平均平方根。副交感神経活動の指標 |
| **pNN50** | {pnn50:.2f} % | 50ms以上異なる連続R-R間隔の割合。副交感神経活動の指標 |

#### SDNN評価基準（参考値）

- **100 ms以上**: 優秀
- **50-100 ms**: 良好
- **20-50 ms**: 低下
- **20 ms未満**: 著しく低下

現在の値: **{sdnn:.2f} ms** - {'優秀' if sdnn >= 100 else '良好' if sdnn >= 50 else '低下' if sdnn >= 20 else '著しく低下'}

### 周波数領域指標

| 指標 | 値 | 説明 |
|:-----|---:|:-----|
| **VLF Power** | {vlf_power:.2f} ms² | 超低周波成分 (0.003-0.04 Hz) |
| **LF Power** | {lf_power:.2f} ms² | 低周波成分 (0.04-0.15 Hz)。交感神経と副交感神経の両方 |
| **HF Power** | {hf_power:.2f} ms² | 高周波成分 (0.15-0.4 Hz)。副交感神経活動 |
| **LF/HF比** | {lf_hf_ratio:.2f} | 交感神経/副交感神経バランスの指標 |
| **LF (n.u.)** | {lf_nu:.2f} % | 正規化LF |
| **HF (n.u.)** | {hf_nu:.2f} % | 正規化HF |

#### LF/HF比の評価

- **1.5-2.0**: バランスが取れている
- **2.0以上**: 交感神経優位（ストレス状態）
- **1.0未満**: 副交感神経優位（リラックス状態）

現在の値: **{lf_hf_ratio:.2f}** - {'交感神経優位' if lf_hf_ratio >= 2.0 else 'リラックス状態' if lf_hf_ratio < 1.0 else 'バランス'}

---

## 可視化

![HRV Analysis](hrv_analysis.png)

---

## 解釈

### 時間領域指標からの評価

SDNNは **{sdnn:.2f} ms** で、{'優秀な' if sdnn >= 100 else '良好な' if sdnn >= 50 else '低下した' if sdnn >= 20 else '著しく低下した'}心拍変動を示しています。

RMSSDは **{rmssd:.2f} ms** で、{'高い' if rmssd >= 40 else '中程度の' if rmssd >= 20 else '低い'}副交感神経活動を示唆しています。

pNN50は **{pnn50:.2f}%** です。{'高い値（20%以上）' if pnn50 >= 20 else '中程度の値（5-20%）' if pnn50 >= 5 else '低い値（5%未満）'}です。

### 周波数領域指標からの評価

LF/HF比は **{lf_hf_ratio:.2f}** で、自律神経系が{'交感神経優位の状態' if lf_hf_ratio >= 2.0 else '副交感神経優位のリラックス状態' if lf_hf_ratio < 1.0 else 'バランスの取れた状態'}にあることを示しています。

---

## 考察

### HRV指標の臨床的意義

1. **SDNN**: 全体的な自律神経機能を反映。運動、ストレス、睡眠の質などに影響される
2. **RMSSD**: 短期的な心拍変動を反映。呼吸や副交感神経活動に敏感
3. **LF/HF比**: 交感神経と副交感神経のバランスを示す

### この測定結果の特徴

- 測定時間: {duration_min:.2f}分
- 平均心拍数: {mean_hr:.1f} bpm
- 心拍変動レベル: {'高い' if sdnn >= 50 else '中程度' if sdnn >= 20 else '低い'}

---

## 技術情報

- **解析ライブラリ**: NeuroKit2
- **HRV指標数**: {len(hrv_indices.columns)} 個
- **データソース**: SelfLoops HRV

---

## 今後の分析課題

1. 時系列での変化を追跡（朝晩の違い、日々の変化など）
2. 瞑想セッション前後でのHRV変化の比較
3. 他の生理指標（EEG、呼吸など）との相関分析
4. ストレスイベントや運動との関連性の検討

---

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ レポート生成: {output_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='HRV (Heart Rate Variability) 分析 - NeuroKit2版'
    )
    parser.add_argument(
        '--data',
        type=Path,
        required=True,
        help='SelfLoops HRV dataファイルパス (.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='出力ディレクトリ（デフォルト: データファイルと同じディレクトリ）'
    )
    parser.add_argument(
        '--warmup',
        type=float,
        default=0.0,
        help='測定開始からの除外期間（秒）'
    )

    args = parser.parse_args()

    # 出力ディレクトリの設定
    if args.output is None:
        args.output = args.data.parent

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('HRV (Heart Rate Variability) 分析 - NeuroKit2版')
    print('='*60)
    print()

    # データ読み込み
    print(f'Loading: {args.data}')
    df = load_selfloops_csv(str(args.data), warmup_seconds=args.warmup)

    print(f'データ形状: {df.shape[0]} 行 × {df.shape[1]} 列')
    print(f'測定時間: {df["Time_sec"].iloc[-1] / 60:.2f} 分')
    if args.warmup > 0:
        print(f'ウォームアップ除外: {args.warmup} 秒')
    print()

    # HRVデータ取得
    hrv_data = get_hrv_data(df)

    # HRV解析実行
    print('計算中: HRV指標（NeuroKit2）...')
    hrv_indices = analyze_hrv(hrv_data, show=False)

    print(f'  SDNN: {hrv_indices["HRV_SDNN"].values[0]:.2f} ms')
    print(f'  RMSSD: {hrv_indices["HRV_RMSSD"].values[0]:.2f} ms')
    print(f'  pNN50: {hrv_indices["HRV_pNN50"].values[0]:.2f} %')
    print(f'  LF Power: {hrv_indices["HRV_LF"].values[0]:.2f} ms²')
    print(f'  HF Power: {hrv_indices["HRV_HF"].values[0]:.2f} ms²')
    print(f'  LF/HF Ratio: {hrv_indices["HRV_LFHF"].values[0]:.2f}')
    print()

    # 可視化
    print('生成中: 可視化プロット...')
    plot_hrv_analysis(df, hrv_indices, args.output / 'hrv_analysis.png')
    print()

    # レポート生成
    print('生成中: マークダウンレポート...')
    generate_report(df, hrv_indices, args.output / 'REPORT.md')
    print()

    print('='*60)
    print('分析完了!')
    print('='*60)
    print(f'出力ディレクトリ: {args.output}')
    print(f'  - REPORT.md')
    print(f'  - hrv_analysis.png')

    return 0


if __name__ == '__main__':
    exit(main())
