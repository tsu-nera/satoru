#!/usr/bin/env python3
"""
HRV分析スクリプト - シンプル版

NeuroKit2の組み込み機能を最大限活用したシンプルなHRV解析。
可視化もNeuroKit2に任せることでコード量を大幅削減。

Usage:
    python analyze_hrv_simple.py --data <CSV_PATH> [--warmup <SECONDS>]
"""

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
from lib.sensors.ecg.analysis import analyze_hrv


def save_neurokit_plots(hrv_data, output_dir):
    """
    NeuroKit2の組み込み可視化を保存

    Args:
        hrv_data: HRVデータ辞書
        output_dir: 出力ディレクトリ
    """
    peaks = nk.intervals_to_peaks(
        hrv_data['rr_intervals'],
        sampling_rate=hrv_data['sampling_rate']
    )

    # 周波数領域（パワースペクトル密度）
    print('生成中: 周波数領域プロット...')
    nk.hrv_frequency(peaks, sampling_rate=hrv_data['sampling_rate'], show=True)
    plt.savefig(output_dir / 'hrv_frequency.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 非線形領域（Poincaréプロット）
    print('生成中: Poincaréプロット...')
    nk.hrv_nonlinear(peaks, sampling_rate=hrv_data['sampling_rate'], show=True)
    plt.savefig(output_dir / 'hrv_poincare.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 時間領域（ヒストグラム）
    print('生成中: 時間領域プロット...')
    nk.hrv_time(peaks, sampling_rate=hrv_data['sampling_rate'], show=True)
    plt.savefig(output_dir / 'hrv_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'✓ NeuroKit2可視化保存完了')


def generate_simple_report(df, hrv_indices, output_path):
    """
    シンプルなマークダウンレポートを生成

    基本的なHRV指標のみに絞った簡潔なレポート。

    Args:
        df: SelfLoopsデータフレーム
        hrv_indices: NeuroKit2のHRV解析結果
        output_path: 出力レポートパス
    """
    duration_min = df['Time_sec'].iloc[-1] / 60
    session_start = df.attrs.get('session_start', 'N/A')

    # 主要指標の取得
    mean_rr = hrv_indices['HRV_MeanNN'].values[0]
    sdnn = hrv_indices['HRV_SDNN'].values[0]
    rmssd = hrv_indices['HRV_RMSSD'].values[0]
    pnn50 = hrv_indices['HRV_pNN50'].values[0]
    lf_power = hrv_indices['HRV_LF'].values[0]
    hf_power = hrv_indices['HRV_HF'].values[0]
    lf_hf_ratio = hrv_indices['HRV_LFHF'].values[0]
    mean_hr = 60000 / mean_rr

    report = f"""# HRV分析レポート（簡易版）

## 測定情報

- **測定日時**: {session_start}
- **測定時間**: {duration_min:.1f}分
- **データポイント数**: {len(df)}点
- **平均心拍数**: {mean_hr:.1f} bpm

---

## 主要HRV指標

### 時間領域

| 指標 | 値 | 評価 |
|:-----|---:|:-----|
| **SDNN** | {sdnn:.1f} ms | {'✅ 良好' if sdnn >= 50 else '⚠️ 低下' if sdnn >= 20 else '❌ 著しく低下'} |
| **RMSSD** | {rmssd:.1f} ms | {'✅ 高い' if rmssd >= 40 else '⭕ 中程度' if rmssd >= 20 else '⚠️ 低い'} |
| **pNN50** | {pnn50:.1f}% | {'✅ 高い' if pnn50 >= 20 else '⭕ 中程度' if pnn50 >= 5 else '⚠️ 低い'} |

### 周波数領域

| 指標 | 値 | 評価 |
|:-----|---:|:-----|
| **LF Power** | {lf_power:.2f} ms² | 低周波成分 (0.04-0.15 Hz) |
| **HF Power** | {hf_power:.2f} ms² | 高周波成分 (0.15-0.4 Hz) |
| **LF/HF比** | {lf_hf_ratio:.2f} | {'⚠️ 交感神経優位' if lf_hf_ratio >= 2.0 else '✅ リラックス' if lf_hf_ratio < 1.0 else '⭕ バランス'} |

---

## 解釈

**自律神経バランス**: LF/HF比が{lf_hf_ratio:.2f}で、{'交感神経優位（ストレス状態）' if lf_hf_ratio >= 2.0 else '副交感神経優位（リラックス状態）' if lf_hf_ratio < 1.0 else 'バランスの取れた状態'}です。

**心拍変動レベル**: SDNN {sdnn:.1f}msは{'良好' if sdnn >= 50 else '低下' if sdnn >= 20 else '著しく低下'}な心拍変動を示しています。

**副交感神経活動**: RMSSD {rmssd:.1f}msは{'高い' if rmssd >= 40 else '中程度の' if rmssd >= 20 else '低い'}副交感神経活動を示唆しています。

---

## 可視化

### 周波数領域分析
![Frequency Domain](hrv_frequency.png)

### Poincaréプロット
![Poincare Plot](hrv_poincare.png)

### 時間領域分析
![Time Domain](hrv_time.png)

---

## 技術情報

- **解析ライブラリ**: NeuroKit2
- **計算されたHRV指標**: {len(hrv_indices.columns)}個
- **データソース**: SelfLoops HRV

---

*生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'✓ レポート生成: {output_path}')


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='HRV分析 - シンプル版（NeuroKit2組み込み機能活用）'
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
        args.output = args.data.parent / 'simple_output'

    # パスの検証
    if not args.data.exists():
        print(f'エラー: データファイルが見つかりません: {args.data}')
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    print('='*60)
    print('HRV分析 - シンプル版')
    print('='*60)
    print()

    # データ読み込み
    print(f'📁 Loading: {args.data.name}')
    df = load_selfloops_csv(str(args.data), warmup_seconds=args.warmup)
    print(f'   データ: {len(df)}点, {df["Time_sec"].iloc[-1] / 60:.1f}分')
    if args.warmup > 0:
        print(f'   ウォームアップ除外: {args.warmup}秒')
    print()

    # HRV解析実行
    print('🔬 計算中: HRV指標（NeuroKit2）...')
    hrv_data = get_hrv_data(df)
    hrv_indices = analyze_hrv(hrv_data, show=False)

    # 主要指標の表示
    print(f'   SDNN: {hrv_indices["HRV_SDNN"].values[0]:.1f} ms')
    print(f'   RMSSD: {hrv_indices["HRV_RMSSD"].values[0]:.1f} ms')
    print(f'   LF/HF: {hrv_indices["HRV_LFHF"].values[0]:.2f}')
    print()

    # NeuroKit2の可視化機能を活用
    print('📊 可視化生成中（NeuroKit2組み込み機能）...')
    save_neurokit_plots(hrv_data, args.output)
    print()

    # レポート生成
    print('📝 レポート生成中...')
    generate_simple_report(df, hrv_indices, args.output / 'REPORT_SIMPLE.md')
    print()

    print('='*60)
    print('✅ 分析完了!')
    print('='*60)
    print(f'📂 出力先: {args.output}')
    print(f'   - REPORT_SIMPLE.md')
    print(f'   - hrv_frequency.png')
    print(f'   - hrv_poincare.png')
    print(f'   - hrv_time.png')

    return 0


if __name__ == '__main__':
    exit(main())
