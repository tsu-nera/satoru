#!/usr/bin/env python3
"""
共鳴周波数（Resonance Frequency, RF）HRV分析スクリプト（簡易版）

QUICK_PROTOCOL.mdに基づく3分測定データの分析
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib.loaders.elite_hrv import load_elite_hrv_txt, get_hrv_data
from lib.sensors.ecg.hrv import (
    calculate_rmssd,
    calculate_hr_stats,
    calculate_lf_power,
    analyze_measurement_order_effect,
)


def analyze_trial(
    trial_info: Dict,
    data_dir: Path,
    clean_artifacts: bool = True
) -> Dict:
    """1つのトライアルを分析"""
    filepath = data_dir / trial_info['file']

    # Elite HRVデータを読み込み
    df = load_elite_hrv_txt(
        str(filepath),
        breathing_rate=trial_info['rate']
    )

    # HRVデータを取得
    hrv_data = get_hrv_data(df, clean_artifacts=clean_artifacts)

    # クリーニング済みRR間隔を使用
    rr = hrv_data['rr_intervals_clean']

    # 指標の計算
    rmssd = calculate_rmssd(rr)
    hr_mean, hr_max_min = calculate_hr_stats(rr)
    lf_power, freqs, psd = calculate_lf_power(rr)

    return {
        'Trial': trial_info['trial'],
        'Rate': trial_info['rate'],
        'Inhale': trial_info['inhale'],
        'Exhale': trial_info['exhale'],
        'RMSSD': rmssd,
        'LF Power': lf_power,
        'HR Mean': hr_mean,
        'HR Max-Min': hr_max_min,
        'Samples': len(rr),
        'freqs': freqs,
        'psd': psd,
        'rr_intervals': rr
    }


def plot_results(
    df_results: pd.DataFrame,
    trial_results: List[Dict],
    output_path: Path
):
    """分析結果を可視化"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. LF Power vs Breathing Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(df_results['Rate'], df_results['LF Power'],
             'o-', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Breathing Rate (breaths/min)', fontsize=11)
    ax1.set_ylabel('LF Power (ms²)', fontsize=11)
    ax1.set_title('LF Power vs Breathing Rate', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()

    # 最大LF Powerをハイライト
    max_idx = df_results['LF Power'].idxmax()
    ax1.plot(df_results.loc[max_idx, 'Rate'],
             df_results.loc[max_idx, 'LF Power'],
             'r*', markersize=20,
             label=f"Peak: {df_results.loc[max_idx, 'Rate']} bpm")
    ax1.legend()

    # 2. RMSSD vs Breathing Rate
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(df_results['Rate'], df_results['RMSSD'],
             's-', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Breathing Rate (breaths/min)', fontsize=11)
    ax2.set_ylabel('RMSSD (ms)', fontsize=11)
    ax2.set_title('RMSSD vs Breathing Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()

    # 3. HR Max-Min vs Breathing Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(df_results['Rate'], df_results['HR Max-Min'],
             '^-', linewidth=2, markersize=8, color='#F18F01')
    ax3.set_xlabel('Breathing Rate (breaths/min)', fontsize=11)
    ax3.set_ylabel('HR Max-Min (bpm)', fontsize=11)
    ax3.set_title('HR Variability Range vs Breathing Rate',
                  fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()

    # 4-9. パワースペクトル密度（各トライアル）
    for idx, trial in enumerate(trial_results):
        row = 1 + idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])

        ax.semilogy(trial['freqs'], trial['psd'], color='#2E86AB')
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('PSD (ms²/Hz)', fontsize=10)
        ax.set_title(f"Trial {trial['Trial']}: {trial['Rate']} bpm",
                     fontsize=11, fontweight='bold')
        ax.set_xlim([0, 0.5])
        ax.grid(True, alpha=0.3)

        # LF帯域をハイライト
        ax.axvspan(0.04, 0.15, alpha=0.2, color='red', label='LF Band')
        ax.legend(fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Figure saved: {output_path}")


def generate_markdown_report(
    df_results: pd.DataFrame,
    correlation: float,
    has_order_effect: bool,
    output_path: Path
):
    """マークダウンレポートを生成"""
    max_idx = df_results['LF Power'].idxmax()
    optimal = df_results.loc[max_idx]

    report = f"""# Resonance Frequency HRV Analysis Report (Quick Protocol)

**分析日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**測定方法**: 簡易版プロトコル（3分×6回）

## 1. 最適な共鳴周波数（Optimal Resonance Frequency）

| 指標 | 値 |
|:-----|---:|
| **呼吸レート (Breathing Rate)** | {optimal['Rate']} breaths/min |
| **吸気時間 (Inhale)** | {optimal['Inhale']} sec |
| **呼気時間 (Exhale)** | {optimal['Exhale']} sec |
| **LF Power** | {optimal['LF Power']:.2f} ms² |
| **RMSSD** | {optimal['RMSSD']:.2f} ms |
| **HR Mean** | {optimal['HR Mean']:.2f} bpm |
| **HR Max-Min** | {optimal['HR Max-Min']:.2f} bpm |

## 2. 全トライアル結果

| Trial | Rate | Inhale | Exhale | RMSSD | LF Power | HR Mean | HR Max-Min | Samples |
|------:|-----:|-------:|-------:|------:|---------:|--------:|-----------:|--------:|
"""

    for _, row in df_results.iterrows():
        report += f"| {int(row['Trial'])} | {row['Rate']} | {row['Inhale']} | {row['Exhale']} | "
        report += f"{row['RMSSD']:.2f} | {row['LF Power']:.2f} | {row['HR Mean']:.2f} | "
        report += f"{row['HR Max-Min']:.2f} | {int(row['Samples'])} |\n"

    report += f"""
## 3. 測定順序効果の分析

- **測定順序とLF Powerの相関係数**: {correlation:.3f}
"""

    if has_order_effect:
        report += """
⚠️ **警告**: 測定順序とLF Powerに中程度以上の相関があります

測定条件が時間経過とともに変化している可能性があります。
"""
    else:
        report += """
✓ 測定順序効果は小さいです（相関係数の絶対値 < 0.5）

ランダム化された測定順序により、測定順序効果が軽減されました。
"""

    report += f"""
## 4. 結果の解釈

### 共鳴周波数の特定

**暫定的な共鳴周波数**: {optimal['Rate']} bpm

LF Powerが最大となる呼吸レートが個人の共鳴周波数です。
この周波数で呼吸することで、HRVが最大化され、自律神経のバランスが最適化されます。

### 注意事項

⚠️ **3分測定の制約**: この結果は簡易版プロトコル（3分測定）によるものです。

- より正確な測定には5分以上の測定を推奨
- 現在の結果は「スクリーニング」として扱う
- 気になるレートがあれば、5分測定で再確認を推奨

### 次のステップ

1. **訓練開始**: {optimal['Rate']} bpmで呼吸訓練を開始
2. **定期測定**: 数週間後に再度測定し、変化を確認
3. **精密測定** (オプション): 5分測定で共鳴周波数を再確認

## 参考文献

1. **Lehrer, P., & Gevirtz, R. (2014).**
   Heart rate variability biofeedback: how and why does it work?
   *Frontiers in psychology*, 5, 756.

2. **Steffen, P. R., et al. (2017).**
   A Practical Guide to Resonance Frequency Assessment for Heart Rate Variability Biofeedback.
   *Frontiers in Neuroscience*, 14, 570400.
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report saved: {output_path}")


def main():
    """メイン処理"""
    # 分析ディレクトリ
    analysis_dir = Path(__file__).parent
    data_dir = analysis_dir / 'data-10-18'
    img_dir = analysis_dir / 'img'

    # 出力パス
    output_img = img_dir / 'rf_hrv_analysis_quick.png'
    output_report = analysis_dir / 'REPORT_QUICK.md'

    # トライアル情報（QUICK_PROTOCOL.mdに基づく）
    trials = [
        {'trial': 1, 'rate': 5.0, 'file': '2026-01-18 10-18-10.txt',
         'inhale': 4.8, 'exhale': 7.2},
        {'trial': 2, 'rate': 3.5, 'file': '2026-01-18 10-27-29.txt',
         'inhale': 6.9, 'exhale': 10.3},
        {'trial': 3, 'rate': 6.0, 'file': '2026-01-18 10-31-32.txt',
         'inhale': 4.0, 'exhale': 6.0},
        {'trial': 4, 'rate': 4.5, 'file': '2026-01-18 10-35-30.txt',
         'inhale': 5.3, 'exhale': 8.0},
        {'trial': 5, 'rate': 5.5, 'file': '2026-01-18 10-39-07.txt',
         'inhale': 4.4, 'exhale': 6.5},
        {'trial': 6, 'rate': 4.0, 'file': '2026-01-18 10-42-52.txt',
         'inhale': 6.0, 'exhale': 9.0},
    ]

    print("=" * 70)
    print("Resonance Frequency HRV Analysis (Quick Protocol)")
    print("=" * 70)
    print()

    # 各トライアルを分析
    print("Analyzing trials...")
    trial_results = []
    for trial in trials:
        print(f"  Processing Trial {trial['trial']} ({trial['rate']} bpm)...")
        result = analyze_trial(trial, data_dir, clean_artifacts=True)
        trial_results.append(result)

    # DataFrameに変換
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ['freqs', 'psd', 'rr_intervals']}
        for r in trial_results
    ])

    print()
    print("Results:")
    print(df_results[['Trial', 'Rate', 'LF Power', 'RMSSD', 'HR Mean']].to_string(index=False))
    print()

    # 測定順序効果を分析
    correlation, has_order_effect = analyze_measurement_order_effect(df_results['LF Power'])
    print(f"Measurement order correlation: {correlation:.3f}")
    if has_order_effect:
        print("⚠️  Warning: Measurement order effect detected")
    else:
        print("✓ Measurement order effect is minimal")
    print()

    # 可視化
    print("Generating visualizations...")
    plot_results(df_results, trial_results, output_img)
    print()

    # レポート生成
    print("Generating markdown report...")
    generate_markdown_report(df_results, correlation, has_order_effect, output_report)
    print()

    # 最適な共鳴周波数を表示
    max_idx = df_results['LF Power'].idxmax()
    optimal = df_results.loc[max_idx]

    print("=" * 70)
    print("Optimal Resonance Frequency")
    print("=" * 70)
    print(f"Breathing Rate: {optimal['Rate']} breaths/min")
    print(f"LF Power: {optimal['LF Power']:.2f} ms²")
    print("=" * 70)


if __name__ == '__main__':
    main()
