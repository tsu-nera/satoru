#!/usr/bin/env python3
"""
共鳴周波数（Resonance Frequency, RF）HRV分析スクリプト

Elite HRVアプリで測定した呼吸ペーシング試験データを分析し、
個人の最適な共鳴周波数を特定します。

主要分析項目:
1. LF Power (0.04-0.15 Hz) - 共鳴周波数の主指標
2. RMSSD - 時間領域HRV指標
3. HR Mean / HR Max-Min - 心拍数統計
4. パワースペクトル密度（PSD）- 周波数領域解析
5. 測定順序効果の評価

参考文献:
- Lehrer, P., & Gevirtz, R. (2014). Heart rate variability biofeedback:
  how and why does it work? Frontiers in psychology, 5, 756.
- Steffen, P. R., et al. (2017). A Practical Guide to Resonance Frequency
  Assessment for Heart Rate Variability Biofeedback. Frontiers in
  Neuroscience, 14, 570400.
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
    """
    1つのトライアルを分析

    Parameters
    ----------
    trial_info : dict
        トライアル情報（trial, rate, file, inhale, exhale）
    data_dir : Path
        データディレクトリ
    clean_artifacts : bool, default True
        外れ値を除外するか

    Returns
    -------
    dict
        分析結果
    """
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
    """
    分析結果を可視化

    Parameters
    ----------
    df_results : pd.DataFrame
        分析結果のDataFrame
    trial_results : list of dict
        各トライアルの詳細結果
    output_path : Path
        出力画像パス
    """
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
    """
    マークダウンレポートを生成

    Parameters
    ----------
    df_results : pd.DataFrame
        分析結果のDataFrame
    correlation : float
        測定順序とLF Powerの相関係数
    has_order_effect : bool
        測定順序効果があるか
    output_path : Path
        出力レポートパス
    """
    max_idx = df_results['LF Power'].idxmax()
    optimal = df_results.loc[max_idx]

    # 同じ呼吸レートの測定があれば比較
    rate_counts = df_results['Rate'].value_counts()
    duplicate_rates = rate_counts[rate_counts > 1].index.tolist()

    report = f"""# Resonance Frequency HRV Analysis Report

**分析日時**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

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

測定条件が時間経過とともに変化している可能性があります:
- ウォーミングアップ効果
- 疲労
- 環境要因の変化

**推奨事項**:
1. 測定順序をランダム化して再測定
2. 各測定間に十分な休憩（2-3分）を挿入
3. 同じ時間帯・環境で測定
"""
    else:
        report += """
✓ 測定順序効果は小さいです（相関係数の絶対値 < 0.5）
"""

    # 再現性の評価
    if duplicate_rates:
        report += """
## 4. 再現性の評価

同じ呼吸レートでの複数測定:

"""
        for rate in duplicate_rates:
            trials = df_results[df_results['Rate'] == rate]
            lf_powers = trials['LF Power'].values
            mean_lf = lf_powers.mean()
            std_lf = lf_powers.std()
            cv = (std_lf / mean_lf * 100) if mean_lf > 0 else 0

            report += f"### {rate} bpm\n\n"
            report += "| Trial | LF Power | RMSSD | HR Mean |\n"
            report += "|------:|---------:|------:|--------:|\n"

            for _, trial in trials.iterrows():
                report += f"| {int(trial['Trial'])} | {trial['LF Power']:.2f} | "
                report += f"{trial['RMSSD']:.2f} | {trial['HR Mean']:.2f} |\n"

            report += f"\n- **LF Power平均**: {mean_lf:.2f} ms²\n"
            report += f"- **標準偏差**: {std_lf:.2f} ms²\n"
            report += f"- **変動係数 (CV)**: {cv:.2f}%\n\n"

            if cv > 20:
                report += "⚠️ 変動係数が20%を超えており、測定の再現性が低い可能性があります。\n\n"

    report += """
## 5. 推奨事項

### 次のステップ
"""

    # LF Powerの傾向から推奨を生成
    min_rate = df_results['Rate'].min()
    max_rate = df_results['Rate'].max()
    min_lf = df_results[df_results['Rate'] == min_rate]['LF Power'].values[0]
    max_lf = df_results[df_results['Rate'] == max_rate]['LF Power'].values[0]

    if optimal['Rate'] == min_rate and min_lf > max_lf:
        report += f"""
1. **さらに低い呼吸レートを測定**: 現在の最低レート（{min_rate} bpm）でLF Powerが最大ですが、
   さらに低いレート（例: {min_rate - 0.5} bpm, {min_rate - 1.0} bpm）でピークがある可能性があります。

2. **測定の再現性確認**: 同じ呼吸レートで複数回測定し、結果の安定性を確認してください。
"""
    elif optimal['Rate'] == max_rate and max_lf > min_lf:
        report += f"""
1. **さらに高い呼吸レートを測定**: 現在の最高レート（{max_rate} bpm）でLF Powerが最大ですが、
   さらに高いレート（例: {max_rate + 0.5} bpm, {max_rate + 1.0} bpm）でピークがある可能性があります。

2. **測定の再現性確認**: 同じ呼吸レートで複数回測定し、結果の安定性を確認してください。
"""
    else:
        report += f"""
1. **最適な呼吸レート周辺の詳細測定**: {optimal['Rate']} bpm付近（例: {optimal['Rate'] - 0.5} bpm, {optimal['Rate'] + 0.5} bpm）
   で追加測定を行い、ピークを精密に特定してください。

2. **測定の再現性確認**: 最適レート（{optimal['Rate']} bpm）で複数回測定し、結果の安定性を確認してください。
"""

    if has_order_effect:
        report += """
3. **測定プロトコルの改善**:
   - 測定順序をランダム化
   - 各測定間に2-3分の休憩
   - 測定前に10分間の安静時間
   - 同じ時間帯・環境で測定
"""

    report += """
## 参考文献

1. **Lehrer, P., & Gevirtz, R. (2014).**
   Heart rate variability biofeedback: how and why does it work?
   *Frontiers in psychology*, 5, 756.

2. **Steffen, P. R., et al. (2017).**
   A Practical Guide to Resonance Frequency Assessment for Heart Rate Variability Biofeedback.
   *Frontiers in Neuroscience*, 14, 570400.
   https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.570400/full
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Report saved: {output_path}")


def main():
    """メイン処理"""
    # 分析ディレクトリ
    analysis_dir = Path(__file__).parent
    data_dir = analysis_dir / 'data'
    img_dir = analysis_dir / 'img'

    # 出力パス
    output_img = img_dir / 'rf_hrv_analysis.png'
    output_report = analysis_dir / 'REPORT.md'

    # トライアル情報（ここを変更して使用）
    trials = [
        {'trial': 1, 'rate': 6.5, 'file': '2026-01-17 17-47-13.txt',
         'inhale': 3.1, 'exhale': 6.2},
        {'trial': 2, 'rate': 6.0, 'file': '2026-01-17 17-51-11.txt',
         'inhale': 3.3, 'exhale': 6.7},
        {'trial': 3, 'rate': 5.5, 'file': '2026-01-17 17-54-41.txt',
         'inhale': 3.6, 'exhale': 7.3},
        {'trial': 4, 'rate': 5.0, 'file': '2026-01-17 17-57-47.txt',
         'inhale': 4.0, 'exhale': 8.0},
        {'trial': 5, 'rate': 4.5, 'file': '2026-01-17 18-00-59.txt',
         'inhale': 4.4, 'exhale': 8.9},
        {'trial': 6, 'rate': 6.5, 'file': '2026-01-17 18-11-56.txt',
         'inhale': 3.1, 'exhale': 6.2},  # 再測定
    ]

    print("=" * 70)
    print("Resonance Frequency HRV Analysis")
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
