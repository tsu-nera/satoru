#!/usr/bin/env python3
"""
Phase 3 ステップ2: クロスセッション統計分析

複数セッションの相関分析結果を統合し、
絶対パワー、相対パワー、バンド比率の再現性を評価します。

Usage:
    python issues/008_sazou/relative_power/multi_sessions/cross_session_stats.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def load_session_correlations(session_dir: Path) -> pd.DataFrame:
    """
    セッションディレクトリから相関結果を読み込む

    Args:
        session_dir: セッションディレクトリのパス

    Returns:
        pd.DataFrame: 相関結果
    """
    corr_path = session_dir / 'correlation_results.csv'
    if corr_path.exists():
        df = pd.read_csv(corr_path)
        df['session_date'] = session_dir.name
        return df
    return None


def calculate_reproducibility(all_correlations: pd.DataFrame, posture_var: str,
                               eeg_var: str, metric_type: str) -> dict:
    """
    特定の相関の再現性を計算

    Args:
        all_correlations: 全セッションの相関結果
        posture_var: 坐相変数名
        eeg_var: EEG変数名
        metric_type: 指標タイプ（'Absolute Power', 'Relative Power', 'Band Ratio'）

    Returns:
        dict: 再現性統計量
    """
    # 該当する相関を抽出
    mask = (
        (all_correlations['posture_var'] == posture_var) &
        (all_correlations['eeg_var'] == eeg_var) &
        (all_correlations['metric_type'] == metric_type)
    )
    subset = all_correlations[mask]

    if len(subset) == 0:
        return None

    n_sessions = len(subset)
    n_significant = sum(subset['significant'])
    reproducibility = n_significant / n_sessions if n_sessions > 0 else 0

    correlations = subset['correlation'].values
    p_values = subset['p_value'].values

    mean_r = np.mean(correlations)
    std_r = np.std(correlations, ddof=1) if n_sessions > 1 else 0
    median_r = np.median(correlations)

    # 信頼区間
    if n_sessions > 1:
        se = std_r / np.sqrt(n_sessions)
        df_t = n_sessions - 1
        t_critical = stats.t.ppf(0.975, df_t)
        ci_lower = mean_r - t_critical * se
        ci_upper = mean_r + t_critical * se
    else:
        ci_lower = mean_r
        ci_upper = mean_r

    return {
        'posture_var': posture_var,
        'eeg_var': eeg_var,
        'metric_type': metric_type,
        'n_sessions': n_sessions,
        'n_significant': n_significant,
        'reproducibility': reproducibility,
        'mean_r': mean_r,
        'median_r': median_r,
        'std_r': std_r,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'correlations': correlations.tolist(),
        'p_values': p_values.tolist(),
    }


def main():
    """メイン実行関数"""
    print("="*80)
    print("Phase 3 ステップ2: クロスセッション統計分析")
    print("="*80)
    print()

    # セッションディレクトリ
    sessions_dir = Path('issues/008_sazou/relative_power/multi_sessions/sessions')

    if not sessions_dir.exists():
        print(f"エラー: {sessions_dir} が見つかりません")
        sys.exit(1)

    # 各セッションの相関結果を読み込む
    print("セッション相関結果を読み込んでいます...")
    all_session_dirs = sorted([d for d in sessions_dir.iterdir() if d.is_dir()])

    all_correlations_list = []
    for session_dir in all_session_dirs:
        print(f"  {session_dir.name}")
        corr_df = load_session_correlations(session_dir)
        if corr_df is not None:
            all_correlations_list.append(corr_df)

    if len(all_correlations_list) == 0:
        print("エラー: 相関結果が見つかりませんでした")
        sys.exit(1)

    # 全相関結果を統合
    all_correlations = pd.concat(all_correlations_list, ignore_index=True)
    print(f"\n読み込んだセッション数: {len(all_correlations_list)}")
    print(f"総相関数: {len(all_correlations)}")
    print()

    # 出力ディレクトリ
    output_dir = Path('issues/008_sazou/relative_power/multi_sessions')

    # ===== 主要な仮説の検証 =====
    print("="*80)
    print("主要な仮説の再現性分析")
    print("="*80)
    print()

    # Yaw RMS ⇔ Beta関連指標の再現性
    print("【Yaw RMS ⇔ Beta関連指標の再現性】")
    print()

    hypotheses = [
        ('yaw_rms', 'Beta', 'Absolute Power', 'Phase 1/2の基準'),
        ('yaw_rms', 'Beta%', 'Relative Power', '相対パワー'),
        ('yaw_rms', 'beta_alpha', 'Band Ratio', 'β/α比（覚醒度）'),
    ]

    yaw_beta_results = []
    for posture_var, eeg_var, metric_type, description in hypotheses:
        result = calculate_reproducibility(all_correlations, posture_var, eeg_var, metric_type)
        if result:
            yaw_beta_results.append(result)
            print(f"{description}（{eeg_var}）:")
            print(f"  再現性: {result['reproducibility']*100:.0f}% ({result['n_significant']}/{result['n_sessions']}セッション)")
            print(f"  平均相関: r={result['mean_r']:+.3f} (SD={result['std_r']:.3f})")
            print(f"  95%CI: [{result['ci_lower']:+.3f}, {result['ci_upper']:+.3f}]")
            print()

    # Yaw RMS ⇔ その他の有望な指標
    print("\n【Yaw RMS ⇔ その他の有望な指標の再現性】")
    print()

    other_hypotheses = [
        ('yaw_rms', 'theta_alpha', 'Band Ratio', 'θ/α比（瞑想深度）'),
        ('yaw_rms', 'low_high', 'Band Ratio', '低周波/高周波比'),
        ('yaw_rms', 'Theta%', 'Relative Power', 'Theta相対パワー'),
        ('yaw_rms', 'Delta%', 'Relative Power', 'Delta相対パワー'),
        ('yaw_rms', 'Alpha%', 'Relative Power', 'Alpha相対パワー'),
        ('yaw_rms', 'Delta', 'Absolute Power', 'Delta絶対パワー'),
        ('yaw_rms', 'Theta', 'Absolute Power', 'Theta絶対パワー'),
    ]

    other_results = []
    for posture_var, eeg_var, metric_type, description in other_hypotheses:
        result = calculate_reproducibility(all_correlations, posture_var, eeg_var, metric_type)
        if result:
            other_results.append(result)
            print(f"{description}（{eeg_var}）:")
            print(f"  再現性: {result['reproducibility']*100:.0f}% ({result['n_significant']}/{result['n_sessions']}セッション)")
            print(f"  平均相関: r={result['mean_r']:+.3f} (SD={result['std_r']:.3f})")
            print()

    # ===== 全体的な再現性サマリー =====
    print("\n" + "="*80)
    print("指標タイプ別の再現性サマリー")
    print("="*80)
    print()

    # ユニークな相関を特定
    unique_correlations = all_correlations.groupby(
        ['posture_var', 'eeg_var', 'metric_type']
    ).size().reset_index(name='count')

    summary_by_type = []
    for metric_type in ['Absolute Power', 'Relative Power', 'Band Ratio']:
        subset = unique_correlations[unique_correlations['metric_type'] == metric_type]

        # 各相関の再現性を計算
        reproducibilities = []
        for _, row in subset.iterrows():
            result = calculate_reproducibility(
                all_correlations,
                row['posture_var'],
                row['eeg_var'],
                metric_type
            )
            if result:
                reproducibilities.append(result['reproducibility'])

        if len(reproducibilities) > 0:
            mean_repro = np.mean(reproducibilities)
            median_repro = np.median(reproducibilities)
            max_repro = np.max(reproducibilities)

            # 60%以上の再現性を持つ相関の数
            high_repro_count = sum(1 for r in reproducibilities if r >= 0.6)

            summary_by_type.append({
                'metric_type': metric_type,
                'n_unique_correlations': len(reproducibilities),
                'mean_reproducibility': mean_repro,
                'median_reproducibility': median_repro,
                'max_reproducibility': max_repro,
                'n_high_reproducibility': high_repro_count,
            })

    summary_df = pd.DataFrame(summary_by_type)
    print(summary_df.to_string(index=False))
    print()

    # サマリーを保存
    summary_path = output_dir / 'reproducibility_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"再現性サマリーを保存: {summary_path}")
    print()

    # ===== 詳細な再現性結果を保存 =====
    all_results = yaw_beta_results + other_results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        # correlationsとp_valuesを除いて保存
        results_df_simple = results_df.drop(columns=['correlations', 'p_values'])
        results_path = output_dir / 'key_hypotheses_reproducibility.csv'
        results_df_simple.to_csv(results_path, index=False)
        print(f"主要仮説の再現性結果を保存: {results_path}")
        print()

    # ===== Phase 2との比較 =====
    print("\n" + "="*80)
    print("Phase 2（絶対パワーのみ）との比較")
    print("="*80)
    print()

    print("Phase 2の結果（参考）:")
    print("  Yaw RMS ⇔ Beta（絶対パワー）: 再現性40% (2/5セッション)")
    print()

    print("Phase 3の結果:")
    for result in yaw_beta_results:
        if result['eeg_var'] == 'Beta':
            print(f"  Yaw RMS ⇔ Beta（絶対パワー）: 再現性{result['reproducibility']*100:.0f}% ({result['n_significant']}/{result['n_sessions']}セッション)")
        elif result['eeg_var'] == 'beta_alpha':
            print(f"  Yaw RMS ⇔ β/α比（バンド比率）: 再現性{result['reproducibility']*100:.0f}% ({result['n_significant']}/{result['n_sessions']}セッション)")
    print()

    # 改善度を計算
    phase2_repro = 0.4  # 40%
    beta_result = next((r for r in yaw_beta_results if r['eeg_var'] == 'Beta'), None)
    ratio_result = next((r for r in yaw_beta_results if r['eeg_var'] == 'beta_alpha'), None)

    if beta_result:
        print(f"絶対パワーの再現性: {phase2_repro*100:.0f}% → {beta_result['reproducibility']*100:.0f}% （変化なし）")

    if ratio_result:
        improvement = (ratio_result['reproducibility'] - phase2_repro) / phase2_repro * 100
        print(f"β/α比の改善度: +{improvement:.0f}% （{phase2_repro*100:.0f}% → {ratio_result['reproducibility']*100:.0f}%）")
    print()

    print("="*80)
    print("クロスセッション統計分析完了")
    print("="*80)


if __name__ == '__main__':
    main()
