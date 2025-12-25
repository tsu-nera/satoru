#!/usr/bin/env python3
"""
クロスセッション統計分析スクリプト

複数セッションの相関分析結果を統合し、仮説の再現性を評価します。

Usage:
    python issues/008_sazou/multi_sessions/cross_session_stats.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


def load_session_correlations(session_dir: Path) -> dict:
    """
    セッションディレクトリから相関結果を読み込む

    Args:
        session_dir: セッションディレクトリのパス

    Returns:
        dict: EEGとfNIRSの相関結果
    """
    result = {
        'session_date': session_dir.name,
        'eeg_corr': None,
        'fnirs_corr': None,
    }

    # EEG相関結果
    eeg_corr_path = session_dir / 'correlation_results.csv'
    if eeg_corr_path.exists():
        result['eeg_corr'] = pd.read_csv(eeg_corr_path)

    # fNIRS相関結果
    fnirs_corr_path = session_dir / 'fnirs_correlation_results.csv'
    if fnirs_corr_path.exists():
        result['fnirs_corr'] = pd.read_csv(fnirs_corr_path)

    return result


def calculate_meta_correlation(correlation_dfs: list, posture_var: str, target_var: str) -> dict:
    """
    複数セッションの相関係数からメタ分析統計量を計算

    Args:
        correlation_dfs: 各セッションの相関結果DataFrameのリスト
        posture_var: 坐相変数名
        target_var: ターゲット変数名（EEGまたはfNIRS）

    Returns:
        dict: メタ分析結果
    """
    correlations = []
    p_values = []
    n_samples_list = []
    sessions_with_data = []

    for i, df in enumerate(correlation_dfs):
        if df is None:
            continue

        # posture_varとtarget_varの組み合わせを検索
        if 'eeg_var' in df.columns:
            mask = (df['posture_var'] == posture_var) & (df['eeg_var'] == target_var)
        elif 'fnirs_var' in df.columns:
            mask = (df['posture_var'] == posture_var) & (df['fnirs_var'] == target_var)
        else:
            continue

        matched = df[mask]
        if len(matched) > 0:
            row = matched.iloc[0]
            correlations.append(row['correlation'])
            p_values.append(row['p_value'])
            n_samples_list.append(row.get('n_samples', 10))
            sessions_with_data.append(i)

    if len(correlations) == 0:
        return None

    correlations = np.array(correlations)
    p_values = np.array(p_values)

    # メタ分析統計量を計算
    mean_r = np.mean(correlations)
    std_r = np.std(correlations, ddof=1) if len(correlations) > 1 else 0
    median_r = np.median(correlations)

    # 有意なセッション数
    n_significant = np.sum(np.array(p_values) < 0.05)

    # 信頼区間（t分布を使用）
    if len(correlations) > 1:
        se = std_r / np.sqrt(len(correlations))
        df_t = len(correlations) - 1
        t_critical = stats.t.ppf(0.975, df_t)
        ci_lower = mean_r - t_critical * se
        ci_upper = mean_r + t_critical * se
    else:
        ci_lower = mean_r
        ci_upper = mean_r

    # Fisher's z変換を用いた統合p値計算
    z_scores = []
    for r, n in zip(correlations, n_samples_list):
        if abs(r) < 1.0:  # r=1または-1の場合を避ける
            z = 0.5 * np.log((1 + r) / (1 - r))  # Fisher's z変換
            z_scores.append(z)

    if len(z_scores) > 0:
        mean_z = np.mean(z_scores)
        se_z = 1 / np.sqrt(np.sum([n - 3 for n in n_samples_list]))  # 標準誤差
        z_stat = mean_z / se_z
        combined_p = 2 * (1 - stats.norm.cdf(abs(z_stat)))  # 両側検定
    else:
        combined_p = 1.0

    return {
        'posture_var': posture_var,
        'target_var': target_var,
        'n_sessions': len(correlations),
        'n_significant': n_significant,
        'mean_r': mean_r,
        'median_r': median_r,
        'std_r': std_r,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'combined_p': combined_p,
        'reproducibility': n_significant / len(correlations) if len(correlations) > 0 else 0,
        'correlations': correlations.tolist(),
        'p_values': p_values.tolist(),
    }


def analyze_hypothesis(all_sessions: list, hypothesis_name: str, posture_var: str,
                        target_var: str, expected_direction: str, data_type: str) -> dict:
    """
    特定の仮説について分析

    Args:
        all_sessions: 全セッションデータのリスト
        hypothesis_name: 仮説名
        posture_var: 坐相変数名
        target_var: ターゲット変数名
        expected_direction: 期待される相関方向（'positive' or 'negative'）
        data_type: データタイプ（'eeg' or 'fnirs'）

    Returns:
        dict: 仮説検証結果
    """
    # 相関データフレームを抽出
    if data_type == 'eeg':
        corr_dfs = [s['eeg_corr'] for s in all_sessions]
    elif data_type == 'fnirs':
        corr_dfs = [s['fnirs_corr'] for s in all_sessions]
    else:
        return None

    # メタ分析を実行
    meta_result = calculate_meta_correlation(corr_dfs, posture_var, target_var)

    if meta_result is None:
        return {
            'hypothesis_name': hypothesis_name,
            'status': 'データなし',
            'details': 'この仮説を検証するためのデータが見つかりませんでした。',
        }

    # 仮説の採択/棄却を判断
    is_significant = meta_result['combined_p'] < 0.05
    is_correct_direction = (
        (expected_direction == 'negative' and meta_result['mean_r'] < 0) or
        (expected_direction == 'positive' and meta_result['mean_r'] > 0)
    )

    # 再現性評価
    reproducibility = meta_result['reproducibility']
    if reproducibility >= 0.8:
        reproducibility_level = '高い'
    elif reproducibility >= 0.6:
        reproducibility_level = '中程度'
    else:
        reproducibility_level = '低い'

    # ステータス判定
    if is_significant and is_correct_direction and reproducibility >= 0.6:
        status = '✓ 支持'
    elif is_significant and is_correct_direction:
        status = '△ 部分的に支持'
    else:
        status = '✗ 棄却'

    return {
        'hypothesis_name': hypothesis_name,
        'status': status,
        'posture_var': posture_var,
        'target_var': target_var,
        'expected_direction': expected_direction,
        'n_sessions': meta_result['n_sessions'],
        'n_significant': meta_result['n_significant'],
        'reproducibility_rate': f"{reproducibility * 100:.0f}%",
        'reproducibility_level': reproducibility_level,
        'mean_r': meta_result['mean_r'],
        'median_r': meta_result['median_r'],
        'std_r': meta_result['std_r'],
        'ci_95': f"[{meta_result['ci_lower']:.3f}, {meta_result['ci_upper']:.3f}]",
        'combined_p': meta_result['combined_p'],
        'is_significant': is_significant,
        'is_correct_direction': is_correct_direction,
        'correlations': meta_result['correlations'],
        'p_values': meta_result['p_values'],
    }


def main():
    """メイン実行関数"""
    print("=" * 80)
    print("クロスセッション統計分析")
    print("=" * 80)
    print()

    # セッションディレクトリ
    sessions_dir = Path('issues/008_sazou/multi_sessions/sessions')

    # 各セッションの相関結果を読み込み
    all_sessions = []
    session_dates = []

    for session_dir in sorted(sessions_dir.iterdir()):
        if session_dir.is_dir():
            session_data = load_session_correlations(session_dir)
            all_sessions.append(session_data)
            session_dates.append(session_dir.name)

    print(f"読み込んだセッション数: {len(all_sessions)}")
    print(f"セッション日付: {', '.join(session_dates)}")
    print()

    # Phase 1で発見された主要な仮説を検証
    hypotheses = [
        {
            'name': '仮説1: Yaw RMS ⇔ HbO（脳血流）の負の相関',
            'posture_var': 'yaw_rms',
            'target_var': 'hbo_mean',
            'expected_direction': 'negative',
            'data_type': 'fnirs',
            'phase1_r': -0.915,
        },
        {
            'name': '仮説2: Yaw RMS ⇔ Beta波（緊張）の正の相関',
            'posture_var': 'yaw_rms',
            'target_var': 'Beta',
            'expected_direction': 'positive',
            'data_type': 'eeg',
            'phase1_r': +0.930,
        },
        {
            'name': '仮説3a: Gyro RMS ⇔ HbO（脳血流）の負の相関（ジャイロ系）',
            'posture_var': 'gyro_rms',
            'target_var': 'hbo_mean',
            'expected_direction': 'negative',
            'data_type': 'fnirs',
            'phase1_r': -0.79,
        },
        {
            'name': '仮説3b: Motion Index ⇔ HbR（脱酸素Hb）の正の相関（加速度系）',
            'posture_var': 'motion_index_mean',
            'target_var': 'hbr_mean',
            'expected_direction': 'positive',
            'data_type': 'fnirs',
            'phase1_r': +0.79,
        },
        {
            'name': '仮説4a: Yaw RMS ⇔ Delta波の正の相関',
            'posture_var': 'yaw_rms',
            'target_var': 'Delta',
            'expected_direction': 'positive',
            'data_type': 'eeg',
            'phase1_r': +0.850,
        },
        {
            'name': '仮説4b: Yaw RMS ⇔ Theta波の正の相関',
            'posture_var': 'yaw_rms',
            'target_var': 'Theta',
            'expected_direction': 'positive',
            'data_type': 'eeg',
            'phase1_r': +0.831,
        },
        {
            'name': '仮説4c: Yaw RMS ⇔ Alpha波の負の相関',
            'posture_var': 'yaw_rms',
            'target_var': 'Alpha',
            'expected_direction': 'negative',
            'data_type': 'eeg',
            'phase1_r': -0.672,
        },
    ]

    print("=" * 80)
    print("仮説検証結果")
    print("=" * 80)
    print()

    results = []

    for hyp in hypotheses:
        result = analyze_hypothesis(
            all_sessions=all_sessions,
            hypothesis_name=hyp['name'],
            posture_var=hyp['posture_var'],
            target_var=hyp['target_var'],
            expected_direction=hyp['expected_direction'],
            data_type=hyp['data_type'],
        )

        if result:
            result['phase1_r'] = hyp.get('phase1_r', None)
            results.append(result)

            # 結果を表示
            print(f"【{result['hypothesis_name']}】")
            print(f"ステータス: {result['status']}")
            if 'details' in result:
                print(f"  {result['details']}")
            else:
                print(f"  Phase 1の相関係数: r={result['phase1_r']:+.3f}")
                print(f"  セッション数: {result['n_sessions']}")
                print(f"  有意なセッション数: {result['n_significant']}/{result['n_sessions']}")
                print(f"  再現性: {result['reproducibility_rate']} ({result['reproducibility_level']})")
                print(f"  平均相関係数: r={result['mean_r']:+.3f}")
                print(f"  中央値: r={result['median_r']:+.3f}")
                print(f"  標準偏差: {result['std_r']:.3f}")
                print(f"  95%信頼区間: {result['ci_95']}")
                print(f"  統合p値: {result['combined_p']:.4f}")
                print(f"  各セッションの相関係数: {[f'{r:+.3f}' for r in result['correlations']]}")
            print()

    # 結果をCSVに保存
    output_dir = Path('issues/008_sazou/multi_sessions')

    if len(results) > 0:
        results_df = pd.DataFrame([
            {
                'hypothesis': r['hypothesis_name'],
                'status': r['status'],
                'posture_var': r.get('posture_var', ''),
                'target_var': r.get('target_var', ''),
                'phase1_r': r.get('phase1_r', None),
                'n_sessions': r.get('n_sessions', 0),
                'n_significant': r.get('n_significant', 0),
                'reproducibility': r.get('reproducibility_rate', ''),
                'mean_r': r.get('mean_r', None),
                'median_r': r.get('median_r', None),
                'std_r': r.get('std_r', None),
                'ci_95': r.get('ci_95', ''),
                'combined_p': r.get('combined_p', None),
            }
            for r in results
        ])

        results_csv_path = output_dir / 'hypothesis_validation_results.csv'
        results_df.to_csv(results_csv_path, index=False)
        print(f"仮説検証結果を保存: {results_csv_path}")

    # サマリー統計
    print()
    print("=" * 80)
    print("サマリー")
    print("=" * 80)
    print()

    supported = sum(1 for r in results if r['status'] == '✓ 支持')
    partial = sum(1 for r in results if r['status'] == '△ 部分的に支持')
    rejected = sum(1 for r in results if r['status'] == '✗ 棄却')

    print(f"検証した仮説数: {len(results)}")
    print(f"  ✓ 支持: {supported}")
    print(f"  △ 部分的に支持: {partial}")
    print(f"  ✗ 棄却: {rejected}")
    print()

    # 再現性が高い仮説
    high_reproducibility = [r for r in results if r.get('reproducibility_level') == '高い']
    if len(high_reproducibility) > 0:
        print("再現性が高い仮説（再現率≥80%）:")
        for r in high_reproducibility:
            print(f"  • {r['hypothesis_name']}")
            print(f"    {r['posture_var']} ⇔ {r['target_var']}: r={r['mean_r']:+.3f}, 再現率={r['reproducibility_rate']}")
        print()

    print("=" * 80)
    print("クロスセッション統計分析完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
