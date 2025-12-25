#!/usr/bin/env python3
"""
Phase 3 ステップ2: 複数セッションでの相対パワー・バンド比率分析

5つのセッションに対して、相対パワーとバンド比率を使った坐相分析を実行し、
絶対パワー（Phase 2）との再現性を比較します。

Usage:
    python issues/008_sazou/relative_power/multi_sessions/batch_analyze.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from lib import load_mind_monitor_csv, prepare_mne_raw
from lib.statistical_dataframe import create_statistical_dataframe


def calculate_relative_power(band_powers_df):
    """
    絶対パワー（dB）から相対パワー（%）を計算

    Parameters:
    -----------
    band_powers_df : pd.DataFrame
        'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma' 列を含む絶対パワーのDataFrame

    Returns:
    --------
    pd.DataFrame
        'Delta%', 'Theta%', 'Alpha%', 'Beta%', 'Gamma%' 列を含む相対パワーのDataFrame
    """
    # dB → 線形パワーに変換
    delta_power = 10 ** (band_powers_df['Delta'] / 10)
    theta_power = 10 ** (band_powers_df['Theta'] / 10)
    alpha_power = 10 ** (band_powers_df['Alpha'] / 10)
    beta_power = 10 ** (band_powers_df['Beta'] / 10)
    gamma_power = 10 ** (band_powers_df['Gamma'] / 10)

    # 総パワー
    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power

    # 相対パワー（%）
    relative_powers = pd.DataFrame({
        'Delta%': (delta_power / total_power) * 100,
        'Theta%': (theta_power / total_power) * 100,
        'Alpha%': (alpha_power / total_power) * 100,
        'Beta%': (beta_power / total_power) * 100,
        'Gamma%': (gamma_power / total_power) * 100,
    }, index=band_powers_df.index)

    return relative_powers


def analyze_correlations(combined_df, posture_vars, eeg_vars, metric_type='EEG'):
    """
    坐相指標とEEG指標の相関分析

    Parameters:
    -----------
    combined_df : pd.DataFrame
        統合データフレーム
    posture_vars : list
        坐相指標のリスト
    eeg_vars : list
        EEG指標のリスト
    metric_type : str
        指標タイプ（'Absolute Power', 'Relative Power', 'Band Ratio'）

    Returns:
    --------
    list
        相関分析結果のリスト
    """
    correlation_results = []

    for posture_var in posture_vars:
        if posture_var not in combined_df.columns:
            continue

        for eeg_var in eeg_vars:
            if eeg_var not in combined_df.columns:
                continue

            # 欠損値を除外
            mask = combined_df[[posture_var, eeg_var]].notna().all(axis=1)
            data_clean = combined_df[mask]

            if len(data_clean) < 3:
                continue

            r, p = stats.pearsonr(data_clean[posture_var], data_clean[eeg_var])

            correlation_results.append({
                'posture_var': posture_var,
                'eeg_var': eeg_var,
                'metric_type': metric_type,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05,
                'n_samples': len(data_clean)
            })

    return correlation_results


def analyze_single_session(csv_path: str, output_dir: Path) -> dict:
    """
    単一セッションの解析を実行

    Args:
        csv_path: Mind Monitor CSVファイルのパス
        output_dir: 結果の出力ディレクトリ

    Returns:
        dict: 解析結果のサマリー
    """
    print(f"\n{'='*80}")
    print(f"セッション解析: {Path(csv_path).name}")
    print(f"{'='*80}\n")

    # 出力ディレクトリを作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ読み込み
    print("データを読み込んでいます...")
    df = load_mind_monitor_csv(csv_path)
    print(f"  サンプル数: {len(df)}")
    duration_minutes = (df['TimeStamp'].iloc[-1] - df['TimeStamp'].iloc[0]).total_seconds() / 60
    print(f"  セッション時間: {duration_minutes:.1f}分")

    # MNE Rawオブジェクトを準備
    print("\nMNE Rawオブジェクトを準備中...")
    raw_data = prepare_mne_raw(df)
    raw = raw_data['raw']

    # Statistical DataFrame生成（fNIRS・心拍数は除外）
    print("\nStatistical DataFrameを生成中...")
    print("  セグメント長: 3分")
    print("  fNIRS・心拍数: 除外")

    stat_df = create_statistical_dataframe(
        raw=raw,
        segment_minutes=3,
        warmup_minutes=0.0,
        session_start=df['TimeStamp'].iloc[0],
        fnirs_results=None,
        hr_data=None,
        df_timestamps=df['TimeStamp'],
        df=df,
    )

    n_segments = len(stat_df['band_powers'])
    print(f"  生成されたセグメント数: {n_segments}")

    # 絶対パワー
    band_powers_df = stat_df['band_powers']
    band_powers_df.to_csv(output_dir / 'band_powers_absolute.csv')

    # 相対パワー
    relative_powers_df = calculate_relative_power(band_powers_df)
    relative_powers_df.to_csv(output_dir / 'band_powers_relative.csv')

    # バンド比率
    band_ratios_df = stat_df['band_ratios']
    band_ratios_df.to_csv(output_dir / 'band_ratios.csv')

    # 坐相統計量
    posture_df = stat_df['posture'].set_index('timestamp')
    posture_df.to_csv(output_dir / 'posture_data.csv')

    # 統合データフレームを作成
    combined_df = band_powers_df.copy()
    combined_df = combined_df.join(relative_powers_df)
    combined_df = combined_df.join(band_ratios_df)
    combined_df = combined_df.join(posture_df, rsuffix='_posture')
    combined_df.to_csv(output_dir / 'combined_data.csv')

    print("\n相関分析を実行中...")

    # 坐相指標
    posture_vars = ['motion_index_mean', 'motion_index_max', 'gyro_rms',
                    'gyro_rms_corrected', 'pitch_angle', 'roll_angle', 'yaw_rms']

    all_correlation_results = []

    # 1. 絶対パワー（dB）
    eeg_abs_vars = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    abs_results = analyze_correlations(combined_df, posture_vars, eeg_abs_vars, 'Absolute Power')
    all_correlation_results.extend(abs_results)

    # 2. 相対パワー（%）
    eeg_rel_vars = ['Delta%', 'Theta%', 'Alpha%', 'Beta%', 'Gamma%']
    rel_results = analyze_correlations(combined_df, posture_vars, eeg_rel_vars, 'Relative Power')
    all_correlation_results.extend(rel_results)

    # 3. バンド比率
    ratio_vars = ['theta_alpha', 'beta_alpha', 'beta_theta', 'delta_beta', 'gamma_theta', 'low_high']
    ratio_results = analyze_correlations(combined_df, posture_vars, ratio_vars, 'Band Ratio')
    all_correlation_results.extend(ratio_results)

    # 相関結果を保存
    if len(all_correlation_results) > 0:
        corr_df = pd.DataFrame(all_correlation_results)
        corr_df.to_csv(output_dir / 'correlation_results.csv', index=False)

        # 有意な相関のカウント
        significant_by_type = {}
        for metric_type in ['Absolute Power', 'Relative Power', 'Band Ratio']:
            subset = corr_df[corr_df['metric_type'] == metric_type]
            n_significant = sum(subset['significant'])
            significant_by_type[metric_type] = n_significant

        print(f"  絶対パワー: {significant_by_type['Absolute Power']}件の有意な相関")
        print(f"  相対パワー: {significant_by_type['Relative Power']}件の有意な相関")
        print(f"  バンド比率: {significant_by_type['Band Ratio']}件の有意な相関")

    # セッションサマリーを作成
    summary = {
        'csv_path': str(csv_path),
        'n_samples': len(df),
        'duration_minutes': duration_minutes,
        'n_segments': n_segments,
        'n_correlations': len(all_correlation_results),
        'n_significant_absolute': significant_by_type['Absolute Power'],
        'n_significant_relative': significant_by_type['Relative Power'],
        'n_significant_ratio': significant_by_type['Band Ratio'],
    }

    print(f"\nセッション解析完了: {output_dir}")

    return summary


def main():
    """メイン実行関数"""
    print("="*80)
    print("Phase 3 ステップ2: 複数セッション分析")
    print("="*80)
    print()

    # 解析対象のセッション（日付順）
    sessions = [
        'data/mindMonitor_2025-12-11--07-36-21_1223728811535012372.csv',
        'data/mindMonitor_2025-12-19--07-38-20_2788494757623296620.csv',
        'data/mindMonitor_2025-12-22--07-34-33_9161536009876231252.csv',
        'data/mindMonitor_2025-12-24--07-44-35_879452356987311069.csv',
        'data/mindMonitor_2025-12-25--07-44-46_823597931075241326.csv',
    ]

    # 基本出力ディレクトリ
    base_output_dir = Path('issues/008_sazou/relative_power/multi_sessions/sessions')
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 各セッションを解析
    all_summaries = []

    for i, csv_path in enumerate(sessions, 1):
        if not Path(csv_path).exists():
            print(f"警告: {csv_path} が見つかりません。スキップします。\n")
            continue

        # 日付を抽出
        session_date = Path(csv_path).name.split('_')[1]

        # セッションごとの出力ディレクトリ
        session_output_dir = base_output_dir / session_date

        print(f"\n[{i}/{len(sessions)}] {session_date}")

        try:
            summary = analyze_single_session(csv_path, session_output_dir)
            summary['session_date'] = session_date
            all_summaries.append(summary)
        except Exception as e:
            print(f"エラー: セッション {session_date} の解析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 全体サマリーを保存
    if len(all_summaries) > 0:
        summary_df = pd.DataFrame(all_summaries)
        summary_csv_path = base_output_dir.parent / 'all_sessions_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"\n全セッションサマリーを保存: {summary_csv_path}")

        print("\n" + "="*80)
        print("全セッション解析完了")
        print("="*80)
        print(f"\n解析したセッション数: {len(all_summaries)}/{len(sessions)}")
        print(f"\n各セッションの結果: {base_output_dir}/")
        print(f"全体サマリー: {summary_csv_path}")
    else:
        print("\n警告: 解析できたセッションがありませんでした。")


if __name__ == '__main__':
    main()
