#!/usr/bin/env python3
"""
複数セッション坐相分析バッチ処理スクリプト

複数のMind Monitor CSVファイルに対して坐相分析を実行し、
各セッションの結果を個別のディレクトリに保存します。

Usage:
    python issues/008_sazou/multi_sessions/batch_analyze_posture.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    get_heart_rate_data,
    get_optics_data,
    analyze_fnirs,
)
from lib.statistical_dataframe import create_statistical_dataframe


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
    print(f"  チャンネル数: {len(raw.ch_names)}")
    print(f"  サンプリングレート: {raw.info['sfreq']} Hz")

    # fNIRS解析
    fnirs_results = None
    has_fnirs = False
    try:
        print("\nfNIRS解析を実行中...")
        optics_data = get_optics_data(df)
        if optics_data and len(optics_data['time']) > 0:
            fnirs_results = analyze_fnirs(optics_data)
            left_hbo_mean = np.nanmean(fnirs_results['left_hbo'])
            right_hbo_mean = np.nanmean(fnirs_results['right_hbo'])
            print(f"  左半球 HbO平均: {left_hbo_mean:.2f}")
            print(f"  右半球 HbO平均: {right_hbo_mean:.2f}")
            has_fnirs = True
    except Exception as e:
        print(f"  警告: fNIRS解析でエラーが発生しました: {e}")
        print("  fNIRS解析をスキップします。")

    # 心拍数解析
    print("\n心拍数解析を実行中...")
    hr_data = get_heart_rate_data(df)
    hr_mean = None
    if 'heart_rate' in hr_data and len(hr_data['heart_rate']) > 0:
        hr_mean = np.nanmean(hr_data['heart_rate'])
        print(f"  平均心拍数: {hr_mean:.1f} bpm")

    # Statistical DataFrame生成
    print("\nStatistical DataFrameを生成中...")
    print("  セグメント長: 3分")
    print("  ウォームアップ: なし")

    stat_df = create_statistical_dataframe(
        raw=raw,
        segment_minutes=3,
        warmup_minutes=0.0,
        session_start=df['TimeStamp'].iloc[0],
        fnirs_results=fnirs_results,
        hr_data=hr_data,
        df_timestamps=df['TimeStamp'],
        df=df,
    )

    n_segments = len(stat_df['band_powers'])
    print(f"  生成されたセグメント数: {n_segments}")

    # データを保存
    stat_df['band_powers'].to_csv(output_dir / 'band_powers.csv')

    if 'fnirs' in stat_df and stat_df['fnirs'] is not None:
        stat_df['fnirs'].to_csv(output_dir / 'fnirs_data.csv')

    if 'hr' in stat_df and stat_df['hr'] is not None:
        stat_df['hr'].to_csv(output_dir / 'hr_data.csv')

    if 'posture' in stat_df and stat_df['posture'] is not None:
        stat_df['posture'].to_csv(output_dir / 'posture_data.csv')

    # 統合データフレームを作成
    combined_df = stat_df['band_powers'].copy()

    if 'fnirs' in stat_df and stat_df['fnirs'] is not None:
        fnirs_df = stat_df['fnirs']
        combined_df = combined_df.join(fnirs_df, rsuffix='_fnirs')

    if 'hr' in stat_df and stat_df['hr'] is not None:
        hr_df = stat_df['hr']
        combined_df = combined_df.join(hr_df, rsuffix='_hr')

    if 'posture' in stat_df and stat_df['posture'] is not None:
        posture_df = stat_df['posture'].set_index('timestamp')
        combined_df = combined_df.join(posture_df, rsuffix='_posture')

    combined_df.to_csv(output_dir / 'combined_data.csv')

    # 相関分析
    print(f"\n{'='*80}")
    print("相関分析")
    print(f"{'='*80}\n")

    posture_vars = ['motion_index_mean', 'motion_index_max', 'gyro_rms',
                    'gyro_rms_corrected', 'pitch_angle', 'roll_angle', 'yaw_rms']
    eeg_vars = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    # EEG相関
    print("【EEG指標】坐相指標 vs EEG指標の相関係数（Pearson）:\n")

    correlation_results = []

    for posture_var in posture_vars:
        if posture_var not in combined_df.columns:
            continue

        print(f"{posture_var}:")
        for eeg_var in eeg_vars:
            if eeg_var not in combined_df.columns:
                continue

            mask = combined_df[[posture_var, eeg_var]].notna().all(axis=1)
            data_clean = combined_df[mask]

            if len(data_clean) < 3:
                continue

            r, p = stats.pearsonr(data_clean[posture_var], data_clean[eeg_var])
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {eeg_var:10s}: r={r:+.3f}, p={p:.4f} {significance}")

            correlation_results.append({
                'posture_var': posture_var,
                'eeg_var': eeg_var,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05,
                'n_samples': len(data_clean)
            })
        print()

    # 相関結果を保存
    if len(correlation_results) > 0:
        corr_df = pd.DataFrame(correlation_results)
        corr_df.to_csv(output_dir / 'correlation_results.csv', index=False)

        significant_corr = corr_df[corr_df['significant']].sort_values('p_value')
        print("\n有意な相関関係（p < 0.05）:")
        if len(significant_corr) > 0:
            for _, row in significant_corr.iterrows():
                direction = "正の相関" if row['correlation'] > 0 else "負の相関"
                print(f"  {row['posture_var']:20s} ⇔ {row['eeg_var']:10s}: {direction} (r={row['correlation']:+.3f}, p={row['p_value']:.4f})")
        else:
            print("  有意な相関は検出されませんでした")

    # fNIRS相関分析
    fnirs_correlation_results = []
    if has_fnirs and 'fnirs' in stat_df and stat_df['fnirs'] is not None:
        print("\n\n【fNIRS指標】坐相指標 vs fNIRS指標の相関係数（Spearman）:")
        print("(fNIRSデータは正規分布しないため、Spearman相関を使用)\n")

        fnirs_vars = [col for col in combined_df.columns if 'hbo' in col or 'hbr' in col or 'hbt' in col or 'hbd' in col]

        for posture_var in posture_vars:
            if posture_var not in combined_df.columns:
                continue

            print(f"{posture_var}:")
            for fnirs_var in fnirs_vars:
                if fnirs_var not in combined_df.columns:
                    continue

                mask = combined_df[[posture_var, fnirs_var]].notna().all(axis=1)
                data_clean = combined_df[mask]

                if len(data_clean) < 3:
                    continue

                rho, p = stats.spearmanr(data_clean[posture_var], data_clean[fnirs_var])
                significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                print(f"  {fnirs_var:18s}: ρ={rho:+.3f}, p={p:.4f} {significance}")

                fnirs_correlation_results.append({
                    'posture_var': posture_var,
                    'fnirs_var': fnirs_var,
                    'correlation': rho,
                    'p_value': p,
                    'significant': p < 0.05,
                    'n_samples': len(data_clean)
                })
            print()

        if len(fnirs_correlation_results) > 0:
            fnirs_corr_df = pd.DataFrame(fnirs_correlation_results)
            fnirs_corr_df.to_csv(output_dir / 'fnirs_correlation_results.csv', index=False)

            significant_fnirs = fnirs_corr_df[fnirs_corr_df['significant']].sort_values('p_value')
            print("\n有意な相関関係（p < 0.05）:")
            if len(significant_fnirs) > 0:
                for _, row in significant_fnirs.iterrows():
                    direction = "正の相関" if row['correlation'] > 0 else "負の相関"
                    print(f"  {row['posture_var']:20s} ⇔ {row['fnirs_var']:18s}: {direction} (ρ={row['correlation']:+.3f}, p={row['p_value']:.4f})")
            else:
                print("  有意な相関は検出されませんでした")

    # セッションサマリーを作成
    summary = {
        'csv_path': str(csv_path),
        'n_samples': len(df),
        'duration_minutes': duration_minutes,
        'n_segments': n_segments,
        'has_fnirs': has_fnirs,
        'hr_mean': hr_mean,
        'n_eeg_correlations': len(correlation_results),
        'n_eeg_significant': sum(1 for r in correlation_results if r['significant']),
        'n_fnirs_correlations': len(fnirs_correlation_results),
        'n_fnirs_significant': sum(1 for r in fnirs_correlation_results if r['significant']),
    }

    # サマリーを保存
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write(f"セッション解析サマリー\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"CSVファイル: {summary['csv_path']}\n")
        f.write(f"サンプル数: {summary['n_samples']}\n")
        f.write(f"セッション時間: {summary['duration_minutes']:.1f}分\n")
        f.write(f"セグメント数: {summary['n_segments']}\n")
        f.write(f"fNIRSデータ: {'あり' if summary['has_fnirs'] else 'なし'}\n")
        if summary['hr_mean']:
            f.write(f"平均心拍数: {summary['hr_mean']:.1f} bpm\n")
        f.write(f"\nEEG相関分析:\n")
        f.write(f"  総相関数: {summary['n_eeg_correlations']}\n")
        f.write(f"  有意な相関: {summary['n_eeg_significant']}\n")
        if summary['has_fnirs']:
            f.write(f"\nfNIRS相関分析:\n")
            f.write(f"  総相関数: {summary['n_fnirs_correlations']}\n")
            f.write(f"  有意な相関: {summary['n_fnirs_significant']}\n")

    print(f"\n{'='*80}")
    print(f"セッション解析完了: {output_dir}")
    print(f"{'='*80}\n")

    return summary


def main():
    """メイン実行関数"""
    print("="*80)
    print("複数セッション坐相分析バッチ処理")
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
    base_output_dir = Path('issues/008_sazou/multi_sessions/sessions')
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # 各セッションを解析
    all_summaries = []

    for i, csv_path in enumerate(sessions, 1):
        if not Path(csv_path).exists():
            print(f"警告: {csv_path} が見つかりません。スキップします。\n")
            continue

        # 日付を抽出（ファイル名からYYYY-MM-DD形式を取得）
        session_date = Path(csv_path).name.split('_')[1]  # mindMonitor_2025-12-11--... から 2025-12-11 を抽出

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
