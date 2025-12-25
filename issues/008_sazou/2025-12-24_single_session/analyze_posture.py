#!/usr/bin/env python3
"""
坐相とほかのパラメータの分析スクリプト

レポートライブラリを使用して、坐相（IMU）とEEG、fNIRS、心拍数の関係を分析します。

Usage:
    python issues/008_sazou/analyze_posture.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from lib import (
    load_mind_monitor_csv,
    prepare_mne_raw,
    get_heart_rate_data,
    get_optics_data,
    analyze_fnirs,
)
from lib.statistical_dataframe import create_statistical_dataframe


def main():
    """メイン実行関数"""

    print("=" * 80)
    print("坐相とほかのパラメータの分析")
    print("=" * 80)
    print()

    # データファイルのパス
    csv_path = 'data/mindMonitor_2025-12-24--07-44-35_879452356987311069.csv'

    if not Path(csv_path).exists():
        print(f"エラー: {csv_path} が見つかりません")
        sys.exit(1)

    print(f"データファイル: {csv_path}")
    print()

    # データ読み込み
    print("データを読み込んでいます...")
    df = load_mind_monitor_csv(csv_path)
    print(f"  サンプル数: {len(df)}")
    print()

    # MNE Rawオブジェクトを準備
    print("MNE Rawオブジェクトを準備中...")
    raw_data = prepare_mne_raw(df)
    raw = raw_data['raw']
    print(f"  チャンネル数: {len(raw.ch_names)}")
    print(f"  サンプリングレート: {raw.info['sfreq']} Hz")
    print()

    # fNIRS解析（Optics列からデータを抽出）
    fnirs_results = None
    try:
        print("fNIRS解析を実行中...")
        optics_data = get_optics_data(df)
        if optics_data and len(optics_data['time']) > 0:
            fnirs_results = analyze_fnirs(optics_data)
            print(f"  左半球 HbO平均: {np.nanmean(fnirs_results['left_hbo']):.2f}")
            print(f"  右半球 HbO平均: {np.nanmean(fnirs_results['right_hbo']):.2f}")
        print()
    except Exception as e:
        print(f"  警告: fNIRS解析でエラーが発生しました: {e}")
        print("  fNIRS解析をスキップします。")
        print()

    # 心拍数解析
    print("心拍数解析を実行中...")
    hr_data = get_heart_rate_data(df)
    if 'heart_rate' in hr_data and len(hr_data['heart_rate']) > 0:
        hr_mean = np.nanmean(hr_data['heart_rate'])
        print(f"  平均心拍数: {hr_mean:.1f} bpm")
    print()

    # Statistical DataFrame生成（IMU統計量を含む）
    print("Statistical DataFrameを生成中...")
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
        df=df,  # IMU統計量計算のため元データを渡す
    )

    print(f"  生成されたデータフレーム:")
    for key, value in stat_df.items():
        if isinstance(value, pd.DataFrame):
            print(f"    - {key}: {len(value)} rows × {len(value.columns)} columns")
        elif isinstance(value, pd.Series):
            print(f"    - {key}: {len(value)} values")
    print()

    # 各データフレームを保存
    output_dir = Path('issues/008_sazou')

    # バンドパワー
    band_powers_path = output_dir / 'band_powers.csv'
    stat_df['band_powers'].to_csv(band_powers_path)
    print(f"バンドパワーを保存: {band_powers_path}")

    # fNIRS
    if 'fnirs' in stat_df and stat_df['fnirs'] is not None:
        fnirs_path = output_dir / 'fnirs_data.csv'
        stat_df['fnirs'].to_csv(fnirs_path)
        print(f"fNIRSデータを保存: {fnirs_path}")

    # 心拍数
    if 'hr' in stat_df and stat_df['hr'] is not None:
        hr_path = output_dir / 'hr_data.csv'
        stat_df['hr'].to_csv(hr_path)
        print(f"心拍数データを保存: {hr_path}")

    # 坐相統計量
    if 'posture' in stat_df and stat_df['posture'] is not None:
        posture_path = output_dir / 'posture_data.csv'
        stat_df['posture'].to_csv(posture_path)
        print(f"坐相統計量を保存: {posture_path}")

        # 坐相データの内容を確認
        print()
        print("坐相統計量の列:")
        for col in stat_df['posture'].columns:
            print(f"  - {col}")

    print()

    # 統合データフレームを作成
    print("統合データフレームを作成中...")
    combined_df = stat_df['band_powers'].copy()

    # fNIRSデータを結合
    if 'fnirs' in stat_df and stat_df['fnirs'] is not None:
        fnirs_df = stat_df['fnirs']
        combined_df = combined_df.join(fnirs_df, rsuffix='_fnirs')
        print(f"  fNIRSデータを結合: {len(fnirs_df.columns)} columns")

    # 心拍数データを結合
    if 'hr' in stat_df and stat_df['hr'] is not None:
        hr_df = stat_df['hr']
        combined_df = combined_df.join(hr_df, rsuffix='_hr')
        print(f"  心拍数データを結合: {len(hr_df.columns)} columns")

    # 坐相データを結合
    if 'posture' in stat_df and stat_df['posture'] is not None:
        posture_df = stat_df['posture'].set_index('timestamp')
        combined_df = combined_df.join(posture_df, rsuffix='_posture')
        print(f"  坐相データを結合: {len(posture_df.columns)} columns")

    # 統合データを保存
    combined_path = output_dir / 'combined_data.csv'
    combined_df.to_csv(combined_path)
    print(f"統合データを保存: {combined_path}")
    print(f"  行数: {len(combined_df)}")
    print(f"  列数: {len(combined_df.columns)}")
    print()

    # 相関分析
    print("=" * 80)
    print("相関分析")
    print("=" * 80)
    print()

    # 坐相指標とEEG指標の相関
    posture_vars = ['motion_index_mean', 'motion_index_max', 'gyro_rms',
                    'gyro_rms_corrected', 'pitch_angle', 'roll_angle', 'yaw_rms']
    eeg_vars = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

    print("【EEG指標】坐相指標 vs EEG指標の相関係数（Pearson）:")
    print()

    correlation_results = []

    for posture_var in posture_vars:
        if posture_var not in combined_df.columns:
            continue

        print(f"{posture_var}:")
        for eeg_var in eeg_vars:
            if eeg_var not in combined_df.columns:
                continue

            # 欠損値を除外
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
                'significant': p < 0.05
            })
        print()

    # 相関結果を保存
    corr_path = output_dir / 'correlation_results.csv'
    if len(correlation_results) > 0:
        corr_df = pd.DataFrame(correlation_results)
        corr_df.to_csv(corr_path, index=False)
        print(f"EEG相関結果を保存: {corr_path}")
        print()

        # 有意な相関のサマリー
        print("有意な相関関係（p < 0.05）:")
        significant_corr = corr_df[corr_df['significant']].sort_values('p_value')
        if len(significant_corr) > 0:
            for _, row in significant_corr.iterrows():
                direction = "正の相関" if row['correlation'] > 0 else "負の相関"
                print(f"  {row['posture_var']:20s} ⇔ {row['eeg_var']:10s}: {direction} (r={row['correlation']:+.3f}, p={row['p_value']:.4f})")
        else:
            print("  有意な相関は検出されませんでした")
    else:
        print("相関分析の結果がありません（データ数不足の可能性）")
    print()

    # fNIRS相関分析
    if 'fnirs' in stat_df and stat_df['fnirs'] is not None:
        print()
        print("【fNIRS指標】坐相指標 vs fNIRS指標の相関係数（Spearman）:")
        print("(fNIRSデータは正規分布しないため、Spearman相関を使用)")
        print()

        fnirs_correlation_results = []
        fnirs_vars = [col for col in combined_df.columns if 'hbo' in col or 'hbr' in col or 'hbt' in col or 'hbd' in col]

        for posture_var in posture_vars:
            if posture_var not in combined_df.columns:
                continue

            print(f"{posture_var}:")
            for fnirs_var in fnirs_vars:
                if fnirs_var not in combined_df.columns:
                    continue

                # 欠損値を除外
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
                    'significant': p < 0.05
                })
            print()

        # fNIRS相関結果を保存
        fnirs_corr_path = output_dir / 'fnirs_correlation_results.csv'
        if len(fnirs_correlation_results) > 0:
            fnirs_corr_df = pd.DataFrame(fnirs_correlation_results)
            fnirs_corr_df.to_csv(fnirs_corr_path, index=False)
            print(f"fNIRS相関結果を保存: {fnirs_corr_path}")
            print()

            # 有意な相関のサマリー
            print("有意な相関関係（p < 0.05）:")
            significant_fnirs = fnirs_corr_df[fnirs_corr_df['significant']].sort_values('p_value')
            if len(significant_fnirs) > 0:
                for _, row in significant_fnirs.iterrows():
                    direction = "正の相関" if row['correlation'] > 0 else "負の相関"
                    print(f"  {row['posture_var']:20s} ⇔ {row['fnirs_var']:18s}: {direction} (ρ={row['correlation']:+.3f}, p={row['p_value']:.4f})")
            else:
                print("  有意な相関は検出されませんでした")
            print()

    print("=" * 80)
    print("分析完了")
    print("=" * 80)


if __name__ == '__main__':
    main()
