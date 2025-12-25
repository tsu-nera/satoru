#!/usr/bin/env python3
"""
Phase 3: 相対パワー・バンド比率による坐相分析

絶対パワー（dB）に加えて、相対パワー（%）とバンド比率を使用し、
坐相指標との相関を分析します。

Usage:
    python issues/008_sazou/relative_power/single_session/analyze_relative_power.py
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
        指標タイプ（'EEG', 'Relative Power', 'Band Ratio'）

    Returns:
    --------
    list
        相関分析結果のリスト
    """
    print(f"【{metric_type}】坐相指標 vs {metric_type}の相関係数（Pearson）:")
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
            print(f"  {eeg_var:20s}: r={r:+.3f}, p={p:.4f} {significance}")

            correlation_results.append({
                'posture_var': posture_var,
                'eeg_var': eeg_var,
                'metric_type': metric_type,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05
            })
        print()

    return correlation_results


def main():
    """メイン実行関数"""

    print("=" * 80)
    print("Phase 3: 相対パワー・バンド比率による坐相分析")
    print("=" * 80)
    print()

    # データファイルのパス
    csv_path = 'data/mindMonitor_2025-12-24--07-44-35_879452356987311069.csv'

    if not Path(csv_path).exists():
        print(f"エラー: {csv_path} が見つかりません")
        sys.exit(1)

    print(f"データファイル: {csv_path}")
    print(f"セッション: 2025-12-24（Phase 1と同じ）")
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

    # Statistical DataFrame生成（fNIRS・心拍数は除外）
    print("Statistical DataFrameを生成中...")
    print("  セグメント長: 3分")
    print("  ウォームアップ: なし")
    print("  fNIRS・心拍数: 除外（Phase 3の焦点は脳波と坐相の関係）")
    print()

    stat_df = create_statistical_dataframe(
        raw=raw,
        segment_minutes=3,
        warmup_minutes=0.0,
        session_start=df['TimeStamp'].iloc[0],
        fnirs_results=None,  # fNIRS除外
        hr_data=None,        # 心拍数除外
        df_timestamps=df['TimeStamp'],
        df=df,               # 坐相統計量計算のため
    )

    print(f"  生成されたデータフレーム:")
    for key, value in stat_df.items():
        if isinstance(value, pd.DataFrame):
            print(f"    - {key}: {len(value)} rows × {len(value.columns)} columns")
        elif isinstance(value, pd.Series):
            print(f"    - {key}: {len(value)} values")
    print()

    # 出力ディレクトリ
    output_dir = Path('issues/008_sazou/relative_power/single_session')

    # ===== 1. 絶対パワー（dB）- Phase 1との比較用 =====
    print("=" * 80)
    print("1. 絶対パワー（dB）の保存")
    print("=" * 80)
    print()

    band_powers_df = stat_df['band_powers']
    band_powers_path = output_dir / 'band_powers_absolute.csv'
    band_powers_df.to_csv(band_powers_path)
    print(f"絶対パワーを保存: {band_powers_path}")
    print()

    # ===== 2. 相対パワー（%）の計算 =====
    print("=" * 80)
    print("2. 相対パワー（%）の計算")
    print("=" * 80)
    print()

    relative_powers_df = calculate_relative_power(band_powers_df)
    relative_powers_path = output_dir / 'band_powers_relative.csv'
    relative_powers_df.to_csv(relative_powers_path)
    print(f"相対パワーを保存: {relative_powers_path}")
    print()
    print("相対パワー統計量:")
    print(relative_powers_df.describe())
    print()

    # ===== 3. バンド比率の取得 =====
    print("=" * 80)
    print("3. バンド比率の取得")
    print("=" * 80)
    print()

    band_ratios_df = stat_df['band_ratios']
    band_ratios_path = output_dir / 'band_ratios.csv'
    band_ratios_df.to_csv(band_ratios_path)
    print(f"バンド比率を保存: {band_ratios_path}")
    print()
    print("バンド比率の列:")
    for col in band_ratios_df.columns:
        print(f"  - {col}")
    print()

    # ===== 4. 坐相統計量の取得 =====
    print("=" * 80)
    print("4. 坐相統計量の取得")
    print("=" * 80)
    print()

    posture_df = stat_df['posture'].set_index('timestamp')
    posture_path = output_dir / 'posture_data.csv'
    posture_df.to_csv(posture_path)
    print(f"坐相統計量を保存: {posture_path}")
    print()
    print("坐相統計量の列:")
    for col in posture_df.columns:
        print(f"  - {col}")
    print()

    # ===== 5. 統合データフレームの作成 =====
    print("=" * 80)
    print("5. 統合データフレームの作成")
    print("=" * 80)
    print()

    # 全データを結合
    combined_df = band_powers_df.copy()
    combined_df = combined_df.join(relative_powers_df)
    combined_df = combined_df.join(band_ratios_df)
    combined_df = combined_df.join(posture_df, rsuffix='_posture')

    combined_path = output_dir / 'combined_data.csv'
    combined_df.to_csv(combined_path)
    print(f"統合データを保存: {combined_path}")
    print(f"  行数: {len(combined_df)}")
    print(f"  列数: {len(combined_df.columns)}")
    print()

    # ===== 6. 相関分析 =====
    print("=" * 80)
    print("6. 相関分析")
    print("=" * 80)
    print()

    # 坐相指標
    posture_vars = ['motion_index_mean', 'motion_index_max', 'gyro_rms',
                    'gyro_rms_corrected', 'pitch_angle', 'roll_angle', 'yaw_rms']

    all_correlation_results = []

    # 6-1. 絶対パワー（dB）
    print("--- 6-1. 絶対パワー（dB）との相関 ---")
    print()
    eeg_abs_vars = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    abs_results = analyze_correlations(combined_df, posture_vars, eeg_abs_vars, 'Absolute Power (dB)')
    all_correlation_results.extend(abs_results)

    # 6-2. 相対パワー（%）
    print("--- 6-2. 相対パワー（%）との相関 ---")
    print()
    eeg_rel_vars = ['Delta%', 'Theta%', 'Alpha%', 'Beta%', 'Gamma%']
    rel_results = analyze_correlations(combined_df, posture_vars, eeg_rel_vars, 'Relative Power (%)')
    all_correlation_results.extend(rel_results)

    # 6-3. バンド比率（実数）
    print("--- 6-3. バンド比率（実数）との相関 ---")
    print()
    ratio_vars = ['theta_alpha', 'beta_alpha', 'beta_theta', 'delta_beta', 'gamma_theta', 'low_high']
    ratio_results = analyze_correlations(combined_df, posture_vars, ratio_vars, 'Band Ratio')
    all_correlation_results.extend(ratio_results)

    # 相関結果を保存
    corr_path = output_dir / 'correlation_results.csv'
    if len(all_correlation_results) > 0:
        corr_df = pd.DataFrame(all_correlation_results)
        corr_df.to_csv(corr_path, index=False)
        print(f"全相関結果を保存: {corr_path}")
        print()

    # ===== 7. サマリー =====
    print("=" * 80)
    print("7. 有意な相関関係のサマリー（p < 0.05）")
    print("=" * 80)
    print()

    significant_corr = corr_df[corr_df['significant']].sort_values('p_value')

    if len(significant_corr) > 0:
        for metric_type in ['Absolute Power (dB)', 'Relative Power (%)', 'Band Ratio']:
            subset = significant_corr[significant_corr['metric_type'] == metric_type]
            if len(subset) == 0:
                continue

            print(f"【{metric_type}】")
            for _, row in subset.iterrows():
                direction = "正の相関" if row['correlation'] > 0 else "負の相関"
                significance = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*"
                print(f"  {row['posture_var']:20s} ⇔ {row['eeg_var']:20s}: {direction} (r={row['correlation']:+.3f}, p={row['p_value']:.4f}) {significance}")
            print()
    else:
        print("  有意な相関は検出されませんでした")
        print()

    # ===== 8. Phase 1との比較 =====
    print("=" * 80)
    print("8. Phase 1（絶対パワーのみ）との比較")
    print("=" * 80)
    print()

    # Yaw RMS ⇔ Betaの相関を抽出（Phase 1で最も強かった相関）
    yaw_beta_abs = corr_df[
        (corr_df['posture_var'] == 'yaw_rms') &
        (corr_df['eeg_var'] == 'Beta')
    ]

    yaw_beta_rel = corr_df[
        (corr_df['posture_var'] == 'yaw_rms') &
        (corr_df['eeg_var'] == 'Beta%')
    ]

    yaw_beta_ratio = corr_df[
        (corr_df['posture_var'] == 'yaw_rms') &
        (corr_df['eeg_var'] == 'beta_alpha')
    ]

    print("Yaw RMS ⇔ Beta指標の比較:")
    print()

    if len(yaw_beta_abs) > 0:
        row = yaw_beta_abs.iloc[0]
        print(f"  絶対パワー（Beta）:     r={row['correlation']:+.3f}, p={row['p_value']:.4f}")

    if len(yaw_beta_rel) > 0:
        row = yaw_beta_rel.iloc[0]
        print(f"  相対パワー（Beta%）:    r={row['correlation']:+.3f}, p={row['p_value']:.4f}")

    if len(yaw_beta_ratio) > 0:
        row = yaw_beta_ratio.iloc[0]
        print(f"  バンド比率（β/α比）:   r={row['correlation']:+.3f}, p={row['p_value']:.4f}")

    print()
    print("Phase 1の結果（参考）:")
    print("  絶対パワー（Beta）:     r=+0.930, p<0.001 ***")
    print()

    print("=" * 80)
    print("Phase 3: 単一セッション分析完了")
    print("=" * 80)
    print()
    print(f"出力ディレクトリ: {output_dir}")
    print()


if __name__ == '__main__':
    main()
