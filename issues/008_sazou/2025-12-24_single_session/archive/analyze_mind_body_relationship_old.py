"""
マインド-ボディ関係解析スクリプト
体軸の揺らぎと脳波・心拍数の関係を解析

使い方:
    python issues/008_sazou/analyze_mind_body_relationship.py

前提条件:
    - data/mindMonitor_*.csv が存在すること
    - issues/008_sazou/motion_intervals.csv が存在すること
      （事前に坐相解析を実行しておく必要があります）
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

# プロジェクトルートをPYTHONPATHに追加（SMR解析のため）
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ====================================
# 1. データ前処理
# ====================================

def remove_outliers(df, column='motion_score', z_threshold=3):
    """
    Z-scoreを用いた異常値除去

    generate_report.pyの前処理方法に準じて、統計的に極端な値を除去します。
    これにより装着時・取り外し時のノイズやアーティファクトを除外します。

    Parameters
    ----------
    df : pd.DataFrame
        入力データフレーム
    column : str
        異常値検出に使用する列名（デフォルト: 'motion_score'）
    z_threshold : float
        Z-scoreの閾値（デフォルト: 3.0）
        |Z| > thresholdの場合に異常値と判定

    Returns
    -------
    df_clean : pd.DataFrame
        異常値を除去したデータフレーム
    outlier_info : dict
        除去された異常値の情報
    """
    # Z-scoreを計算
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)

    # 異常値マスク
    outlier_mask = z_scores > z_threshold

    # 異常値情報を記録
    outliers = df[outlier_mask].copy()
    outlier_info = {
        'count': len(outliers),
        'percentage': len(outliers) / len(df) * 100,
        'outliers': outliers[['interval', column]].copy() if len(outliers) > 0 else pd.DataFrame()
    }

    # 異常値を除去
    df_clean = df[~outlier_mask].copy()

    return df_clean, outlier_info


# ====================================
# 2. データ統合
# ====================================

def integrate_motion_eeg_data(csv_path, motion_intervals_path, output_path,
                              output_path_fnirs=None, remove_outliers_flag=True):
    """
    モーションデータとEEG・心拍数・fNIRS・SMRデータを統合

    Parameters
    ----------
    csv_path : str
        Mind Monitor CSVファイルのパス
    motion_intervals_path : str
        モーション解析結果CSVのパス
    output_path : str
        統合データ(EEG+HR)の出力パス
    output_path_fnirs : str, optional
        統合データ(EEG+HR+fNIRS+SMR)の出力パス
    remove_outliers_flag : bool
        異常値除去を実施するかどうか（デフォルト: True）

    Returns
    -------
    combined : pd.DataFrame
        統合データフレーム（異常値除去済み）
    """
    print("=" * 80)
    print("データ統合と前処理")
    print("=" * 80)

    # データ読み込み
    df = pd.read_csv(csv_path)
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    # 10秒間隔でEEGデータを集計
    df['interval'] = df['TimeStamp'].dt.floor('10s')

    # 脳波バンド
    eeg_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # 各バンドの全チャネル平均を計算
    agg_dict = {}
    for band in eeg_bands:
        cols = [f"{band}_{ch}" for ch in channels]
        agg_dict[f'{band}_mean'] = df[cols].mean(axis=1)

    # 心拍数も追加
    agg_dict['Heart_Rate'] = df['Heart_Rate']

    # DataFrame作成
    df_temp = pd.DataFrame(agg_dict, index=df.index)
    df_temp['interval'] = df['interval']

    # 間隔でグループ化
    eeg_intervals = df_temp.groupby('interval').agg({
        'Delta_mean': 'mean',
        'Theta_mean': 'mean',
        'Alpha_mean': 'mean',
        'Beta_mean': 'mean',
        'Gamma_mean': 'mean',
        'Heart_Rate': 'mean'
    }).reset_index()

    # 既存のモーション解析データを読み込み
    motion_intervals = pd.read_csv(motion_intervals_path)
    motion_intervals['interval'] = pd.to_datetime(motion_intervals['interval'])

    # EEGとモーションデータを結合
    combined = pd.merge(
        motion_intervals,
        eeg_intervals,
        on='interval',
        how='inner'
    )

    # 保存（異常値除去前）
    combined.to_csv(output_path, index=False)

    print(f"\n統合データ保存(異常値除去前): {output_path}")
    print(f"インターバル数: {len(combined)}")

    # fNIRS・SMRデータの統合（指定された場合）
    if output_path_fnirs:
        # fNIRSデータの読み込みと解析
        fnirs_path = Path('issues/008_sazou/fnirs_intervals.csv')
        if fnirs_path.exists():
            fnirs_intervals = pd.read_csv(fnirs_path)
            fnirs_intervals['interval'] = pd.to_datetime(fnirs_intervals['interval'])

            # fNIRSデータを結合
            combined_fnirs = pd.merge(
                combined,
                fnirs_intervals,
                on='interval',
                how='inner'
            )

            # fNIRSデータも含めたDataFrameを使用
            combined = combined_fnirs
            print(f"fNIRS統合後インターバル数: {len(combined)}")
        else:
            print(f"警告: {fnirs_path} が見つかりません。fNIRS統合をスキップします。")

        # SMR解析: smr.pyライブラリを使用（mne依存）
        print("\nSMR (12-15Hz) 解析を実施中...")
        try:
            # sys.pathに追加済みなので、通常のimportを使用
            from lib.sensors.eeg.smr import calculate_smr

            # SMRデータを計算
            smr_result = calculate_smr(df, resample_interval='10S')
            smr_series = smr_result.time_series

            # SMRデータをDataFrameに変換（10秒間隔）
            smr_df = pd.DataFrame({
                'interval': smr_series.index,
                'smr_power': smr_series.values
            })
            smr_df['interval'] = pd.to_datetime(smr_df['interval'])

            # SMRデータを結合
            combined = pd.merge(
                combined,
                smr_df,
                on='interval',
                how='inner'
            )
            print(f"SMR統合後インターバル数: {len(combined)}")
        except ModuleNotFoundError as e:
            print(f"警告: SMR解析に必要なモジュールがありません: {e}")
            print("SMR統合をスキップします。mneモジュールのインストールが必要です。")
        except Exception as e:
            print(f"警告: SMR解析でエラーが発生しました: {e}")
            print("SMR統合をスキップします。")

        # 保存
        combined.to_csv(output_path_fnirs, index=False)
        print(f"\n統合データ保存(fNIRS+SMR含む, 異常値除去前): {output_path_fnirs}")

    # 異常値除去
    if remove_outliers_flag:
        print("\n異常値除去を実施...")
        combined_clean, outlier_info = remove_outliers(combined, column='motion_score', z_threshold=3)

        print(f"  除去された異常値: {outlier_info['count']}個 ({outlier_info['percentage']:.1f}%)")

        if outlier_info['count'] > 0:
            print("  異常値の詳細:")
            for _, row in outlier_info['outliers'].iterrows():
                # 経過時間を計算
                elapsed = (row['interval'] - combined['interval'].min()).total_seconds() / 60
                print(f"    - {elapsed:.1f}分: motion_score={row['motion_score']:.2f}")

        combined = combined_clean

    print(f"\n最終インターバル数: {len(combined)}")
    print(f"欠損値なし: {combined.isna().sum().sum() == 0}")
    print()

    return combined


# ====================================
# 3. 相関分析（EEG・心拍数）
# ====================================

def analyze_correlations(combined_df, output_path):
    """
    揺らぎと脳波・心拍数の相関分析

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム
    output_path : str
        相関結果の出力パス

    Returns
    -------
    corr_df : pd.DataFrame
        相関分析結果
    """
    print("=" * 80)
    print("相関分析")
    print("=" * 80)

    # 相関分析の対象変数
    motion_vars = ['motion_score', 'acc_magnitude_std', 'gyro_magnitude_mean']
    eeg_vars = ['Delta_mean', 'Theta_mean', 'Alpha_mean', 'Beta_mean', 'Gamma_mean']
    physio_vars = ['Heart_Rate']

    print("\n【Pearson相関係数】")
    print("(揺らぎスコアが高いほど、各脳波バンドがどう変化するか)\n")

    correlation_results = []

    for motion_var in motion_vars:
        print(f"\n{motion_var}との相関:")
        for eeg_var in eeg_vars:
            r, p = stats.pearsonr(combined_df[motion_var], combined_df[eeg_var])
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {eeg_var:12s}: r={r:+.3f}, p={p:.4f} {significance}")

            correlation_results.append({
                'motion_var': motion_var,
                'brain_var': eeg_var,
                'correlation': r,
                'p_value': p,
                'significant': p < 0.05
            })

    # Spearman順位相関
    print("\n\n【Spearman順位相関係数】")
    print("(順位での関係性 - 非線形関係も検出)\n")

    for motion_var in motion_vars:
        print(f"\n{motion_var}との相関:")
        for eeg_var in eeg_vars:
            rho, p = stats.spearmanr(combined_df[motion_var], combined_df[eeg_var])
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {eeg_var:12s}: ρ={rho:+.3f}, p={p:.4f} {significance}")

    # 心拍数との相関
    print("\n\n【揺らぎ指標と心拍数の相関】\n")

    for motion_var in motion_vars:
        r, p = stats.pearsonr(combined_df[motion_var], combined_df['Heart_Rate'])
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{motion_var:25s}: r={r:+.3f}, p={p:.4f} {significance}")

        correlation_results.append({
            'motion_var': motion_var,
            'brain_var': 'Heart_Rate',
            'correlation': r,
            'p_value': p,
            'significant': p < 0.05
        })

    # 相関結果をDataFrameに保存
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(output_path, index=False)

    print(f"\n相関結果保存: {output_path}")

    # 有意な相関のサマリー
    print("\n\n【有意な相関関係の要約】")
    significant_corr = corr_df[corr_df['significant']]
    if len(significant_corr) > 0:
        for _, row in significant_corr.iterrows():
            direction = "正の相関" if row['correlation'] > 0 else "負の相関"
            print(f"  {row['motion_var']} ⇔ {row['brain_var']}: {direction} (r={row['correlation']:+.3f}, p={row['p_value']:.4f})")
    else:
        print("  有意な相関は検出されませんでした(p < 0.05)")

    print()

    return corr_df


# ====================================
# 4. 相関分析（fNIRS）
# ====================================

def analyze_fnirs_correlations(combined_df, output_path):
    """
    揺らぎとfNIRS指標の相関分析

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム（fNIRS含む）
    output_path : str
        相関結果の出力パス

    Returns
    -------
    corr_df : pd.DataFrame
        相関分析結果
    """
    # fNIRS列が存在するか確認
    fnirs_vars = ['bilateral_hbo', 'bilateral_hbr', 'bilateral_hbt',
                  'left_hbo_outer', 'right_hbo_outer', 'left_hbd', 'right_hbd']

    if not all(var in combined_df.columns for var in fnirs_vars):
        print("fNIRS列が見つかりません。fNIRS相関分析をスキップします。")
        return None

    print("=" * 80)
    print("fNIRS相関分析")
    print("=" * 80)

    # 相関分析の対象変数
    motion_vars = ['motion_score', 'acc_magnitude_std', 'gyro_magnitude_mean']

    print("\n【Spearman順位相関係数】")
    print("(fNIRSデータは正規分布しないため、Spearman相関を使用)\n")

    correlation_results = []

    for motion_var in motion_vars:
        print(f"\n{motion_var}との相関:")
        for fnirs_var in fnirs_vars:
            rho, p = stats.spearmanr(combined_df[motion_var], combined_df[fnirs_var])
            significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            print(f"  {fnirs_var:18s}: ρ={rho:+.3f}, p={p:.4f} {significance}")

            correlation_results.append({
                'motion_var': motion_var,
                'fnirs_var': fnirs_var,
                'correlation': rho,
                'p_value': p,
                'significant': p < 0.05
            })

    # 相関結果をDataFrameに保存
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(output_path, index=False)

    print(f"\nfNIRS相関結果保存: {output_path}")

    # 有意な相関のサマリー
    print("\n【有意な相関関係の要約】")
    significant_corr = corr_df[corr_df['significant']]
    if len(significant_corr) > 0:
        for _, row in significant_corr.iterrows():
            direction = "正の相関" if row['correlation'] > 0 else "負の相関"
            print(f"  {row['motion_var']} ⇔ {row['fnirs_var']}: {direction} (ρ={row['correlation']:+.3f}, p={row['p_value']:.4f})")
    else:
        print("  有意な相関は検出されませんでした(p < 0.05)")

    print()

    return corr_df


# ====================================
# 4-2. 相関分析（SMR）
# ====================================

def analyze_smr_correlations(combined_df, output_path):
    """
    揺らぎとSMR（12-15Hz）の相関分析

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム（SMR含む）
    output_path : str
        相関結果の出力パス

    Returns
    -------
    corr_df : pd.DataFrame
        相関分析結果
    """
    # SMR列が存在するか確認
    if 'smr_power' not in combined_df.columns:
        print("SMR列が見つかりません。SMR相関分析をスキップします。")
        return None

    print("=" * 80)
    print("SMR相関分析")
    print("=" * 80)

    # 相関分析の対象変数
    motion_vars = ['motion_score', 'acc_magnitude_std', 'gyro_magnitude_mean']

    print("\n【Pearson相関係数】")
    print("(SMRは身体の静止・運動抑制と関連)\n")

    correlation_results = []

    for motion_var in motion_vars:
        r, p = stats.pearsonr(combined_df[motion_var], combined_df['smr_power'])
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{motion_var:25s}: r={r:+.3f}, p={p:.4f} {significance}")

        correlation_results.append({
            'motion_var': motion_var,
            'smr_var': 'smr_power',
            'correlation': r,
            'p_value': p,
            'significant': p < 0.05
        })

    # Spearman順位相関
    print("\n\n【Spearman順位相関係数】\n")

    for motion_var in motion_vars:
        rho, p = stats.spearmanr(combined_df[motion_var], combined_df['smr_power'])
        significance = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"{motion_var:25s}: ρ={rho:+.3f}, p={p:.4f} {significance}")

    # 相関結果をDataFrameに保存
    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(output_path, index=False)

    print(f"\nSMR相関結果保存: {output_path}")

    # 有意な相関のサマリー
    print("\n【有意な相関関係の要約】")
    significant_corr = corr_df[corr_df['significant']]
    if len(significant_corr) > 0:
        for _, row in significant_corr.iterrows():
            direction = "正の相関" if row['correlation'] > 0 else "負の相関"
            print(f"  {row['motion_var']} ⇔ SMR: {direction} (r={row['correlation']:+.3f}, p={row['p_value']:.4f})")
    else:
        print("  有意な相関は検出されませんでした(p < 0.05)")

    print()

    return corr_df


# ====================================
# 5. 高低揺らぎ期間の比較（EEG）
# ====================================

def compare_high_low_sway(combined_df, output_path):
    """
    高揺らぎ期間と低揺らぎ期間の脳波比較

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム
    output_path : str
        比較結果の出力パス

    Returns
    -------
    comparison_df : pd.DataFrame
        比較結果
    """
    print("=" * 80)
    print("高揺らぎ期間 vs 低揺らぎ期間の比較")
    print("=" * 80)

    # 揺らぎスコアの上位25%と下位25%で分類
    q75 = combined_df['motion_score'].quantile(0.75)
    q25 = combined_df['motion_score'].quantile(0.25)

    high_sway = combined_df[combined_df['motion_score'] >= q75]
    low_sway = combined_df[combined_df['motion_score'] <= q25]

    print(f"\n高揺らぎ期間(上位25%): 揺らぎスコア >= {q75:.4f} ({len(high_sway)}インターバル)")
    print(f"低揺らぎ期間(下位25%): 揺らぎスコア <= {q25:.4f} ({len(low_sway)}インターバル)\n")

    # 脳波バンドの比較
    eeg_vars = ['Delta_mean', 'Theta_mean', 'Alpha_mean', 'Beta_mean', 'Gamma_mean', 'Heart_Rate']

    print("【脳波バンドと心拍数の平均値比較】\n")
    print(f"{'変数':15s} {'低揺らぎ':>12s} {'高揺らぎ':>12s} {'差分':>12s} {'t値':>8s} {'p値':>10s}")
    print("-" * 80)

    comparison_results = []

    for var in eeg_vars:
        low_mean = low_sway[var].mean()
        high_mean = high_sway[var].mean()
        diff = high_mean - low_mean

        # t検定
        t_stat, p_val = stats.ttest_ind(high_sway[var], low_sway[var])
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        print(f"{var:15s} {low_mean:12.4f} {high_mean:12.4f} {diff:+12.4f} {t_stat:8.3f} {p_val:10.6f} {significance}")

        comparison_results.append({
            'variable': var,
            'low_sway_mean': low_mean,
            'high_sway_mean': high_mean,
            'difference': diff,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    # 効果量(Cohen's d)を計算
    print("\n\n【効果量 (Cohen's d)】")
    print("(0.2=小, 0.5=中, 0.8=大)\n")

    for var in eeg_vars:
        pooled_std = np.sqrt((high_sway[var].std()**2 + low_sway[var].std()**2) / 2)
        cohens_d = (high_sway[var].mean() - low_sway[var].mean()) / pooled_std
        effect_size = "大" if abs(cohens_d) >= 0.8 else "中" if abs(cohens_d) >= 0.5 else "小" if abs(cohens_d) >= 0.2 else "極小"
        print(f"{var:15s}: d = {cohens_d:+.3f} ({effect_size})")

    # 比較結果を保存
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_path, index=False)

    print(f"\n比較結果保存: {output_path}\n")

    return comparison_df


# ====================================
# 6. 高低揺らぎ期間の比較（fNIRS）
# ====================================

def compare_high_low_sway_fnirs(combined_df, output_path):
    """
    高揺らぎ期間と低揺らぎ期間のfNIRS比較

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム（fNIRS含む）
    output_path : str
        比較結果の出力パス

    Returns
    -------
    comparison_df : pd.DataFrame
        比較結果
    """
    # fNIRS列が存在するか確認
    fnirs_vars = ['bilateral_hbo', 'bilateral_hbr', 'bilateral_hbt',
                  'left_hbo_outer', 'right_hbo_outer', 'left_hbd', 'right_hbd']

    if not all(var in combined_df.columns for var in fnirs_vars):
        print("fNIRS列が見つかりません。fNIRS比較分析をスキップします。")
        return None

    print("=" * 80)
    print("高揺らぎ期間 vs 低揺らぎ期間の比較（fNIRS）")
    print("=" * 80)

    # 揺らぎスコアの上位25%と下位25%で分類
    q75 = combined_df['motion_score'].quantile(0.75)
    q25 = combined_df['motion_score'].quantile(0.25)

    high_sway = combined_df[combined_df['motion_score'] >= q75]
    low_sway = combined_df[combined_df['motion_score'] <= q25]

    print(f"\n高揺らぎ期間(上位25%): 揺らぎスコア >= {q75:.4f} ({len(high_sway)}インターバル)")
    print(f"低揺らぎ期間(下位25%): 揺らぎスコア <= {q25:.4f} ({len(low_sway)}インターバル)\n")

    print("【fNIRS指標の平均値比較】\n")
    print(f"{'変数':20s} {'低揺らぎ':>12s} {'高揺らぎ':>12s} {'差分':>12s} {'t値':>8s} {'p値':>10s}")
    print("-" * 80)

    comparison_results = []

    for var in fnirs_vars:
        low_mean = low_sway[var].mean()
        high_mean = high_sway[var].mean()
        diff = high_mean - low_mean

        # t検定
        t_stat, p_val = stats.ttest_ind(high_sway[var], low_sway[var])
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        print(f"{var:20s} {low_mean:12.4f} {high_mean:12.4f} {diff:+12.4f} {t_stat:8.3f} {p_val:10.6f} {significance}")

        comparison_results.append({
            'variable': var,
            'low_sway_mean': low_mean,
            'high_sway_mean': high_mean,
            'difference': diff,
            't_statistic': t_stat,
            'p_value': p_val,
            'significant': p_val < 0.05
        })

    # 効果量(Cohen's d)を計算
    print("\n\n【効果量 (Cohen's d)】")
    print("(0.2=小, 0.5=中, 0.8=大)\n")

    for var in fnirs_vars:
        pooled_std = np.sqrt((high_sway[var].std()**2 + low_sway[var].std()**2) / 2)
        cohens_d = (high_sway[var].mean() - low_sway[var].mean()) / pooled_std
        effect_size = "大" if abs(cohens_d) >= 0.8 else "中" if abs(cohens_d) >= 0.5 else "小" if abs(cohens_d) >= 0.2 else "極小"
        print(f"{var:20s}: d = {cohens_d:+.3f} ({effect_size})")

    # 比較結果を保存
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df.to_csv(output_path, index=False)

    print(f"\nfNIRS比較結果保存: {output_path}\n")

    return comparison_df


# ====================================
# 6-2. 高低揺らぎ期間の比較（SMR）
# ====================================

def compare_high_low_sway_smr(combined_df, output_path):
    """
    高揺らぎ期間と低揺らぎ期間のSMR比較

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム（SMR含む）
    output_path : str
        比較結果の出力パス

    Returns
    -------
    comparison_df : pd.DataFrame
        比較結果
    """
    # SMR列が存在するか確認
    if 'smr_power' not in combined_df.columns:
        print("SMR列が見つかりません。SMR比較分析をスキップします。")
        return None

    print("=" * 80)
    print("高揺らぎ期間 vs 低揺らぎ期間の比較（SMR）")
    print("=" * 80)

    # 揺らぎスコアの上位25%と下位25%で分類
    q75 = combined_df['motion_score'].quantile(0.75)
    q25 = combined_df['motion_score'].quantile(0.25)

    high_sway = combined_df[combined_df['motion_score'] >= q75]
    low_sway = combined_df[combined_df['motion_score'] <= q25]

    print(f"\n高揺らぎ期間(上位25%): 揺らぎスコア >= {q75:.4f} ({len(high_sway)}インターバル)")
    print(f"低揺らぎ期間(下位25%): 揺らぎスコア <= {q25:.4f} ({len(low_sway)}インターバル)\n")

    print("【SMR指標の平均値比較】\n")
    print(f"{'変数':20s} {'低揺らぎ':>12s} {'高揺らぎ':>12s} {'差分':>12s} {'t値':>8s} {'p値':>10s}")
    print("-" * 80)

    low_mean = low_sway['smr_power'].mean()
    high_mean = high_sway['smr_power'].mean()
    diff = high_mean - low_mean

    # t検定
    t_stat, p_val = stats.ttest_ind(high_sway['smr_power'], low_sway['smr_power'])
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

    print(f"{'smr_power':20s} {low_mean:12.4f} {high_mean:12.4f} {diff:+12.4f} {t_stat:8.3f} {p_val:10.6f} {significance}")

    comparison_result = {
        'variable': 'smr_power',
        'low_sway_mean': low_mean,
        'high_sway_mean': high_mean,
        'difference': diff,
        't_statistic': t_stat,
        'p_value': p_val,
        'significant': p_val < 0.05
    }

    # 効果量(Cohen's d)を計算
    print("\n\n【効果量 (Cohen's d)】")
    print("(0.2=小, 0.5=中, 0.8=大)\n")

    pooled_std = np.sqrt((high_sway['smr_power'].std()**2 + low_sway['smr_power'].std()**2) / 2)
    cohens_d = (high_sway['smr_power'].mean() - low_sway['smr_power'].mean()) / pooled_std
    effect_size = "大" if abs(cohens_d) >= 0.8 else "中" if abs(cohens_d) >= 0.5 else "小" if abs(cohens_d) >= 0.2 else "極小"
    print(f"{'smr_power':20s}: d = {cohens_d:+.3f} ({effect_size})")

    # 比較結果を保存
    comparison_df = pd.DataFrame([comparison_result])
    comparison_df.to_csv(output_path, index=False)

    print(f"\nSMR比較結果保存: {output_path}\n")

    return comparison_df


# ====================================
# 7. 時系列パターン分析
# ====================================

def analyze_time_series_patterns(combined_df):
    """
    時系列での揺らぎと脳波の変化を分析

    Parameters
    ----------
    combined_df : pd.DataFrame
        統合データフレーム
    """
    print("=" * 80)
    print("時系列での揺らぎと脳波の変化")
    print("=" * 80)

    # 前半・後半の比較
    mid_point = len(combined_df) // 2
    first_half = combined_df.iloc[:mid_point]
    second_half = combined_df.iloc[mid_point:]

    print(f"\n前半({mid_point}インターバル):")
    print(f"  揺らぎスコア: {first_half['motion_score'].mean():.4f}")
    print(f"  Delta波:      {first_half['Delta_mean'].mean():.4f}")
    print(f"  Theta波:      {first_half['Theta_mean'].mean():.4f}")
    print(f"  Alpha波:      {first_half['Alpha_mean'].mean():.4f}")
    print(f"  心拍数:       {first_half['Heart_Rate'].mean():.2f} bpm")

    print(f"\n後半({len(combined_df)-mid_point}インターバル):")
    print(f"  揺らぎスコア: {second_half['motion_score'].mean():.4f}")
    print(f"  Delta波:      {second_half['Delta_mean'].mean():.4f}")
    print(f"  Theta波:      {second_half['Theta_mean'].mean():.4f}")
    print(f"  Alpha波:      {second_half['Alpha_mean'].mean():.4f}")
    print(f"  心拍数:       {second_half['Heart_Rate'].mean():.2f} bpm")

    print(f"\n変化率:")
    print(f"  揺らぎスコア: {(second_half['motion_score'].mean() - first_half['motion_score'].mean()) / first_half['motion_score'].mean() * 100:+.1f}%")
    print(f"  Delta波:      {(second_half['Delta_mean'].mean() - first_half['Delta_mean'].mean()) / abs(first_half['Delta_mean'].mean()) * 100:+.1f}%")
    print(f"  Theta波:      {(second_half['Theta_mean'].mean() - first_half['Theta_mean'].mean()) / abs(first_half['Theta_mean'].mean()) * 100:+.1f}%")
    print(f"  Alpha波:      {(second_half['Alpha_mean'].mean() - first_half['Alpha_mean'].mean()) / first_half['Alpha_mean'].mean() * 100:+.1f}%")
    print()


# ====================================
# メイン実行
# ====================================

def main():
    """メイン実行関数"""

    # パス設定
    base_dir = Path(__file__).parent

    # データファイルのパスを指定
    csv_path = 'data/mindMonitor_2025-12-24--07-44-35_879452356987311069.csv'
    motion_intervals_path = base_dir / 'motion_intervals.csv'

    # 出力パス
    combined_output = base_dir / 'combined_motion_eeg_hr.csv'
    combined_output_fnirs = base_dir / 'combined_motion_eeg_hr_fnirs.csv'
    correlation_output = base_dir / 'correlation_results.csv'
    fnirs_correlation_output = base_dir / 'fnirs_correlation_results.csv'
    smr_correlation_output = base_dir / 'smr_correlation_results.csv'
    comparison_output = base_dir / 'high_low_sway_comparison.csv'
    fnirs_comparison_output = base_dir / 'fnirs_high_low_sway_comparison.csv'
    smr_comparison_output = base_dir / 'smr_high_low_sway_comparison.csv'

    # ファイル存在確認
    if not Path(csv_path).exists():
        print(f"エラー: {csv_path} が見つかりません")
        sys.exit(1)

    if not motion_intervals_path.exists():
        print(f"エラー: {motion_intervals_path} が見つかりません")
        print("先に坐相解析を実行してください")
        sys.exit(1)

    print("\n")
    print("=" * 80)
    print("マインド-ボディ関係解析スクリプト")
    print("=" * 80)
    print(f"\nデータファイル: {csv_path}")
    print(f"出力ディレクトリ: {base_dir}")
    print(f"異常値除去: 有効（Z-score閾値=3.0）\n")

    # 1. データ統合（異常値除去含む）
    combined_df = integrate_motion_eeg_data(
        csv_path,
        motion_intervals_path,
        combined_output,
        output_path_fnirs=combined_output_fnirs,
        remove_outliers_flag=True
    )

    # 2. EEG相関分析
    corr_df = analyze_correlations(combined_df, correlation_output)

    # 3. fNIRS相関分析（fNIRSデータがある場合）
    fnirs_corr_df = analyze_fnirs_correlations(combined_df, fnirs_correlation_output)

    # 4. SMR相関分析（SMRデータがある場合）
    smr_corr_df = analyze_smr_correlations(combined_df, smr_correlation_output)

    # 5. EEG高低揺らぎ比較
    comparison_df = compare_high_low_sway(combined_df, comparison_output)

    # 6. fNIRS高低揺らぎ比較（fNIRSデータがある場合）
    fnirs_comparison_df = compare_high_low_sway_fnirs(combined_df, fnirs_comparison_output)

    # 7. SMR高低揺らぎ比較（SMRデータがある場合）
    smr_comparison_df = compare_high_low_sway_smr(combined_df, smr_comparison_output)

    # 8. 時系列パターン分析
    analyze_time_series_patterns(combined_df)

    print("=" * 80)
    print("分析完了")
    print("=" * 80)
    print("\n生成されたファイル:")
    print(f"  - {combined_output}")
    if fnirs_corr_df is not None or smr_corr_df is not None:
        print(f"  - {combined_output_fnirs}")
    print(f"  - {correlation_output}")
    if fnirs_corr_df is not None:
        print(f"  - {fnirs_correlation_output}")
    if smr_corr_df is not None:
        print(f"  - {smr_correlation_output}")
    print(f"  - {comparison_output}")
    if fnirs_comparison_df is not None:
        print(f"  - {fnirs_comparison_output}")
    if smr_comparison_df is not None:
        print(f"  - {smr_comparison_output}")
    print()


if __name__ == '__main__':
    main()
