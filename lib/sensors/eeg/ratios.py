"""
EEGバンド比率計算モジュール（リラックス度、集中度、瞑想深度）
"""

import numpy as np
import pandas as pd
from scipy import stats

from .constants import FREQ_BANDS


def calculate_band_ratios(df, resample_interval='10S', bands=None):
    """
    バンド比率を計算（リラックス度、集中度、瞑想深度）

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム（バンドパワー列を含む）
    resample_interval : str
        リサンプリング間隔（例: '10S'）
    bands : dict, optional
        バンド定義辞書（デフォルト: FREQ_BANDS）

    Returns
    -------
    ratios_dict : dict
        {
            'ratios': pd.DataFrame,  # 比率の時系列データ（リサンプル済み）
            'statistics': pd.DataFrame,  # 統計サマリー
            'spike_analysis': pd.DataFrame  # スパイク分析
        }
    """
    if bands is None:
        bands = FREQ_BANDS

    ratio_df = pd.DataFrame({'TimeStamp': df['TimeStamp']})

    # 比率定義（英語名、日本語表示名、分子、分母）
    ratio_configs = [
        ('Alpha/Beta', 'リラックス度 (α/β)', 'Alpha', 'Beta'),
        ('Beta/Theta', '集中度 (β/θ)', 'Beta', 'Theta'),
        ('Theta/Alpha', '瞑想深度 (θ/α)', 'Theta', 'Alpha'),
    ]

    # 各比率を計算
    for ratio_name, display_name, num_band, den_band in ratio_configs:
        ratios = []
        for col in [c for c in df.columns if c.startswith(f'{num_band}_')]:
            electrode = col.split('_', 1)[1]
            den_col = f'{den_band}_{electrode}'
            if den_col in df.columns:
                num_vals = pd.to_numeric(df[col], errors='coerce').abs()
                den_vals = pd.to_numeric(df[den_col], errors='coerce').abs()
                den_vals = den_vals.replace(0, np.nan)

                ratio = (num_vals / den_vals).replace([np.inf, -np.inf], np.nan)
                ratios.append(ratio)

        if ratios:
            ratio_df[ratio_name] = pd.concat(ratios, axis=1).mean(axis=1)

    # リサンプリング
    ratio_df_resampled = ratio_df.set_index('TimeStamp').resample(resample_interval).mean().reset_index()

    # 統計サマリー（外れ値除去版）→縦長形式（Metric/Value/Unit）
    summary_scores = []
    z_threshold = 3.0

    for ratio_name, display_name, _, _ in ratio_configs:
        if ratio_name in ratio_df.columns:
            values = ratio_df[ratio_name].dropna()

            if len(values) > 0:
                # Z-scoreによる外れ値除去
                z_scores = np.abs(stats.zscore(values))
                filtered_values = values[z_scores < z_threshold]
                n_outliers = len(values) - len(filtered_values)

                # 外れ値除去後の統計量を計算
                if len(filtered_values) > 0:
                    q1 = filtered_values.quantile(0.25)
                    q3 = filtered_values.quantile(0.75)
                    iqr = q3 - q1

                    # 縦長形式で各統計量を追加
                    summary_scores.extend([
                        {'Metric': f'{ratio_name} Mean', 'Value': filtered_values.mean(), 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Median', 'Value': filtered_values.median(), 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Std Dev', 'Value': filtered_values.std(), 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Min', 'Value': filtered_values.min(), 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Max', 'Value': filtered_values.max(), 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} IQR', 'Value': iqr, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Outliers', 'Value': n_outliers, 'Unit': 'count', 'DisplayName': display_name},
                    ])
                else:
                    # 全て外れ値の場合
                    summary_scores.extend([
                        {'Metric': f'{ratio_name} Mean', 'Value': np.nan, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Median', 'Value': np.nan, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Std Dev', 'Value': np.nan, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Min', 'Value': np.nan, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Max', 'Value': np.nan, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} IQR', 'Value': np.nan, 'Unit': 'ratio', 'DisplayName': display_name},
                        {'Metric': f'{ratio_name} Outliers', 'Value': n_outliers, 'Unit': 'count', 'DisplayName': display_name},
                    ])

    # スパイク分析
    spike_info = []
    for ratio_name, display_name, _, _ in ratio_configs:
        if ratio_name in ratio_df_resampled.columns:
            values = ratio_df_resampled[ratio_name].dropna()

            if len(values) > 0:
                z_scores = np.abs(stats.zscore(values))
                outliers = values[z_scores > 3]
                cv = (values.std() / values.mean()) * 100

                spike_info.append({
                    'Metric': ratio_name,
                    'Outlier Count': len(outliers),
                    'Outlier Ratio (%)': len(outliers) / len(values) * 100,
                    'CV (%)': cv,
                    'DisplayName': display_name
                })

    return {
        'ratios': ratio_df_resampled,
        'statistics': pd.DataFrame(summary_scores),
        'spike_analysis': pd.DataFrame(spike_info)
    }
