"""
テンプレート用データフォーマッタ

データオブジェクトをテンプレートで使用可能なDataFrame形式に変換する。
プレゼンテーション層のデータ変換ロジックを集約。
"""

from typing import Any

import pandas as pd


def format_respiratory_stats(respiration_result: Any) -> pd.DataFrame:
    """
    RespirationResultをテーブル用DataFrameに変換

    Parameters
    ----------
    respiration_result : RespirationResult
        呼吸分析結果オブジェクト

    Returns
    -------
    pd.DataFrame
        Metric/Value/Unitの3カラムを持つDataFrame

    Examples
    --------
    >>> df = format_respiratory_stats(respiration_result)
    >>> print(df.columns.tolist())
    ['Metric', 'Value', 'Unit']
    """
    stats = []

    # 主推定値（通常はスペクトル法）を先頭に置く
    stats.append({
        'Metric': 'Mean Breathing Rate',
        'Value': respiration_result.breathing_rate,
        'Unit': 'bpm'
    })

    # RP (Respiratory Period) = 60 / BR
    from lib.sensors.ecg.respiration import calculate_respiratory_period
    stats.append({
        'Metric': 'Respiratory Period',
        'Value': calculate_respiratory_period(respiration_result.breathing_rate),
        'Unit': 's'
    })

    if hasattr(respiration_result, 'spectral_breathing_rate'):
        stats.append({
            'Metric': 'Breathing Rate (Spectral)',
            'Value': respiration_result.spectral_breathing_rate,
            'Unit': 'bpm'
        })

    if hasattr(respiration_result, 'breathing_rate_trough'):
        stats.append({
            'Metric': 'Breathing Rate (Trough)',
            'Value': respiration_result.breathing_rate_trough,
            'Unit': 'bpm'
        })

    stats.append({
        'Metric': 'Breathing Rate (Std)',
        'Value': respiration_result.breathing_rate_std,
        'Unit': 'bpm'
    })

    stats.append({
        'Metric': 'Peak Count',
        'Value': respiration_result.peak_count,
        'Unit': 'count'
    })

    stats.append({
        'Metric': 'Trough Count',
        'Value': respiration_result.trough_count,
        'Unit': 'count'
    })

    if getattr(respiration_result, 'rsa_cycle_count', 0):
        stats.append({
            'Metric': 'RSA Amplitude (peak-valley)',
            'Value': respiration_result.rsa_amplitude_mean,
            'Unit': 'ms'
        })
        stats.append({
            'Metric': 'RSA Amplitude (median)',
            'Value': respiration_result.rsa_amplitude_median,
            'Unit': 'ms'
        })
        stats.append({
            'Metric': 'RSA Amplitude (std)',
            'Value': respiration_result.rsa_amplitude_std,
            'Unit': 'ms'
        })
        stats.append({
            'Metric': 'RSA Cycles',
            'Value': respiration_result.rsa_cycle_count,
            'Unit': 'count'
        })

    return pd.DataFrame(stats)
