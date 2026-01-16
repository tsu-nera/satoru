"""
テンプレート用データフォーマッタ

データオブジェクトをテンプレートで使用可能なDataFrame形式に変換する。
プレゼンテーション層のデータ変換ロジックを集約。
"""

import pandas as pd
from typing import Any


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

    stats.append({
        'Metric': 'Breathing Rate (Std)',
        'Value': respiration_result.breathing_rate_std,
        'Unit': 'bpm'
    })

    if hasattr(respiration_result, 'spectral_breathing_rate') and \
       respiration_result.spectral_breathing_rate is not None:
        stats.append({
            'Metric': 'Breathing Rate (Spectral)',
            'Value': respiration_result.spectral_breathing_rate,
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

    return pd.DataFrame(stats)
