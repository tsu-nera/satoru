"""
EEG時系列データの統計計算モジュール

前半/後半比較、基本統計量の計算など、共通の統計処理を提供する。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def calculate_half_comparison(series: pd.Series) -> Dict[str, float]:
    """
    時系列データの前半と後半を比較する。

    Parameters
    ----------
    series : pd.Series
        時系列データ（DatetimeIndexを持つ）

    Returns
    -------
    dict
        {
            'first_half_mean': float,
            'second_half_mean': float,
            'change_db': float,  # dB差分（後半 - 前半）
            'change_percent': float,  # 変化率（%）
        }
    """
    if series.empty:
        return {
            'first_half_mean': np.nan,
            'second_half_mean': np.nan,
            'change_db': np.nan,
            'change_percent': np.nan,
        }

    midpoint = series.index[0] + (series.index[-1] - series.index[0]) / 2
    first_half = series[series.index <= midpoint]
    second_half = series[series.index > midpoint]

    first_mean = first_half.mean() if not first_half.empty else np.nan
    second_mean = second_half.mean() if not second_half.empty else np.nan

    # dB差分（対数スケールなので差分が意味を持つ）
    if pd.notna(first_mean) and pd.notna(second_mean):
        change_db = second_mean - first_mean
    else:
        change_db = np.nan

    # パーセント変化（線形スケール換算）
    if pd.notna(first_mean) and pd.notna(second_mean) and first_mean != 0:
        # dBの差分をパーセント変化に変換: 10^(dB/10) - 1
        # 近似: dB差分が小さい場合、約 23% per dB
        change_percent = ((second_mean - first_mean) / abs(first_mean)) * 100
    else:
        change_percent = np.nan

    return {
        'first_half_mean': first_mean,
        'second_half_mean': second_mean,
        'change_db': change_db,
        'change_percent': change_percent,
    }


def create_statistics_dataframe(
    series: pd.Series,
    name: str = 'Value',
    unit: str = 'dB',
    include_half_comparison: bool = True,
) -> pd.DataFrame:
    """
    時系列データの基本統計量をDataFrameとして生成する。

    Parameters
    ----------
    series : pd.Series
        時系列データ
    name : str
        値のカラム名
    unit : str
        単位
    include_half_comparison : bool
        前半/後半比較を含めるか

    Returns
    -------
    pd.DataFrame
        Metric, Value, Unit カラムを持つDataFrame
    """
    if series.empty:
        return pd.DataFrame(columns=['Metric', 'Value', 'Unit'])

    rows = [
        {'Metric': 'Mean', 'Value': series.mean(), 'Unit': unit},
        {'Metric': 'Median', 'Value': series.median(), 'Unit': unit},
        {'Metric': 'Std Dev', 'Value': series.std(), 'Unit': unit},
    ]

    if include_half_comparison:
        half_stats = calculate_half_comparison(series)
        rows.extend([
            {'Metric': 'First Half Mean', 'Value': half_stats['first_half_mean'], 'Unit': unit},
            {'Metric': 'Second Half Mean', 'Value': half_stats['second_half_mean'], 'Unit': unit},
            {'Metric': 'Change (2nd-1st)', 'Value': half_stats['change_db'], 'Unit': unit},
        ])

    return pd.DataFrame(rows)


def create_metadata(
    series: pd.Series,
    band: Tuple[float, float],
    channels: List[str],
    sfreq: float,
    processing_params: Dict,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    解析結果のメタデータを生成する。

    Parameters
    ----------
    series : pd.Series
        時系列データ
    band : Tuple[float, float]
        周波数帯域
    channels : List[str]
        使用チャネル
    sfreq : float
        サンプリングレート
    processing_params : Dict
        処理パラメータ（resample_interval等）
    extra : Dict, optional
        追加のメタデータ

    Returns
    -------
    Dict
        メタデータ辞書
    """
    half_stats = calculate_half_comparison(series)

    metadata = {
        'channels': channels,
        'band': band,
        'sfreq': sfreq,
        'first_half_mean': half_stats['first_half_mean'],
        'second_half_mean': half_stats['second_half_mean'],
        'change_db': half_stats['change_db'],
        'change_percent': half_stats['change_percent'],
        'unit': 'dB',
        'method': 'mne_hilbert_db',
        'filter_settings': {
            'l_freq': band[0],
            'h_freq': band[1],
            'fir_design': 'firwin',
            'phase': 'zero',
        },
        'processing': processing_params,
    }

    if extra:
        metadata.update(extra)

    return metadata
