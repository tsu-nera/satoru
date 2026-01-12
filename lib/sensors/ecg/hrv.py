"""
心拍変動（HRV）解析モジュール

Selfloops ECGデータからNeuroKit2を使用して標準HRV指標セットを計算し、
時系列（RMSSD、LF/HF ratio）と統計テーブルを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import neurokit2 as nk


@dataclass
class HRVResult:
    """
    HRV解析結果を保持するデータクラス

    Attributes
    ----------
    time_series : dict
        時系列データ（RMSSD, LF/HF ratio）
        {
            'rmssd': pd.Series,      # RMSSD時系列（インデックス: datetime）
            'lfhf_ratio': pd.Series  # LF/HF ratio時系列（インデックス: datetime）
        }
    statistics : pd.DataFrame
        統計テーブル（Domain, Metric, Value, Unit, Interpretation列）
    metadata : dict
        メタデータ（window_seconds, step_seconds, session_start等）
    full_metrics : pd.DataFrame
        全HRV指標（NeuroKit2の完全出力、124指標）
    """

    time_series: Dict[str, pd.Series]
    statistics: pd.DataFrame
    metadata: dict
    full_metrics: pd.DataFrame


def calculate_hrv_standard_set(
    hrv_data: Dict[str, Any],
    window_seconds: float = 180.0,
    step_seconds: float = 30.0,
    min_rr_count: int = 50,
) -> HRVResult:
    """
    標準HRV指標セット（~15指標）を計算

    時間領域（Time Domain）:
    - SDNN: R-R間隔の標準偏差
    - RMSSD: R-R間隔の二乗平均平方根
    - pNN50: 50ms以上異なる隣接R-R間隔の割合
    - MeanNN: R-R間隔の平均
    - MedianNN: R-R間隔の中央値
    - CVNN: 変動係数
    - SDSD: R-R間隔差分の標準偏差

    周波数領域（Frequency Domain）:
    - VLF: 超低周波成分（0.003-0.04 Hz）
    - LF: 低周波成分（0.04-0.15 Hz）
    - HF: 高周波成分（0.15-0.4 Hz）
    - LF/HF ratio: 交感神経・副交感神経バランス
    - Total Power: 全周波数帯域のパワー

    非線形（Nonlinear）:
    - SD1: Poincaréプロット短軸標準偏差
    - SD2: Poincaréプロット長軸標準偏差
    - SD1/SD2: 非線形バランス指標

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()の戻り値
        - rr_intervals_clean: クリーニング済みR-R間隔（ms）
        - time: 相対時間（秒）
        - session_start: セッション開始datetime
        - sampling_rate: サンプリングレート（通常1000Hz）
    window_seconds : float
        スライディングウィンドウのサイズ（秒）
        デフォルト180秒（3分）- HRV解析の標準ウィンドウサイズ
    step_seconds : float
        スライディングウィンドウのステップ（秒）
        デフォルト30秒 - 平滑な時系列を得るため重複を許容
    min_rr_count : int
        各ウィンドウに必要な最小R-R間隔数
        不足する場合は該当ウィンドウをスキップ

    Returns
    -------
    HRVResult
        時系列・統計情報・メタデータを含む解析結果

    Raises
    ------
    ValueError
        R-R間隔データが不足している場合（セッション時間 < 180秒）

    Notes
    -----
    - セッション全体の統計値: full_metricsに格納
    - 時系列データ: RMSSDとLF/HF ratioのみ生成（可視化対象）
    - 短時間記録への対応: 最小180秒（3分）必要
    """
    rr_intervals = hrv_data['rr_intervals_clean']
    time_sec = hrv_data['time']
    session_start = hrv_data['session_start']
    sampling_rate = hrv_data.get('sampling_rate', 1000)

    # セッション時間チェック
    total_duration = time_sec[-1] - time_sec[0]
    if total_duration < 180:
        raise ValueError(
            f"Recording too short for HRV analysis: {total_duration:.0f}s < 180s minimum. "
            "HRV analysis requires at least 3 minutes of data."
        )

    # 短時間記録（180-300秒）の場合、ウィンドウサイズを動的調整
    if total_duration < 300:
        window_seconds = total_duration / 2
        step_seconds = window_seconds / 6
        print(f'⚠️  短時間記録のためウィンドウサイズ調整: {window_seconds:.0f}秒')

    # 1. セッション全体のHRV指標計算（NeuroKit2使用）
    try:
        peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=sampling_rate)
        full_metrics = nk.hrv(peaks, sampling_rate=sampling_rate, show=False)
    except Exception as e:
        raise ValueError(f"Failed to calculate HRV metrics: {e}")

    # 2. スライディングウィンドウでRMSSD・LF/HF時系列生成
    time_series = _calculate_sliding_window_hrv(
        rr_intervals=rr_intervals,
        time_sec=time_sec,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
        sampling_rate=sampling_rate,
        min_rr_count=min_rr_count,
        session_start=session_start,
    )

    # 3. 統計テーブル作成
    statistics = _create_hrv_statistics_table(
        full_metrics=full_metrics,
        rmssd_series=time_series['rmssd'],
        lfhf_series=time_series['lfhf_ratio'],
    )

    # 4. メタデータ作成
    metadata = {
        'window_seconds': window_seconds,
        'step_seconds': step_seconds,
        'min_rr_count': min_rr_count,
        'session_start': session_start,
        'total_duration_sec': total_duration,
        'sampling_rate': sampling_rate,
        'rr_intervals_count': len(rr_intervals),
        'mean_rmssd': time_series['rmssd'].mean(),
        'mean_lfhf': time_series['lfhf_ratio'].mean(),
    }

    return HRVResult(
        time_series=time_series,
        statistics=statistics,
        metadata=metadata,
        full_metrics=full_metrics,
    )


def _calculate_sliding_window_hrv(
    rr_intervals: np.ndarray,
    time_sec: np.ndarray,
    window_seconds: float,
    step_seconds: float,
    sampling_rate: int,
    min_rr_count: int,
    session_start: datetime,
) -> Dict[str, pd.Series]:
    """
    スライディングウィンドウでHRV時系列を計算（内部関数）

    各ウィンドウでNeuroKit2を呼び出し、RMSSDとLF/HF ratioを抽出。
    タイムスタンプはウィンドウの中央時刻を使用。

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R間隔（ms）
    time_sec : np.ndarray
        相対時間（秒）
    window_seconds : float
        ウィンドウサイズ（秒）
    step_seconds : float
        ステップサイズ（秒）
    sampling_rate : int
        サンプリングレート（Hz）
    min_rr_count : int
        最小R-R間隔数
    session_start : datetime
        セッション開始時刻

    Returns
    -------
    dict
        {
            'rmssd': pd.Series,      # インデックス: datetime
            'lfhf_ratio': pd.Series
        }
    """
    total_duration = time_sec[-1] - time_sec[0]
    num_windows = int((total_duration - window_seconds) / step_seconds) + 1

    rmssd_values = []
    lfhf_values = []
    timestamps = []
    consecutive_nans = 0

    for i in range(num_windows):
        # ウィンドウ範囲
        window_start = i * step_seconds
        window_end = window_start + window_seconds

        # R-R間隔抽出
        mask = (time_sec >= window_start) & (time_sec < window_end)
        window_rr = rr_intervals[mask]

        # 最小R-R間隔数チェック
        if len(window_rr) < min_rr_count:
            rmssd_values.append(np.nan)
            lfhf_values.append(np.nan)
            consecutive_nans += 1
            if consecutive_nans >= 3:
                print(f'⚠️  Warning: 連続{consecutive_nans}ウィンドウでデータ不足（ウィンドウ{i}）')
        else:
            consecutive_nans = 0

            # NeuroKit2でHRV計算
            try:
                peaks = nk.intervals_to_peaks(window_rr, sampling_rate=sampling_rate)

                # 時間領域（RMSSD）
                hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
                rmssd = hrv_time['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in hrv_time.columns else np.nan

                # 周波数領域（LF, HF）
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False)
                lf = hrv_freq['HRV_LF'].iloc[0] if 'HRV_LF' in hrv_freq.columns else np.nan
                hf = hrv_freq['HRV_HF'].iloc[0] if 'HRV_HF' in hrv_freq.columns else np.nan
                lfhf = lf / hf if hf > 0 and not np.isnan(hf) else np.nan

                rmssd_values.append(rmssd)
                lfhf_values.append(lfhf)

            except Exception as e:
                print(f'⚠️  Warning: ウィンドウ{i}のHRV計算失敗: {e}')
                rmssd_values.append(np.nan)
                lfhf_values.append(np.nan)

        # ウィンドウ中央時刻をタイムスタンプに
        window_center = window_start + window_seconds / 2
        timestamp = session_start + pd.Timedelta(seconds=window_center)
        timestamps.append(timestamp)

    # pd.Series作成
    return {
        'rmssd': pd.Series(rmssd_values, index=timestamps, name='RMSSD'),
        'lfhf_ratio': pd.Series(lfhf_values, index=timestamps, name='LF/HF Ratio'),
    }


def _create_hrv_statistics_table(
    full_metrics: pd.DataFrame,
    rmssd_series: pd.Series,
    lfhf_series: pd.Series,
) -> pd.DataFrame:
    """
    HRV統計テーブルを生成（内部関数）

    Parameters
    ----------
    full_metrics : pd.DataFrame
        NeuroKit2の完全HRV指標
    rmssd_series : pd.Series
        RMSSD時系列
    lfhf_series : pd.Series
        LF/HF ratio時系列

    Returns
    -------
    pd.DataFrame
        列: ['Domain', 'Metric', 'Value', 'Unit', 'Interpretation']
    """
    stats_rows = []

    # 時間領域指標
    time_metrics = [
        ('HRV_SDNN', 'SDNN', 'ms'),
        ('HRV_RMSSD', 'RMSSD', 'ms'),
        ('HRV_pNN50', 'pNN50', '%'),
        ('HRV_MeanNN', 'MeanNN', 'ms'),
        ('HRV_MedianNN', 'MedianNN', 'ms'),
        ('HRV_CVNN', 'CVNN', '%'),
        ('HRV_SDSD', 'SDSD', 'ms'),
    ]

    for col_name, metric_name, unit in time_metrics:
        if col_name in full_metrics.columns:
            value = full_metrics[col_name].iloc[0]
            interpretation = _interpret_hrv_metric(col_name, value)
            stats_rows.append({
                'Domain': 'Time Domain',
                'Metric': metric_name,
                'Value': value,
                'Unit': unit,
                'Interpretation': interpretation,
            })

    # 周波数領域指標
    freq_metrics = [
        ('HRV_VLF', 'VLF', 'ms²'),
        ('HRV_LF', 'LF', 'ms²'),
        ('HRV_HF', 'HF', 'ms²'),
        ('HRV_LFHF', 'LF/HF', '-'),
        ('HRV_TP', 'Total Power', 'ms²'),
    ]

    for col_name, metric_name, unit in freq_metrics:
        if col_name in full_metrics.columns:
            value = full_metrics[col_name].iloc[0]
            interpretation = _interpret_hrv_metric(col_name, value)
            stats_rows.append({
                'Domain': 'Frequency Domain',
                'Metric': metric_name,
                'Value': value,
                'Unit': unit,
                'Interpretation': interpretation,
            })

    # 非線形指標
    nonlinear_metrics = [
        ('HRV_SD1', 'SD1', 'ms'),
        ('HRV_SD2', 'SD2', 'ms'),
        ('HRV_SD1SD2', 'SD1/SD2', '-'),
    ]

    for col_name, metric_name, unit in nonlinear_metrics:
        if col_name in full_metrics.columns:
            value = full_metrics[col_name].iloc[0]
            interpretation = _interpret_hrv_metric(col_name, value)
            stats_rows.append({
                'Domain': 'Nonlinear',
                'Metric': metric_name,
                'Value': value,
                'Unit': unit,
                'Interpretation': interpretation,
            })

    return pd.DataFrame(stats_rows)


def _interpret_hrv_metric(metric_name: str, value: float) -> str:
    """
    HRV指標値の解釈文字列を生成（内部関数）

    Parameters
    ----------
    metric_name : str
        HRV指標名（例: 'HRV_RMSSD'）
    value : float
        指標値

    Returns
    -------
    str
        解釈文字列

    Examples
    --------
    >>> _interpret_hrv_metric('HRV_RMSSD', 50.0)
    '高い = リラックス'
    >>> _interpret_hrv_metric('HRV_LFHF', 0.8)
    '<1 = 副交感神経優位'

    References
    ----------
    - Task Force (1996): Heart rate variability standards
    - Shaffer & Ginsberg (2017): An Overview of HRV Metrics
    """
    if pd.isna(value):
        return '-'

    interpretations = {
        'HRV_SDNN': [
            (0, 50, '低い（<50ms）'),
            (50, 100, '標準（50-100ms）'),
            (100, float('inf'), '高い（>100ms）'),
        ],
        'HRV_RMSSD': [
            (0, 20, '低い = ストレス'),
            (20, 50, '標準'),
            (50, float('inf'), '高い = リラックス'),
        ],
        'HRV_pNN50': [
            (0, 5, '低い（<5%）'),
            (5, 20, '標準（5-20%）'),
            (20, float('inf'), '高い（>20%）'),
        ],
        'HRV_LFHF': [
            (0, 1.0, '<1 = 副交感神経優位'),
            (1.0, 2.5, '1-2.5 = バランス'),
            (2.5, float('inf'), '>2.5 = 交感神経優位'),
        ],
        'HRV_SD1': [
            (0, 20, '低い（<20ms）'),
            (20, 50, '標準（20-50ms）'),
            (50, float('inf'), '高い（>50ms）'),
        ],
        'HRV_SD2': [
            (0, 50, '低い（<50ms）'),
            (50, 100, '標準（50-100ms）'),
            (100, float('inf'), '高い（>100ms）'),
        ],
        'HRV_SD1SD2': [
            (0, 0.5, '<0.5 = 長期変動優位'),
            (0.5, 1.5, '0.5-1.5 = バランス'),
            (1.5, float('inf'), '>1.5 = 短期変動優位'),
        ],
    }

    if metric_name not in interpretations:
        return '-'

    ranges = interpretations[metric_name]
    for low, high, interpretation in ranges:
        if low <= value < high:
            return interpretation

    return '-'
