"""
マインドワンダリング指標の計算

タップ打刻ログ（`lib.loaders.tap_log.load_tap_log_csv()` の戻り値）から、
マインドワンダリングの主観指標を計算する。タップは「注意がそれたことに
気づいた瞬間」の記録であり、逸脱そのものの開始時刻ではないことに注意。
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def calculate_mind_wandering_stats(tap_df: pd.DataFrame) -> Dict[str, Any]:
    """
    タップ打刻ログからマインドワンダリング指標を計算する。

    Parameters
    ----------
    tap_df : pd.DataFrame
        `load_tap_log_csv()` の戻り値。`event`, `elapsed_s` 列を持つ。

    Returns
    -------
    dict
        以下のキーを持つ辞書。
        - tap_count : int
        - duration_min : float
        - tap_rate_per_min : float
        - time_to_first_tap_s : float or None
        - median_iti_s : float or None
        - iti_cv : float or None
    """
    start_rows = tap_df.loc[tap_df['event'] == 'start', 'elapsed_s']
    start_elapsed = float(start_rows.iloc[0]) if len(start_rows) else 0.0

    stop_rows = tap_df.loc[tap_df['event'] == 'stop', 'elapsed_s']
    if len(stop_rows):
        end_elapsed = float(stop_rows.iloc[0])
    else:
        end_elapsed = float(tap_df['elapsed_s'].iloc[-1])

    duration_s = end_elapsed - start_elapsed
    duration_min = duration_s / 60.0

    tap_rows = tap_df.loc[tap_df['event'] == 'tap']
    tap_times = tap_rows['elapsed_s'].astype(float).sort_values().to_numpy()
    tap_count = len(tap_times)

    tap_rate_per_min = tap_count / duration_min if duration_min > 0 else float('nan')

    if tap_count == 0:
        time_to_first_tap_s = None
    else:
        time_to_first_tap_s = float(tap_times[0] - start_elapsed)

    if tap_count < 2:
        median_iti_s = None
        iti_cv = None
    else:
        itis = np.diff(tap_times)
        median_iti_s = float(np.median(itis))
        iti_mean = float(np.mean(itis))
        iti_std = float(np.std(itis))
        iti_cv = iti_std / iti_mean if iti_mean != 0 else None

    return {
        'tap_count': tap_count,
        'duration_min': duration_min,
        'tap_rate_per_min': tap_rate_per_min,
        'time_to_first_tap_s': time_to_first_tap_s,
        'median_iti_s': median_iti_s,
        'iti_cv': iti_cv,
    }


def calculate_segment_tap_counts(tap_df: pd.DataFrame, segments: pd.DataFrame) -> pd.DataFrame:
    """
    セグメントごとのタップ数を集計する。

    セグメント境界は自分で計算せず、呼び出し側から渡された `segments` の
    `segment_start` / `segment_end` をそのまま使う。

    Parameters
    ----------
    tap_df : pd.DataFrame
        `load_tap_log_csv()` の戻り値。`TimeStamp`, `event` 列を持つ。
    segments : pd.DataFrame
        `segment_start` / `segment_end` 列（pandas Timestamp）を持つDataFrame。
        `segment_result.segments`（`lib.report.steps_summary.analyze_segments()`
        の戻り値）を想定。

    Returns
    -------
    pd.DataFrame
        `segment_index`, `segment_start`, `segment_end`, `tap_count` の4列。
    """
    tap_times = tap_df.loc[tap_df['event'] == 'tap', 'TimeStamp']

    rows = []
    for segment_index, row in segments.iterrows():
        seg_start = row['segment_start']
        seg_end = row['segment_end']
        count = int(((tap_times >= seg_start) & (tap_times < seg_end)).sum())
        rows.append({
            'segment_index': segment_index,
            'segment_start': seg_start,
            'segment_end': seg_end,
            'tap_count': count,
        })

    return pd.DataFrame(rows)
