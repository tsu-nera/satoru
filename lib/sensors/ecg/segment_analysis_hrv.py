"""
HRV時間セグメント分析モジュール

HRVデータを固定時間セグメントに分割し、各セグメントでHRV指標を計算する。
瞑想レポートのフォーマットに準拠。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import neurokit2 as nk


@dataclass
class HRVSegmentAnalysisResult:
    """
    HRV時間セグメント分析の結果を保持

    Attributes
    ----------
    segments : pd.DataFrame
        セグメント別の全データ（segment_index, segment_start, segment_end, 各指標）
    table : pd.DataFrame
        レポート用テーブル（min, RMSSD, SDNN, LF Power, HF Power, LF/HF, DFA α1, DFA α2）
    metadata : dict
        メタデータ（segment_minutes, session_start, session_end）
    """

    segments: pd.DataFrame
    table: pd.DataFrame
    metadata: Dict[str, Any]

    def to_markdown(self, floatfmt: str = '.2f') -> str:
        """テーブルをMarkdown文字列として返す"""
        return self.table.to_markdown(index=False, floatfmt=floatfmt)


def calculate_segment_hrv_analysis(
    hrv_data: Dict[str, Any],
    segment_minutes: float = 3.0,
    warmup_seconds: float = 0.0,
) -> HRVSegmentAnalysisResult:
    """
    HRVデータを固定時間セグメントに分割し、各セグメントでHRV指標を計算

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()の戻り値
        - rr_intervals_clean: クリーニング済みR-R間隔（ms）
        - time: 相対時間（秒）
        - session_start: セッション開始datetime
        - sampling_rate: サンプリングレート（通常1000Hz）
    segment_minutes : float, default 3.0
        セグメント長（分単位）
    warmup_seconds : float, default 0.0
        セッション開始後の除外期間（秒単位）

    Returns
    -------
    HRVSegmentAnalysisResult
        セグメント別HRV指標を含む分析結果

    Notes
    -----
    各セグメントで以下の指標を計算：
    - 時間領域: RMSSD, SDNN
    - 周波数領域: LF Power, HF Power, LF/HF Ratio
    - 非線形: DFA α1, DFA α2
    """
    rr_intervals = hrv_data['rr_intervals_clean']
    time_array = hrv_data['time']
    session_start = hrv_data.get('session_start')
    sampling_rate = hrv_data.get('sampling_rate', 1000.0)

    if len(rr_intervals) == 0:
        raise ValueError('R-R間隔データが空です')

    # ウォームアップ期間を除外
    time_array = np.array(time_array)
    rr_intervals = np.array(rr_intervals)

    # time_arrayとrr_intervalsは同じ長さ（各R-R間隔に対応する時刻）
    valid_mask = time_array >= warmup_seconds
    time_array = time_array[valid_mask]
    rr_intervals = rr_intervals[valid_mask]

    if len(time_array) == 0:
        raise ValueError(f'ウォームアップ期間（{warmup_seconds}秒）除外後、データが空です')

    # セッション開始・終了時刻
    if session_start:
        session_start_adjusted = session_start + pd.Timedelta(seconds=warmup_seconds)
        session_end = session_start + pd.Timedelta(seconds=time_array[-1])
    else:
        session_start_adjusted = None
        session_end = None

    # セグメント境界を計算
    segment_seconds = segment_minutes * 60.0
    total_duration = time_array[-1] - time_array[0]
    num_segments = int(np.ceil(total_duration / segment_seconds))

    if num_segments == 0:
        raise ValueError('セグメント数が0です。データ期間が短すぎます。')

    records = []

    for idx in range(num_segments):
        segment_start_sec = time_array[0] + idx * segment_seconds
        segment_end_sec = segment_start_sec + segment_seconds

        # セグメント内のR-R間隔を抽出
        # time_arrayは各R-R間隔が発生した時刻を表す
        segment_mask = (time_array >= segment_start_sec) & (time_array < segment_end_sec)
        segment_rr = rr_intervals[segment_mask]

        # セグメント開始・終了時刻（datetime）
        if session_start:
            segment_start_dt = session_start + pd.Timedelta(seconds=segment_start_sec)
            segment_end_dt = session_start + pd.Timedelta(seconds=segment_end_sec)
        else:
            segment_start_dt = None
            segment_end_dt = None

        # セグメント中心時刻（分）- RSAレポートの形式（1.50, 4.50など）
        elapsed_min = (segment_start_sec - time_array[0]) / 60.0 + segment_minutes / 2.0

        # R-R間隔が不足している場合はスキップ
        if len(segment_rr) < 50:
            records.append({
                'segment_index': idx + 1,
                'segment_start': segment_start_dt,
                'segment_end': segment_end_dt,
                'elapsed_min': elapsed_min,
                'rmssd': np.nan,
                'sdnn': np.nan,
                'lf_power': np.nan,
                'hf_power': np.nan,
                'lf_hf_ratio': np.nan,
                'dfa_alpha1': np.nan,
                'dfa_alpha2': np.nan,
            })
            continue

        # NeuroKit2でHRV指標を計算
        # R-R間隔をピークインデックスに変換
        try:
            peaks = nk.intervals_to_peaks(segment_rr, sampling_rate=sampling_rate)

            # 時間領域
            hrv_time = nk.hrv_time(peaks, sampling_rate=sampling_rate, show=False)
            rmssd = hrv_time['HRV_RMSSD'].iloc[0] if 'HRV_RMSSD' in hrv_time.columns else np.nan
            sdnn = hrv_time['HRV_SDNN'].iloc[0] if 'HRV_SDNN' in hrv_time.columns else np.nan

            # 周波数領域
            hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False)
            lf_power = hrv_freq['HRV_LF'].iloc[0] if 'HRV_LF' in hrv_freq.columns else np.nan
            hf_power = hrv_freq['HRV_HF'].iloc[0] if 'HRV_HF' in hrv_freq.columns else np.nan
            lf_hf_ratio = hrv_freq['HRV_LFHF'].iloc[0] if 'HRV_LFHF' in hrv_freq.columns else np.nan

            # 非線形（DFA）- Hoshiyama論文準拠パラメータ
            # integrate=False, order=1
            try:
                dfa_alpha1, _ = nk.fractal_dfa(
                    segment_rr,
                    scale=range(4, 17),
                    integrate=False,
                    order=1,
                    show=False
                )
                dfa_alpha2, _ = nk.fractal_dfa(
                    segment_rr,
                    scale=range(16, 65),
                    integrate=False,
                    order=1,
                    show=False
                )
            except:
                dfa_alpha1 = np.nan
                dfa_alpha2 = np.nan

        except Exception as e:
            # HRV計算失敗時はNaNを設定
            rmssd = np.nan
            sdnn = np.nan
            lf_power = np.nan
            hf_power = np.nan
            lf_hf_ratio = np.nan
            dfa_alpha1 = np.nan
            dfa_alpha2 = np.nan

        records.append({
            'segment_index': idx + 1,
            'segment_start': segment_start_dt,
            'segment_end': segment_end_dt,
            'elapsed_min': elapsed_min,
            'rmssd': rmssd,
            'sdnn': sdnn,
            'lf_power': lf_power,
            'hf_power': hf_power,
            'lf_hf_ratio': lf_hf_ratio,
            'dfa_alpha1': dfa_alpha1,
            'dfa_alpha2': dfa_alpha2,
        })

    segments_df = pd.DataFrame(records)

    if segments_df.empty:
        raise ValueError('セグメント分析の結果が空です')

    # レポート用テーブル（英語カラム名）
    table_rows = []
    for _, row in segments_df.iterrows():
        table_rows.append({
            'Time (min)': row['elapsed_min'],
            'RMSSD (ms)': row['rmssd'],
            'SDNN (ms)': row['sdnn'],
            'LF Power (ms²)': row['lf_power'],
            'HF Power (ms²)': row['hf_power'],
            'LF/HF': row['lf_hf_ratio'],
            'DFA α1': row['dfa_alpha1'],
            'DFA α2': row['dfa_alpha2'],
        })

    table_df = pd.DataFrame(table_rows)

    metadata = {
        'segment_minutes': segment_minutes,
        'session_start': session_start_adjusted,
        'session_end': session_end,
        'num_segments': num_segments,
    }

    return HRVSegmentAnalysisResult(
        segments=segments_df,
        table=table_df,
        metadata=metadata,
    )
