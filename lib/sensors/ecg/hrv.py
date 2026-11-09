"""
心拍変動（HRV）解析モジュール

Selfloops ECGデータからNeuroKit2を使用して標準HRV指標セットを計算し、
時系列（RMSSD、LF/HF ratio）と統計テーブルを生成する。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline
from scipy.stats import linregress


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
        統計テーブル（Domain, Metric, Value, Unit列）
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
    - DFA α1: 短期スケールのDFA指数（4-11拍）
    - DFA α2: 長期スケールのDFA指数（>11拍）
    - 1/f β: パワースペクトルの傾き（ゆらぎ指数）

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

        # 周波数領域を実単位（ms²）で再計算して上書き
        # nk.hrv() は hrv_frequency に kwargs を渡さないため normalize=True が効き、
        # VLF/LF/HF/TP がPSD最大値で正規化された無次元量になる（LF/HF比のみ無影響）。
        full_metrics.update(
            nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False, normalize=False)
        )

        # DFAを論文準拠パラメータで再計算
        # Hoshiyama et al. (2008, 2010) 準拠: integrate=False, order=1
        # スケール: α1 = 4-16拍, α2 = 16-64拍
        # integrate=False: R-R間隔を直接使用（cumsumなし）
        # order=1: 線形デトレンド
        dfa_alpha1, _ = nk.fractal_dfa(
            rr_intervals,
            scale=range(4, 17),
            integrate=False,
            order=1,
            show=False
        )
        dfa_alpha2, _ = nk.fractal_dfa(
            rr_intervals,
            scale=range(16, 65),
            integrate=False,
            order=1,
            show=False
        )

        # 論文準拠DFA値で上書き
        full_metrics['HRV_DFA_alpha1'] = dfa_alpha1
        full_metrics['HRV_DFA_alpha2'] = dfa_alpha2

    except Exception as e:
        raise ValueError(f"Failed to calculate HRV metrics: {e}") from e

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

    # 2.5. 1/fゆらぎ解析とピーク周波数解析（統計テーブル作成前に実行）
    try:
        freqs, power = calculate_power_spectrum(rr_intervals, fs=4.0)
        slope_result = calculate_1f_slope(freqs, power, freq_range=(0.01, 0.4))

        # LF/HF帯域のピーク周波数を計算
        lf_peak_freq = find_peak_frequency(freqs, power, freq_range=(0.04, 0.15))
        hf_peak_freq = find_peak_frequency(freqs, power, freq_range=(0.15, 0.4))

        # full_metricsに追加（統計テーブル生成で参照可能に）
        if lf_peak_freq is not None:
            full_metrics['HRV_LFPeak'] = lf_peak_freq
        if hf_peak_freq is not None:
            full_metrics['HRV_HFPeak'] = hf_peak_freq

        # 1/f β指数も追加
        full_metrics['HRV_1f_beta'] = slope_result['beta']
    except Exception as e:
        print(f'⚠️  1/f解析をスキップ: {e}')

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
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=sampling_rate, show=False, normalize=False)
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
        列: ['Domain', 'Metric', 'Value', 'Unit']
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
            stats_rows.append({
                'Domain': 'Time Domain',
                'Metric': metric_name,
                'Value': value,
                'Unit': unit,
            })

    # 周波数領域指標
    freq_metrics = [
        ('HRV_VLF', 'VLF', 'ms²'),
        ('HRV_LF', 'LF', 'ms²'),
        ('HRV_LFPeak', 'LF Peak', 'Hz'),
        ('HRV_HF', 'HF', 'ms²'),
        ('HRV_HFPeak', 'HF Peak', 'Hz'),
        ('HRV_LFHF', 'LF/HF', '-'),
        ('HRV_TP', 'Total Power', 'ms²'),
    ]

    for col_name, metric_name, unit in freq_metrics:
        if col_name in full_metrics.columns:
            value = full_metrics[col_name].iloc[0]
            stats_rows.append({
                'Domain': 'Frequency Domain',
                'Metric': metric_name,
                'Value': value,
                'Unit': unit,
            })

    # 非線形指標
    nonlinear_metrics = [
        ('HRV_SD1', 'SD1', 'ms'),
        ('HRV_SD2', 'SD2', 'ms'),
        ('HRV_SD1SD2', 'SD1/SD2', '-'),
        ('HRV_DFA_alpha1', 'DFA α1', '-'),
        ('HRV_DFA_alpha2', 'DFA α2', '-'),
        ('HRV_1f_beta', '1/f β (Exponent)', '-'),
    ]

    for col_name, metric_name, unit in nonlinear_metrics:
        if col_name in full_metrics.columns:
            value = full_metrics[col_name].iloc[0]
            stats_rows.append({
                'Domain': 'Nonlinear',
                'Metric': metric_name,
                'Value': value,
                'Unit': unit,
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
        'HRV_DFA_alpha1': [
            (0, 0.5, '<0.5 = 反相関'),
            (0.5, 0.55, '0.5-0.55 = 熟練瞑想者レベル (Hoshiyama 2008)'),
            (0.55, 0.70, '0.55-0.70 = 中級瞑想者レベル'),
            (0.70, 0.85, '0.70-0.85 = 初心者レベル (Hoshiyama 2008)'),
            (0.85, 1.0, '0.85-1.0 = 強い長期相関'),
            (1.0, float('inf'), '≥1.0 = 1/fノイズ'),
        ],
        'HRV_DFA_alpha2': [
            (0, 0.5, '<0.5 = 反相関'),
            (0.5, 1.0, '0.5-1.0 = 長期相関'),
            (1.0, 1.5, '1.0-1.5 = 1/fノイズ'),
            (1.5, float('inf'), '>1.5 = ブラウンノイズ'),
        ],
    }

    if metric_name not in interpretations:
        return '-'

    ranges = interpretations[metric_name]
    for low, high, interpretation in ranges:
        if low <= value < high:
            return interpretation

    return '-'



# RSAバンド（呼吸追従帯域）の半値幅。呼吸周波数 f0 に対し相対幅を基本とし、
# 超低速呼吸で幅が潰れないよう絶対下限を設ける。
RSA_BAND_REL_HALF_WIDTH = 0.25
RSA_BAND_MIN_HALF_WIDTH_HZ = 0.02
RSA_BAND_FLOOR_HZ = 0.003


def calculate_rsa_band_power(
    rr_intervals: np.ndarray,
    breathing_rate_bpm: float,
    sampling_rate: int = 1000,
) -> dict | None:
    """
    呼吸周波数に追従するRSAバンドのパワーを計算

    固定のHF帯（0.15-0.4Hz）は「人は12-24回/分で呼吸する」前提で作られており、
    瞑想の超低速呼吸ではRSAがHFではなくLF帯に落ちて、HF・LF/HF比が意味を失う。
    呼吸周波数 f0 を中心にバンドを切り直すことで、呼吸数によらず同じ構成概念
    （呼吸性洞性不整脈の大きさ）を測る。

    バンド幅は f0 に対する相対幅（±25%）を基本とし、超低速呼吸で幅が潰れない
    よう絶対下限（±0.02Hz）を設ける。

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R間隔（ms）
    breathing_rate_bpm : float
        呼吸数（bpm）。RespirationResult.breathing_rate を渡す。
    sampling_rate : int
        R-R間隔の元となるサンプリングレート（Hz）

    Returns
    -------
    dict or None
        - center_hz: バンド中心 = 呼吸周波数（Hz）
        - band_low_hz / band_high_hz: バンド境界（Hz）
        - power: バンドパワー（ms²）
        呼吸数が無効な場合や計算に失敗した場合はNone。

    Notes
    -----
    パワーは nk.hrv_frequency の ``lf`` スロットにRSAバンドを差し込んで得る。
    これは統計テーブルのVLF/LF/HF/TPと同一のPSD推定・同一スケール（normalize=False）
    になるため、「LFのうち何割がRSAバンドか」をそのまま比較できる。
    この呼び出しの戻り値のうち HRV_LF 以外（TP等）は帯域が標準と異なるため使わない。
    """
    if not np.isfinite(breathing_rate_bpm) or breathing_rate_bpm <= 0:
        return None

    center_hz = breathing_rate_bpm / 60.0
    half_width = max(RSA_BAND_REL_HALF_WIDTH * center_hz, RSA_BAND_MIN_HALF_WIDTH_HZ)
    band_low = max(center_hz - half_width, RSA_BAND_FLOOR_HZ)
    band_high = center_hz + half_width

    try:
        peaks = nk.intervals_to_peaks(rr_intervals, sampling_rate=sampling_rate)
        metrics = nk.hrv_frequency(
            peaks,
            sampling_rate=sampling_rate,
            show=False,
            normalize=False,
            lf=(band_low, band_high),
        )
        power = float(metrics['HRV_LF'].iloc[0])
    except Exception as e:
        print(f'⚠️  RSAバンドパワーの計算に失敗: {e}')
        return None

    return {
        'center_hz': center_hz,
        'band_low_hz': band_low,
        'band_high_hz': band_high,
        'power': power,
    }


def find_peak_frequency(freqs, power, freq_range):
    """
    指定周波数範囲内のピーク周波数を見つける

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）
    power : np.ndarray
        パワースペクトル密度
    freq_range : tuple
        周波数範囲 (min_freq, max_freq)

    Returns
    -------
    float or None
        ピーク周波数（Hz）、見つからない場合はNone
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    if not np.any(mask):
        return None

    freqs_filtered = freqs[mask]
    power_filtered = power[mask]

    if len(power_filtered) == 0:
        return None

    peak_idx = np.argmax(power_filtered)
    return freqs_filtered[peak_idx]


def calculate_power_spectrum(rr_intervals, fs=4.0):
    """
    R-R間隔からパワースペクトルを計算

    Parameters
    ----------
    rr_intervals : array-like
        R-R間隔データ（ms）
    fs : float, default=4.0
        リサンプリング周波数（Hz）

    Returns
    -------
    freqs : np.ndarray
        周波数配列（Hz）
    power : np.ndarray
        パワースペクトル密度
    """
    # R-R間隔を秒に変換
    rr_sec = np.array(rr_intervals) / 1000.0

    # 時間軸を生成
    time = np.cumsum(rr_sec)
    time = time - time[0]

    # 等間隔にリサンプリング
    time_regular = np.arange(0, time[-1], 1/fs)
    rr_interp = np.interp(time_regular, time, rr_sec)

    # デトレンド（線形トレンド除去）
    rr_detrend = signal.detrend(rr_interp)

    # ハニング窓を適用
    window = signal.windows.hann(len(rr_detrend))
    rr_windowed = rr_detrend * window

    # Welchメソッドでパワースペクトル密度を計算
    freqs, power = signal.welch(
        rr_windowed,
        fs=fs,
        nperseg=min(256, len(rr_windowed)),
        scaling='density'
    )

    return freqs, power


def calculate_1f_slope(freqs, power, freq_range=(0.01, 0.4)):
    """
    パワースペクトルから1/f^β指数を計算

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）
    power : np.ndarray
        パワースペクトル密度
    freq_range : tuple, default=(0.01, 0.4)
        解析周波数範囲（Hz）

    Returns
    -------
    dict
        - beta: 1/f^β の指数β（ピンクノイズなら1に近い）
        - slope: 傾き（= -beta、後方互換性のため保持）
        - intercept: 切片
        - r_value: 相関係数
        - p_value: p値
        - std_err: 標準誤差
        - freq_range: 解析に使用した周波数範囲
    """
    # 周波数範囲でフィルタ
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1]) & (freqs > 0) & (power > 0)
    freqs_filtered = freqs[mask]
    power_filtered = power[mask]

    # 対数変換
    log_freq = np.log10(freqs_filtered)
    log_power = np.log10(power_filtered)

    # 線形回帰
    # log(PSD) = log(A) - β * log(f)
    # slope = -β
    slope, intercept, r_value, p_value, std_err = linregress(log_freq, log_power)
    beta = -slope

    return {
        'beta': beta,
        'slope': slope,  # 後方互換性のため保持
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'freq_range': freq_range
    }


def classify_noise_type(beta):
    """
    1/f^β指数からノイズタイプを分類

    Parameters
    ----------
    beta : float
        1/f^β の指数β

    Returns
    -------
    str
        ノイズタイプ分類

    Notes
    -----
    - White Noise: β < 0.5
    - Pink Noise (1/f): 0.5 ≤ β < 1.5
    - Brown Noise (1/f²): 1.5 ≤ β < 2.5
    - Black Noise: β ≥ 2.5
    """
    if beta < 0.5:
        return "White Noise"
    elif 0.5 <= beta < 1.5:
        return "Pink Noise (1/f)"
    elif 1.5 <= beta < 2.5:
        return "Brown Noise (1/f²)"
    else:
        return "Black Noise"


def analyze_1f_fluctuation(hrv_data):
    """
    HRVデータから1/fゆらぎを解析

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()の戻り値
        - rr_intervals_clean: クリーニング済みR-R間隔（ms）

    Returns
    -------
    dict
        - freqs: 周波数配列
        - power: パワースペクトル密度
        - slope_result: calculate_1f_slope()の結果
        - noise_type: ノイズタイプ分類
        - statistics: 統計情報（DataFrame）
    """
    rr_intervals = hrv_data['rr_intervals_clean']

    # パワースペクトル計算
    freqs, power = calculate_power_spectrum(rr_intervals, fs=4.0)

    # 1/f^β 指数計算
    slope_result = calculate_1f_slope(freqs, power, freq_range=(0.01, 0.4))

    # ノイズタイプ分類
    noise_type = classify_noise_type(slope_result['beta'])

    # 統計情報をDataFrameに変換
    stats = []
    stats.append({
        'Metric': 'β (Exponent)',
        'Value': slope_result['beta'],
        'Unit': '-'
    })
    stats.append({
        'Metric': 'R²',
        'Value': slope_result['r_value'] ** 2,
        'Unit': '-'
    })
    stats.append({
        'Metric': 'Noise Type',
        'Value': noise_type,
        'Unit': '-'
    })
    stats.append({
        'Metric': 'Frequency Range',
        'Value': f"{slope_result['freq_range'][0]}-{slope_result['freq_range'][1]}",
        'Unit': 'Hz'
    })

    statistics_df = pd.DataFrame(stats)

    return {
        'freqs': freqs,
        'power': power,
        'slope_result': slope_result,
        'noise_type': noise_type,
        'statistics': statistics_df
    }



# ---------------------------------------------------------------------------
# 共鳴呼吸（Elite HRV）向け HRV 基本統計
# ---------------------------------------------------------------------------

LF_BAND_HZ = (0.04, 0.15)
HF_BAND_HZ = (0.15, 0.4)


def resample_rr(rr_ms: np.ndarray, fs: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """
    RR間隔を cubic spline で等間隔サンプリングに変換

    各 RR 値は区間の終端（拍の時刻 = cumsum(rr)/1000）に対応づける。

    Returns
    -------
    (t_sec, rr_interp_ms) : 等間隔の時刻（秒）と補間した RR（ms）
    """
    rr = np.asarray(rr_ms, dtype=float)
    t = np.cumsum(rr) / 1000.0
    t_grid = np.arange(t[0], t[-1], 1.0 / fs)
    return t_grid, CubicSpline(t, rr)(t_grid)


@dataclass
class HRVBasicStats:
    """1 reading 全体の HRV 基本統計（周波数領域が算出不能なら NaN）"""

    mean_hr_bpm: float
    min_hr_bpm: float
    max_hr_bpm: float
    mean_rr_ms: float
    rmssd_ms: float
    ln_rmssd: float
    sdnn_ms: float
    pnn50_pct: float
    lf_power_ms2: float = np.nan
    hf_power_ms2: float = np.nan
    total_power_ms2: float = np.nan
    lf_hf_ratio: float = np.nan
    lf_peak_hz: float = np.nan
    hf_peak_hz: float = np.nan

    def to_dict(self) -> Dict[str, float]:
        """CSV 列名をキーにした dict"""
        return {
            'mean_hr_bpm': self.mean_hr_bpm,
            'min_hr_bpm': self.min_hr_bpm,
            'max_hr_bpm': self.max_hr_bpm,
            'mean_rr_ms': self.mean_rr_ms,
            'rmssd_ms': self.rmssd_ms,
            'ln_rmssd': self.ln_rmssd,
            'sdnn_ms': self.sdnn_ms,
            'pnn50_pct': self.pnn50_pct,
            'lf_power_ms2': self.lf_power_ms2,
            'hf_power_ms2': self.hf_power_ms2,
            'total_power_ms2': self.total_power_ms2,
            'lf_hf_ratio': self.lf_hf_ratio,
            'lf_peak_hz': self.lf_peak_hz,
            'hf_peak_hz': self.hf_peak_hz,
        }


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> float:
    """PSD をバンド内で台形積分（ms²）"""
    mask = (freqs >= band[0]) & (freqs < band[1])
    return float(trapezoid(psd[mask], freqs[mask]))


def calculate_hrv_basic_stats(
    rr_ms: np.ndarray,
    fs: float = 4.0,
    min_freq_duration_sec: float = 240.0,
) -> HRVBasicStats:
    """
    補正済み RR から時間領域・周波数領域の基本統計を一度に計算する

    周波数領域は等間隔化（cubic spline）→ 平均除去 → Welch（nperseg=記録全長,
    hann, density）。LF 0.04-0.15 Hz / HF 0.15-0.4 Hz を台形積分し
    Total = LF + HF（VLF を含めない）。単位は ms²。
    記録長が min_freq_duration_sec 未満なら周波数領域は NaN。
    """
    rr = np.asarray(rr_ms, dtype=float)
    hr = 60000.0 / rr
    diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diffs**2)))
    stats = HRVBasicStats(
        mean_hr_bpm=float(np.mean(hr)),
        min_hr_bpm=float(np.min(hr)),
        max_hr_bpm=float(np.max(hr)),
        mean_rr_ms=float(np.mean(rr)),
        rmssd_ms=rmssd,
        ln_rmssd=float(np.log(rmssd)),
        sdnn_ms=float(np.std(rr, ddof=1)),
        pnn50_pct=float(np.sum(np.abs(diffs) > 50) / len(diffs) * 100),
    )

    if rr.sum() / 1000.0 < min_freq_duration_sec:
        return stats

    _, series = resample_rr(rr, fs)
    freqs, psd = signal.welch(
        series - series.mean(), fs=fs, nperseg=len(series),
        window='hann', scaling='density',
    )
    stats.lf_power_ms2 = _band_power(freqs, psd, LF_BAND_HZ)
    stats.hf_power_ms2 = _band_power(freqs, psd, HF_BAND_HZ)
    stats.total_power_ms2 = stats.lf_power_ms2 + stats.hf_power_ms2
    stats.lf_hf_ratio = (
        stats.lf_power_ms2 / stats.hf_power_ms2 if stats.hf_power_ms2 > 0 else np.nan
    )
    lf_peak = find_peak_frequency(freqs, psd, LF_BAND_HZ)
    hf_peak = find_peak_frequency(freqs, psd, HF_BAND_HZ)
    stats.lf_peak_hz = float(lf_peak) if lf_peak is not None else np.nan
    stats.hf_peak_hz = float(hf_peak) if hf_peak is not None else np.nan
    return stats
