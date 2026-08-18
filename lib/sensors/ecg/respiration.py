"""
呼吸解析モジュール（ECG-Derived Respiration）

ECGのR-R間隔データからNeuroKit2を使用して呼吸パターンを推定し、
呼吸数（BR）および共鳴呼吸回数（RBP）を算出する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy import interpolate

# スペクトル法で呼吸ピークを探索する周波数帯域（Hz）
# 下限0.04Hz=2.4bpm。瞑想の超低速呼吸（4-6bpm、0.07-0.10Hz）を含む一方、
# VLF帯のドリフト（<0.04Hz）は除外する。
BR_SEARCH_BAND_HZ = (0.04, 0.40)

# LF帯（0.04-0.15Hz）の上限。呼吸がこれを下回るとRSAがHFではなくLFに計上され、
# LF/HF比が自律神経バランスの指標として成立しなくなる。
LF_BAND_UPPER_HZ = 0.15
SLOW_BREATHING_THRESHOLD_BPM = LF_BAND_UPPER_HZ * 60  # 9.0 bpm


def calculate_respiratory_period(respiratory_rate):
    """
    呼吸数(bpm)から呼吸周期(秒)を計算

    Parameters
    ----------
    respiratory_rate : float or np.ndarray
        呼吸数 (breaths per minute)

    Returns
    -------
    float or np.ndarray
        呼吸周期 (seconds)

    Examples
    --------
    >>> calculate_respiratory_period(15)
    4.0
    >>> calculate_respiratory_period(12)
    5.0
    >>> calculate_respiratory_period(20)
    3.0
    """
    return 60.0 / respiratory_rate


@dataclass
class RespirationResult:
    """
    呼吸解析結果を保持するデータクラス

    Attributes
    ----------
    breathing_rate : float
        呼吸数の主推定値（bpm）。スペクトル法が有効ならその値、
        失敗時はトラフ法にフォールバックする。
    breathing_rate_method : str
        breathing_rate の由来（'spectral' または 'trough'）
    breathing_rate_trough : float
        トラフ法による平均呼吸数（bpm）。peak_distance依存が強いため補助指標。
    breathing_rate_std : float
        トラフ法による瞬時呼吸数の標準偏差
    peak_count : int
        検出された呼吸ピーク数
    trough_count : int
        検出された呼吸トラフ（谷）数
    spectral_breathing_rate : float
        スペクトル法による呼吸数（bpm）。検出できない場合はNaN。
    is_slow_breathing : bool
        呼吸が0.15Hz（9bpm）を下回るか。TrueならLF/HF比は解釈不能。
    rsa_amplitude_mean : float
        peak-valley法によるRSA振幅の平均（ms）。周波数帯域の定義に依存しないため、
        超低速呼吸でもHF/LFの帯域境界に影響されずに副交感神経活動を評価できる。
    rsa_amplitude_median : float
        RSA振幅の中央値（ms）
    rsa_amplitude_std : float
        RSA振幅の標準偏差（ms）
    rsa_cycle_count : int
        RSA振幅を算出できた呼吸周期の数
    rsa_cycles : pd.DataFrame
        呼吸周期ごとの内訳。列: Time (min), RSA Amplitude (ms), EDR Amplitude (a.u.), Beats
    time_series : pd.DataFrame
        時系列メトリクス（Time, HR, RMSSD, LF/HF, LF Power, HF Power, BR）
    metadata : dict
        メタデータ（測定時間、サンプリングレート等）
    """

    breathing_rate: float
    breathing_rate_method: str
    breathing_rate_trough: float
    breathing_rate_std: float
    peak_count: int
    trough_count: int
    spectral_breathing_rate: float
    is_slow_breathing: bool
    rsa_amplitude_mean: float
    rsa_amplitude_median: float
    rsa_amplitude_std: float
    rsa_cycle_count: int
    rsa_cycles: pd.DataFrame
    time_series: pd.DataFrame
    metadata: dict


@dataclass
class ResonanceBreathingPaceResult:
    """
    共鳴呼吸回数（RBP）推定結果を保持するデータクラス

    Attributes
    ----------
    optimal_rmssd : dict
        RMSSD基準の最適呼吸数範囲
        - range: 呼吸数範囲（文字列）
        - center: 中心値（bpm）
        - value: 平均RMSSD値（ms）
    optimal_lf : dict
        LF Power基準の最適呼吸数範囲
        - range: 呼吸数範囲（文字列）
        - center: 中心値（bpm）
        - value: 平均LF Power値（ms²）
    bin_statistics : pd.DataFrame
        呼吸数ビンごとの統計
    raw_correlation_data : pd.DataFrame
        生の相関データ
    """

    optimal_rmssd: dict
    optimal_lf: dict
    bin_statistics: pd.DataFrame
    raw_correlation_data: pd.DataFrame


def calculate_breathing_rate(
    hrv_data: Dict[str, Any],
    target_fs: float = 8.0,
    peak_distance: Optional[float] = None,
    window_minutes: float = 3.0,
    edr_method: str = 'soni2019',
) -> RespirationResult:
    """
    ECG-Derived Respiration（EDR）法で呼吸数を計算

    NeuroKit2を使用してR-R間隔から呼吸成分を抽出し、呼吸数を推定します。
    主推定値はスペクトル法（EDRのPSDピーク）です。トラフ法は peak_distance の
    設定に強く依存するため補助指標として併記します。

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()の戻り値
        - rr_intervals_clean: クリーニング済みR-R間隔（ms）
        - time: 相対時間（秒）
        - session_start: セッション開始datetime
        - sampling_rate: サンプリングレート（通常1000Hz）
    target_fs : float
        リサンプリング周波数（Hz）
        デフォルト8.0Hz（NeuroKit2のフィルタに対応）
    peak_distance : float, optional
        呼吸ピーク間の最小距離（秒）
        Noneの場合、スペクトル推定値から呼吸周期の半分を自動設定する。
        固定値を与えると検出可能な呼吸数に上限（60/peak_distance bpm）が
        生じるため、通常はNoneのままにすること。
    window_minutes : float
        時系列メトリクス計算のウィンドウサイズ（分）
        デフォルト3分
    edr_method : str
        NeuroKit2のEDR抽出フィルタ。デフォルト'soni2019'（0-0.5Hz）。
        'vangent2019'（0.1-0.4Hz = 6-24bpm）は瞑想の超低速呼吸を
        通過帯域外で除去してしまうため使わないこと。

    Returns
    -------
    RespirationResult
        呼吸数、ピーク数、時系列データを含む解析結果

    Raises
    ------
    ValueError
        R-R間隔データが不足している場合

    Examples
    --------
    >>> from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
    >>> df = load_selfloops_csv('data.csv')
    >>> hrv_data = get_hrv_data(df)
    >>> result = calculate_breathing_rate(hrv_data)
    >>> print(f"平均呼吸数: {result.breathing_rate:.1f} bpm")
    """
    rr_intervals = hrv_data['rr_intervals_clean']
    rr_time = hrv_data['time']

    if len(rr_intervals) < 10:
        raise ValueError("R-R間隔データが不足しています（最低10個必要）")

    # 1. R-R間隔から心拍数を計算
    hr_signal = 60000.0 / rr_intervals  # bpm

    # 2. 等間隔にリサンプリング
    time_original = np.cumsum(rr_intervals) / 1000  # 秒
    time_resampled = np.arange(0, time_original[-1], 1.0 / target_fs)

    f = interpolate.interp1d(time_original, hr_signal, kind='cubic', fill_value='extrapolate')
    hr_resampled = f(time_resampled)

    # 3. ECG-Derived Respiration（EDR）を抽出
    edr_signal = nk.ecg_rsp(hr_resampled, sampling_rate=int(target_fs), method=edr_method)

    # 4. 周波数解析（スペクトル法）— 主推定値
    # nperseg は 0.04Hz を分解できる長さが必要（2048サンプル @8Hz = 256秒 → 0.004Hz分解能）。
    # 短すぎると超低速呼吸のピークが隣接ビンに埋もれる。
    from scipy import signal as sp_signal
    freqs, psd = sp_signal.welch(
        edr_signal,
        fs=target_fs,
        nperseg=min(2048, len(edr_signal)),
        scaling='density'
    )

    br_low, br_high = BR_SEARCH_BAND_HZ
    br_mask = (freqs >= br_low) & (freqs <= br_high)
    br_freqs = freqs[br_mask]
    br_psd = psd[br_mask]

    if len(br_psd) > 0:
        breathing_rate_spectral = br_freqs[np.argmax(br_psd)] * 60
    else:
        breathing_rate_spectral = np.nan

    # 5. ピーク間最小距離を決定
    # 固定値は検出可能な呼吸数に上限を課す（8秒 → 7.5bpm以下しか測れない）。
    # スペクトル推定値から呼吸周期の半分を取り、速い呼吸にも遅い呼吸にも追随させる。
    if peak_distance is None:
        if np.isfinite(breathing_rate_spectral) and breathing_rate_spectral > 0:
            peak_distance = float(np.clip(0.5 * 60.0 / breathing_rate_spectral, 1.0, 15.0))
        else:
            peak_distance = 2.0

    # 6. 呼吸信号のクリーニングとピーク検出
    rsp_cleaned = nk.rsp_clean(edr_signal, sampling_rate=target_fs)
    rsp_peaks_dict = nk.rsp_findpeaks(
        rsp_cleaned,
        sampling_rate=target_fs,
        method='scipy',
        peak_distance=peak_distance
    )

    peaks_idx = rsp_peaks_dict['RSP_Peaks']
    troughs_idx = rsp_peaks_dict['RSP_Troughs']

    rsp_rate = nk.rsp_rate(
        rsp_cleaned,
        troughs=rsp_peaks_dict,
        sampling_rate=target_fs,
        method='trough'
    )

    trough_rate = np.nanmean(rsp_rate)
    std_rate = np.nanstd(rsp_rate)

    # スペクトル法を主推定値とし、失敗時のみトラフ法へフォールバック
    if np.isfinite(breathing_rate_spectral):
        mean_rate = breathing_rate_spectral
        rate_method = 'spectral'
    else:
        mean_rate = trough_rate
        rate_method = 'trough'

    # 7. RSA振幅（peak-valley法）を呼吸周期ごとに算出
    # トラフ間を1呼吸周期とみなす。帯域定義に依存しないため、超低速呼吸でも
    # LF/HF比のように破綻せず副交感神経活動の推移を追える。
    rsa_cycles = calculate_rsa_amplitude(
        rr_intervals=rr_intervals,
        rr_time_sec=time_original,
        cycle_bounds_sec=np.asarray(troughs_idx, dtype=float) / target_fs,
        edr_signal=rsp_cleaned,
        edr_fs=target_fs,
    )
    rsa_values = rsa_cycles['RSA Amplitude (ms)'].to_numpy(dtype=float)

    # 8. 時系列メトリクス計算
    metrics_df = _calculate_windowed_metrics(
        hrv_data,
        hr_resampled,
        rsp_rate,
        time_resampled,
        target_fs,
        window_minutes
    )

    # 9. メタデータ
    metadata = {
        'total_duration_minutes': rr_time[-1] / 60,
        'sampling_rate': target_fs,
        'peak_distance_seconds': peak_distance,
        'window_minutes': window_minutes,
        'resampled_samples': len(hr_resampled),
        'edr_method': edr_method,
        'br_search_band_hz': BR_SEARCH_BAND_HZ,
    }

    return RespirationResult(
        breathing_rate=mean_rate,
        breathing_rate_method=rate_method,
        breathing_rate_trough=trough_rate,
        breathing_rate_std=std_rate,
        peak_count=len(peaks_idx),
        trough_count=len(troughs_idx),
        spectral_breathing_rate=breathing_rate_spectral,
        is_slow_breathing=bool(
            np.isfinite(mean_rate) and mean_rate < SLOW_BREATHING_THRESHOLD_BPM
        ),
        rsa_amplitude_mean=float(np.mean(rsa_values)) if len(rsa_values) else np.nan,
        rsa_amplitude_median=float(np.median(rsa_values)) if len(rsa_values) else np.nan,
        rsa_amplitude_std=float(np.std(rsa_values)) if len(rsa_values) else np.nan,
        rsa_cycle_count=len(rsa_values),
        rsa_cycles=rsa_cycles,
        time_series=metrics_df,
        metadata=metadata
    )


def calculate_rsa_amplitude(
    rr_intervals: np.ndarray,
    rr_time_sec: np.ndarray,
    cycle_bounds_sec: np.ndarray,
    edr_signal: Optional[np.ndarray] = None,
    edr_fs: Optional[float] = None,
    min_beats: int = 3,
) -> pd.DataFrame:
    """
    peak-valley法（Grossman）でRSA振幅を呼吸周期ごとに算出

    各呼吸周期に含まれるR-R間隔の最大値と最小値の差を取る。周波数帯域の
    定義を一切使わないため、呼吸が0.15Hz（9bpm）を下回りRSAがHF帯から
    外れるセッションでも、通常呼吸のセッションと同じ意味で比較できる。

    Parameters
    ----------
    rr_intervals : np.ndarray
        R-R間隔（ms）
    rr_time_sec : np.ndarray
        各R-R間隔に対応する時刻（秒）。rr_intervals と同じ長さ。
    cycle_bounds_sec : np.ndarray
        呼吸周期の境界時刻（秒）。通常はEDRのトラフ時刻。
        隣接する2要素が1呼吸周期を成すため、N個の境界からN-1周期が得られる。
    edr_signal : np.ndarray, optional
        クリーニング済みEDR信号。与えると周期ごとの振幅（呼吸の深さの代理指標）
        も算出する。単位は任意（a.u.）でセッション内の相対推移のみ意味を持つ。
    edr_fs : float, optional
        edr_signal のサンプリング周波数（Hz）。edr_signal を与える場合は必須。
    min_beats : int
        1周期に必要な最小拍数。これ未満の周期は除外する。

    Returns
    -------
    pd.DataFrame
        列: Time (min), RSA Amplitude (ms), EDR Amplitude (a.u.), Beats
        算出できる周期が無い場合は空のDataFrame。

    References
    ----------
    - Grossman, P., van Beek, J., & Wientjes, C. (1990). A comparison of three
      quantification methods for estimation of respiratory sinus arrhythmia.
      Psychophysiology, 27(6), 702-714.
    """
    rr_intervals = np.asarray(rr_intervals, dtype=float)
    rr_time_sec = np.asarray(rr_time_sec, dtype=float)
    cycle_bounds_sec = np.asarray(cycle_bounds_sec, dtype=float)

    if edr_signal is not None and edr_fs is None:
        raise ValueError('edr_signal を指定する場合は edr_fs も必要です')

    rows = []
    for start, end in zip(cycle_bounds_sec[:-1], cycle_bounds_sec[1:]):
        mask = (rr_time_sec >= start) & (rr_time_sec < end)
        cycle_rr = rr_intervals[mask]
        if len(cycle_rr) < min_beats:
            continue

        edr_amplitude = np.nan
        if edr_signal is not None and edr_fs is not None:
            i0 = int(round(start * edr_fs))
            i1 = int(round(end * edr_fs))
            cycle_edr = np.asarray(edr_signal, dtype=float)[i0:i1]
            if len(cycle_edr) > 0:
                edr_amplitude = float(np.max(cycle_edr) - np.min(cycle_edr))

        rows.append({
            'Time (min)': start / 60.0,
            'RSA Amplitude (ms)': float(np.max(cycle_rr) - np.min(cycle_rr)),
            'EDR Amplitude (a.u.)': edr_amplitude,
            'Beats': len(cycle_rr),
        })

    if not rows:
        return pd.DataFrame(
            columns=['Time (min)', 'RSA Amplitude (ms)', 'EDR Amplitude (a.u.)', 'Beats']
        )

    return pd.DataFrame(rows)


def _calculate_windowed_metrics(
    hrv_data: Dict[str, Any],
    hr_resampled: np.ndarray,
    rsp_rate: np.ndarray,
    time_resampled: np.ndarray,
    target_fs: float,
    window_minutes: float
) -> pd.DataFrame:
    """
    ウィンドウごとにHR, RMSSD, LF/HF, LF Power, HF Power, BRを計算

    Parameters
    ----------
    hrv_data : dict
        HRVデータ
    hr_resampled : np.ndarray
        リサンプリング済み心拍数
    rsp_rate : np.ndarray
        瞬時呼吸数
    time_resampled : np.ndarray
        リサンプリング済み時間軸（秒）
    target_fs : float
        サンプリング周波数
    window_minutes : float
        ウィンドウ幅（分）

    Returns
    -------
    pd.DataFrame
        時系列メトリクステーブル
    """
    window_sec = window_minutes * 60
    total_duration = time_resampled[-1]

    results = []

    rr_intervals = hrv_data['rr_intervals_clean']
    rr_time = hrv_data['time']

    for start_time in np.arange(0, total_duration, window_sec):
        end_time = min(start_time + window_sec, total_duration)
        timestamp_min = (start_time + end_time) / 2 / 60

        # 心拍数（リサンプリング済みデータから）
        hr_mask = (time_resampled >= start_time) & (time_resampled < end_time)
        hr_window = hr_resampled[hr_mask]
        hr_mean = np.mean(hr_window) if len(hr_window) > 0 else np.nan

        # 呼吸数（リサンプリング済みデータから）
        br_window = rsp_rate[hr_mask]
        br_mean = np.nanmean(br_window) if len(br_window) > 0 else np.nan

        # R-R間隔のウィンドウを取得（元データから）
        rr_mask = (rr_time >= start_time) & (rr_time < end_time)
        rr_window = rr_intervals[rr_mask]

        # RMSSD, LF/HF, LF Power, HF Powerを計算
        if len(rr_window) > 10:
            try:
                peaks = nk.intervals_to_peaks(rr_window, sampling_rate=1000)

                # 時間領域HRV
                hrv_time = nk.hrv_time(peaks, sampling_rate=1000, show=False)
                rmssd = hrv_time['HRV_RMSSD'].values[0]

                # 周波数領域HRV（normalize=Falseで実際のパワー値を取得）
                hrv_freq = nk.hrv_frequency(peaks, sampling_rate=1000, show=False, normalize=False)
                lf_hf = hrv_freq['HRV_LFHF'].values[0]
                lf_power = hrv_freq['HRV_LF'].values[0]
                hf_power = hrv_freq['HRV_HF'].values[0]
            except Exception:
                rmssd = np.nan
                lf_hf = np.nan
                lf_power = np.nan
                hf_power = np.nan
        else:
            rmssd = np.nan
            lf_hf = np.nan
            lf_power = np.nan
            hf_power = np.nan

        results.append({
            'Time (min)': timestamp_min,
            'HR (bpm)': hr_mean,
            'RMSSD (ms)': rmssd,
            'LF/HF': lf_hf,
            'LF Power (ms^2)': lf_power,
            'HF Power (ms^2)': hf_power,
            'BR (bpm)': br_mean
        })

    return pd.DataFrame(results)


def analyze_breathing_hrv_correlation(
    metrics_df: pd.DataFrame,
    bin_width: float = 0.5
) -> Optional[ResonanceBreathingPaceResult]:
    """
    呼吸数とHRV振幅の相関を分析し、最適呼吸数範囲を推定

    Parameters
    ----------
    metrics_df : pd.DataFrame
        時系列メトリクステーブル（BR, RMSSD, LF Power含む）
    bin_width : float
        呼吸数ビンの幅（bpm）
        デフォルト0.5 bpm

    Returns
    -------
    ResonanceBreathingPaceResult or None
        分析結果（ビン統計、最適範囲など）
        データが不足している場合はNone
    """
    # NaNを除外
    df_clean = metrics_df.dropna(subset=['BR (bpm)', 'RMSSD (ms)', 'LF Power (ms^2)'])

    if len(df_clean) == 0:
        return None

    # 呼吸数をビン分け
    br_min = max(3.0, np.floor(df_clean['BR (bpm)'].min() * 2) / 2)
    br_max = min(10.0, np.ceil(df_clean['BR (bpm)'].max() * 2) / 2)
    br_bins = np.arange(br_min, br_max + bin_width, bin_width)

    # 各ビンでHRV指標の平均を計算
    bin_stats = []
    for i in range(len(br_bins) - 1):
        bin_start = br_bins[i]
        bin_end = br_bins[i + 1]
        bin_center = (bin_start + bin_end) / 2

        mask = (df_clean['BR (bpm)'] >= bin_start) & (df_clean['BR (bpm)'] < bin_end)
        bin_data = df_clean[mask]

        if len(bin_data) > 0:
            bin_stats.append({
                'BR Range': f'{bin_start:.1f}-{bin_end:.1f}',
                'BR Center (bpm)': bin_center,
                'Count': len(bin_data),
                'RMSSD Mean (ms)': bin_data['RMSSD (ms)'].mean(),
                'RMSSD Std (ms)': bin_data['RMSSD (ms)'].std(),
                'LF Power Mean (ms^2)': bin_data['LF Power (ms^2)'].mean(),
                'LF Power Std (ms^2)': bin_data['LF Power (ms^2)'].std(),
            })

    bin_stats_df = pd.DataFrame(bin_stats)

    if len(bin_stats_df) == 0:
        return None

    # 最適呼吸数範囲を特定
    optimal_rmssd_idx = bin_stats_df['RMSSD Mean (ms)'].idxmax()
    optimal_lf_idx = bin_stats_df['LF Power Mean (ms^2)'].idxmax()

    optimal_rmssd = {
        'range': bin_stats_df.loc[optimal_rmssd_idx, 'BR Range'],
        'center': bin_stats_df.loc[optimal_rmssd_idx, 'BR Center (bpm)'],
        'value': bin_stats_df.loc[optimal_rmssd_idx, 'RMSSD Mean (ms)']
    }

    optimal_lf = {
        'range': bin_stats_df.loc[optimal_lf_idx, 'BR Range'],
        'center': bin_stats_df.loc[optimal_lf_idx, 'BR Center (bpm)'],
        'value': bin_stats_df.loc[optimal_lf_idx, 'LF Power Mean (ms^2)']
    }

    return ResonanceBreathingPaceResult(
        optimal_rmssd=optimal_rmssd,
        optimal_lf=optimal_lf,
        bin_statistics=bin_stats_df,
        raw_correlation_data=df_clean
    )


def estimate_resonance_breathing_pace(
    hrv_data: Dict[str, Any],
    target_fs: float = 8.0,
    peak_distance: Optional[float] = None,
    window_minutes: float = 3.0,
    bin_width: float = 0.5
) -> tuple[RespirationResult, Optional[ResonanceBreathingPaceResult]]:
    """
    呼吸数を計算し、共鳴呼吸回数（RBP）を推定

    この関数は calculate_breathing_rate() と analyze_breathing_hrv_correlation()
    を組み合わせた便利関数です。

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()の戻り値
    target_fs : float
        リサンプリング周波数（Hz）
    peak_distance : float, optional
        呼吸ピーク間の最小距離（秒）。Noneでスペクトル推定値から自動設定。
    window_minutes : float
        時系列メトリクス計算のウィンドウサイズ（分）
    bin_width : float
        呼吸数ビンの幅（bpm）

    Returns
    -------
    tuple[RespirationResult, ResonanceBreathingPaceResult or None]
        (呼吸数解析結果, 共鳴呼吸回数推定結果)
        相関分析に十分なデータがない場合、2番目の要素はNone

    Examples
    --------
    >>> from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
    >>> df = load_selfloops_csv('data.csv')
    >>> hrv_data = get_hrv_data(df)
    >>> resp_result, rbp_result = estimate_resonance_breathing_pace(hrv_data)
    >>> print(f"平均呼吸数: {resp_result.breathing_rate:.1f} bpm")
    >>> if rbp_result:
    ...     print(f"推奨呼吸数: {rbp_result.optimal_rmssd['range']} bpm")
    """
    # 1. 呼吸数計算
    resp_result = calculate_breathing_rate(
        hrv_data,
        target_fs=target_fs,
        peak_distance=peak_distance,
        window_minutes=window_minutes
    )

    # 2. 相関分析
    rbp_result = analyze_breathing_hrv_correlation(
        resp_result.time_series,
        bin_width=bin_width
    )

    return resp_result, rbp_result
