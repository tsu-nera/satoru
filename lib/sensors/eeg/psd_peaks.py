"""
PSDピーク分析モジュール

PSDからピークを検出し、周波数帯域で分類する。
SMR (12-15Hz) も帯域として識別される。
4Hzの整数倍はMuse/Mind Monitor由来のアーチファクトとしてフラグ付けされる。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


class PeakType(Enum):
    """ピークの種別"""
    FUNDAMENTAL = "基本リズム"
    INDEPENDENT = "独立リズム"
    HARMONIC = "ハーモニクス"
    UNKNOWN = "不明"


# 周波数帯域定義（SMRを含む詳細版）
DETAILED_FREQ_BANDS: Dict[str, Tuple[float, float, str]] = {
    'Delta': (0.5, 4.0, '深い睡眠'),
    'Theta': (4.0, 8.0, '瞑想・内省'),
    'Low Alpha': (8.0, 10.0, 'リラックス'),
    'High Alpha': (10.0, 13.0, '閉眼安静'),
    'SMR': (12.0, 15.0, '感覚運動リズム'),
    'Low Beta': (15.0, 20.0, '軽い集中'),
    'Mid Beta': (20.0, 25.0, '活発な思考'),
    'High Beta': (25.0, 30.0, '興奮・不安'),
    'Gamma': (30.0, 50.0, '認知処理'),
}


def _is_4hz_harmonic(freq: float, tolerance: float = 0.5) -> bool:
    """
    周波数が4Hzの整数倍かどうかを判定

    4Hz倍数（4, 8, 12, 16, 20, 24, 28, 32...）はMuse/Mind Monitor由来の
    アーチファクトである可能性が高い。

    Parameters
    ----------
    freq : float
        判定する周波数 (Hz)
    tolerance : float
        許容誤差 (Hz)

    Returns
    -------
    bool
        4Hzの整数倍であればTrue
    """
    if freq < 4.0:
        return False
    remainder = freq % 4.0
    return remainder < tolerance or (4.0 - remainder) < tolerance


@dataclass
class PeakInfo:
    """検出されたピークの情報"""
    frequency: float
    power_db: float
    prominence: float
    band_name: str
    band_description: str
    peak_type: PeakType
    harmonic_of: Optional[float] = None
    harmonic_number: Optional[int] = None
    is_4hz_harmonic: bool = False  # 4Hz倍数アーチファクトの可能性


@dataclass
class PsdPeaksResult:
    """PSDピーク分析結果"""
    peaks: List[PeakInfo]
    peaks_table: pd.DataFrame
    fundamental_candidates: pd.DataFrame
    best_fundamental: Optional[float]
    statistics: pd.DataFrame
    metadata: dict


# 後方互換性のためのエイリアス
HarmonicsResult = PsdPeaksResult


def _classify_frequency_band(freq: float) -> Tuple[str, str]:
    """
    周波数を帯域に分類

    Parameters
    ----------
    freq : float
        周波数 (Hz)

    Returns
    -------
    band_name : str
        帯域名
    description : str
        帯域の説明
    """
    for band_name, (low, high, desc) in DETAILED_FREQ_BANDS.items():
        if low <= freq < high:
            return band_name, desc

    if freq < 0.5:
        return 'Sub-Delta', '極低周波'
    return 'High Gamma', '高次認知'


def _find_psd_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float] = (1.0, 45.0),
    prominence: float = 0.5,
    distance: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PSDからピークを検出

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列
    psd : np.ndarray
        PSD配列（μV²/Hz）
    freq_range : tuple
        検出する周波数範囲
    prominence : float
        ピーク検出の突出度閾値（dB）
    distance : int
        ピーク間の最小距離（サンプル数）

    Returns
    -------
    peak_freqs : np.ndarray
        ピーク周波数
    peak_powers : np.ndarray
        ピークパワー（dB）
    prominences : np.ndarray
        各ピークの突出度
    """
    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freq_subset = freqs[mask]
    psd_subset = psd[mask]

    # dBスケールに変換
    psd_db = 10 * np.log10(psd_subset + 1e-10)

    # ピーク検出
    peaks, properties = find_peaks(psd_db, prominence=prominence, distance=distance)

    if len(peaks) == 0:
        return np.array([]), np.array([]), np.array([])

    peak_freqs = freq_subset[peaks]
    peak_powers = psd_db[peaks]
    prominences = properties['prominences']

    return peak_freqs, peak_powers, prominences


def _classify_peaks(
    peak_freqs: np.ndarray,
    peak_powers: np.ndarray,
    prominences: np.ndarray,
    iaf: Optional[float] = None,
) -> List[PeakInfo]:
    """
    ピークを帯域で分類

    Parameters
    ----------
    peak_freqs : np.ndarray
        ピーク周波数
    peak_powers : np.ndarray
        ピークパワー（dB）
    prominences : np.ndarray
        各ピークの突出度
    iaf : float, optional
        Individual Alpha Frequency

    Returns
    -------
    peaks : list of PeakInfo
        分類されたピーク情報
    """
    peaks = []

    for freq, power, prom in zip(peak_freqs, peak_powers, prominences):
        band_name, band_desc = _classify_frequency_band(freq)

        # ピーク種別の判定（シンプル版）
        peak_type = PeakType.INDEPENDENT

        # IAFに近いピークは基本リズム
        if iaf is not None and abs(freq - iaf) < 1.0:
            peak_type = PeakType.FUNDAMENTAL

        # 4Hz倍数アーチファクトの判定
        is_artifact = _is_4hz_harmonic(freq)

        peaks.append(PeakInfo(
            frequency=float(freq),
            power_db=float(power),
            prominence=float(prom),
            band_name=band_name,
            band_description=band_desc,
            peak_type=peak_type,
            harmonic_of=None,
            harmonic_number=None,
            is_4hz_harmonic=is_artifact,
        ))

    # パワー順にソート
    peaks.sort(key=lambda p: -p.power_db)

    return peaks


def analyze_psd_peaks(
    psd_dict: dict,
    iaf: Optional[float] = None,
    freq_range: Tuple[float, float] = (1.0, 45.0),
    prominence: float = 0.5,
    max_peaks: int = 15,
) -> PsdPeaksResult:
    """
    PSDのピーク分析を実行

    PSDからピークを検出し、周波数帯域で分類する。
    SMR (12-15Hz) も帯域として識別される。
    4Hzの整数倍はアーチファクトとしてフラグ付けされる。

    Parameters
    ----------
    psd_dict : dict
        calculate_psd() の戻り値
        {'freqs': np.ndarray, 'psds': np.ndarray, 'channels': list}
    iaf : float, optional
        Individual Alpha Frequency
    freq_range : tuple
        ピーク検出の周波数範囲
    prominence : float
        ピーク検出の突出度閾値（dB）
    max_peaks : int
        レポートに含める最大ピーク数

    Returns
    -------
    PsdPeaksResult
        ピーク分析結果
    """
    freqs = psd_dict['freqs']
    psds = psd_dict['psds']

    # チャネル平均PSD
    psd_avg = np.mean(psds, axis=0)

    # ピーク検出
    peak_freqs, peak_powers, prominences = _find_psd_peaks(
        freqs, psd_avg, freq_range=freq_range, prominence=prominence
    )

    if len(peak_freqs) == 0:
        empty_df = pd.DataFrame(columns=[
            '周波数 (Hz)', 'パワー (dB)', '帯域', '備考'
        ])
        return PsdPeaksResult(
            peaks=[],
            peaks_table=empty_df,
            fundamental_candidates=pd.DataFrame(),
            best_fundamental=iaf,
            statistics=pd.DataFrame([{'Metric': 'Peak Count', 'Value': 0}]),
            metadata={'freq_range': freq_range, 'prominence': prominence, 'iaf': iaf},
        )

    # ピーク分類（帯域ベース）
    peaks = _classify_peaks(peak_freqs, peak_powers, prominences, iaf)

    # ピークテーブル作成
    table_rows = []
    for p in peaks[:max_peaks]:
        note = ""
        if p.peak_type == PeakType.FUNDAMENTAL:
            note = "IAF"
        elif p.band_name == 'SMR':
            note = "SMR"

        table_rows.append({
            '周波数 (Hz)': round(p.frequency, 1),
            'パワー (dB)': round(p.power_db, 1),
            '帯域': p.band_name,
            '備考': note,
            'is_4hz_harmonic': p.is_4hz_harmonic,  # 内部フラグ（フィルタ用）
        })

    peaks_table = pd.DataFrame(table_rows)

    # 統計情報
    stats_rows = [
        {'Metric': 'Total Peaks', 'Value': len(peaks)},
    ]

    if iaf is not None:
        stats_rows.append({'Metric': 'IAF (Hz)', 'Value': round(iaf, 2)})

    # SMRピークの情報を追加
    smr_peaks = [p for p in peaks if p.band_name == 'SMR']
    if smr_peaks:
        smr_max = max(smr_peaks, key=lambda p: p.power_db)
        stats_rows.append({'Metric': 'SMR Peak (Hz)', 'Value': round(smr_max.frequency, 1)})
        stats_rows.append({'Metric': 'SMR Power (dB)', 'Value': round(smr_max.power_db, 1)})

    statistics = pd.DataFrame(stats_rows)

    metadata = {
        'freq_range': freq_range,
        'prominence': prominence,
        'iaf': iaf,
        'n_channels': len(psd_dict['channels']),
        'channels': psd_dict['channels'],
    }

    return PsdPeaksResult(
        peaks=peaks,
        peaks_table=peaks_table,
        fundamental_candidates=pd.DataFrame(),
        best_fundamental=iaf,
        statistics=statistics,
        metadata=metadata,
    )


# 後方互換性のためのエイリアス
analyze_harmonics = analyze_psd_peaks
