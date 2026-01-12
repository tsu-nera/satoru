"""
ECGデータからのHRV解析 - NeuroKit2ラッパー

このモジュールはNeuroKit2を使用してECGデータからHRV解析を実行します。
デバイス非依存の設計により、どのECGセンサーからのデータでも同じAPIで解析可能です。
"""

import neurokit2 as nk
import pandas as pd
from typing import Dict, Any


def analyze_hrv(hrv_data: Dict[str, Any], show: bool = False) -> pd.DataFrame:
    """
    HRVデータを包括的に解析

    時間領域、周波数領域、非線形領域の全てのHRV指標を計算します。
    NeuroKit2の`hrv()`関数のラッパーです。

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()が返すHRVデータ辞書
        以下のキーを含む必要があります:
        - 'rr_intervals_clean': R-R間隔配列（ms、クリーニング済み）
          または 'rr_intervals': R-R間隔配列（ms、生データ）
        - 'sampling_rate': サンプリングレート（Hz）
    show : bool, default False
        結果を可視化するか（Poincaréプロット等）

    Returns
    -------
    pd.DataFrame
        HRV指標（124種類の指標を含む）
        - 時間領域: RMSSD, SDNN, pNN50等
        - 周波数領域: LF, HF, LF/HF ratio等
        - 非線形: SD1, SD2, Sample Entropy等

    Examples
    --------
    >>> from lib.loaders.selfloops import load_selfloops_csv, get_hrv_data
    >>> df = load_selfloops_csv('data/selfloops_2026-01-10.csv')
    >>> hrv_data = get_hrv_data(df)  # clean_artifacts=Trueがデフォルト
    >>> result = analyze_hrv(hrv_data, show=True)
    >>> print(result['HRV_RMSSD'])

    Notes
    -----
    - クリーニング済みR-R間隔（'rr_intervals_clean'）を優先的に使用
    - 外れ値除外済みのデータで、より正確なHRV指標を計算
    """
    # R-R間隔をNeuroKit2のpeaks形式に変換
    # クリーニング済みデータを優先的に使用
    rr_intervals = hrv_data.get('rr_intervals_clean', hrv_data['rr_intervals'])

    peaks = nk.intervals_to_peaks(
        rr_intervals,
        sampling_rate=hrv_data['sampling_rate']
    )

    # HRV解析実行
    hrv_indices = nk.hrv(
        peaks,
        sampling_rate=hrv_data['sampling_rate'],
        show=show
    )

    return hrv_indices


def analyze_hrv_time_domain(hrv_data: Dict[str, Any],
                           show: bool = False) -> pd.DataFrame:
    """
    時間領域HRV解析

    RMSSD, SDNN, pNN50等の時間領域指標を計算します。

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()が返すHRVデータ辞書
    show : bool, default False
        R-R間隔のヒストグラムを表示するか

    Returns
    -------
    pd.DataFrame
        時間領域HRV指標
        - HRV_RMSSD: R-R間隔の二乗平均平方根
        - HRV_SDNN: R-R間隔の標準偏差
        - HRV_pNN50: 50ms以上異なる隣接R-R間隔の割合
        - HRV_MeanNN: R-R間隔の平均
        等

    Examples
    --------
    >>> result = analyze_hrv_time_domain(hrv_data, show=True)
    >>> print(f"RMSSD: {result['HRV_RMSSD'][0]:.2f} ms")
    """
    # クリーニング済みデータを優先的に使用
    rr_intervals = hrv_data.get('rr_intervals_clean', hrv_data['rr_intervals'])

    peaks = nk.intervals_to_peaks(
        rr_intervals,
        sampling_rate=hrv_data['sampling_rate']
    )

    return nk.hrv_time(
        peaks,
        sampling_rate=hrv_data['sampling_rate'],
        show=show
    )


def analyze_hrv_frequency_domain(hrv_data: Dict[str, Any],
                                show: bool = False) -> pd.DataFrame:
    """
    周波数領域HRV解析

    LF, HF, LF/HF ratio等の周波数領域指標を計算します。

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()が返すHRVデータ辞書
    show : bool, default False
        パワースペクトル密度を表示するか

    Returns
    -------
    pd.DataFrame
        周波数領域HRV指標
        - HRV_LF: 低周波成分（0.04-0.15 Hz）
        - HRV_HF: 高周波成分（0.15-0.4 Hz）
        - HRV_LFHF: LF/HF比
        - HRV_VLF: 超低周波成分（0.003-0.04 Hz）
        等

    Examples
    --------
    >>> result = analyze_hrv_frequency_domain(hrv_data, show=True)
    >>> print(f"LF/HF ratio: {result['HRV_LFHF'][0]:.2f}")
    """
    # クリーニング済みデータを優先的に使用
    rr_intervals = hrv_data.get('rr_intervals_clean', hrv_data['rr_intervals'])

    peaks = nk.intervals_to_peaks(
        rr_intervals,
        sampling_rate=hrv_data['sampling_rate']
    )

    return nk.hrv_frequency(
        peaks,
        sampling_rate=hrv_data['sampling_rate'],
        show=show
    )


def analyze_hrv_nonlinear(hrv_data: Dict[str, Any],
                         show: bool = False) -> pd.DataFrame:
    """
    非線形HRV解析

    SD1, SD2, Sample Entropy等の非線形指標を計算します。

    Parameters
    ----------
    hrv_data : dict
        get_hrv_data()が返すHRVデータ辞書
    show : bool, default False
        Poincaréプロットを表示するか

    Returns
    -------
    pd.DataFrame
        非線形HRV指標
        - HRV_SD1: Poincaréプロット短軸標準偏差
        - HRV_SD2: Poincaréプロット長軸標準偏差
        - HRV_SD1SD2: SD1/SD2比
        - HRV_SampEn: Sample Entropy
        - HRV_ApEn: Approximate Entropy
        等

    Examples
    --------
    >>> result = analyze_hrv_nonlinear(hrv_data, show=True)
    >>> print(f"SD1: {result['HRV_SD1'][0]:.2f} ms")
    """
    # クリーニング済みデータを優先的に使用
    rr_intervals = hrv_data.get('rr_intervals_clean', hrv_data['rr_intervals'])

    peaks = nk.intervals_to_peaks(
        rr_intervals,
        sampling_rate=hrv_data['sampling_rate']
    )

    return nk.hrv_nonlinear(
        peaks,
        sampling_rate=hrv_data['sampling_rate'],
        show=show
    )
