"""
ECGデータからのHRV解析 - NeuroKit2ラッパー

このモジュールはNeuroKit2を使用してECGデータからHRV解析を実行します。
デバイス非依存の設計により、どのECGセンサーからのデータでも同じAPIで解析可能です。
"""

from typing import Any, Dict

import neurokit2 as nk
import pandas as pd


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
