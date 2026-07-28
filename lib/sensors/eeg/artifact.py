"""
振幅ベースのEEGアーチファクト検出

HSI（Horseshoe Signal Index）は電極の接触インピーダンス指標であり、
アーチファクト検出器ではない。眼球運動・まばたき・前額の筋電・発汗ドリフト・
電極の微小な滑りはいずれもHSIをGoodのまま通過するため、
振幅による独立した品質ゲートが必要になる。

検出の粒度はPSDのWelch窓（既定2048サンプル = 8秒 @256Hz）に揃える。
PSDが8秒単位で汚染される以上、それより細かい粒度で除去しても意味がないため。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

# 絶対振幅閾値（μV, peak-to-peak）。MNEの慣例的なEEG rejectデフォルト 150μV に合わせる
ARTIFACT_P2P_THRESHOLD_UV = 150.0

# 相対閾値の係数。チャネル別のp2p中央値の何倍を外れ値とみなすか
#
# 絶対閾値だけでは不十分。バンドパスフィルタ後のMuse信号は振幅が小さく、
# 実測（2026-07-28セッション）ではAF7/AF8のp2p中央値が19-23μVしかない。
# 絶対150μVは中央値の6.5-7.8倍に相当し、明らかな眼球運動バーストを素通しする。
# 逆にTP10は中央値69.5μVで、相対基準だけでは緩くなる。両者のANDではなくORで判定する。
ARTIFACT_RELATIVE_K = 4.0

# 相対閾値の下限（μV）。極端にクリーンなチャネルで正常なEEG変動まで落とさないため
ARTIFACT_RELATIVE_FLOOR_UV = 50.0

# PSDのWelch窓長（サンプル）。statistical_dataframe側のn_fftと一致させること
ARTIFACT_WINDOW_SAMPLES = 2048

# セグメントを有効とみなす最小の残存率。これを下回るセグメントは指標をNaNにする
SEGMENT_MIN_VALID_RATIO = 0.5

# ピーク判定・best値算出に使える最大の除外率。
# 指標として表示はするが、データの1/5超を捨てた区間を「最高パフォーマンス区間」に
# 選ばせない。除去は8秒窓単位なので、閾値未満の残留汚染は必ずある。
PEAK_MAX_EXCLUDED_RATIO = 0.2


def _window_p2p(data_uv: np.ndarray, window_samples: int) -> np.ndarray:
    """チャネル×窓のpeak-to-peak振幅（μV）を返す。"""
    n_channels, n_samples = data_uv.shape
    n_windows = n_samples // window_samples

    if n_windows == 0:
        return np.zeros((n_channels, 0))

    windowed = data_uv[:, : n_windows * window_samples].reshape(
        n_channels, n_windows, window_samples
    )
    return windowed.max(axis=2) - windowed.min(axis=2)


def channel_thresholds(
    p2p: np.ndarray,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
    relative_k: float = ARTIFACT_RELATIVE_K,
    relative_floor_uv: float = ARTIFACT_RELATIVE_FLOOR_UV,
) -> np.ndarray:
    """
    チャネルごとの実効閾値（μV）を返す。

    絶対閾値と、チャネル別p2p中央値のrelative_k倍のうち厳しい方を採用する。
    相対側にはrelative_floor_uvの下限を設け、極端にクリーンなチャネルで
    正常なEEG変動まで除外しないようにする。
    """
    medians = np.median(p2p, axis=1) if p2p.size else np.array([])
    relative = np.maximum(medians * relative_k, relative_floor_uv)
    return np.minimum(relative, threshold_uv)


def detect_artifact_windows(
    data_uv: np.ndarray,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
    relative_k: float = ARTIFACT_RELATIVE_K,
    relative_floor_uv: float = ARTIFACT_RELATIVE_FLOOR_UV,
) -> np.ndarray:
    """
    チャネル×窓の単位でアーチファクトを判定する。

    Parameters
    ----------
    data_uv : np.ndarray
        EEGデータ（n_channels, n_samples）、単位μV
    window_samples : int
        1窓のサンプル数
    threshold_uv : float
        絶対的なp2p閾値（μV）
    relative_k : float
        チャネル別p2p中央値に対する倍率
    relative_floor_uv : float
        相対閾値の下限（μV）

    Returns
    -------
    valid : np.ndarray
        bool配列（n_channels, n_windows）。Trueが有効な窓

    Notes
    -----
    チャネル単位で判定するのは、Museでは耳側（TP9/TP10）だけが恒常的に汚く、
    前頭（AF7/AF8）はクリーンというケースが頻出するため。
    全チャネル一括で落とすとセッション全体が失われる。
    """
    p2p = _window_p2p(data_uv, window_samples)
    if p2p.size == 0:
        return np.zeros((data_uv.shape[0], 0), dtype=bool)

    thresholds = channel_thresholds(p2p, threshold_uv, relative_k, relative_floor_uv)
    return p2p <= thresholds[:, np.newaxis]


def calculate_amplitude_statistics(
    data_uv: np.ndarray,
    channel_names: list,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
) -> pd.DataFrame:
    """
    チャネル別のRAW振幅統計を計算する（接続品質セクション表示用）。

    Parameters
    ----------
    data_uv : np.ndarray
        EEGデータ（n_channels, n_samples）、単位μV
    channel_names : list
        チャネル名のリスト
    window_samples : int
        窓のサンプル数
    threshold_uv : float
        p2p閾値（μV）

    Returns
    -------
    pd.DataFrame
        Channel / Std / P2P (median) / Threshold / Rejected の各列
    """
    p2p = _window_p2p(data_uv, window_samples)
    valid = detect_artifact_windows(data_uv, window_samples, threshold_uv)
    thresholds = channel_thresholds(p2p, threshold_uv) if p2p.size else None

    rows = []
    for idx, name in enumerate(channel_names):
        if p2p.size:
            p2p_median = float(np.median(p2p[idx]))
            threshold = float(thresholds[idx])
            rejected_pct = float((~valid[idx]).mean() * 100)
        else:
            p2p_median = np.nan
            threshold = np.nan
            rejected_pct = np.nan

        rows.append({
            'Channel': name.replace('RAW_', ''),
            'Std': float(np.nanstd(data_uv[idx])),
            'P2P (median)': p2p_median,
            'Threshold': threshold,
            'Rejected': rejected_pct,
        })

    return pd.DataFrame(rows)


def summarize_artifacts(
    data_uv: np.ndarray,
    channel_names: list,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
) -> Dict[str, object]:
    """
    アーチファクト検出結果をまとめて返す。

    Returns
    -------
    dict
        {
            'valid_mask': np.ndarray,        # (n_channels, n_windows) bool
            'amplitude_statistics': DataFrame,
            'rejected_ratio': float,          # 全体の除外率 (0.0-1.0)
            'window_samples': int,
            'threshold_uv': float,
            'channels': list,
        }
    """
    valid = detect_artifact_windows(data_uv, window_samples, threshold_uv)
    rejected_ratio = float((~valid).mean()) if valid.size else 0.0

    return {
        'valid_mask': valid,
        'amplitude_statistics': calculate_amplitude_statistics(
            data_uv, channel_names, window_samples, threshold_uv
        ),
        'rejected_ratio': rejected_ratio,
        'window_samples': window_samples,
        'threshold_uv': threshold_uv,
        'channels': [name.replace('RAW_', '') for name in channel_names],
    }


def segment_valid_ratio(
    valid_mask: np.ndarray,
    start_window: int,
    end_window: int,
) -> float:
    """
    指定した窓レンジにおける有効セル率（チャネル×窓）を返す。

    Parameters
    ----------
    valid_mask : np.ndarray
        (n_channels, n_windows) のbool配列
    start_window, end_window : int
        窓インデックスの範囲 [start, end)

    Returns
    -------
    float
        有効率（0.0-1.0）。範囲が空の場合は0.0
    """
    if valid_mask.size == 0 or end_window <= start_window:
        return 0.0

    sliced = valid_mask[:, start_window:end_window]
    if sliced.size == 0:
        return 0.0

    return float(sliced.mean())
