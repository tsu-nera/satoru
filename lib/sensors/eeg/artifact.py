"""
EEGアーチファクト検出（振幅判定と窓単位のHSI判定）

HSI（Horseshoe Signal Index）は電極の接触インピーダンス指標であり、
アーチファクト検出器ではない。眼球運動・まばたき・前額の筋電・発汗ドリフト・
電極の微小な滑りはいずれもHSIをGoodのまま通過するため、
振幅による独立した品質ゲートが必要になる。

検出の粒度はPSDのWelch窓（既定1024サンプル = 4秒 @256Hz）に揃える。
PSDは窓単位で汚染されるため、それより細かい粒度で除去しても意味がない。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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

# PSDのWelch窓長（サンプル）。statistical_dataframe側のn_fftと一致させること。
# 窓は1点でも汚れると丸ごと落ちるので、窓長がそのまま巻き添え損失になる。
# 長くする方向の変更は除外率を大きく押し上げるため、docs/adr/001 を読んでから。
ARTIFACT_WINDOW_SAMPLES = 1024

# セグメントを有効とみなす最小の残存率。これを下回るセグメントは指標をNaNにする
SEGMENT_MIN_VALID_RATIO = 0.5

# ピーク判定・best値算出に使える最大の除外率。
# 指標として表示はするが、データの1/5超を捨てた区間を「最高パフォーマンス区間」に
# 選ばせない。除去は8秒窓単位なので、閾値未満の残留汚染は必ずある。
PEAK_MAX_EXCLUDED_RATIO = 0.2

# 窓の中でHSIがGood以外だった時間の割合がこれを超えたら、そのチャネル×窓を無効にする。
#
# 振幅判定だけでは接触不良を防げない。2026-10-27はTP9/TP10の接触不良でHSIがほぼ全時間Mediumになり、
# TP9は振幅判定を79%の窓が通過したが、その窓にも約7Hzの倍音列（+17〜+30dB）が乗っていた。
# 2026-10-28はTP10が記録の途中で外れ、Good以外の割合が1分刻みで0%→57%→100%と変わった。
# セッション全体の割合で判定すると途中で外れたチャネルを拾えないため、窓単位で判定する。
HSI_WINDOW_MAX_BAD_RATIO = 0.5


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


def detect_hsi_windows(
    hsi: np.ndarray,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    max_bad_ratio: float = HSI_WINDOW_MAX_BAD_RATIO,
) -> np.ndarray:
    """
    チャネル×窓の単位でHSIによる接触不良を判定する。

    Parameters
    ----------
    hsi : np.ndarray
        HSI値（n_channels, n_samples）。RAWと同じサンプル並び。1.0=Good, 2.0=Medium, 4.0=Bad。
        NaNはGoodとして扱う（HSIの欠損は接触不良の証拠にならないため）
    window_samples : int
        1窓のサンプル数（detect_artifact_windowsと揃える）
    max_bad_ratio : float
        窓内でGood以外が占めてよい割合の上限

    Returns
    -------
    valid : np.ndarray
        bool配列（n_channels, n_windows）。Trueが有効な窓
    """
    n_channels, n_samples = hsi.shape
    n_windows = n_samples // window_samples
    if n_windows == 0:
        return np.zeros((n_channels, 0), dtype=bool)

    bad = np.nan_to_num(hsi[:, : n_windows * window_samples], nan=1.0) > 1.0
    bad_ratio = bad.reshape(n_channels, n_windows, window_samples).mean(axis=2)
    return bad_ratio <= max_bad_ratio


def detect_valid_windows(
    data_uv: np.ndarray,
    hsi: Optional[np.ndarray] = None,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
) -> np.ndarray:
    """
    振幅判定とHSI判定をANDで合成した有効マスク（n_channels, n_windows）を返す。

    hsiがNoneなら振幅判定のみ。セッション全体で接触不良のチャネルは全窓が無効になる。
    """
    valid = detect_artifact_windows(data_uv, window_samples, threshold_uv)
    if hsi is not None:
        valid &= detect_hsi_windows(hsi, window_samples)
    return valid


def expand_window_mask(window_mask: np.ndarray, n_samples: int, window_samples: int) -> np.ndarray:
    """
    窓単位のbool配列（n_windows,）をサンプル単位（n_samples,）に展開する。

    窓に収まらない末尾のサンプルは最後の窓の判定を引き継ぐ（窓が無ければFalse）。
    """
    per_sample = np.repeat(window_mask, window_samples)[:n_samples]
    if per_sample.size < n_samples:
        tail = bool(window_mask[-1]) if window_mask.size else False
        per_sample = np.concatenate(
            [per_sample, np.full(n_samples - per_sample.size, tail)]
        )
    return per_sample


def calculate_amplitude_statistics(
    data_uv: np.ndarray,
    channel_names: list,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
    hsi: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    チャネル別のRAW振幅統計と有効率を計算する（接続品質セクション表示用）。

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
    hsi : np.ndarray, optional
        HSI値（n_channels, n_samples）。指定するとHSIによる除外率と合成後の有効率も出す

    Returns
    -------
    pd.DataFrame
        Channel / Std / P2P (median) / Threshold / Rejected / HSI Rejected / Valid の各列
        （Rejectedは振幅判定のみ、Validは振幅とHSIを合成した後の有効率）
    """
    p2p = _window_p2p(data_uv, window_samples)
    valid = detect_artifact_windows(data_uv, window_samples, threshold_uv)
    hsi_valid = detect_hsi_windows(hsi, window_samples) if hsi is not None else None
    combined = valid & hsi_valid if hsi_valid is not None else valid
    thresholds = channel_thresholds(p2p, threshold_uv) if p2p.size else None

    rows = []
    for idx, name in enumerate(channel_names):
        if p2p.size:
            p2p_median = float(np.median(p2p[idx]))
            threshold = float(thresholds[idx])
            rejected_pct = float((~valid[idx]).mean() * 100)
            hsi_rejected_pct = (
                float((~hsi_valid[idx]).mean() * 100) if hsi_valid is not None else np.nan
            )
            valid_pct = float(combined[idx].mean() * 100)
        else:
            p2p_median = np.nan
            threshold = np.nan
            rejected_pct = np.nan
            hsi_rejected_pct = np.nan
            valid_pct = np.nan

        row = {
            'Channel': name.replace('RAW_', ''),
            'Std': float(np.nanstd(data_uv[idx])),
            'P2P (median)': p2p_median,
            'Threshold': threshold,
            'Rejected': rejected_pct,
        }
        if hsi is not None:
            row['HSI Rejected'] = hsi_rejected_pct
        row['Valid'] = valid_pct
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_artifacts(
    data_uv: np.ndarray,
    channel_names: list,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
    hsi: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    アーチファクト検出結果をまとめて返す。

    hsiを渡すと、valid_mask・rejected_ratioは振幅判定とHSI判定を合成した結果になる。

    Returns
    -------
    dict
        {
            'valid_mask': np.ndarray,        # (n_channels, n_windows) bool（合成マスク）
            'amplitude_statistics': DataFrame,
            'rejected_ratio': float,          # 全体の除外率 (0.0-1.0)
            'window_samples': int,
            'threshold_uv': float,
            'channels': list,
        }
    """
    valid = detect_valid_windows(data_uv, hsi, window_samples, threshold_uv)
    rejected_ratio = float((~valid).mean()) if valid.size else 0.0

    return {
        'valid_mask': valid,
        'amplitude_statistics': calculate_amplitude_statistics(
            data_uv, channel_names, window_samples, threshold_uv, hsi=hsi
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
