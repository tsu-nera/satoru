"""
EEGアーチファクト検出（振幅判定とクリップ判定）

除外は信号の中身だけで決める。チャネル×窓（PSDのWelch窓）ごとに、
- 振幅: 窓内peak-to-peak振幅が閾値を超えた（眼球運動・筋電・ドリフトなど）
- クリップ: フィルタ前の生値が増幅器の飽和値（0 / 1450 μV）に張り付いたサンプルを含む
のどちらかに当たる窓を無効にする。

HSI（Horseshoe Signal Index）は電極の接触インピーダンス指標であり、除外には使わない。
HSIがGoodでも汚れている信号（2026-10-29の前頭は線スペクトルが全時間乗っていた）と、
HSIが非Goodでも内容で救える信号があり、HSIは窓の価値を決める指標にならないため。
表示用の「HSI Rejected」列だけ参考値として残す。

線スペクトル（電源・機器由来の狭帯域ノイズ）は窓を捨てず、PSD側で除去する
（lib/sensors/eeg/spectral_lines.py）。

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

# Mind Monitor のRAWは増幅器が飽和するとこの値に張り付く（2026-10-29確認:
# 後半の数分間、TP9/TP10は27〜34%のサンプルが0または1450に張り付き、AF7は一度も張り付かなかった）。
# 検出にはフィルタ前の生値を使うこと。バンドパス後は張り付き値が見えなくなる。
CLIP_LOW_UV = 0.0
CLIP_HIGH_UV = 1450.0

# 張り付き値との一致判定の許容幅（μV）。float変換の丸め誤差を吸収する
CLIP_TOLERANCE_UV = 0.5

# 窓の中でHSIがGood以外だった時間の割合がこれを超えたら、HSI Rejectedとして数える。
#
# 表示専用（参考値）。除外には使わない。理由:
# - 2026-10-29の前頭はHSIが98〜100% Goodのまま、4.5Hz間隔の線スペクトルが全時間乗っていた
# - 同日のTPでHSI非Goodだった窓の大半は実際にはクリップしており、
#   捨てる理由はHSIではなく窓の内容（クリップ）であるべき
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
    チャネル×窓の単位でHSIによる接触不良を判定する（表示用の参考値。除外には使わない）。

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


def detect_clipped_windows(
    raw_uv_unfiltered: np.ndarray,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    tolerance_uv: float = CLIP_TOLERANCE_UV,
) -> np.ndarray:
    """
    チャネル×窓の単位で、増幅器の飽和（0 / 1450 μVへの張り付き）を判定する。

    Parameters
    ----------
    raw_uv_unfiltered : np.ndarray
        フィルタ前のEEGデータ（n_channels, n_samples）、単位μV。
        バンドパス後の値では張り付きが見えないため、必ずフィルタ前を渡す
    window_samples : int
        1窓のサンプル数
    tolerance_uv : float
        張り付き値との差がこれ以内ならクリップとみなす

    Returns
    -------
    valid : np.ndarray
        bool配列（n_channels, n_windows）。Trueが有効（クリップ無し）な窓。
        NaNはクリップとみなさない
    """
    n_channels, n_samples = raw_uv_unfiltered.shape
    n_windows = n_samples // window_samples
    if n_windows == 0:
        return np.zeros((n_channels, 0), dtype=bool)

    data = raw_uv_unfiltered[:, : n_windows * window_samples]
    with np.errstate(invalid='ignore'):
        clipped = (np.abs(data - CLIP_LOW_UV) <= tolerance_uv) | (
            np.abs(data - CLIP_HIGH_UV) <= tolerance_uv
        )
    return ~clipped.reshape(n_channels, n_windows, window_samples).any(axis=2)


def detect_valid_windows(
    data_uv: np.ndarray,
    unfiltered_uv: Optional[np.ndarray] = None,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
) -> np.ndarray:
    """
    振幅判定とクリップ判定をANDで合成した有効マスク（n_channels, n_windows）を返す。

    unfiltered_uv（フィルタ前の生値）がNoneならクリップ判定は行わない。
    HSIは判定に使わない。
    """
    valid = detect_artifact_windows(data_uv, window_samples, threshold_uv)
    if unfiltered_uv is not None:
        valid &= detect_clipped_windows(unfiltered_uv, window_samples)
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
    unfiltered_uv: Optional[np.ndarray] = None,
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
    unfiltered_uv : np.ndarray, optional
        フィルタ前の生値（n_channels, n_samples）。指定するとクリップ率も出し、Validに反映する
    hsi : np.ndarray, optional
        HSI値（n_channels, n_samples）。指定すると参考値の「HSI Rejected」列を出す
        （Validには影響しない）

    Returns
    -------
    pd.DataFrame
        Channel / Std / P2P (median) / Threshold / Rejected / Clipped / HSI Rejected / Valid の各列
        （Rejectedは振幅判定のみ、Clippedはunfiltered_uv指定時、HSI Rejectedはhsi指定時。
        Validは振幅とクリップを合成した有効率）
    """
    p2p = _window_p2p(data_uv, window_samples)
    amp_valid = detect_artifact_windows(data_uv, window_samples, threshold_uv)
    clip_valid = (
        detect_clipped_windows(unfiltered_uv, window_samples)
        if unfiltered_uv is not None
        else None
    )
    hsi_valid = detect_hsi_windows(hsi, window_samples) if hsi is not None else None
    combined = amp_valid & clip_valid if clip_valid is not None else amp_valid
    thresholds = channel_thresholds(p2p, threshold_uv) if p2p.size else None

    def _pct_rejected(mask: np.ndarray, idx: int) -> float:
        return float((~mask[idx]).mean() * 100) if p2p.size else np.nan

    rows = []
    for idx, name in enumerate(channel_names):
        has_windows = bool(p2p.size)
        row = {
            'Channel': name.replace('RAW_', ''),
            'Std': float(np.nanstd(data_uv[idx])),
            'P2P (median)': float(np.median(p2p[idx])) if has_windows else np.nan,
            'Threshold': float(thresholds[idx]) if has_windows else np.nan,
            'Rejected': _pct_rejected(amp_valid, idx),
        }
        if clip_valid is not None:
            row['Clipped'] = _pct_rejected(clip_valid, idx)
        if hsi_valid is not None:
            row['HSI Rejected'] = _pct_rejected(hsi_valid, idx)
        row['Valid'] = float(combined[idx].mean() * 100) if has_windows else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_artifacts(
    data_uv: np.ndarray,
    channel_names: list,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
    unfiltered_uv: Optional[np.ndarray] = None,
    hsi: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    アーチファクト検出結果をまとめて返す。

    unfiltered_uvを渡すと、valid_mask・rejected_ratioは振幅判定とクリップ判定を合成した結果になる。
    hsiは表示用の参考列にだけ使い、マスクには影響しない。

    Returns
    -------
    dict
        {
            'valid_mask': np.ndarray,        # (n_channels, n_windows) bool（振幅AND クリップ）
            'amplitude_statistics': DataFrame,
            'rejected_ratio': float,          # 全体の除外率 (0.0-1.0)
            'window_samples': int,
            'threshold_uv': float,
            'channels': list,
        }
    """
    valid = detect_valid_windows(data_uv, unfiltered_uv, window_samples, threshold_uv)
    rejected_ratio = float((~valid).mean()) if valid.size else 0.0

    return {
        'valid_mask': valid,
        'amplitude_statistics': calculate_amplitude_statistics(
            data_uv, channel_names, window_samples, threshold_uv,
            unfiltered_uv=unfiltered_uv, hsi=hsi,
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
