"""
線スペクトル（狭帯域の鋭いピーク）の検出と除去

電源・機器由来のノイズは、PSD上で基本周波数の整数倍に並ぶ細い線（櫛状）として現れる。
2026-10-29の前頭はHSIがGoodのまま4.5Hz間隔の線が全時間に乗っており、窓単位で除外すると
前頭のデータが全て失われる。そのため窓は捨てず、PSD上の線のビンだけを両側からの補間で置き換える。

幅のある山（本物のα波、幅およそ2Hz）は線ではないので除去しない。判定は周波数（Hz）で行い、
PSDの周波数分解能（セッションPSDは0.125Hz、窓ごとのPSDは0.25Hz）に依存しない。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

# 線とみなす局所中央値からの突出量（dB）。
# 実測: 2026-10-29の前頭の櫛は局所より+15〜20dB、2026-10-27のTP9の7Hz櫛は+17〜30dB。
# 6dBは、窓平均PSDの揺らぎ（数dB）と、幅のある本物のピークの肩を線と誤認しない下限。
LINE_MIN_PROMINENCE_DB = 6.0

# 線の裾を数えるときの局所中央値からの突出量（dB）。マスクする範囲の決定に使う
LINE_SKIRT_DB = 3.0

# 線の幅の上限（Hz）。実測の線は約1ビン幅（0.25Hz以下）、αのピークは約2Hz幅。
# 幅はピークから LINE_WIDTH_DROP_DB 下がった高さで測る（下記）。
LINE_MAX_WIDTH_HZ = 0.75

# 幅を測る高さ（ピークからの低下量dB）。局所中央値基準の3dB幅だと、強い線（+30dB）では
# Hann窓のサイドローブで裾が数ビン広がって誤判定し、αの肩に乗った線は肩と連結して幅広になる。
# ピーク基準にすれば、どちらも線自身の幅だけが測れる。突出量が小さい線では低下量を半分にする
LINE_WIDTH_DROP_DB = 6.0

# 局所中央値を取る半幅（Hz）
LINE_MEDIAN_HALF_WIDTH_HZ = 2.0

# 線の裾としてマスクする範囲の上限（ピークからの片側Hz）。肩に乗った線が幅広の山を巻き込まないため
LINE_MAX_SKIRT_HZ = 0.5


def _run_bounds(above: np.ndarray, peak: int) -> Tuple[int, int]:
    """peakを含む連続したTrueの区間 [lo, hi] を返す。"""
    lo = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak
    while hi < above.size - 1 and above[hi + 1]:
        hi += 1
    return lo, hi


def detect_spectral_lines(
    psd_1d: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 3.0,
    fmax: float = 45.0,
) -> np.ndarray:
    """
    PSD上の線スペクトルのビンを検出する。

    dB変換したPSDを±2Hzの移動中央値と比べ、LINE_MIN_PROMINENCE_DBを超えて突出し、
    かつ幅（ピークからLINE_WIDTH_DROP_DB下がった高さで測る）がLINE_MAX_WIDTH_HZ以下のピークを
    線とみなす。線のビン（裾と両隣1ビンを含む）をTrueにする。

    Parameters
    ----------
    psd_1d : np.ndarray
        1チャネルのPSD（n_freqs,）。線形パワー
    freqs : np.ndarray
        周波数配列（n_freqs,）、等間隔
    fmin, fmax : float
        線を探す周波数範囲（Hz）

    Returns
    -------
    np.ndarray
        bool配列（n_freqs,）。Trueが線のビン
    """
    mask = np.zeros(len(freqs), dtype=bool)
    if len(freqs) < 5:
        return mask
    df = float(freqs[1] - freqs[0])

    psd_db = 10 * np.log10(np.maximum(np.asarray(psd_1d, dtype=float), 1e-30))
    half = max(1, int(round(LINE_MEDIAN_HALF_WIDTH_HZ / df)))
    local = median_filter(psd_db, size=2 * half + 1, mode='nearest')
    excess = psd_db - local

    in_range = (freqs >= fmin) & (freqs <= fmax)
    candidates = np.flatnonzero(in_range & (excess > LINE_MIN_PROMINENCE_DB))
    # 局所最大のみを候補にし、突出の大きい順に処理する
    peaks = [
        i for i in candidates
        if excess[i] >= excess[max(i - 1, 0)] and excess[i] >= excess[min(i + 1, len(freqs) - 1)]
    ]
    peaks.sort(key=lambda i: -excess[i])

    max_skirt = int(round(LINE_MAX_SKIRT_HZ / df))
    for peak in peaks:
        if mask[peak]:
            continue
        drop = min(LINE_WIDTH_DROP_DB, excess[peak] / 2)
        lo, hi = _run_bounds(psd_db > psd_db[peak] - drop, peak)
        if (hi - lo + 1) * df > LINE_MAX_WIDTH_HZ:
            continue
        # 裾は局所中央値+3dBまで、ピークから離れる向きに単調減少する範囲だけ足す
        # （αの肩など隣の山に乗り上げる手前で止める）。さらに両隣1ビンを足す
        lo = hi = peak
        while (
            lo > max(peak - max_skirt, 0)
            and excess[lo - 1] > LINE_SKIRT_DB
            and psd_db[lo - 1] <= psd_db[lo] + 1.0
        ):
            lo -= 1
        while (
            hi < min(peak + max_skirt, len(freqs) - 1)
            and excess[hi + 1] > LINE_SKIRT_DB
            and psd_db[hi + 1] <= psd_db[hi] + 1.0
        ):
            hi += 1
        mask[max(lo - 1, 0) : min(hi + 2, len(freqs))] = True

    return mask


def line_frequencies(line_mask: np.ndarray, freqs: np.ndarray) -> List[float]:
    """線のビンの連続区間ごとに中心周波数（Hz）を返す。ログ表示用。"""
    result: List[float] = []
    idx = np.flatnonzero(line_mask)
    if idx.size == 0:
        return result
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        result.append(float(freqs[run].mean()))
    return result


def suppress_spectral_lines(
    psds: np.ndarray,
    freqs: np.ndarray,
    detect_from: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[float]]:
    """
    線のビンを、隣接する非線ビンからの線形補間（線形パワー）に置き換える。

    Parameters
    ----------
    psds : np.ndarray
        1チャネルのPSD（..., n_freqs）。複数行（窓やセグメント）を渡してよい
    freqs : np.ndarray
        周波数配列（n_freqs,）
    detect_from : np.ndarray, optional
        線の検出に使う参照PSD（n_freqs,）。例: 有効窓の平均PSD。
        Noneなら psds 自身（1次元のとき）で検出する。窓ごとのPSDは揺らぐので、
        複数行を渡すときは平均した参照PSDを渡すこと

    Returns
    -------
    cleaned : np.ndarray
        psds と同じshape。線が無ければ入力のコピー
    line_freqs : list of float
        検出した線の中心周波数（Hz）
    """
    psds = np.asarray(psds, dtype=float)
    reference = detect_from if detect_from is not None else psds
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 1:
        raise ValueError('detect_from を指定するか、1次元のPSDを渡してください')

    line_mask = detect_spectral_lines(reference, freqs)
    cleaned = psds.copy()
    if not line_mask.any() or line_mask.all():
        return cleaned, []

    rows = cleaned.reshape(-1, cleaned.shape[-1])
    for row in rows:
        row[line_mask] = np.interp(freqs[line_mask], freqs[~line_mask], row[~line_mask])
    return cleaned, line_frequencies(line_mask, freqs)
