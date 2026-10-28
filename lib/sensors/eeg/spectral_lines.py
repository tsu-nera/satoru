"""
線スペクトル（等間隔の櫛状ピーク）の検出と除去

電源・機器由来のノイズは、PSD上で基本周波数の整数倍に並ぶ細い線（櫛状）として現れる。
2026-10-29の前頭はHSIがGoodのまま4.5Hz間隔の線が全時間に乗っており、窓単位で除外すると
前頭のデータが全て失われる。そのため窓は捨てず、PSD上の線のビンだけを両側からの補間で置き換える。

除去するのは「等間隔の倍音列に属する狭いピーク」だけにする。狭いピーク単体では本物の脳波と
区別できないため（単独ルールの誤検出、2026-10-27 AF8の8.5Hz・2026-09-29 AF7の10.9Hzは
この人の本物の前頭αピークで、26分平均では幅が狭く出る）。孤立した狭いピークは、α帯域でも触らない。

判定は周波数（Hz）で行い、PSDの周波数分解能（セッションPSDは0.125Hz、窓ごとのPSDは0.25Hz）に
依存しない。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

# 線の候補とみなす局所中央値からの突出量（dB）。
# 実測: 2026-10-29の前頭の櫛は局所より+15〜20dB、2026-10-27のTP9の7Hz櫛は+17〜30dB。
# 6dBは、窓平均PSDの揺らぎ（数dB）と、幅のある本物のピークの肩を候補にしない下限。
LINE_MIN_PROMINENCE_DB = 6.0

# 櫛が確定した後、予想位置（k·f0）の取りこぼしを拾うときの突出量（dB）。
# 2026-10-29のAF8は4.5Hzの基本波だけが候補の基準に届かなかった
LINE_COMB_MIN_PROMINENCE_DB = 3.0

# 取りこぼしを探す位置の許容幅（Hz）。広げると櫛の隣の本物のαピーク
# （例: 9.0Hzの倍音の隣の8.5Hz）を拾ってしまうため、候補の照合より狭くする
LINE_COMB_RECHECK_TOL_HZ = 0.3

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

# マスクする範囲の上限（ピークからの片側Hz、両隣1ビンを含む）。Hann窓の線の裾は±1ビン強なので、
# これ以上は広げない。櫛の隣の狭い本物のαピーク（9.0Hzの倍音の0.5Hz隣の8.5Hz）を巻き込まないため
LINE_MAX_REACH_HZ = 0.375

# 櫛の基本周波数の探索範囲（Hz）と刻み。実測の基本波は4.5Hz（10-29）と約7.1Hz（10-27）
LINE_COMB_F0_RANGE_HZ = (2.0, 15.0)
LINE_COMB_F0_STEP_HZ = 0.01

# 候補が倍音 k·f0 に合致する許容幅: max(LINE_COMB_TOL_HZ, LINE_COMB_TOL_RATIO·f)。
# 2026-10-27のTP9の7Hz櫛は倍音の位置が±0.5Hz揺れ、高次ほど大きくずれた
LINE_COMB_TOL_HZ = 0.5
LINE_COMB_TOL_RATIO = 0.03

# 櫛と認める条件: 合致した倍音の種類数の下限と、探索範囲内の倍音位置のうち合致した割合の下限。
# 基本波が小さいほど倍音位置が密になり偶然の合致が増えるため、割合でも縛る
LINE_COMB_MIN_HARMONICS = 3
LINE_COMB_MIN_FILL = 0.5


def _run_bounds(above: np.ndarray, peak: int) -> Tuple[int, int]:
    """peakを含む連続したTrueの区間 [lo, hi] を返す。"""
    lo = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak
    while hi < above.size - 1 and above[hi + 1]:
        hi += 1
    return lo, hi


def _narrow_peak_extent(
    psd_db: np.ndarray, excess: np.ndarray, peak: int, df: float
) -> Optional[Tuple[int, int]]:
    """
    peakが狭いピークなら、マスクする範囲 [lo, hi]（裾と両隣1ビンを含む）を返す。幅広ならNone。

    幅はピークから LINE_WIDTH_DROP_DB（突出量が小さければその半分）下がった高さで測る。
    裾は局所中央値+3dBまで、ピークから離れる向きに単調減少する範囲だけ足す
    （αの肩など隣の山に乗り上げる手前で止める）。
    """
    n = len(psd_db)
    drop = min(LINE_WIDTH_DROP_DB, excess[peak] / 2)
    lo, hi = _run_bounds(psd_db > psd_db[peak] - drop, peak)
    if (hi - lo + 1) * df > LINE_MAX_WIDTH_HZ:
        return None

    max_skirt = max(0, int(LINE_MAX_REACH_HZ / df + 1e-9) - 1)
    lo = hi = peak
    while (
        lo > max(peak - max_skirt, 0)
        and excess[lo - 1] > LINE_SKIRT_DB
        and psd_db[lo - 1] <= psd_db[lo] + 1.0
    ):
        lo -= 1
    while (
        hi < min(peak + max_skirt, n - 1)
        and excess[hi + 1] > LINE_SKIRT_DB
        and psd_db[hi + 1] <= psd_db[hi] + 1.0
    ):
        hi += 1
    return max(lo - 1, 0), min(hi + 1, n - 1)


def _find_comb(candidates: np.ndarray, fmin: float, fmax: float) -> Optional[Tuple[float, np.ndarray]]:
    """
    候補周波数（Hz）から等間隔の櫛を探す。見つかれば (基本周波数f0, 合致した候補のbool配列)。

    f0ごとに、倍音 k·f0 に許容幅内で合致する候補の「倍音の種類数」を数え、最多のf0を採る
    （同数なら最大のf0。f0/2のような下位の倍音は同数で合致するため、大きい方が本物）。
    採用の条件は種類数が LINE_COMB_MIN_HARMONICS 以上、かつ探索範囲内の倍音位置の
    LINE_COMB_MIN_FILL 以上が埋まっていること。
    """
    if len(candidates) < LINE_COMB_MIN_HARMONICS:
        return None

    tol = np.maximum(LINE_COMB_TOL_HZ, LINE_COMB_TOL_RATIO * candidates)
    f0_grid = np.arange(
        LINE_COMB_F0_RANGE_HZ[0], LINE_COMB_F0_RANGE_HZ[1] + 1e-9, LINE_COMB_F0_STEP_HZ
    )
    best: Optional[Tuple[int, float, np.ndarray]] = None
    for f0 in f0_grid:
        k = np.rint(candidates / f0)
        matched = (k >= 1) & (np.abs(candidates - k * f0) <= tol)
        n_harmonics = len(np.unique(k[matched]))
        if n_harmonics < LINE_COMB_MIN_HARMONICS:
            continue
        n_slots = max(1, int(np.floor(fmax / f0)) - int(np.ceil(fmin / f0)) + 1)
        if n_harmonics / n_slots < LINE_COMB_MIN_FILL:
            continue
        if best is None or n_harmonics > best[0] or (n_harmonics == best[0] and f0 > best[1]):
            best = (n_harmonics, float(f0), matched)

    if best is None:
        return None

    # 合致した候補で f0 を最小二乗で精密化する（高次の倍音位置のずれを小さくする）
    _, f0, matched = best
    k = np.rint(candidates[matched] / f0)
    f0 = float((k * candidates[matched]).sum() / (k * k).sum())
    return f0, matched


def detect_spectral_lines(
    psd_1d: np.ndarray,
    freqs: np.ndarray,
    fmin: float = 3.0,
    fmax: float = 45.0,
) -> np.ndarray:
    """
    PSD上の線スペクトル（等間隔の倍音列に属する狭いピーク）のビンを検出する。

    1. dB変換したPSDを±2Hzの移動中央値と比べ、LINE_MIN_PROMINENCE_DBを超えて突出し、
       幅がLINE_MAX_WIDTH_HZ以下のピークを候補にする
    2. 候補が等間隔の倍音列（基本周波数2〜15Hz）を3種類以上作るときだけ櫛と認め、合致した候補を線とする
    3. 櫛が確定したら、予想位置 k·f0 の近くの取りこぼしを LINE_COMB_MIN_PROMINENCE_DB で拾う

    櫛に属さない孤立した狭いピークは、α帯域でも除去しない。

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
        bool配列（n_freqs,）。Trueが線のビン（裾と両隣1ビンを含む）
    """
    mask = np.zeros(len(freqs), dtype=bool)
    if len(freqs) < 5:
        return mask
    df = float(freqs[1] - freqs[0])
    n = len(freqs)

    psd_db = 10 * np.log10(np.maximum(np.asarray(psd_1d, dtype=float), 1e-30))
    half = max(1, int(round(LINE_MEDIAN_HALF_WIDTH_HZ / df)))
    local = median_filter(psd_db, size=2 * half + 1, mode='nearest')
    excess = psd_db - local

    in_range = (freqs >= fmin) & (freqs <= fmax)
    peaks = [
        int(i)
        for i in np.flatnonzero(in_range & (excess > LINE_MIN_PROMINENCE_DB))
        if excess[i] >= excess[max(i - 1, 0)] and excess[i] >= excess[min(i + 1, n - 1)]
    ]
    peaks.sort(key=lambda i: -excess[i])  # 突出の大きい順（裾に入る副次ピークを飛ばすため）

    # 狭い候補だけ残す
    extents = {}
    taken = np.zeros(n, dtype=bool)
    for peak in peaks:
        if taken[peak]:
            continue
        extent = _narrow_peak_extent(psd_db, excess, peak, df)
        if extent is None:
            continue
        extents[peak] = extent
        taken[extent[0] : extent[1] + 1] = True

    candidate_idx = np.array(sorted(extents), dtype=int)
    comb = _find_comb(freqs[candidate_idx], fmin, fmax) if candidate_idx.size else None
    if comb is None:
        return mask

    f0, matched = comb
    for peak in candidate_idx[matched]:
        lo, hi = extents[int(peak)]
        mask[lo : hi + 1] = True

    # 櫛が確定した後は、予想位置の近くにある弱い線も拾う
    tol_bins = max(1, int(round(LINE_COMB_RECHECK_TOL_HZ / df)))
    for k in range(int(np.ceil(fmin / f0)), int(np.floor(fmax / f0)) + 1):
        center = int(np.argmin(np.abs(freqs - k * f0)))
        window = np.arange(max(center - tol_bins, 0), min(center + tol_bins, n - 1) + 1)
        window = window[(excess[window] > LINE_COMB_MIN_PROMINENCE_DB) & ~mask[window]]
        if window.size == 0:
            continue
        peak = int(window[np.argmax(excess[window])])
        if excess[peak] < excess[max(peak - 1, 0)] or excess[peak] < excess[min(peak + 1, n - 1)]:
            continue
        extent = _narrow_peak_extent(psd_db, excess, peak, df)
        if extent is not None:
            mask[extent[0] : extent[1] + 1] = True

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
