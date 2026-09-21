"""
タップ連動のEEG解析

タップ打刻を基準に、その前後のバンドパワーを比べる。単一セッションの主観指標は
`lib.mind_wandering` が扱い、ここではEEGとの対応付けだけを担当する。

1セッション内ではタップ頻度もバンドパワーも時間とともに動くため、素の値で
比べると「タップ前後の差」と「時間経過」を分離できない。`detrend_and_zscore()`
で時間の一次成分を落としてから比較すること。

窓はオーバーラップしており隣接窓が強く相関するので、窓数をそのまま標本数として
検定に使わない。検定の単位はセッションにする。
"""

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

# MW窓: タップ直前のどこを「逸脱に気づく直前」とみなすか（秒）。タップは
# 気づいた瞬間であり逸脱の開始ではないため、直前側に幅を取る。
TAP_MW_WINDOW_S = (1.0, 15.0)
# 対照窓: どのタップからもこの秒数以上離れた窓だけを対照に使う
TAP_CONTROL_MIN_GAP_S = 40.0
# セッション冒頭の除外秒数。レポート本体の warmup と揃える
TAP_ANALYSIS_WARMUP_S = 60.0
# ばらつきが「実質ゼロ」と判定する相対許容値。浮動小数点誤差しか残っていない
# 残差をz化すると、誤差が増幅されて意味のない値になる。
NEGLIGIBLE_SPREAD_RATIO = 1e-9

#
# MW窓: タップ直前のどこを「逸脱に気づく直前」とみなすか（秒）。タップは
# 気づいた瞬間であり逸脱の開始ではないため、直前側に幅を取る。
TAP_MW_WINDOW_S = (1.0, 15.0)
# 対照窓: どのタップからもこの秒数以上離れた窓だけを対照に使う
TAP_CONTROL_MIN_GAP_S = 40.0
# セッション冒頭の除外秒数。レポート本体の warmup と揃える
TAP_ANALYSIS_WARMUP_S = 60.0
# ばらつきが「実質ゼロ」と判定する相対許容値。浮動小数点誤差しか残っていない
# 残差をz化すると、誤差が増幅されて意味のない値になる。
NEGLIGIBLE_SPREAD_RATIO = 1e-9


def detrend_and_zscore(power_df: pd.DataFrame, bands: list) -> pd.DataFrame:
    """
    セッション内の時間トレンドを除いてからz化する。

    1セッション内ではタップ頻度もバンドパワーも時間とともに変化するため、
    素の値で比較すると「タップ前後の差」と「時間経過」が分離できない。
    各バンドを `t_sec` に対して線形回帰し、その残差をz化することで
    時間の一次成分を落とす。

    Parameters
    ----------
    power_df : pd.DataFrame
        `compute_sliding_band_power()` の戻り値。`t_sec` と各バンド列を持つ。
    bands : list of str
        対象の列名。

    Returns
    -------
    pd.DataFrame
        `{band}_z` 列を追加したコピー。有効値が2点未満、または残差のばらつきが
        元の値のスケールに対して無視できるバンド（完全な直線など）は NaN になる。
    """
    out = power_df.copy()
    for band in bands:
        values = out[band].to_numpy(dtype=float)
        elapsed = out['t_sec'].to_numpy(dtype=float)
        valid = np.isfinite(values)

        residual = np.full(len(values), np.nan)
        if valid.sum() >= 2:
            slope, intercept = np.polyfit(elapsed[valid], values[valid], 1)
            residual[valid] = values[valid] - (slope * elapsed[valid] + intercept)

        # 残差のばらつきが元の値のスケールに対して無視できる場合（完全な直線など）、
        # z化すると浮動小数点誤差だけを増幅することになるので NaN にする。
        std = np.nanstd(residual)
        scale = np.nanstd(values[valid]) if valid.any() else 0.0
        negligible = std <= max(abs(scale), 1.0) * NEGLIGIBLE_SPREAD_RATIO
        out[f'{band}_z'] = np.nan if negligible else (residual - np.nanmean(residual)) / std

    return out


def label_tap_context(
    power_df: pd.DataFrame,
    tap_times: Sequence,
    mw_window_s: tuple = TAP_MW_WINDOW_S,
    control_min_gap_s: float = TAP_CONTROL_MIN_GAP_S,
) -> pd.DataFrame:
    """
    各窓をタップとの時間関係で MW / 対照 / その他に分類する。

    Parameters
    ----------
    power_df : pd.DataFrame
        `center_ts` 列を持つ窓ごとのDataFrame。
    tap_times : sequence of datetime-like
        タップ時刻。`event == 'tap'` の行のみを渡すこと。
    mw_window_s : tuple of float
        MW窓の下限・上限（秒）。窓中心から見て「次のタップまで」がこの範囲。
    control_min_gap_s : float
        対照窓の条件。前後どちらのタップからもこの秒数以上離れていること。

    Returns
    -------
    pd.DataFrame
        以下を追加したコピー。
        - to_next_tap : 次のタップまでの秒数（無ければ inf）
        - abs_nearest_tap : 最も近いタップまでの秒数（絶対値）
        - tap_context : 'mw' / 'control' / 'other'

        `tap_times` が空の場合、全窓が 'other' になる。
    """
    out = power_df.copy()
    taps = pd.to_datetime(pd.Series(list(tap_times))).to_numpy(dtype='datetime64[ns]')
    centers = pd.to_datetime(out['center_ts']).to_numpy(dtype='datetime64[ns]')

    if len(taps) == 0:
        out['to_next_tap'] = np.inf
        out['abs_nearest_tap'] = np.inf
        out['tap_context'] = 'other'
        return out

    # 窓中心から各タップまでの秒数（正 = タップはこの窓より後）
    delta = (taps[None, :] - centers[:, None]) / np.timedelta64(1, 's')
    upcoming = np.where(delta > 0, delta, np.inf)
    out['to_next_tap'] = upcoming.min(axis=1)
    out['abs_nearest_tap'] = np.abs(delta).min(axis=1)

    low, high = mw_window_s
    is_mw = out['to_next_tap'].between(low, high)
    is_control = out['abs_nearest_tap'] >= control_min_gap_s
    out['tap_context'] = np.where(is_mw, 'mw', np.where(is_control, 'control', 'other'))

    return out


def contrast_mw_vs_control(labeled_df: pd.DataFrame, bands: list) -> Dict[str, float]:
    """
    MW窓と対照窓の差（z化済みの値の平均差）をバンドごとに返す。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        `detrend_and_zscore()` と `label_tap_context()` を通したDataFrame。
    bands : list of str
        対象バンド名。`{band}_z` 列を参照する。

    Returns
    -------
    dict
        バンド名 -> MW平均 − 対照平均（単位はSD）。どちらかの窓が
        1つも無いバンドは NaN。
    """
    mw = labeled_df[labeled_df['tap_context'] == 'mw']
    control = labeled_df[labeled_df['tap_context'] == 'control']

    result = {}
    for band in bands:
        column = f'{band}_z'
        if mw[column].notna().any() and control[column].notna().any():
            result[band] = float(mw[column].mean() - control[column].mean())
        else:
            result[band] = float('nan')

    return result


def peri_event_average(
    labeled_df: pd.DataFrame,
    tap_times: Sequence,
    band: str,
    lags_s: Sequence,
    tolerance_s: float = 1.5,
) -> np.ndarray:
    """
    タップ時刻を0とした各ラグの平均値（z化済み）を返す。

    Parameters
    ----------
    labeled_df : pd.DataFrame
        `detrend_and_zscore()` を通したDataFrame。`center_ts` を持つこと。
    tap_times : sequence of datetime-like
        タップ時刻。
    band : str
        バンド名。`{band}_z` 列を参照する。
    lags_s : sequence of float
        タップからのラグ（秒）。負がタップ前。
    tolerance_s : float
        ラグ位置に最も近い窓を採用する際の許容ずれ（秒）。これを超える場合は
        その（タップ, ラグ）を欠測として捨てる。

    Returns
    -------
    np.ndarray
        `lags_s` と同じ長さ。該当窓が1つも無いラグは NaN。
    """
    values = labeled_df[f'{band}_z'].to_numpy(dtype=float)
    centers = pd.to_datetime(labeled_df['center_ts']).to_numpy(dtype='datetime64[ns]')
    taps = pd.to_datetime(pd.Series(list(tap_times))).to_numpy(dtype='datetime64[ns]')

    means = []
    for lag in lags_s:
        picked = []
        for tap in taps:
            target = tap + np.timedelta64(int(lag * 1000), 'ms')
            offsets = np.abs((centers - target) / np.timedelta64(1, 's'))
            nearest = int(offsets.argmin())
            if offsets[nearest] <= tolerance_s and np.isfinite(values[nearest]):
                picked.append(values[nearest])
        means.append(float(np.mean(picked)) if picked else np.nan)

    return np.asarray(means)


def required_sessions(effects: Sequence, target_effect_sd: float) -> Optional[int]:
    """
    セッション別の効果量から、目的の効果を検出するのに必要なセッション数を返す。

    両側5%・検出力80%のペアード検定を前提にした近似（係数2.8）。効果量の
    標準偏差自体が少数サンプルからの推定なので、返る値の不確かさは大きい。
    区間ではなく点推定であることを前提に読むこと。

    Parameters
    ----------
    effects : sequence of float
        セッションごとの効果量（`contrast_mw_vs_control()` の値など）。
    target_effect_sd : float
        検出したい効果の大きさ（SD単位）。正の値。

    Returns
    -------
    int or None
        必要セッション数。有効な効果量が2つ未満、またはばらつきが実質ゼロの場合は None。

    Raises
    ------
    ValueError
        `target_effect_sd` が0以下の場合。
    """
    if target_effect_sd <= 0:
        raise ValueError(f'target_effect_sd は正の値であること: {target_effect_sd}')

    finite = np.asarray([e for e in effects if np.isfinite(e)], dtype=float)
    if len(finite) < 2:
        return None

    sd = float(np.std(finite, ddof=1))
    if sd <= max(abs(float(np.mean(finite))), 1.0) * NEGLIGIBLE_SPREAD_RATIO:
        return None

    return int(np.ceil((2.8 * sd / target_effect_sd) ** 2))
