"""
非周期成分（1/f）分離モジュール

specparam（旧FOOOF）でPSDを非周期成分（1/f）と周期成分（ピーク）に分離する。
バンドパワーやargmaxベースのピーク検出は、1/fが単調減少する帯域では
探索窓の端に張り付いた値を返してしまう（Issue #31）。specparamは
反復的にピークを除去しながら非周期成分を当てるため、この問題を避けられる。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from specparam import SpectralModel

# フィット範囲（Hz）。前処理で1Hzハイパスフィルタ・50/60Hzノッチフィルタが
# 掛かっているため、フィルタの遷移帯を避けるために2-40Hzを起点にする。
FIT_RANGE_HZ: Tuple[float, float] = (2.0, 40.0)

# フィット範囲の端からこの距離（Hz）以内のピークは、実際のピークではなく
# フィット窓の境界への張り付きとみなして棄却する。
PEAK_EDGE_MARGIN_HZ = 0.5

# この高さ（log10パワー単位）未満のピークは「検出なし」として扱う。
# specparamのmin_peak_height未満のピークも通過することがあるため、
# ここでも同じ閾値でガードする。
MIN_REPORT_PEAK_HEIGHT = 0.15


@dataclass
class AperiodicResult:
    """非周期成分フィット結果"""

    offset: float
    exponent: float
    r_squared: float
    error: float
    peaks: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=['center_hz', 'height', 'bandwidth_hz'])
    )
    fit_range: Tuple[float, float] = FIT_RANGE_HZ
    n_peaks: int = 0


def fit_aperiodic(
    freqs: np.ndarray,
    psd: np.ndarray,
    freq_range: Tuple[float, float] = FIT_RANGE_HZ,
    peak_width_limits: Tuple[float, float] = (1.0, 8.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.15,
    peak_threshold: float = 2.0,
) -> Optional[AperiodicResult]:
    """
    PSDに非周期成分（1/f）＋周期成分（ピーク）モデルをフィットする。

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）。
    psd : np.ndarray
        パワースペクトル密度（線形スケール、μV²/Hz等）。
    freq_range : tuple
        フィット範囲（Hz）。
    peak_width_limits, max_n_peaks, min_peak_height, peak_threshold
        specparam.SpectralModel に渡すパラメータ。

    Returns
    -------
    AperiodicResult or None
        フィットに失敗した場合はNoneを返す。

    Notes
    -----
    ピークは2段のガードを通過したもののみ `peaks` に含める:
    (a) フィット範囲の下限+PEAK_EDGE_MARGIN_HZ未満、または
        上限-PEAK_EDGE_MARGIN_HZ超のピークは棄却（窓端への張り付き対策）
    (b) 高さがMIN_REPORT_PEAK_HEIGHT未満のピークは棄却
    """
    try:
        fm = SpectralModel(
            peak_width_limits=peak_width_limits,
            max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height,
            peak_threshold=peak_threshold,
            aperiodic_mode='fixed',
            verbose=False,
        )
        fm.fit(freqs, psd, freq_range=freq_range)

        ap_params = fm.get_params('aperiodic')
        offset = float(ap_params[0])
        exponent = float(ap_params[1])

        metrics = fm.results.metrics.results
        r_squared = float(metrics.get('gof_rsquared', np.nan))
        error = float(metrics.get('error_mae', np.nan))
    except Exception as e:
        print(f'⚠️ specparamによる非周期成分フィットに失敗しました: {e}')
        return None

    low, high = freq_range
    peak_params = fm.get_params('peak')

    peak_rows = []
    if peak_params is not None and np.asarray(peak_params).size > 0:
        for cf, pw, bw in np.atleast_2d(peak_params):
            if cf < low + PEAK_EDGE_MARGIN_HZ or cf > high - PEAK_EDGE_MARGIN_HZ:
                continue
            if pw < MIN_REPORT_PEAK_HEIGHT:
                continue
            peak_rows.append({
                'center_hz': float(cf),
                'height': float(pw),
                'bandwidth_hz': float(bw),
            })

    peaks_df = pd.DataFrame(peak_rows, columns=['center_hz', 'height', 'bandwidth_hz'])

    return AperiodicResult(
        offset=offset,
        exponent=exponent,
        r_squared=r_squared,
        error=error,
        peaks=peaks_df,
        fit_range=(float(low), float(high)),
        n_peaks=len(peaks_df),
    )


def find_band_peak(result: AperiodicResult, band: Tuple[float, float]) -> Optional[dict]:
    """
    指定帯域内でガード通過済みの最も高いピークを返す。

    Parameters
    ----------
    result : AperiodicResult
        fit_aperiodic() の戻り値。
    band : tuple
        探索する周波数帯域（Hz）。

    Returns
    -------
    dict or None
        {'center_hz', 'height', 'bandwidth_hz'}。該当ピークが無ければNone。
    """
    if result is None or result.peaks.empty:
        return None

    low, high = band
    in_band = result.peaks[
        (result.peaks['center_hz'] >= low) & (result.peaks['center_hz'] <= high)
    ]
    if in_band.empty:
        return None

    best = in_band.loc[in_band['height'].idxmax()]
    return {
        'center_hz': float(best['center_hz']),
        'height': float(best['height']),
        'bandwidth_hz': float(best['bandwidth_hz']),
    }


def oscillatory_band_power(
    freqs: np.ndarray,
    psd: np.ndarray,
    result: AperiodicResult,
    band: Tuple[float, float],
) -> float:
    """
    帯域の振動性パワーを dB で返す（実測バンドパワー ÷ 非周期成分のみのバンドパワー）。

    非周期成分は specparam の fixed モードの定義に従って再構成する:
    log10(A(f)) = offset - exponent * log10(f)  すなわち  A(f) = 10^offset / f^exponent
    （specparam.modes.definitions の powerlaw_function に同じ。線形freq/対数power空間の線形モデル）

    Parameters
    ----------
    freqs : np.ndarray
        周波数配列（Hz）。
    psd : np.ndarray
        パワースペクトル密度（線形スケール）。
    result : AperiodicResult
        fit_aperiodic() の戻り値。
    band : tuple
        対象帯域（Hz）。

    Returns
    -------
    float
        振動性パワー（dB）。帯域内にデータが無い場合はNaN。
    """
    if result is None:
        return float('nan')

    low, high = band
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return float('nan')

    measured_power = float(np.mean(psd[mask]))
    aperiodic_component = (10 ** result.offset) / (freqs[mask] ** result.exponent)
    aperiodic_power = float(np.mean(aperiodic_component))

    if measured_power <= 0 or aperiodic_power <= 0:
        return float('nan')

    return float(10 * np.log10(measured_power / aperiodic_power))
