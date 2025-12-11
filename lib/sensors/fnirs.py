"""
fNIRS（functional Near-Infrared Spectroscopy）センサー解析ライブラリ
Modified Beer-Lambert LawによるHbO/HbR計算

fNIRSセンサーによる近赤外光計測データから脳血流を解析します。
"""

import numpy as np


# Modified Beer-Lambert Lawのパラメータ
EXTINCTION_COEF = {
    730: {'HbO': 1.4866, 'HbR': 3.8437},  # 730nm
    850: {'HbO': 2.5264, 'HbR': 1.7989},  # 850nm
}

# Differential Pathlength Factor (前頭葉・成人)
DPF = 6.0

# ソース-ディテクタ間距離 (cm)
SOURCE_DETECTOR_DISTANCE = 3.0

# Muse App準拠のスケール調整係数
SCALE_FACTOR = 10000.0


def calculate_hbo_hbr(optics_730nm, optics_850nm, baseline_duration=10, sampling_rate=64):
    """
    Modified Beer-Lambert Lawを使ってHbOとHbRを計算

    Parameters
    ----------
    optics_730nm : np.ndarray
        730nmの光強度 (µA)
    optics_850nm : np.ndarray
        850nmの光強度 (µA)
    baseline_duration : float
        ベースライン期間 (秒)
    sampling_rate : float
        サンプリングレート (Hz)

    Returns
    -------
    hbo : np.ndarray
        酸素化ヘモグロビン濃度変化 (µM)
    hbr : np.ndarray
        脱酸素化ヘモグロビン濃度変化 (µM)
    """
    optics_730nm = np.asarray(optics_730nm, dtype=float)
    optics_850nm = np.asarray(optics_850nm, dtype=float)

    # ベースライン計算用のサンプル数
    baseline_samples = int(baseline_duration * sampling_rate)

    def _compute_baseline(arr):
        finite_indices = np.flatnonzero(np.isfinite(arr) & (arr > 0))
        if finite_indices.size == 0:
            return None
        take_indices = finite_indices[:baseline_samples]
        baseline = np.nanmean(arr[take_indices]) if take_indices.size > 0 else np.nan
        if not np.isfinite(baseline) or baseline <= 0:
            # 先頭で十分なサンプルが取れない場合、全体の有限正値から平均を取る
            finite_values = arr[np.isfinite(arr) & (arr > 0)]
            baseline = np.nanmean(finite_values) if finite_values.size > 0 else np.nan
        if not np.isfinite(baseline) or baseline <= 0:
            return None
        return baseline

    baseline_730 = _compute_baseline(optics_730nm)
    baseline_850 = _compute_baseline(optics_850nm)

    if baseline_730 is None or baseline_850 is None:
        # 有効ベースラインが取得できない場合はNaN配列を返す
        length = optics_730nm.shape[0]
        nan_arr = np.full(length, np.nan)
        return nan_arr, nan_arr

    # 光学密度変化 (ΔOD) を計算
    delta_od_730 = np.full_like(optics_730nm, np.nan)
    delta_od_850 = np.full_like(optics_850nm, np.nan)

    valid_730 = np.isfinite(optics_730nm) & (optics_730nm > 0)
    valid_850 = np.isfinite(optics_850nm) & (optics_850nm > 0)

    delta_od_730[valid_730] = -np.log10(
        np.clip(optics_730nm[valid_730] / baseline_730, 1e-6, None)
    )
    delta_od_850[valid_850] = -np.log10(
        np.clip(optics_850nm[valid_850] / baseline_850, 1e-6, None)
    )

    # 消衰係数マトリックス
    eps_matrix = np.array([
        [EXTINCTION_COEF[730]['HbO'], EXTINCTION_COEF[730]['HbR']],
        [EXTINCTION_COEF[850]['HbO'], EXTINCTION_COEF[850]['HbR']]
    ])

    eps_inv = np.linalg.inv(eps_matrix)

    # 各時点でHbO, HbRを計算（ベクトル化）
    delta_od = np.column_stack([delta_od_730, delta_od_850])
    concentrations = (delta_od @ eps_inv.T) / (DPF * SOURCE_DETECTOR_DISTANCE)
    hbo = concentrations[:, 0]
    hbr = concentrations[:, 1]

    # 非有限値をNaNに置換
    hbo = np.where(np.isfinite(hbo), hbo, np.nan)
    hbr = np.where(np.isfinite(hbr), hbr, np.nan)

    # Muse App準拠のスケールに調整
    hbo = hbo * SCALE_FACTOR
    hbr = hbr * SCALE_FACTOR

    return hbo, hbr


def analyze_fnirs(optics_data):
    """
    左右半球のfNIRS解析を実行

    Parameters
    ----------
    optics_data : dict
        get_optics_data()の戻り値
        {
            'left_730': np.ndarray,
            'left_850': np.ndarray,
            'right_730': np.ndarray,
            'right_850': np.ndarray,
            'time': np.ndarray
        }

    Returns
    -------
    fnirs_results : dict
        {
            'left_hbo': np.ndarray,
            'left_hbr': np.ndarray,
            'right_hbo': np.ndarray,
            'right_hbr': np.ndarray,
            'time': np.ndarray,
            'stats': dict
        }
    """
    # 左半球
    left_hbo, left_hbr = calculate_hbo_hbr(
        optics_data['left_730'],
        optics_data['left_850']
    )

    # 右半球
    right_hbo, right_hbr = calculate_hbo_hbr(
        optics_data['right_730'],
        optics_data['right_850']
    )

    # 統計情報
    def safe_stats(arr):
        if np.all(~np.isfinite(arr)):
            return {
                'mean': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
            }
        return {
            'mean': np.nanmean(arr),
            'std': np.nanstd(arr),
            'min': np.nanmin(arr),
            'max': np.nanmax(arr),
        }

    left_hbo_stats = safe_stats(left_hbo)
    left_hbr_stats = safe_stats(left_hbr)
    right_hbo_stats = safe_stats(right_hbo)
    right_hbr_stats = safe_stats(right_hbr)

    # HbT (Total Hemoglobin) と HbD (Hemoglobin Difference) を計算
    left_hbt = left_hbo + left_hbr
    left_hbd = left_hbo - left_hbr
    right_hbt = right_hbo + right_hbr
    right_hbd = right_hbo - right_hbr

    left_hbt_stats = safe_stats(left_hbt)
    left_hbd_stats = safe_stats(left_hbd)
    right_hbt_stats = safe_stats(right_hbt)
    right_hbd_stats = safe_stats(right_hbd)

    # Laterality Index (LI) を計算
    # LI = (Right - Left) / (Right + Left)
    # HbOベースとHbDベースの両方を計算
    hbo_left_mean = left_hbo_stats['mean']
    hbo_right_mean = right_hbo_stats['mean']
    hbd_left_mean = left_hbd_stats['mean']
    hbd_right_mean = right_hbd_stats['mean']

    # HbOベースのLI
    if np.isfinite(hbo_left_mean) and np.isfinite(hbo_right_mean):
        denominator = hbo_left_mean + hbo_right_mean
        if abs(denominator) > 1e-6:
            li_hbo = (hbo_right_mean - hbo_left_mean) / denominator
        else:
            li_hbo = np.nan
    else:
        li_hbo = np.nan

    # HbDベースのLI
    if np.isfinite(hbd_left_mean) and np.isfinite(hbd_right_mean):
        denominator = hbd_left_mean + hbd_right_mean
        if abs(denominator) > 1e-6:
            li_hbd = (hbd_right_mean - hbd_left_mean) / denominator
        else:
            li_hbd = np.nan
    else:
        li_hbd = np.nan

    stats = {
        'left': {
            'hbo_mean': left_hbo_stats['mean'],
            'hbo_min': left_hbo_stats['min'],
            'hbo_max': left_hbo_stats['max'],
            'hbr_mean': left_hbr_stats['mean'],
            'hbr_min': left_hbr_stats['min'],
            'hbr_max': left_hbr_stats['max'],
            'hbt_mean': left_hbt_stats['mean'],
            'hbd_mean': left_hbd_stats['mean'],
        },
        'right': {
            'hbo_mean': right_hbo_stats['mean'],
            'hbo_min': right_hbo_stats['min'],
            'hbo_max': right_hbo_stats['max'],
            'hbr_mean': right_hbr_stats['mean'],
            'hbr_min': right_hbr_stats['min'],
            'hbr_max': right_hbr_stats['max'],
            'hbt_mean': right_hbt_stats['mean'],
            'hbd_mean': right_hbd_stats['mean'],
        },
        'laterality': {
            'li_hbo': li_hbo,
            'li_hbd': li_hbd,
        }
    }

    return {
        'left_hbo': left_hbo,
        'left_hbr': left_hbr,
        'right_hbo': right_hbo,
        'right_hbr': right_hbr,
        'time': optics_data['time'],
        'stats': stats
    }
