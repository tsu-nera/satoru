"""
Alpha Power（Brain Recharge Score）解析モジュール

Mind MonitorのAlpha帯域データから、Muse Brain Recharge Score相当の
Alpha Power指標を算出する。

計算方式:
1. オフセット方式: Alpha_dB + offset
2. 線形方式（推奨）: slope × Alpha_dB + intercept

参考:
- Muse公式: Brain Recharge Scoreは Alpha brainwave activity を追跡
- Alpha帯域: 8-13 Hz
- 単位: dBx（Museアプリ独自のスケール）

注意:
- 係数は4セッションのデータから推定（2025-11時点）
- データが増えたら係数の再検証を推奨
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class AlphaPowerMethod(Enum):
    """Alpha Power計算方式"""

    OFFSET = 'offset'  # Alpha_dB + offset
    LINEAR = 'linear'  # slope × Alpha_dB + intercept


# デフォルトパラメータ（4セッションから推定、2025-11時点）
DEFAULT_PARAMS = {
    AlphaPowerMethod.OFFSET: {
        'offset': 54.0,  # 平均オフセット
    },
    AlphaPowerMethod.LINEAR: {
        'slope': 5.07,  # 線形回帰の傾き
        'intercept': 36.9,  # 線形回帰の切片
    },
}


@dataclass
class AlphaPowerResult:
    """Alpha Power解析結果を保持するデータクラス。"""

    score: float  # セッション全体のスコア (dBx)
    time_series: pd.Series  # Alpha Power時系列 (dBx)
    alpha_db: float  # 元のAlpha Power平均 (dB)
    statistics: pd.DataFrame
    metadata: dict


def calculate_alpha_power(
    df: pd.DataFrame,
    method: AlphaPowerMethod = AlphaPowerMethod.LINEAR,
    channels: Optional[List[str]] = None,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    offset: Optional[float] = None,
    resample_interval: str = '10s',
) -> AlphaPowerResult:
    """
    Alpha Power（Brain Recharge Score相当）を計算する。

    Parameters
    ----------
    df : pd.DataFrame
        Mind MonitorのCSVデータ（Alpha_TP9, Alpha_AF7等を含む）。
    method : AlphaPowerMethod
        計算方式。LINEAR（線形）またはOFFSET。デフォルトはLINEAR。
    channels : list, optional
        使用するAlphaチャネル。デフォルトは全4チャネル。
    slope : float, optional
        線形方式の傾き。Noneの場合はデフォルト値を使用。
    intercept : float, optional
        線形方式の切片。Noneの場合はデフォルト値を使用。
    offset : float, optional
        オフセット方式のオフセット値。Noneの場合はデフォルト値を使用。
    resample_interval : str
        時系列のリサンプル間隔。

    Returns
    -------
    AlphaPowerResult
        スコア、時系列、統計情報を含む解析結果。

    Notes
    -----
    Mind MonitorのAlpha値はBels単位で記録されている。
    dBへの変換: dB = Bels × 10

    線形方式（推奨）:
        score = slope × alpha_db + intercept
        デフォルト: score = 5.07 × alpha_db + 36.9
        R² = 0.927（4セッションから推定）

    オフセット方式:
        score = alpha_db + offset
        デフォルト: score = alpha_db + 54
    """
    if channels is None:
        channels = ['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']

    # チャネル存在確認
    missing = [ch for ch in channels if ch not in df.columns]
    if missing:
        raise ValueError(f'指定されたチャネルが見つかりません: {missing}')

    # Alpha値の取得（Bels単位）
    alpha_bels_df = df[channels].copy()

    # 時系列計算（各行の平均）
    alpha_bels_series = alpha_bels_df.mean(axis=1)

    # dBに変換
    alpha_db_series = alpha_bels_series * 10

    # タイムスタンプをインデックスに設定
    if 'TimeStamp' in df.columns:
        time_index = pd.to_datetime(df['TimeStamp'])
        alpha_db_series.index = time_index

    # リサンプリング
    if resample_interval:
        alpha_db_series = alpha_db_series.resample(resample_interval).mean()

    alpha_db_series = alpha_db_series.dropna()

    if alpha_db_series.empty:
        raise ValueError('Alpha Power時系列が空です。')

    # 平均Alpha dB
    alpha_db_mean = alpha_db_series.mean()

    # スコア計算
    if method == AlphaPowerMethod.LINEAR:
        _slope = slope if slope is not None else DEFAULT_PARAMS[method]['slope']
        _intercept = (
            intercept
            if intercept is not None
            else DEFAULT_PARAMS[method]['intercept']
        )
        score = _slope * alpha_db_mean + _intercept
        score_series = _slope * alpha_db_series + _intercept
        method_params = {'slope': _slope, 'intercept': _intercept}
    else:  # OFFSET
        _offset = offset if offset is not None else DEFAULT_PARAMS[method]['offset']
        score = alpha_db_mean + _offset
        score_series = alpha_db_series + _offset
        method_params = {'offset': _offset}

    # 統計計算
    stats_df = pd.DataFrame(
        [
            {'Metric': 'Score', 'Value': score},
            {'Metric': 'Alpha Power', 'Value': alpha_db_mean},
            {'Metric': 'Score Min', 'Value': score_series.min()},
            {'Metric': 'Score Max', 'Value': score_series.max()},
            {'Metric': 'Score Std', 'Value': score_series.std()},
        ]
    )

    # 前半・後半比較
    midpoint = score_series.index[0] + (score_series.index[-1] - score_series.index[0]) / 2
    first_half = score_series[score_series.index <= midpoint]
    second_half = score_series[score_series.index > midpoint]

    first_mean = first_half.mean() if not first_half.empty else np.nan
    second_mean = second_half.mean() if not second_half.empty else np.nan

    metadata = {
        'method': method.value,
        'method_params': method_params,
        'channels': channels,
        'alpha_db_mean': alpha_db_mean,
        'first_half_mean': first_mean,
        'second_half_mean': second_mean,
        'resample_interval': resample_interval,
        'estimation_note': '係数は4セッション(2025-11)から推定。追加データで要検証。',
    }

    return AlphaPowerResult(
        score=score,
        time_series=score_series,
        alpha_db=alpha_db_mean,
        statistics=stats_df,
        metadata=metadata,
    )


def calculate_alpha_power_from_raw(
    df: pd.DataFrame,
    method: AlphaPowerMethod = AlphaPowerMethod.LINEAR,
    channels: Optional[List[str]] = None,
    alpha_band: Tuple[float, float] = (8, 13),
    sfreq: int = 256,
    window_sec: float = 2.0,
    slope: Optional[float] = None,
    intercept: Optional[float] = None,
    offset: Optional[float] = None,
) -> AlphaPowerResult:
    """
    RAW EEGデータからAlpha Powerを計算する。

    Mind MonitorのAlpha列を使用せず、RAWデータからFFTで直接計算する方式。
    Alpha列が利用できない場合や、異なる帯域幅で計算したい場合に使用。

    Parameters
    ----------
    df : pd.DataFrame
        Mind MonitorのCSVデータ（RAW_TP9, RAW_AF7等を含む）。
    method : AlphaPowerMethod
        計算方式。デフォルトはLINEAR。
    channels : list, optional
        使用するRAWチャネル。デフォルトは全4チャネル。
    alpha_band : tuple
        Alpha帯域 (Hz)。デフォルトは (8, 13)。
    sfreq : int
        サンプリングレート (Hz)。デフォルトは256。
    window_sec : float
        FFT窓サイズ (秒)。
    slope : float, optional
        線形方式の傾き。
    intercept : float, optional
        線形方式の切片。
    offset : float, optional
        オフセット方式のオフセット値。

    Returns
    -------
    AlphaPowerResult
        スコア、時系列、統計情報を含む解析結果。
    """
    from scipy import signal

    if channels is None:
        channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

    # チャネル存在確認
    missing = [ch for ch in channels if ch not in df.columns]
    if missing:
        raise ValueError(f'指定されたチャネルが見つかりません: {missing}')

    # RAWデータ取得
    raw_data = {}
    min_len = float('inf')
    for ch in channels:
        data = df[ch].dropna().values
        raw_data[ch] = data
        min_len = min(min_len, len(data))

    min_len = int(min_len)

    # チャネル平均
    combined = np.zeros(min_len)
    for ch in channels:
        combined += raw_data[ch][:min_len]
    combined /= len(channels)

    # 時系列でAlpha Power計算
    window_samples = int(window_sec * sfreq)
    hop = window_samples // 2

    alpha_powers_db = []
    timestamps = []

    start_time = pd.to_datetime(df['TimeStamp'].iloc[0])

    for start in range(0, len(combined) - window_samples, hop):
        segment = combined[start:start + window_samples]
        freqs, psd = signal.welch(segment, fs=sfreq, nperseg=min(512, len(segment)))

        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        alpha_power_uv2 = np.trapz(psd[alpha_mask], freqs[alpha_mask])

        # dB変換（Bels × 10 と同等のスケールに調整）
        # Mind MonitorのAlpha列との整合性のため
        alpha_db = np.log10(alpha_power_uv2 + 1e-12)  # Bels相当
        alpha_powers_db.append(alpha_db * 10)  # dBに変換

        time_offset = start / sfreq
        timestamps.append(start_time + pd.Timedelta(seconds=time_offset))

    alpha_db_series = pd.Series(alpha_powers_db, index=timestamps)
    alpha_db_mean = alpha_db_series.mean()

    # スコア計算
    if method == AlphaPowerMethod.LINEAR:
        _slope = slope if slope is not None else DEFAULT_PARAMS[method]['slope']
        _intercept = (
            intercept
            if intercept is not None
            else DEFAULT_PARAMS[method]['intercept']
        )
        score = _slope * alpha_db_mean + _intercept
        score_series = _slope * alpha_db_series + _intercept
        method_params = {'slope': _slope, 'intercept': _intercept}
    else:  # OFFSET
        _offset = offset if offset is not None else DEFAULT_PARAMS[method]['offset']
        score = alpha_db_mean + _offset
        score_series = alpha_db_series + _offset
        method_params = {'offset': _offset}

    # 統計計算
    stats_df = pd.DataFrame(
        [
            {'Metric': 'Score', 'Value': score},
            {'Metric': 'Alpha Power', 'Value': alpha_db_mean},
            {'Metric': 'Score Min', 'Value': score_series.min()},
            {'Metric': 'Score Max', 'Value': score_series.max()},
            {'Metric': 'Score Std', 'Value': score_series.std()},
        ]
    )

    metadata = {
        'method': method.value,
        'method_params': method_params,
        'channels': channels,
        'alpha_band': alpha_band,
        'sfreq': sfreq,
        'window_sec': window_sec,
        'alpha_db_mean': alpha_db_mean,
        'source': 'raw_fft',
        'estimation_note': '係数は4セッション(2025-11)から推定。追加データで要検証。',
    }

    return AlphaPowerResult(
        score=score,
        time_series=score_series,
        alpha_db=alpha_db_mean,
        statistics=stats_df,
        metadata=metadata,
    )
