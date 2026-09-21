"""
スライディング窓のバンドパワー時系列

時間セグメント分析（既定3分）より細かい分解能で脳波を追うための計算。
タップ打刻のようなイベントの前後数秒〜数十秒を見る用途を想定する。

アーチファクト判定は `lib.sensors.eeg.artifact` と同じ閾値を使い、窓ごとに
クリーンなチャネルだけを平均する。窓長も既定で `ARTIFACT_WINDOW_SAMPLES` に
揃えてあるため、レポート本体のバンドパワーと同じ前提で読める。
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.signal import welch

from .artifact import ARTIFACT_WINDOW_SAMPLES, _window_p2p, channel_thresholds
from .constants import FREQ_BANDS

# スライディング窓の刻み（サンプル）。256Hz で1秒。
# 窓長1024サンプル（4秒）に対して75%オーバーラップになるため、隣接窓は強く
# 相関する。窓数をそのまま標本数として検定に使わないこと。
SLIDING_STEP_SAMPLES = 256


def compute_sliding_band_power(
    data_uv: np.ndarray,
    sfreq: float,
    timestamps: Optional[Sequence] = None,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
    step_samples: int = SLIDING_STEP_SAMPLES,
) -> pd.DataFrame:
    """
    チャネル×サンプルの EEG から、スライディング窓のバンドパワーを計算する。

    各窓でチャネルごとに peak-to-peak 振幅を取り、`channel_thresholds()` の
    実効閾値を超えたチャネルをその窓から除外する。残ったチャネルの PSD を
    バンド内で平均し、dB に変換したうえでチャネル平均を返す。

    Parameters
    ----------
    data_uv : np.ndarray
        shape = (n_channels, n_samples)。単位はμV。フィルタ適用後を想定。
    sfreq : float
        サンプリングレート（Hz）。
    timestamps : sequence, optional
        `data_uv` の各サンプルに対応する時刻。与えると窓中心の時刻を
        `center_ts` 列に入れる。長さは `data_uv` のサンプル数と一致すること。
    window_samples : int
        窓長（サンプル）。既定は PSD の Welch 窓と同じ 1024。
    step_samples : int
        窓の刻み（サンプル）。

    Returns
    -------
    pd.DataFrame
        1窓1行。列は以下。
        - t_sec : 窓先頭の経過秒
        - center_ts : 窓中心の時刻（`timestamps` を与えた場合のみ）
        - n_clean_ch : その窓で採用したチャネル数
        - delta / theta / alpha / beta / gamma : バンドパワー（dB）

        クリーンなチャネルが1つも無い窓は、バンドパワーが NaN になる。

    Raises
    ------
    ValueError
        `data_uv` が2次元でない場合、窓長より短い場合、または
        `timestamps` の長さがサンプル数と一致しない場合。
    """
    if data_uv.ndim != 2:
        raise ValueError(f'data_uv は (n_channels, n_samples) の2次元配列であること: {data_uv.shape}')

    n_samples = data_uv.shape[1]
    if n_samples < window_samples:
        raise ValueError(f'サンプル数 {n_samples} が窓長 {window_samples} より短い')

    if timestamps is not None and len(timestamps) != n_samples:
        raise ValueError(f'timestamps の長さ {len(timestamps)} がサンプル数 {n_samples} と一致しない')

    # 実効閾値はレポートと同じく非重複窓の p2p 中央値から決める。
    # スライディング窓ごとに中央値を取り直すと、閾値が窓の汚れに引きずられる。
    thresholds = channel_thresholds(_window_p2p(data_uv, window_samples))

    band_names = [name.lower() for name in FREQ_BANDS]
    starts = np.arange(0, n_samples - window_samples + 1, step_samples)
    records = []

    for start in starts:
        segment = data_uv[:, start:start + window_samples]
        p2p = segment.max(axis=1) - segment.min(axis=1)
        clean = p2p <= thresholds

        record = {'t_sec': start / sfreq, 'n_clean_ch': int(clean.sum())}
        if timestamps is not None:
            record['center_ts'] = timestamps[start + window_samples // 2]

        if clean.any():
            freqs, psd = welch(segment[clean], fs=sfreq, nperseg=window_samples)
            for name, (low, high, _) in FREQ_BANDS.items():
                mask = (freqs >= low) & (freqs < high)
                band_mean = np.mean(psd[:, mask], axis=1)
                record[name.lower()] = float(np.mean(10 * np.log10(band_mean)))
        else:
            for name in band_names:
                record[name] = np.nan

        records.append(record)

    columns = ['t_sec'] + (['center_ts'] if timestamps is not None else []) + ['n_clean_ch'] + band_names
    return pd.DataFrame(records, columns=columns)
