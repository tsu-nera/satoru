"""
RAW EEGからバンドパワーを計算するモジュール

Mind Monitor互換のバンドパワー列（Delta_TP9, Theta_AF7等）を
RAW EEGデータから計算して付与する。

Mind Monitorの仕様:
- FFTウィンドウ: 256サンプル（1秒 @ 256Hz）
- 出力: log10(バンド内パワーの合計)
- 更新頻度: 約2-3Hz（前方フィル）
- 値域: 約0-2
"""

import numpy as np
import pandas as pd
from scipy.signal import welch

from .constants import FREQ_BANDS, DEFAULT_SFREQ

# チャネル名マッピング: RAW_XX -> XX
RAW_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']


def compute_band_powers_from_raw(
    df: pd.DataFrame,
    sfreq: float = DEFAULT_SFREQ,
    window_sec: float = 1.0,
    step_sec: float = 0.5,
) -> pd.DataFrame:
    """
    RAW EEGからバンドパワー列を計算してDataFrameに付与する。

    Mind Monitor互換のlog10(absolute_power)形式で出力。

    Parameters
    ----------
    df : pd.DataFrame
        RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10 列を含むDataFrame。
    sfreq : float
        サンプリングレート（Hz）。
    window_sec : float
        FFTウィンドウ長（秒）。
    step_sec : float
        FFTウィンドウのステップ（秒）。

    Returns
    -------
    pd.DataFrame
        バンドパワー列が追加されたDataFrame（元のDataFrameは変更しない）。
    """
    df = df.copy()
    window_samples = int(sfreq * window_sec)
    step_samples = int(sfreq * step_sec)
    n_samples = len(df)

    for ch in RAW_CHANNELS:
        raw_col = f'RAW_{ch}'
        if raw_col not in df.columns:
            continue

        raw_data = pd.to_numeric(df[raw_col], errors='coerce').values

        # 各バンドのパワー時系列を初期化
        band_values = {band: np.full(n_samples, np.nan) for band in FREQ_BANDS}

        # スライディングウィンドウでFFT計算
        for start in range(0, n_samples - window_samples + 1, step_samples):
            end = start + window_samples
            segment = raw_data[start:end]

            if np.any(np.isnan(segment)):
                continue

            # Welch PSD
            freqs, psd = welch(
                segment,
                fs=sfreq,
                nperseg=min(window_samples, 256),
                noverlap=None,
            )

            # 各バンドのパワー（周波数帯域内のPSD合計）
            freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            for band, (f_low, f_high, _) in FREQ_BANDS.items():
                mask = (freqs >= f_low) & (freqs < f_high)
                power = np.sum(psd[mask]) * freq_res
                if power > 0:
                    band_values[band][start] = np.log10(power)

        # 前方フィルで埋める（Mind Monitorと同様）
        for band in FREQ_BANDS:
            col_name = f'{band}_{ch}'
            series = pd.Series(band_values[band])
            df[col_name] = series.ffill().values

    return df


def needs_band_power_computation(df: pd.DataFrame) -> bool:
    """
    バンドパワー列が空かどうかを判定する。

    Parameters
    ----------
    df : pd.DataFrame
        チェック対象のDataFrame。

    Returns
    -------
    bool
        バンドパワー列が空（NaN/0のみ）の場合True。
    """
    for band in FREQ_BANDS:
        for ch in RAW_CHANNELS:
            col = f'{band}_{ch}'
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce')
                if vals.notna().any() and (vals != 0).any():
                    return False
    return True
