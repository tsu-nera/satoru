"""
周波数解析モジュール（PSD、スペクトログラム）
"""

import warnings
import numpy as np
import mne
from mne.time_frequency import AverageTFRArray

# Warningsとログを抑制
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


def calculate_psd(raw, fmin=0.5, fmax=50.0, n_fft=512):
    """
    パワースペクトル密度（PSD）を計算

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE RawArrayオブジェクト
    fmin : float
        最小周波数（Hz）
    fmax : float
        最大周波数（Hz）
    n_fft : int
        FFT window size

    Returns
    -------
    psd_dict : dict
        {
            'freqs': np.ndarray,  # 周波数配列
            'psds': np.ndarray,   # PSD配列（チャネル × 周波数）（μV²/Hz）
            'channels': list,     # チャネル名
            'spectrum': mne.Spectrum  # MNE Spectrumオブジェクト
        }
    """
    sfreq = raw.info['sfreq']
    nyquist = sfreq / 2.0
    fmax = min(fmax, max(nyquist - 1e-6, nyquist * 0.9))

    n_fft = min(n_fft, len(raw.times))

    spectrum = raw.compute_psd(
        method='welch',
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        fmax=fmax,
        verbose=False,
    )

    freqs = spectrum.freqs
    psds = spectrum.get_data()
    psds = np.maximum(psds, np.finfo(float).eps)

    # V²/Hz を μV²/Hz へ変換
    psds_uv = psds * 1e12

    return {
        'freqs': freqs,
        'psds': psds_uv,
        'channels': raw.ch_names,
        'spectrum': spectrum
    }


def calculate_spectrogram(raw, freqs=None, channel='RAW_TP9', fmax=50.0):
    """
    スペクトログラムを計算（Time-Frequency Representation）

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE RawArrayオブジェクト
    freqs : np.ndarray, optional
        周波数配列（Noneの場合は自動生成）
    channel : str
        解析するチャネル名
    fmax : float
        最大周波数（Hz）

    Returns
    -------
    tfr_dict : dict or None
        {
            'power': np.ndarray,  # パワー配列（周波数 × 時間）（μV²）
            'freqs': np.ndarray,  # 周波数配列
            'times': np.ndarray,  # 時間配列
            'channel': str        # チャネル名
        }
        チャネルが見つからない場合はNone
    """
    if channel not in raw.ch_names:
        return None

    sfreq = raw.info['sfreq']
    nyquist = sfreq / 2.0
    fmax = min(fmax, max(nyquist - 1e-6, nyquist * 0.9))

    # 周波数配列の生成
    if freqs is None:
        freqs = np.arange(1.0, fmax, 0.5)

    n_cycles = freqs / 2.0

    # 指定チャネルのデータを取得
    ch_idx = raw.ch_names.index(channel)
    data_3d = raw.get_data()[ch_idx:ch_idx+1, :][np.newaxis, :, :]

    # TFRスペクトログラムを計算
    power = mne.time_frequency.tfr_array_morlet(
        data_3d,
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=n_cycles,
        output='power',
        verbose=False,
    )

    # V² を μV² へ変換
    power_uv = power[0, 0] * 1e12

    tfr_info = mne.create_info([channel], sfreq, ch_types=['eeg'])
    tfr_data = power_uv[np.newaxis, :, :]
    tfr_object = AverageTFRArray(
        info=tfr_info,
        data=tfr_data,
        times=raw.times,
        freqs=freqs,
        nave=1,
        method='morlet'
    )

    return {
        'power': power_uv,
        'freqs': freqs,
        'times': raw.times,
        'channel': channel,
        'tfr': tfr_object
    }


def calculate_spectrogram_all_channels(raw, freqs=None, fmax=50.0):
    """
    全チャネルのスペクトログラムを計算

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE RawArrayオブジェクト
    freqs : np.ndarray, optional
        周波数配列（Noneの場合は自動生成）
    fmax : float
        最大周波数（Hz）

    Returns
    -------
    results : dict
        {
            'RAW_TP9': tfr_dict,
            'RAW_AF7': tfr_dict,
            'RAW_AF8': tfr_dict,
            'RAW_TP10': tfr_dict
        }
        各チャネルのスペクトログラム結果の辞書
    """
    channels = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']
    results = {}

    for channel in channels:
        if channel in raw.ch_names:
            tfr_dict = calculate_spectrogram(raw, freqs=freqs, channel=channel, fmax=fmax)
            if tfr_dict is not None:
                results[channel] = tfr_dict

    return results
