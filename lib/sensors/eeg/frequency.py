"""
周波数解析モジュール（PSD、スペクトログラム）
"""

import warnings

import mne
import numpy as np
from mne.time_frequency import AverageTFRArray

from .spectral_lines import suppress_spectral_lines

# Warningsとログを抑制
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


def calculate_psd(raw, fmin=0.5, fmax=50.0, n_fft=2048):
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
        FFT window size。大きいほど周波数解像度が向上（デフォルト2048で約0.125Hz解像度）

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


def calculate_psd_masked(raw, valid_mask, window_samples, fmax=50.0):
    """
    有効な窓だけを使ってパワースペクトル密度（PSD）を計算する

    calculate_psd と同じく n_fft = 2 × window_samples、50%オーバーラップのWelch法で、
    Welchの各セグメントはマスクの連続する2窓にちょうど重なる。
    チャネルごとに、両方の窓が有効なセグメントだけを平均する（無効窓をゼロ埋めしない）。
    平均後のPSDから線スペクトル（電源・機器由来の狭帯域ノイズ）を検出し、
    線のビンを補間で置き換える。線が無いチャネルは値が変わらない。

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE RawArrayオブジェクト
    valid_mask : np.ndarray
        (n_channels, n_windows) のbool配列。rawの先頭から window_samples ごとの窓
    window_samples : int
        マスクの窓長（サンプル）
    fmax : float
        最大周波数（Hz）

    Returns
    -------
    psd_dict : dict or None
        calculate_psd と同じキー（'spectrum' は None）に加え、
        'n_segments'（チャネル別の使用セグメント数）と
        'line_freqs'（チャネル別の除去した線の周波数Hz、無ければ空リスト）を持つ。
        有効なセグメントが1つも無いチャネルは出力から外す。全チャネルが該当すればNone
    """
    sfreq = raw.info['sfreq']
    nyquist = sfreq / 2.0
    fmax = min(fmax, max(nyquist - 1e-6, nyquist * 0.9))

    n_fft = 2 * window_samples
    n_windows = valid_mask.shape[1]
    n_segments = n_windows - 1
    if n_segments < 1:
        return None

    data = raw.get_data()[:, : n_windows * window_samples]
    segments = np.stack(
        [data[:, k * window_samples : k * window_samples + n_fft] for k in range(n_segments)]
    )
    segment_psds, freqs = mne.time_frequency.psd_array_welch(
        segments,
        sfreq=sfreq,
        fmax=fmax,
        n_fft=n_fft,
        n_per_seg=n_fft,
        n_overlap=0,
        verbose=False,
    )

    segment_valid = valid_mask[:, :-1] & valid_mask[:, 1:]  # (n_channels, n_segments)
    keep = [idx for idx in range(segment_valid.shape[0]) if segment_valid[idx].any()]
    if not keep:
        return None

    psds = np.stack(
        [segment_psds[segment_valid[idx], idx].mean(axis=0) for idx in keep]
    )
    psds = np.maximum(psds, np.finfo(float).eps)

    line_freqs = {}
    for row, idx in enumerate(keep):
        psds[row], lines = suppress_spectral_lines(psds[row], freqs)
        line_freqs[raw.ch_names[idx]] = lines
        if lines:
            print(
                f'  {raw.ch_names[idx]}: 線スペクトルを除去 '
                f'({", ".join(f"{f:.2f}" for f in lines)} Hz)'
            )

    return {
        'freqs': freqs,
        'psds': psds * 1e12,  # V²/Hz を μV²/Hz へ変換
        'channels': [raw.ch_names[idx] for idx in keep],
        'spectrum': None,
        'n_segments': {raw.ch_names[idx]: int(segment_valid[idx].sum()) for idx in keep},
        'line_freqs': line_freqs,
    }


def mask_spectrogram(tfr_dict, window_valid, window_samples, sfreq):
    """
    スペクトログラムの無効窓の時間帯をNaNにする（プロットでは空白になる）。

    Parameters
    ----------
    tfr_dict : dict
        calculate_spectrogram の戻り値（times は raw の先頭からの秒）
    window_valid : np.ndarray
        そのチャネルの (n_windows,) bool配列
    window_samples : int
        マスクの窓長（サンプル、マスク作成時のサンプリングレート基準）
    sfreq : float
        マスク作成時のサンプリングレート（Hz）
    """
    window_idx = np.floor(tfr_dict['times'] * sfreq / window_samples).astype(int)
    window_idx = np.clip(window_idx, 0, len(window_valid) - 1)
    invalid = ~window_valid[window_idx]
    tfr_dict['power'][:, invalid] = np.nan
    # MNEの描画経路はtfrオブジェクトのデータを使うので、そちらにも反映する
    tfr_obj = tfr_dict.get('tfr')
    if tfr_obj is not None:
        tfr_obj.data[0][:, invalid] = np.nan
    return tfr_dict


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
