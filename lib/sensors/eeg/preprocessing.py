"""
EEGデータの前処理（MNE RawArray準備）
"""

import warnings
import pandas as pd
import mne

from .constants import DEFAULT_SFREQ

# Warningsとログを抑制
warnings.filterwarnings('ignore')
mne.set_log_level('ERROR')


def _estimate_sampling_rate(frame):
    """
    前処理後のデータフレームから正確なサンプリングレートを推定

    Parameters
    ----------
    frame : pd.DataFrame
        TimeStampをインデックスとするデータフレーム（重複除去・補間済み）

    Returns
    -------
    sfreq : float
        推定されたサンプリングレート（Hz）

    Notes
    -----
    従来の中央値ベースの推定方法は、Mind Monitor CSVのような
    異なるサンプリングレートが混在するデータでは不正確。
    実際のサンプル数と時間長から直接計算することで、
    MNEが認識する時間軸と実際の記録時間が一致する。
    """
    if frame.empty or len(frame) < 2:
        return DEFAULT_SFREQ

    duration_seconds = (frame.index.max() - frame.index.min()).total_seconds()
    if duration_seconds <= 0:
        return DEFAULT_SFREQ

    sfreq = len(frame) / duration_seconds
    return sfreq


def filter_eeg_quality(df, require_all_good=False):
    """
    EEG固有のHSI品質に基づいてデータをフィルタ

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム（HeadBandOnで事前フィルタ済みを推奨）
    require_all_good : bool, default False
        True: 全HSIチャネルが1.0 (Good) の行のみ抽出（高精度分析用）
        False: 全HSIチャネルが≤2.0 (Good/Medium) の行を抽出（日常分析用、推奨）

    Returns
    -------
    filtered : pd.DataFrame
        フィルタ後のデータフレーム
    quality_mask : pd.Series
        品質マスク（Trueが良好な行）
    """
    quality_mask = pd.Series(True, index=df.index)

    hsi_cols = [c for c in df.columns if c.startswith('HSI_')]
    if hsi_cols:
        hsi_values = df[hsi_cols].apply(pd.to_numeric, errors='coerce')
        if require_all_good:
            quality_mask &= hsi_values.eq(1).all(axis=1)
        else:
            quality_mask &= hsi_values.le(2).all(axis=1)

    if quality_mask.any():
        filtered = df.loc[quality_mask].copy()
        if filtered.empty:
            filtered = df.copy()
    else:
        filtered = df.copy()

    return filtered, quality_mask


def prepare_mne_raw(df, sfreq=None, apply_bandpass=True, apply_notch=True,
                    l_freq=1.0, h_freq=50.0, notch_freqs=(50, 60)):
    """
    MNE RawArrayの準備

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム（RAW_*列を含む、HeadBandOnで事前フィルタ済みを推奨）
    sfreq : float, optional
        サンプリングレート（Noneの場合は自動推定）
    apply_bandpass : bool, default True
        バンドパスフィルタを適用するか
    apply_notch : bool, default True
        ノッチフィルタを適用するか（電源ノイズ除去）
    l_freq : float, default 1.0
        バンドパスフィルタの下限周波数（Hz）
    h_freq : float, default 50.0
        バンドパスフィルタの上限周波数（Hz）
    notch_freqs : tuple, default (50, 60)
        ノッチフィルタで除去する周波数（Hz）

    Returns
    -------
    mne_dict : dict or None
        {
            'raw': mne.io.RawArray,
            'channels': list,
            'sfreq': float,
            'n_samples': int
        }
        RAWチャネルが見つからない場合はNone

    Notes
    -----
    **サンプリングレート**:
    Interaxon社とMind Monitor開発者の公式見解により、タイムスタンプは
    Bluetoothバッファリングの影響で不正確（重複や不均一な間隔）です。
    ハードウェアは正確に256Hzでサンプリングしているため、
    sfreq=Noneの場合は256Hz固定と仮定します。

    **フィルタリング**:
    デフォルトで以下のフィルタが適用されます：
    - バンドパスフィルタ: 1-50Hz（Museの有効帯域）
    - ノッチフィルタ: 50Hz, 60Hz（電源ノイズ除去）

    フィルタの適用順序：
    1. バンドパスフィルタ（ベースラインドリフト除去 + 高周波ノイズカット）
    2. ノッチフィルタ（電源ノイズの狭帯域除去）

    参考：
    - Mind Monitor Forums: https://mind-monitor.com/forums0/viewtopic.php?p=4644
    - muse-lsl: examples/utils.py (60Hzノッチフィルタ)
    - MNE-Python: Filtering and resampling tutorial
    """
    raw_cols = [c for c in df.columns if c.startswith('RAW_')]
    if not raw_cols:
        return None

    df_filtered, _ = filter_eeg_quality(df)

    # 数値変換
    numeric = df_filtered[raw_cols].apply(pd.to_numeric, errors='coerce')

    # 欠損値を補間（前後の値から線形補間）
    numeric = numeric.interpolate(method='linear').ffill().bfill()

    # サンプリングレートの決定
    # Interaxon社公式推奨：タイムスタンプは不正確（Bluetoothバッファリング起因）
    # ハードウェアは正確に256Hzでサンプリングしているため、256Hz固定と仮定
    # 参考: https://mind-monitor.com/forums0/viewtopic.php?p=4644
    if sfreq is None:
        sfreq = 256.0  # Muse標準サンプリングレート（MU-01は220Hz）

    # MNE RawArrayの作成
    ch_names = list(numeric.columns)
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    # μVスケールをVに変換
    data = numeric.to_numpy().T * 1e-6

    raw = mne.io.RawArray(data, info, copy='auto', verbose=False)

    # フィルタリング
    if sfreq > 2.0:
        # 1. バンドパスフィルタ（ベースラインドリフト + 高周波ノイズ除去）
        if apply_bandpass:
            # Museの有効周波数帯域に制限（デフォルト: 1-50Hz）
            nyquist = sfreq / 2.0
            safety_margin = max(0.5, nyquist * 0.05)

            effective_l_freq = l_freq if (l_freq is not None and l_freq < nyquist) else None
            effective_h_freq = h_freq if h_freq is not None else None

            if effective_h_freq is not None:
                max_h_freq = max(nyquist - safety_margin, 0)
                if max_h_freq <= 0:
                    effective_h_freq = None
                else:
                    effective_h_freq = min(effective_h_freq, max_h_freq)

            # Nyquist制限により同時指定できない場合はハイパスのみを優先
            if (effective_l_freq is not None and effective_h_freq is not None
                    and effective_h_freq <= effective_l_freq):
                effective_h_freq = None

            if effective_l_freq is not None or effective_h_freq is not None:
                raw = raw.filter(
                    l_freq=effective_l_freq,
                    h_freq=effective_h_freq,
                    fir_design='firwin',
                    verbose=False,
                )

        # 2. ノッチフィルタ（電源ノイズ除去）
        if apply_notch and notch_freqs:
            # 50Hz/60Hz電源ノイズを除去
            nyquist = sfreq / 2.0
            valid_notch_freqs = [freq for freq in notch_freqs if 0 < freq < nyquist]
            if valid_notch_freqs:
                raw = raw.notch_filter(freqs=valid_notch_freqs, verbose=False)

    return {
        'raw': raw,
        'channels': ch_names,
        'sfreq': sfreq,
        'n_samples': len(numeric)
    }
