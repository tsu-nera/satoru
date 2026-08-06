"""
統一的なStatistical DataFrameレイヤー

EEG解析のための統一的なバンドパワーおよび比率計算を提供する。
全ての解析でこのレイヤーを使用することで、計算の一貫性と効率性を確保する。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .sensors.eeg.artifact import (
    ARTIFACT_P2P_THRESHOLD_UV,
    ARTIFACT_WINDOW_SAMPLES,
    SEGMENT_MIN_VALID_RATIO,
    detect_artifact_windows,
    segment_valid_ratio,
)

# Spectral Entropy計算関数をインポート
from .sensors.eeg.spectral_entropy import _calculate_shannon_entropy

if TYPE_CHECKING:
    import mne

try:
    import mne
    from mne.time_frequency import psd_array_welch
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


def _remove_outliers(values: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Z-score外れ値除去。除去後0件なら元の値をそのまま返す。"""
    if len(values) > 3:
        z_scores = np.abs(stats.zscore(values))
        filtered = values[z_scores < threshold]
        if len(filtered) > 0:
            return filtered
    return values


def _append_stats(
    rows: list,
    series: pd.Series,
    *,
    category: str,
    metric_prefix: str,
    unit: str,
    display_prefix: str,
    display_suffix: str = '',
    include_median: bool = True,
    with_cv: bool = False,
) -> None:
    """dropna → 外れ値除去 → Mean/Median/Std（+CV）の統計行を rows に追加する。"""
    values = series.dropna()
    if len(values) == 0:
        return
    values = _remove_outliers(values)

    rows.append({
        'Category': category,
        'Metric': f'{metric_prefix}_Mean',
        'Value': values.mean(),
        'Unit': unit,
        'DisplayName': f'{display_prefix}平均{display_suffix}',
    })
    if include_median:
        rows.append({
            'Category': category,
            'Metric': f'{metric_prefix}_Median',
            'Value': values.median(),
            'Unit': unit,
            'DisplayName': f'{display_prefix}中央値{display_suffix}',
        })
    rows.append({
        'Category': category,
        'Metric': f'{metric_prefix}_Std',
        'Value': values.std(),
        'Unit': unit,
        'DisplayName': f'{display_prefix}標準偏差{display_suffix}',
    })
    if with_cv:
        mean = values.mean()
        rows.append({
            'Category': category,
            'Metric': f'{metric_prefix}_CV',
            'Value': values.std() / mean if mean > 0 else np.nan,
            'Unit': 'ratio',
            'DisplayName': f'{display_prefix}変動係数',
        })


def create_statistical_dataframe(
    raw: 'mne.io.RawArray',
    segment_minutes: int = 3,
    warmup_minutes: float = 0.0,
    session_start: Optional[pd.Timestamp] = None,
    fnirs_results: Optional[Dict] = None,
    hr_data: Optional[Dict] = None,
    df_timestamps: Optional[pd.Series] = None,
    df: Optional[pd.DataFrame] = None,
    hrv_result: Optional[object] = None,
    respiration_result: Optional[object] = None,
    artifact_threshold_uv: float = ARTIFACT_P2P_THRESHOLD_UV,
    window_samples: int = ARTIFACT_WINDOW_SAMPLES,
) -> Dict[str, pd.DataFrame]:
    """
    統一的なStatistical DataFrameを生成する。

    MNE Epochsを使用してセグメントごとのバンドパワーと比率を計算し、
    統計量とともに返す。全ての解析でこの関数を使用することで、
    一貫した計算結果を保証する。

    Parameters
    ----------
    raw : mne.io.RawArray
        MNE RawArrayオブジェクト
    segment_minutes : int, default 3
        セグメント長（分単位）
    warmup_minutes : float, default 0.0
        ウォームアップ期間（分単位）
    session_start : pd.Timestamp, optional
        セッション開始時刻（Noneの場合は現在時刻を使用）
    fnirs_results : dict, optional
        analyze_fnirs()の戻り値（fNIRSデータ）
    hr_data : dict, optional
        get_heart_rate_data()の戻り値（心拍データ）
    df_timestamps : pd.Series, optional
        元データのTimeStamp列（fNIRS/HR計算に必要）
    df : pd.DataFrame, optional
        Mind Monitor形式の元データ（Posture統計量計算に必要）
    hrv_result : HRVResult, optional
        calculate_hrv_standard_set()の戻り値（HRVデータ）
    respiration_result : RespirationResult, optional
        calculate_breathing_rate()の戻り値（呼吸データ）
    artifact_threshold_uv : float, default ARTIFACT_P2P_THRESHOLD_UV
        Welch窓内peak-to-peak振幅の除外閾値（μV）
    window_samples : int, default ARTIFACT_WINDOW_SAMPLES
        Welch窓長（サンプル）。アーチファクト判定の粒度も兼ねる。
        窓長を変えると除外率が大きく動くため、比較実験用に上書きできるようにしてある

    Returns
    -------
    dict
        {
            'band_powers': DataFrame,      # セグメント別バンドパワー時系列（dB）
            'band_ratios': DataFrame,      # セグメント別バンド比率時系列
            'spectral_entropy': DataFrame, # セグメント別Spectral Entropy時系列（正規化済み）
            'iaf': Series,                 # Individual Alpha Frequency時系列（Hz）
            'itf': Series,                 # Individual Theta Frequency時系列（Hz）
            'fnirs': DataFrame,            # セグメント別fNIRS時系列（HbO/HbR平均）
            'hr': DataFrame,               # セグメント別心拍数時系列
            'posture': DataFrame,          # セグメント別Posture統計量時系列
            'hrv': DataFrame,              # セグメント別HRV時系列（RMSSD）
            'respiration': DataFrame,      # セグメント別呼吸時系列（BR）
            'statistics': DataFrame,       # 統計サマリー（縦長形式）
            'quality': DataFrame           # セグメント別のアーチファクト除去率
        }

    Notes
    -----
    - PSDはWelch窓（既定1024サンプル = 4秒 @256Hz）単位で計算し、
      窓内peak-to-peak振幅が閾値を超えたチャネル×窓を除外してから平均する。
      チャネル単位で判定するため、耳側（TP9/TP10）だけが汚いセッションでも
      前頭（AF7/AF8）のデータは保持される
    - 残存率がSEGMENT_MIN_VALID_RATIOを下回るセグメントは全指標をNaNにする
    - バンドパワーはdB（10*log10(μV²)）で表現
    - 比率（対数）はdB差分（10*log10(A/B) = 10*log10(A) - 10*log10(B)）
    - 比率（実数）は10^(dB差分/10)で計算
    - 統計量計算時にZ-score外れ値除去（閾値3.0）を適用
    """
    if not MNE_AVAILABLE:
        raise ImportError('MNE-Pythonが必要です。pip install mne でインストールしてください。')

    # セッション開始時刻のデフォルト値
    if session_start is None:
        session_start = pd.Timestamp.now()

    # ウォームアップ期間をスキップ
    tmin_sec = warmup_minutes * 60.0
    tmax_sec = raw.times[-1]

    if tmin_sec >= tmax_sec:
        raise ValueError(f'ウォームアップ期間（{warmup_minutes}分）がデータ長を超えています。')

    # ウォームアップ後のデータをクロップ
    raw_cropped = raw.copy().crop(tmin=tmin_sec, tmax=tmax_sec)

    duration_sec = segment_minutes * 60.0

    # PSD計算（Welch法）
    sfreq = raw_cropped.info['sfreq']
    nyquist = sfreq / 2.0
    fmax = min(50.0, nyquist * 0.95)  # 安全マージン5%

    # μV単位のデータ（MNEはV単位で保持）
    data_uv = raw_cropped.get_data() * 1e6

    # 振幅ベースのアーチファクト検出（チャネル×Welch窓）
    valid_mask = detect_artifact_windows(
        data_uv,
        window_samples=window_samples,
        threshold_uv=artifact_threshold_uv,
    )
    n_windows = valid_mask.shape[1]

    if n_windows == 0:
        raise ValueError(
            f'PSD窓（{window_samples}サンプル）を構成できるだけのデータがありません。'
        )

    # 窓ごと・チャネルごとのPSDを一括計算
    # shape: (n_windows, n_channels, window_samples)
    windowed = (
        data_uv[:, : n_windows * ARTIFACT_WINDOW_SAMPLES]
        .reshape(data_uv.shape[0], n_windows, ARTIFACT_WINDOW_SAMPLES)
        .transpose(1, 0, 2)
    )
    window_psds, freqs = psd_array_welch(
        windowed,
        sfreq=sfreq,
        fmin=1.0,
        fmax=fmax,
        n_fft=ARTIFACT_WINDOW_SAMPLES,
        n_per_seg=ARTIFACT_WINDOW_SAMPLES,
        n_overlap=0,
        verbose=False,
    )

    # セグメント境界を窓インデックスで表現
    windows_per_segment = max(1, int(round(duration_sec * sfreq / ARTIFACT_WINDOW_SAMPLES)))
    n_segments = n_windows // windows_per_segment

    if n_segments == 0:
        raise ValueError('セグメントを構成できるだけのデータがありません。')

    # セグメントごとに、有効窓のみを平均してPSDを作る
    # 不良チャネル（そのセグメントで全窓が不良）はチャネル平均から除外する
    psds = np.full((n_segments, data_uv.shape[0], len(freqs)), np.nan)
    segment_valid_ratios = []

    for seg_idx in range(n_segments):
        w_start = seg_idx * windows_per_segment
        w_end = w_start + windows_per_segment
        seg_mask = valid_mask[:, w_start:w_end]
        segment_valid_ratios.append(segment_valid_ratio(valid_mask, w_start, w_end))

        for ch_idx in range(data_uv.shape[0]):
            ch_valid = seg_mask[ch_idx]
            if not ch_valid.any():
                continue
            psds[seg_idx, ch_idx] = window_psds[w_start:w_end][ch_valid, ch_idx].mean(axis=0)

    segment_valid_ratios = np.asarray(segment_valid_ratios)

    # 残存率が閾値を下回るセグメントは指標を算出しない（NaNのまま返す）
    segment_usable = segment_valid_ratios >= SEGMENT_MIN_VALID_RATIO

    # バンド定義（全バンド）
    bands = {
        'Delta': (1, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 50),
    }

    # タイムスタンプ生成
    timestamps = [
        session_start + pd.Timedelta(minutes=warmup_minutes) + pd.Timedelta(seconds=i * duration_sec)
        for i in range(n_segments)
    ]

    # バンドパワー計算（有効チャネル平均、dB変換）
    band_powers_dict = {}
    for band_name, (fmin_band, fmax_band) in bands.items():
        freq_mask = (freqs >= fmin_band) & (freqs < fmax_band)

        band_power_db = np.full(n_segments, np.nan)
        for seg_idx in np.flatnonzero(segment_usable):
            # 全窓が不良だったチャネルはNaNなのでnanmeanで除外する
            band_power = np.nanmean(psds[seg_idx][:, freq_mask])
            band_power_db[seg_idx] = 10 * np.log10(band_power + 1e-12)  # ゼロ除算回避

        band_powers_dict[band_name] = band_power_db

    # DataFrameに変換
    band_powers_df = pd.DataFrame(band_powers_dict, index=timestamps)

    # セグメント別の品質（アーチファクト除去の結果）
    quality_df = pd.DataFrame(
        {
            'valid_ratio': segment_valid_ratios[:n_segments],
            'excluded_ratio': 1.0 - segment_valid_ratios[:n_segments],
            'usable': segment_usable[:n_segments],
        },
        index=timestamps,
    )

    # Spectral Entropy計算（全チャネル平均）
    # 周波数範囲でマスク（1-40Hz）
    freq_range = (1.0, 40.0)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

    se_values = []
    for seg_idx in range(n_segments):
        if not segment_usable[seg_idx]:
            se_values.append(np.nan)
            continue

        # このセグメントの有効チャネル平均PSD (n_freqs,)
        psd_avg = np.nanmean(psds[seg_idx], axis=0)

        # 周波数範囲でマスク
        psd_masked = psd_avg[freq_mask]

        # Shannon Entropy計算（spectral_entropy.pyの関数を使用）
        se = _calculate_shannon_entropy(psd_masked, normalize=True)
        se_values.append(se)

    # DataFrameに変換
    se_df = pd.DataFrame({'spectral_entropy': se_values}, index=timestamps)

    # IAF（Individual Alpha Frequency）計算
    # セグメントごとにアルファ帯域のピーク周波数を計算
    iaf_values = []
    alpha_range = (8.0, 13.0)

    for seg_idx in range(n_segments):
        if not segment_usable[seg_idx]:
            iaf_values.append(np.nan)
            continue

        # アルファ帯域のマスク
        alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
        alpha_freqs = freqs[alpha_mask]

        # 有効チャネルの平均PSD（アルファ帯域）
        psd_alpha_avg = np.nanmean(psds[seg_idx][:, alpha_mask], axis=0)

        # ピーク周波数を検出
        peak_idx = psd_alpha_avg.argmax()
        iaf = alpha_freqs[peak_idx]
        iaf_values.append(iaf)

    # IAF時系列をSeriesに変換
    iaf_series = pd.Series(iaf_values, index=timestamps)

    # ITF（Individual Theta Frequency）計算
    # セグメントごとにシータ帯域のピーク周波数を計算
    itf_values = []
    theta_range = (5.0, 7.0)

    for seg_idx in range(n_segments):
        if not segment_usable[seg_idx]:
            itf_values.append(np.nan)
            continue

        # シータ帯域のマスク
        theta_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
        theta_freqs = freqs[theta_mask]

        # 有効チャネルの平均PSD（シータ帯域）
        psd_theta_avg = np.nanmean(psds[seg_idx][:, theta_mask], axis=0)

        # ピーク周波数を検出
        peak_idx = psd_theta_avg.argmax()
        itf = theta_freqs[peak_idx]
        itf_values.append(itf)

    # ITF時系列をSeriesに変換
    itf_series = pd.Series(itf_values, index=timestamps)

    # バンド比率計算
    ratios_dict = {}

    # β/α比（覚醒度）
    ratios_dict['beta_alpha_db'] = band_powers_df['Beta'] - band_powers_df['Alpha']
    ratios_dict['beta_alpha'] = 10 ** (ratios_dict['beta_alpha_db'] / 10)

    # β/θ比（覚醒度・注意）
    ratios_dict['beta_theta_db'] = band_powers_df['Beta'] - band_powers_df['Theta']
    ratios_dict['beta_theta'] = 10 ** (ratios_dict['beta_theta_db'] / 10)

    # θ/α比（瞑想深度）
    ratios_dict['theta_alpha_db'] = band_powers_df['Theta'] - band_powers_df['Alpha']
    ratios_dict['theta_alpha'] = 10 ** (ratios_dict['theta_alpha_db'] / 10)

    # δ/β比（睡眠傾向）
    ratios_dict['delta_beta_db'] = band_powers_df['Delta'] - band_powers_df['Beta']
    ratios_dict['delta_beta'] = 10 ** (ratios_dict['delta_beta_db'] / 10)

    # γ/θ比（認知負荷）
    ratios_dict['gamma_theta_db'] = band_powers_df['Gamma'] - band_powers_df['Theta']
    ratios_dict['gamma_theta'] = 10 ** (ratios_dict['gamma_theta_db'] / 10)

    # 低周波数/高周波数比（瞑想深度・睡眠傾向）
    # 低周波数: δ+θ (1-8Hz) = 深いリラックス・睡眠・深い瞑想
    # 高周波数: α+β+γ (8-50Hz) = 覚醒・認知活動・注意
    low_freq_power = 10 ** (band_powers_df['Delta'] / 10) + 10 ** (band_powers_df['Theta'] / 10)
    high_freq_power = 10 ** (band_powers_df['Alpha'] / 10) + 10 ** (band_powers_df['Beta'] / 10) + 10 ** (band_powers_df['Gamma'] / 10)
    ratios_dict['low_high'] = low_freq_power / high_freq_power
    ratios_dict['low_high_db'] = 10 * np.log10(ratios_dict['low_high'])

    # DataFrameに変換
    band_ratios_df = pd.DataFrame(ratios_dict, index=timestamps)

    # fNIRSセグメント計算（オプション）
    fnirs_df = None
    if fnirs_results is not None and df_timestamps is not None:
        fnirs_records = []
        segment_delta = pd.Timedelta(seconds=duration_sec)

        # fNIRS時系列データを取得
        left_hbo = fnirs_results['left_hbo']
        left_hbr = fnirs_results['left_hbr']
        right_hbo = fnirs_results['right_hbo']
        right_hbr = fnirs_results['right_hbr']
        fnirs_time = fnirs_results['time']

        # 左右の平均を計算
        avg_hbo = (np.array(left_hbo) + np.array(right_hbo)) / 2.0
        avg_hbr = (np.array(left_hbr) + np.array(right_hbr)) / 2.0

        # fNIRS時系列にタイムスタンプを付与
        df_timestamps_dt = pd.to_datetime(df_timestamps)
        fnirs_session_start = df_timestamps_dt.iloc[0]
        fnirs_timestamps = fnirs_session_start + pd.to_timedelta(fnirs_time, unit='s')

        for ts in timestamps:
            end_ts = ts + segment_delta

            # 該当セグメントのfNIRSデータを取得
            mask = (fnirs_timestamps >= ts) & (fnirs_timestamps < end_ts)
            hbo_segment = avg_hbo[mask]
            hbr_segment = avg_hbr[mask]

            # NaN以外の値で平均を計算
            hbo_mean = np.nanmean(hbo_segment) if len(hbo_segment) > 0 else np.nan
            hbr_mean = np.nanmean(hbr_segment) if len(hbr_segment) > 0 else np.nan

            fnirs_records.append({
                'hbo_mean': hbo_mean,
                'hbr_mean': hbr_mean,
            })

        fnirs_df = pd.DataFrame(fnirs_records, index=timestamps)

    # HRセグメント計算（オプション）
    hr_df = None
    if hr_data is not None:
        hr_records = []
        segment_delta = pd.Timedelta(seconds=duration_sec)

        heart_rate = hr_data['heart_rate']
        hr_timestamps = pd.to_datetime(hr_data['timestamps'])

        for ts in timestamps:
            end_ts = ts + segment_delta

            # 該当セグメントの心拍データを取得
            mask = (hr_timestamps >= ts) & (hr_timestamps < end_ts)
            hr_segment = heart_rate[mask]

            # 平均を計算
            hr_mean = np.nanmean(hr_segment) if len(hr_segment) > 0 else np.nan

            hr_records.append({
                'hr_mean': hr_mean,
            })

        hr_df = pd.DataFrame(hr_records, index=timestamps)

    # 統計量計算（縦長形式）
    statistics_rows = []

    # バンドパワー統計
    for band_name in bands.keys():
        _append_stats(
            statistics_rows,
            band_powers_df[band_name],
            category='BandPower',
            metric_prefix=band_name,
            unit='dB',
            display_prefix=band_name,
            display_suffix=' (dB)',
        )

    # Spectral Entropy統計
    _append_stats(
        statistics_rows,
        se_df['spectral_entropy'],
        category='SpectralEntropy',
        metric_prefix='spectral_entropy',
        unit='normalized',
        display_prefix='Spectral Entropy',
        display_suffix=' (集中度)',
    )

    # IAF統計
    _append_stats(
        statistics_rows,
        iaf_series,
        category='IAF',
        metric_prefix='iaf',
        unit='Hz',
        display_prefix='IAF',
        display_suffix=' (Hz)',
        with_cv=True,
    )

    # ITF統計
    _append_stats(
        statistics_rows,
        itf_series,
        category='ITF',
        metric_prefix='itf',
        unit='Hz',
        display_prefix='ITF',
        display_suffix=' (Hz)',
        with_cv=True,
    )

    # バンド比率統計
    ratio_configs = [
        ('beta_alpha', 'β/α比', 'ratio', '覚醒度'),
        ('beta_theta', 'β/θ比', 'ratio', '覚醒度・注意'),
        ('theta_alpha', 'θ/α比', 'ratio', '瞑想深度'),
        ('delta_beta', 'δ/β比', 'ratio', '睡眠傾向'),
        ('gamma_theta', 'γ/θ比', 'ratio', '認知負荷'),
        ('beta_alpha_db', 'β/α比', 'dB', '覚醒度（対数）'),
        ('beta_theta_db', 'β/θ比', 'dB', '覚醒度・注意（対数）'),
        ('theta_alpha_db', 'θ/α比', 'dB', '瞑想深度（対数）'),
    ]

    for metric_key, ratio_name, unit, description in ratio_configs:
        _append_stats(
            statistics_rows,
            band_ratios_df[metric_key],
            category='BandRatio',
            metric_prefix=metric_key,
            unit=unit,
            display_prefix=ratio_name,
            display_suffix=f' ({unit}) - {description}',
        )

    # fNIRS統計（オプション）
    if fnirs_df is not None:
        # HbO統計
        _append_stats(
            statistics_rows,
            fnirs_df['hbo_mean'],
            category='fNIRS',
            metric_prefix='hbo',
            unit='µM',
            display_prefix='HbO',
            display_suffix=' (µM)',
            include_median=False,
        )

        # HbR統計
        _append_stats(
            statistics_rows,
            fnirs_df['hbr_mean'],
            category='fNIRS',
            metric_prefix='hbr',
            unit='µM',
            display_prefix='HbR',
            display_suffix=' (µM)',
            include_median=False,
        )

    # HR統計（オプション）
    if hr_df is not None:
        _append_stats(
            statistics_rows,
            hr_df['hr_mean'],
            category='HR',
            metric_prefix='hr',
            unit='bpm',
            display_prefix='HR',
            display_suffix=' (bpm)',
            include_median=False,
        )

    # Posture統計量計算（オプション）
    posture_df = pd.DataFrame()  # デフォルトは空のDataFrame
    if df is not None:
        try:
            from .sensors.imu import compute_posture_statistics
            posture_df = compute_posture_statistics(df, timestamps, segment_minutes)

            # Posture統計量をstatistics_rowsに追加
            posture_metrics = {
                'motion_index_mean': ('Posture', 'モーション指数(mean)', 'g'),
                'motion_index_max': ('Posture', 'モーション指数(max)', 'g'),
                'gyro_rms': ('Posture', 'ジャイロRMS', 'deg/s'),
                'gyro_rms_corrected': ('Posture', 'ジャイロRMS(補正)', 'deg/s'),
                'pitch_rms': ('Posture', 'Pitch RMS', 'deg/s'),
                'roll_rms': ('Posture', 'Roll RMS', 'deg/s'),
                'yaw_rms': ('Posture', 'Yaw RMS', 'deg/s'),
            }

            for col, (category, display_name, unit) in posture_metrics.items():
                if col in posture_df.columns:
                    values = posture_df[col].dropna()
                    if len(values) > 0:
                        # Z-score外れ値除去
                        if len(values) > 3:
                            z_scores = np.abs(stats.zscore(values))
                            filtered_values = values[z_scores < 3.0]
                            if len(filtered_values) > 0:
                                values = filtered_values

                        statistics_rows.extend([
                            {
                                'Category': category,
                                'Metric': f'{col}_Mean',
                                'Value': values.mean(),
                                'Unit': unit,
                                'DisplayName': f'{display_name}平均 ({unit})',
                            },
                            {
                                'Category': category,
                                'Metric': f'{col}_Std',
                                'Value': values.std(),
                                'Unit': unit,
                                'DisplayName': f'{display_name}標準偏差 ({unit})',
                            },
                        ])
        except Exception as e:
            # エラー時は空のDataFrameのまま継続
            print(f'警告: Posture統計量の計算に失敗しました ({e})')

    statistics_df = pd.DataFrame(statistics_rows)

    # HRVデータのセグメント別集計
    hrv_df = pd.DataFrame()
    if hrv_result is not None and hasattr(hrv_result, 'time_series'):
        try:
            hrv_ts = hrv_result.time_series
            # time_seriesは辞書形式で、'rmssd'キーがpd.Series（indexはdatetime）
            if isinstance(hrv_ts, dict) and 'rmssd' in hrv_ts:
                rmssd_series = hrv_ts['rmssd']
                # ウォームアップ期間を除外
                rmssd_series = rmssd_series[rmssd_series.index >= session_start + pd.Timedelta(minutes=warmup_minutes)]

                # セグメント別に集計（timestampsを使用）
                hrv_rows = []
                for start_ts in timestamps:
                    end_ts = start_ts + pd.Timedelta(minutes=segment_minutes)
                    mask = (rmssd_series.index >= start_ts) & (rmssd_series.index < end_ts)
                    segment_data = rmssd_series.loc[mask]

                    if len(segment_data) > 0:
                        hrv_rows.append({
                            'timestamp': start_ts,
                            'rmssd_mean': segment_data.mean(),
                            'rmssd_std': segment_data.std(),
                        })

                if hrv_rows:
                    hrv_df = pd.DataFrame(hrv_rows).set_index('timestamp')
        except Exception as e:
            print(f'警告: HRV統計量の計算に失敗しました ({e})')

    # 呼吸データのセグメント別集計
    respiration_df = pd.DataFrame()
    if respiration_result is not None and hasattr(respiration_result, 'time_series'):
        try:
            resp_ts = respiration_result.time_series
            # time_seriesはDataFrame形式で、'Time (min)'列と'BR (bpm)'列を持つ
            if isinstance(resp_ts, pd.DataFrame) and 'Time (min)' in resp_ts.columns and 'BR (bpm)' in resp_ts.columns:
                # 時刻をDatetimeIndexに変換（Time (min)は経過時間（分）なので、session_startからの経過時間として変換）
                resp_ts_indexed = resp_ts.copy()
                resp_ts_indexed.index = session_start + pd.to_timedelta(resp_ts_indexed['Time (min)'], unit='m')
                # ウォームアップ期間を除外
                resp_ts_indexed = resp_ts_indexed[resp_ts_indexed.index >= session_start + pd.Timedelta(minutes=warmup_minutes)]

                # セグメント別に集計（timestampsを使用）
                resp_rows = []
                for start_ts in timestamps:
                    end_ts = start_ts + pd.Timedelta(minutes=segment_minutes)
                    mask = (resp_ts_indexed.index >= start_ts) & (resp_ts_indexed.index < end_ts)
                    segment_data = resp_ts_indexed.loc[mask, 'BR (bpm)']

                    if len(segment_data) > 0:
                        resp_rows.append({
                            'timestamp': start_ts,
                            'br_mean': segment_data.mean(),
                            'br_std': segment_data.std(),
                        })

                if resp_rows:
                    respiration_df = pd.DataFrame(resp_rows).set_index('timestamp')
        except Exception as e:
            print(f'警告: 呼吸統計量の計算に失敗しました ({e})')

    return {
        'band_powers': band_powers_df,
        'band_ratios': band_ratios_df,
        'spectral_entropy': se_df,
        'iaf': iaf_series,
        'itf': itf_series,
        'fnirs': fnirs_df,
        'hr': hr_df,
        'posture': posture_df,
        'hrv': hrv_df,
        'respiration': respiration_df,
        'statistics': statistics_df,
        'quality': quality_df,
    }


def get_band_power_at_time(
    band_powers_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    band: str,
) -> float:
    """
    指定時間範囲のバンドパワー平均を取得する。

    Parameters
    ----------
    band_powers_df : pd.DataFrame
        create_statistical_dataframe()が返すband_powers DataFrame
    start_time : pd.Timestamp
        開始時刻
    end_time : pd.Timestamp
        終了時刻
    band : str
        バンド名（'Alpha', 'Beta', 'Theta', 'Delta', 'Gamma'）

    Returns
    -------
    float
        指定範囲の平均値（dB）、データがない場合はnp.nan
    """
    if band not in band_powers_df.columns:
        return np.nan

    mask = (band_powers_df.index >= start_time) & (band_powers_df.index < end_time)
    values = band_powers_df.loc[mask, band]

    if len(values) == 0:
        return np.nan

    return values.mean()


def get_band_ratio_at_time(
    band_ratios_df: pd.DataFrame,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    ratio: str,
) -> float:
    """
    指定時間範囲のバンド比率平均を取得する。

    Parameters
    ----------
    band_ratios_df : pd.DataFrame
        create_statistical_dataframe()が返すband_ratios DataFrame
    start_time : pd.Timestamp
        開始時刻
    end_time : pd.Timestamp
        終了時刻
    ratio : str
        比率名（'beta_alpha', 'beta_theta', 'theta_alpha', etc.）
        対数スケールの場合は'beta_alpha_db'など

    Returns
    -------
    float
        指定範囲の平均値、データがない場合はnp.nan
    """
    if ratio not in band_ratios_df.columns:
        return np.nan

    mask = (band_ratios_df.index >= start_time) & (band_ratios_df.index < end_time)
    values = band_ratios_df.loc[mask, ratio]

    if len(values) == 0:
        return np.nan

    return values.mean()
