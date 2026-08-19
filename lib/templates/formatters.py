"""
テンプレート用データフォーマッタ

データオブジェクトをテンプレートで使用可能なDataFrame形式に変換する。
プレゼンテーション層のデータ変換ロジックを集約。
"""

from typing import Any

import pandas as pd


def format_respiratory_stats(respiration_result: Any) -> pd.DataFrame:
    """
    RespirationResultをテーブル用DataFrameに変換

    Parameters
    ----------
    respiration_result : RespirationResult
        呼吸分析結果オブジェクト

    Returns
    -------
    pd.DataFrame
        Metric/Value/Unitの3カラムを持つDataFrame

    Examples
    --------
    >>> df = format_respiratory_stats(respiration_result)
    >>> print(df.columns.tolist())
    ['Metric', 'Value', 'Unit']
    """
    stats = []

    # 主推定値（通常はスペクトル法）を先頭に置く
    stats.append({
        'Metric': 'Mean Breathing Rate',
        'Value': respiration_result.breathing_rate,
        'Unit': 'bpm'
    })

    # RP (Respiratory Period) = 60 / BR
    from lib.sensors.ecg.respiration import calculate_respiratory_period
    stats.append({
        'Metric': 'Respiratory Period',
        'Value': calculate_respiratory_period(respiration_result.breathing_rate),
        'Unit': 's'
    })

    if hasattr(respiration_result, 'spectral_breathing_rate'):
        stats.append({
            'Metric': 'Breathing Rate (Spectral)',
            'Value': respiration_result.spectral_breathing_rate,
            'Unit': 'bpm'
        })

    if hasattr(respiration_result, 'breathing_rate_trough'):
        stats.append({
            'Metric': 'Breathing Rate (Trough)',
            'Value': respiration_result.breathing_rate_trough,
            'Unit': 'bpm'
        })

    stats.append({
        'Metric': 'Breathing Rate (Std)',
        'Value': respiration_result.breathing_rate_std,
        'Unit': 'bpm'
    })

    stats.append({
        'Metric': 'Peak Count',
        'Value': respiration_result.peak_count,
        'Unit': 'count'
    })

    stats.append({
        'Metric': 'Trough Count',
        'Value': respiration_result.trough_count,
        'Unit': 'count'
    })

    if getattr(respiration_result, 'rsa_cycle_count', 0):
        stats.append({
            'Metric': 'RSA Amplitude (peak-valley)',
            'Value': respiration_result.rsa_amplitude_mean,
            'Unit': 'ms'
        })
        stats.append({
            'Metric': 'RSA Amplitude (median)',
            'Value': respiration_result.rsa_amplitude_median,
            'Unit': 'ms'
        })
        stats.append({
            'Metric': 'RSA Amplitude (std)',
            'Value': respiration_result.rsa_amplitude_std,
            'Unit': 'ms'
        })
        stats.append({
            'Metric': 'RSA Cycles',
            'Value': respiration_result.rsa_cycle_count,
            'Unit': 'count'
        })

    return pd.DataFrame(stats)


def format_aperiodic_stats(aperiodic_info: dict) -> pd.DataFrame:
    """
    results['aperiodic'] dictをテーブル用DataFrameに変換

    Parameters
    ----------
    aperiodic_info : dict
        lib.report.steps_eeg.prepare_mne_and_spectral() が
        results['aperiodic'] に格納する辞書。
        {'offset', 'exponent', 'r_squared', 'error', 'n_peaks',
         'theta_peak', 'alpha_peak', 'theta_osc_db', 'alpha_osc_db'}

    Returns
    -------
    pd.DataFrame
        Metric/Value/Unitの3カラムを持つDataFrame

    Examples
    --------
    >>> df = format_aperiodic_stats(results['aperiodic'])
    >>> print(df.columns.tolist())
    ['Metric', 'Value', 'Unit']
    """
    def _decimal(value) -> str:
        """小数3桁。未測定はN/A（既存の format_score と同じ表記）"""
        return 'N/A' if value is None or pd.isna(value) else f'{value:.3f}'

    def _count(value) -> str:
        return 'N/A' if value is None or pd.isna(value) else f'{int(value)}'

    def _peak_cf(peak) -> str:
        """ピーク未検出は中心周波数が定義できない。数値を捏造せずN/Aを返す"""
        return 'N/A' if peak is None else _decimal(peak['center_hz'])

    stats = [
        {'Metric': 'Exponent', 'Value': _decimal(aperiodic_info['exponent']), 'Unit': 'a.u.'},
        {'Metric': 'Offset', 'Value': _decimal(aperiodic_info['offset']), 'Unit': 'a.u.'},
        {'Metric': 'Fit R²', 'Value': _decimal(aperiodic_info['r_squared']), 'Unit': 'ratio'},
        {'Metric': 'Fit Error (MAE)', 'Value': _decimal(aperiodic_info['error']), 'Unit': 'a.u.'},
        {'Metric': 'Detected Peaks', 'Value': _count(aperiodic_info['n_peaks']), 'Unit': 'count'},
        {'Metric': 'Theta Peak (CF)', 'Value': _peak_cf(aperiodic_info.get('theta_peak')), 'Unit': 'Hz'},
        {'Metric': 'Alpha Peak (CF)', 'Value': _peak_cf(aperiodic_info.get('alpha_peak')), 'Unit': 'Hz'},
        {'Metric': 'Theta Oscillatory Power', 'Value': _decimal(aperiodic_info['theta_osc_db']), 'Unit': 'dB'},
        {'Metric': 'Alpha Oscillatory Power', 'Value': _decimal(aperiodic_info['alpha_osc_db']), 'Unit': 'dB'},
    ]

    return pd.DataFrame(stats)


def format_aperiodic_peaks(peaks: pd.DataFrame) -> pd.DataFrame:
    """
    検出ピーク一覧を表示用の列名に整える

    Parameters
    ----------
    peaks : pd.DataFrame
        AperiodicResult.peaks（列: center_hz, height, bandwidth_hz）

    Returns
    -------
    pd.DataFrame
        表示用に列名を整えたDataFrame。入力が空なら空のまま返す。
    """
    if peaks is None or peaks.empty:
        return peaks

    return peaks.rename(columns={
        'center_hz': 'Center (Hz)',
        'height': 'Height',
        'bandwidth_hz': 'Bandwidth (Hz)',
    })
