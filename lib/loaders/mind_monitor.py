"""
Mind Monitor データローダー
Mind Monitor CSVの読み込みとデータ抽出の統合モジュール

このモジュールはMind Monitor固有のCSVフォーマットに依存しています。
"""

import pandas as pd
import numpy as np
from pathlib import Path

# EEG定数のインポート（get_eeg_data用）
from ..sensors.eeg.constants import DEFAULT_SFREQ


def load_mind_monitor_csv(csv_path, filter_headband=True, warmup_seconds=60.0):
    """
    Mind Monitor CSVファイルを読み込む（全センサー共通の基本前処理）

    Mind Monitor固有の仕様:
    - TimeStampカラムの存在を前提
    - HeadBandOnによるデバイス装着状態フィルタリング（全センサー共通）
    - Time_sec相対時間カラムの自動追加
    - ウォームアップ期間の除外（フィルタ・処理のエッジ効果回避）

    Parameters
    ----------
    csv_path : str or Path
        CSVファイルのパス
    filter_headband : bool
        True の場合、HeadBandOn=1（装着中）のデータのみ抽出
        全センサー（EEG, Optics, Heart Rate等）に共通のフィルタ
    warmup_seconds : float, default 60.0
        記録開始からの除外期間（秒）。信号処理のエッジ効果とデバイス装着直後の
        不安定性を回避するため、セッション冒頭のデータを除外します。
        EEG研究の標準的な慣行に従い、60秒を推奨。0にすると除外しません。

    Returns
    -------
    df : pd.DataFrame
        読み込んだデータフレーム（Time_sec列を追加）
    """
    df = pd.read_csv(csv_path, low_memory=False)

    # タイムスタンプをDatetime型に変換
    df['TimeStamp'] = pd.to_datetime(df['TimeStamp'])

    # 相対時間 (秒) を計算
    df['Time_sec'] = (df['TimeStamp'] - df['TimeStamp'].iloc[0]).dt.total_seconds()

    # HeadBandOnフィルタリング（全センサー共通）
    if filter_headband:
        df = df[df['HeadBandOn'] == 1].copy()

    # ウォームアップ期間の除外（全センサー共通）
    if warmup_seconds and warmup_seconds > 0:
        start_time = df['TimeStamp'].min()
        warmup_cutoff = start_time + pd.Timedelta(seconds=warmup_seconds)
        df = df[df['TimeStamp'] >= warmup_cutoff].copy()

    return df


def get_eeg_data(df):
    """
    EEGデータ（RAWチャネル）を取得

    Mind Monitor固有の仕様:
    - RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10カラムを検出
    - Time_secカラムを使用（存在しない場合はフォールバック）
    - TimeStampカラムを使用

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム

    Returns
    -------
    eeg_dict : dict or None
        {
            'TP9': np.ndarray,
            'AF7': np.ndarray,
            'AF8': np.ndarray,
            'TP10': np.ndarray,
            'time': np.ndarray,
            'timestamps': pd.DatetimeIndex
        }
        RAWチャネルが見つからない場合はNone
    """
    raw_cols = [c for c in df.columns if c.startswith('RAW_')]

    if not raw_cols:
        return None

    # タイムスタンプが存在しない場合は相対時間を生成
    if 'Time_sec' in df.columns:
        time = df['Time_sec'].values
    else:
        time = np.arange(len(df)) / DEFAULT_SFREQ

    eeg_dict = {'time': time}

    # タイムスタンプ情報
    if 'TimeStamp' in df.columns:
        eeg_dict['timestamps'] = df['TimeStamp']

    # 各チャネルのデータを取得
    for col in raw_cols:
        channel_name = col.replace('RAW_', '')
        eeg_dict[channel_name] = pd.to_numeric(df[col], errors='coerce').values

    return eeg_dict


def get_optics_data(df):
    """
    Opticsデータを取得（fNIRS/PPG用）

    Mind Monitor固有の仕様:
    - Optics1-16カラムを使用
    - 特定のチャネルマッピング（Optics1=730nm Left等）

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム

    Returns
    -------
    optics_dict : dict
        {
            'left_730': np.ndarray,
            'left_850': np.ndarray,
            'right_730': np.ndarray,
            'right_850': np.ndarray,
            'time': np.ndarray
        }
    """
    return {
        'left_730': df['Optics1'].values,   # 730nm Left Inner
        'left_850': df['Optics3'].values,   # 850nm Left Inner
        'right_730': df['Optics2'].values,  # 730nm Right Inner
        'right_850': df['Optics4'].values,  # 850nm Right Inner
        'time': df['Time_sec'].values
    }


def get_heart_rate_data(df):
    """
    心拍数データを取得

    Mind Monitor固有の仕様:
    - Heart_Rateカラムを使用
    - 0値を除外（Mind Monitor特有の動作）

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム

    Returns
    -------
    hr_dict : dict
        {
            'heart_rate': np.ndarray,
            'time': np.ndarray,
            'timestamps': np.ndarray
        }
    """
    df_hr = df[df['Heart_Rate'] > 0].copy()

    return {
        'heart_rate': df_hr['Heart_Rate'].values,
        'time': df_hr['Time_sec'].values,
        'timestamps': df_hr['TimeStamp'].values
    }


def get_data_summary(df):
    """
    データの概要統計を取得

    Mind Monitor固有の仕様:
    - Time_sec, Optics*, Heart_Rateカラムを使用

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム

    Returns
    -------
    summary : dict
        統計情報の辞書
        {
            'total_records': int,
            'duration_sec': float,
            'duration_min': float,
            'sampling_rate_hz': float,
            'optics_channels': int,
            'valid_heart_rate_records': int
        }
    """
    duration = df['Time_sec'].max()

    return {
        'total_records': len(df),
        'duration_sec': duration,
        'duration_min': duration / 60,
        'sampling_rate_hz': len(df) / duration if duration > 0 else 0,
        'optics_channels': sum(1 for i in range(1, 17) if f'Optics{i}' in df.columns and df[f'Optics{i}'].sum() > 0),
        'valid_heart_rate_records': (df['Heart_Rate'] > 0).sum(),
    }
