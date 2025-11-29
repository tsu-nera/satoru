"""
モーションセンサー（加速度計・ジャイロスコープ）解析ライブラリ

頭部の動きを検出し、EEGアーティファクトの識別に利用します。
"""

import numpy as np
import pandas as pd


# センサー定数
DEFAULT_SAMPLING_RATE = 52.0  # Hz (Muse S)

# 動作検出閾値
MOTION_THRESHOLDS = {
    'acc_std': 0.02,      # 加速度標準偏差の閾値 (g)
    'gyro_mean': 3.0,     # ジャイロ平均の閾値 (deg/s)
    'gyro_max': 10.0,     # ジャイロ最大値の閾値 (deg/s)
}

# カラム名
ACC_COLUMNS = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
GYRO_COLUMNS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']


def compute_magnitude(x, y, z):
    """
    3軸データからマグニチュード（大きさ）を計算

    Parameters
    ----------
    x, y, z : np.ndarray
        各軸のデータ

    Returns
    -------
    magnitude : np.ndarray
        ベクトルの大きさ sqrt(x^2 + y^2 + z^2)
    """
    return np.sqrt(x**2 + y**2 + z**2)


def get_motion_data(df):
    """
    DataFrameからモーションセンサーデータを抽出

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム

    Returns
    -------
    motion_dict : dict
        {
            'acc_x': np.ndarray,
            'acc_y': np.ndarray,
            'acc_z': np.ndarray,
            'acc_magnitude': np.ndarray,
            'gyro_x': np.ndarray,
            'gyro_y': np.ndarray,
            'gyro_z': np.ndarray,
            'gyro_magnitude': np.ndarray,
            'time': np.ndarray,
            'timestamps': pd.DatetimeIndex
        }
    """
    acc_x = df['Accelerometer_X'].values
    acc_y = df['Accelerometer_Y'].values
    acc_z = df['Accelerometer_Z'].values

    gyro_x = df['Gyro_X'].values
    gyro_y = df['Gyro_Y'].values
    gyro_z = df['Gyro_Z'].values

    return {
        'acc_x': acc_x,
        'acc_y': acc_y,
        'acc_z': acc_z,
        'acc_magnitude': compute_magnitude(acc_x, acc_y, acc_z),
        'gyro_x': gyro_x,
        'gyro_y': gyro_y,
        'gyro_z': gyro_z,
        'gyro_magnitude': compute_magnitude(gyro_x, gyro_y, gyro_z),
        'time': df['Time_sec'].values if 'Time_sec' in df.columns else np.arange(len(df)) / DEFAULT_SAMPLING_RATE,
        'timestamps': df['TimeStamp'] if 'TimeStamp' in df.columns else None
    }


def detect_motion(acc_std, gyro_mean, gyro_max=None, thresholds=None):
    """
    動作の有無を判定

    Parameters
    ----------
    acc_std : float
        加速度マグニチュードの標準偏差
    gyro_mean : float
        ジャイロマグニチュードの平均
    gyro_max : float, optional
        ジャイロマグニチュードの最大値
    thresholds : dict, optional
        カスタム閾値辞書

    Returns
    -------
    is_motion : bool
        動作が検出されたかどうか
    """
    if thresholds is None:
        thresholds = MOTION_THRESHOLDS

    is_motion = (
        acc_std > thresholds['acc_std'] or
        gyro_mean > thresholds['gyro_mean']
    )

    if gyro_max is not None and 'gyro_max' in thresholds:
        is_motion = is_motion or (gyro_max > thresholds['gyro_max'])

    return is_motion


def compute_motion_score(acc_std, gyro_mean, weights=None):
    """
    動作の強度を示すスコアを計算

    Parameters
    ----------
    acc_std : float
        加速度マグニチュードの標準偏差
    gyro_mean : float
        ジャイロマグニチュードの平均
    weights : dict, optional
        重み付け {'acc': float, 'gyro': float}

    Returns
    -------
    score : float
        動作スコア（高いほど動きが大きい）
    """
    if weights is None:
        weights = {'acc': 10.0, 'gyro': 0.1}

    return acc_std * weights['acc'] + gyro_mean * weights['gyro']


def analyze_motion_intervals(df, interval='10s', thresholds=None):
    """
    指定間隔で動作を解析

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム（TimeStamp列を含む）
    interval : str
        集計間隔（pandasのfreq文字列、例: '10s', '1min'）
    thresholds : dict, optional
        動作検出閾値

    Returns
    -------
    motion_df : pd.DataFrame
        間隔ごとの動作解析結果
        columns: ['interval', 'acc_magnitude_mean', 'acc_magnitude_std',
                  'gyro_magnitude_mean', 'gyro_magnitude_max',
                  'motion_score', 'is_motion']
    """
    if thresholds is None:
        thresholds = MOTION_THRESHOLDS

    # マグニチュード計算
    df = df.copy()
    df['acc_magnitude'] = compute_magnitude(
        df['Accelerometer_X'],
        df['Accelerometer_Y'],
        df['Accelerometer_Z']
    )
    df['gyro_magnitude'] = compute_magnitude(
        df['Gyro_X'],
        df['Gyro_Y'],
        df['Gyro_Z']
    )

    # 間隔でグループ化
    df['interval'] = df['TimeStamp'].dt.floor(interval)

    # 集計
    grouped = df.groupby('interval').agg({
        'acc_magnitude': ['mean', 'std'],
        'gyro_magnitude': ['mean', 'max'],
    }).reset_index()

    grouped.columns = [
        'interval',
        'acc_magnitude_mean', 'acc_magnitude_std',
        'gyro_magnitude_mean', 'gyro_magnitude_max'
    ]

    # 動作スコアと判定
    grouped['motion_score'] = grouped.apply(
        lambda row: compute_motion_score(row['acc_magnitude_std'], row['gyro_magnitude_mean']),
        axis=1
    )

    grouped['is_motion'] = grouped.apply(
        lambda row: detect_motion(
            row['acc_magnitude_std'],
            row['gyro_magnitude_mean'],
            row['gyro_magnitude_max'],
            thresholds
        ),
        axis=1
    )

    return grouped


def get_motion_epochs(motion_df):
    """
    動作が検出されたエポック（区間）を抽出

    Parameters
    ----------
    motion_df : pd.DataFrame
        analyze_motion_intervals()の戻り値

    Returns
    -------
    motion_epochs : list of dict
        動作エポックのリスト
        [{'start': Timestamp, 'end': Timestamp, 'score': float}, ...]
    """
    motion_rows = motion_df[motion_df['is_motion']].copy()

    if len(motion_rows) == 0:
        return []

    epochs = []
    for _, row in motion_rows.iterrows():
        epochs.append({
            'start': row['interval'],
            'end': row['interval'],  # 単一インターバルの場合
            'score': row['motion_score'],
            'acc_std': row['acc_magnitude_std'],
            'gyro_mean': row['gyro_magnitude_mean'],
            'gyro_max': row['gyro_magnitude_max']
        })

    return epochs


def analyze_motion(df, interval='10s', thresholds=None):
    """
    包括的なモーション解析を実行

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitorデータフレーム
    interval : str
        集計間隔
    thresholds : dict, optional
        動作検出閾値

    Returns
    -------
    results : dict
        {
            'motion_df': pd.DataFrame,  # 間隔ごとの解析結果
            'motion_epochs': list,      # 動作検出エポック
            'stats': dict,              # 統計情報
            'motion_ratio': float       # 動作検出割合
        }
    """
    motion_df = analyze_motion_intervals(df, interval, thresholds)
    motion_epochs = get_motion_epochs(motion_df)

    total_intervals = len(motion_df)
    motion_intervals = motion_df['is_motion'].sum()

    stats = {
        'total_intervals': total_intervals,
        'motion_intervals': motion_intervals,
        'still_intervals': total_intervals - motion_intervals,
        'acc_std_mean': motion_df['acc_magnitude_std'].mean(),
        'acc_std_max': motion_df['acc_magnitude_std'].max(),
        'gyro_mean_mean': motion_df['gyro_magnitude_mean'].mean(),
        'gyro_mean_max': motion_df['gyro_magnitude_mean'].max(),
    }

    return {
        'motion_df': motion_df,
        'motion_epochs': motion_epochs,
        'stats': stats,
        'motion_ratio': motion_intervals / total_intervals if total_intervals > 0 else 0.0
    }
