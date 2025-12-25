"""
坐相統計量計算ライブラリ

坐禅中の姿勢安定性を定量的に評価するための統計量を計算します。
デイリーレポート用に標準的な指標（RMS、角度など）を提供します。
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional

# 共通ユーティリティをインポート
from .movement import (
    compute_magnitude,
    remove_dc_offset,
    compute_rms,
    extract_sensor_data,
)


def compute_posture_statistics(
    df: pd.DataFrame,
    timestamps: pd.Series,
    segment_minutes: int = 3
) -> pd.DataFrame:
    """
    セグメントごとの坐相統計量を計算（Statistical DataFrame パターン）

    この関数は create_statistical_dataframe() から呼び出され、
    EEG/fNIRS/HR と同じタイムスタンプで坐相統計量を計算します。

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitor形式のデータフレーム
        必須カラム: Accelerometer_X/Y/Z, Gyro_X/Y/Z, TimeStamp
    timestamps : pd.Series
        セグメントのタイムスタンプ（create_statistical_dataframe で生成）
    segment_minutes : int, default 3
        セグメント長（分単位）

    Returns
    -------
    posture_df : pd.DataFrame
        セグメントごとの坐相統計量
        columns: ['timestamp', 'motion_index_mean', 'motion_index_max',
                 'gyro_rms', 'gyro_rms_corrected', 'pitch_angle',
                 'roll_angle', 'yaw_rms']

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> # サンプルデータ作成
    >>> df = pd.DataFrame({
    ...     'TimeStamp': pd.date_range('2025-01-01', periods=100, freq='1s'),
    ...     'Accelerometer_X': np.random.randn(100) * 0.1,
    ...     'Accelerometer_Y': np.random.randn(100) * 0.1,
    ...     'Accelerometer_Z': np.random.randn(100) * 0.1 + 1.0,
    ...     'Gyro_X': np.random.randn(100),
    ...     'Gyro_Y': np.random.randn(100),
    ...     'Gyro_Z': np.random.randn(100),
    ... })
    >>> timestamps = pd.Series([df['TimeStamp'].iloc[0]])
    >>> result = compute_posture_statistics(df, timestamps, segment_minutes=1)
    >>> result.shape[0] == 1
    True
    """
    results = []
    segment_delta = pd.Timedelta(minutes=segment_minutes)

    for ts in timestamps:
        # セグメント範囲のデータを抽出
        segment_end = ts + segment_delta
        segment_data = df[
            (df['TimeStamp'] >= ts) & (df['TimeStamp'] < segment_end)
        ]

        if len(segment_data) == 0:
            # データがない場合はNaNで埋める
            results.append({
                'timestamp': ts,
                'motion_index_mean': np.nan,
                'motion_index_max': np.nan,
                'gyro_rms': np.nan,
                'gyro_rms_corrected': np.nan,
                'pitch_angle': np.nan,
                'roll_angle': np.nan,
                'yaw_rms': np.nan,
            })
            continue

        # センサーデータ抽出
        sensor_data = extract_sensor_data(segment_data)
        acc_x = sensor_data['acc_x']
        acc_y = sensor_data['acc_y']
        acc_z = sensor_data['acc_z']
        gyro_x = sensor_data['gyro_x']
        gyro_y = sensor_data['gyro_y']
        gyro_z = sensor_data['gyro_z']

        # モーション指数（DC除去版）
        acc_x_dc = remove_dc_offset(acc_x)
        acc_y_dc = remove_dc_offset(acc_y)
        acc_z_dc = remove_dc_offset(acc_z)
        motion_magnitude = compute_magnitude(acc_x_dc, acc_y_dc, acc_z_dc)
        motion_index_mean = np.nanmean(motion_magnitude)
        motion_index_max = np.nanmax(motion_magnitude)

        # ジャイロマグニチュード
        gyro_magnitude = compute_magnitude(gyro_x, gyro_y, gyro_z)
        gyro_rms_val = compute_rms(gyro_magnitude)

        # ゼロ点補正版ジャイロRMS
        gyro_x_dc = remove_dc_offset(gyro_x)
        gyro_y_dc = remove_dc_offset(gyro_y)
        gyro_z_dc = remove_dc_offset(gyro_z)
        gyro_magnitude_corrected = compute_magnitude(gyro_x_dc, gyro_y_dc, gyro_z_dc)
        gyro_rms_corrected = compute_rms(gyro_magnitude_corrected)

        # Pitch角度（前後傾き）
        ax_mean = np.nanmean(acc_x)
        ay_mean = np.nanmean(acc_y)
        az_mean = np.nanmean(acc_z)
        pitch_rad = np.arctan2(ax_mean, np.sqrt(ay_mean**2 + az_mean**2))
        pitch_angle = np.degrees(pitch_rad)

        # Roll角度（左右傾き）
        roll_rad = np.arctan2(ay_mean, az_mean)
        roll_angle = np.degrees(roll_rad)

        # Yaw RMS
        yaw_rms = compute_rms(gyro_z)

        results.append({
            'timestamp': ts,
            'motion_index_mean': motion_index_mean,
            'motion_index_max': motion_index_max,
            'gyro_rms': gyro_rms_val,
            'gyro_rms_corrected': gyro_rms_corrected,
            'pitch_angle': pitch_angle,
            'roll_angle': roll_angle,
            'yaw_rms': yaw_rms,
        })

    return pd.DataFrame(results)


class PostureAnalyzer:
    """
    坐相統計量を計算するクラス

    Tier 1（必須）：デイリーレポート用の5指標
    - acc_rms: 加速度RMS（g）
    - gyro_rms: ジャイロRMS（deg/s）
    - pitch_angle: Pitch角度（前後傾き、deg）
    - roll_angle: Roll角度（左右傾き、deg）
    - heart_rate: 心拍数（BPM）※別途計算
    """

    def __init__(self):
        """初期化"""
        pass

    # compute_rms は movement.py から使用するため削除

    def compute_motion_index(self, acc_x: np.ndarray, acc_y: np.ndarray,
                            acc_z: np.ndarray) -> tuple[float, float]:
        """
        モーション指数を計算（論文ベース）

        Step 1: 生データ抽出
        Step 2: DCオフセット除去（median を引く）
        Step 3: RMS計算（√(x²+y²+z²)）
        Step 4: 平滑化（省略可）
        Step 5: 特徴量抽出（mean, max）

        Parameters
        ----------
        acc_x, acc_y, acc_z : np.ndarray
            各軸の加速度データ（g単位）

        Returns
        -------
        motion_mean : float
            平均モーション指数（I1）
        motion_max : float
            最大モーション指数（I2）
        """
        # Step 2: DCオフセット除去（重力成分を除去）- movement.py の関数を使用
        acc_x_dc = remove_dc_offset(acc_x)
        acc_y_dc = remove_dc_offset(acc_y)
        acc_z_dc = remove_dc_offset(acc_z)

        # Step 3: マグニチュード計算（純粋な動き成分のみ）- movement.py の関数を使用
        motion_magnitude = compute_magnitude(acc_x_dc, acc_y_dc, acc_z_dc)

        # Step 5: 特徴量抽出
        motion_mean = np.nanmean(motion_magnitude)  # I1: 平均モーション指数
        motion_max = np.nanmax(motion_magnitude)    # I2: 最大モーション指数

        return motion_mean, motion_max

    def compute_gyro_corrected(self, gyro_x: np.ndarray, gyro_y: np.ndarray,
                               gyro_z: np.ndarray) -> float:
        """
        ゼロ点補正版のジャイロRMSを計算

        ジャイロセンサーのゼロ点ドリフト（バイアス）を除去し、
        純粋な回転運動成分のみを抽出します。

        Parameters
        ----------
        gyro_x, gyro_y, gyro_z : np.ndarray
            各軸のジャイロデータ（deg/s単位）

        Returns
        -------
        gyro_rms_corrected : float
            ゼロ点補正後のジャイロRMS（deg/s）
        """
        # ゼロ点補正（median除去）- movement.py の関数を使用
        gyro_x_dc = remove_dc_offset(gyro_x)
        gyro_y_dc = remove_dc_offset(gyro_y)
        gyro_z_dc = remove_dc_offset(gyro_z)

        # マグニチュード計算（純粋な回転成分のみ）- movement.py の関数を使用
        gyro_magnitude_corrected = compute_magnitude(gyro_x_dc, gyro_y_dc, gyro_z_dc)

        return compute_rms(gyro_magnitude_corrected)

    def compute_pitch_angle(self, acc_x: np.ndarray, acc_y: np.ndarray,
                           acc_z: np.ndarray) -> float:
        """
        Pitch角度（前後傾き）を計算

        加速度センサーの重力成分から姿勢角度を推定します。

        Parameters
        ----------
        acc_x, acc_y, acc_z : np.ndarray
            各軸の加速度データ（g単位）

        Returns
        -------
        pitch : float
            Pitch角度（度）、正=前傾、負=後傾
        """
        # 平均値を使用（DCオフセット = 重力成分）
        ax_mean = np.nanmean(acc_x)
        ay_mean = np.nanmean(acc_y)
        az_mean = np.nanmean(acc_z)

        # Pitch角度の計算
        pitch_rad = np.arctan2(ax_mean, np.sqrt(ay_mean**2 + az_mean**2))
        return np.degrees(pitch_rad)

    def compute_roll_angle(self, acc_y: np.ndarray, acc_z: np.ndarray) -> float:
        """
        Roll角度（左右傾き）を計算

        Parameters
        ----------
        acc_y, acc_z : np.ndarray
            Y軸・Z軸の加速度データ（g単位）

        Returns
        -------
        roll : float
            Roll角度（度）、正=右傾き、負=左傾き
        """
        # 平均値を使用
        ay_mean = np.nanmean(acc_y)
        az_mean = np.nanmean(acc_z)

        # Roll角度の計算
        roll_rad = np.arctan2(ay_mean, az_mean)
        return np.degrees(roll_rad)

    def compute_essential(self, acc_x: np.ndarray, acc_y: np.ndarray,
                         acc_z: np.ndarray, gyro_x: np.ndarray,
                         gyro_y: np.ndarray, gyro_z: np.ndarray) -> Dict[str, float]:
        """
        Tier 1：Essential指標を計算（デイリーレポート用）

        Parameters
        ----------
        acc_x, acc_y, acc_z : np.ndarray
            加速度データ（g単位）
        gyro_x, gyro_y, gyro_z : np.ndarray
            ジャイロデータ（deg/s単位）

        Returns
        -------
        metrics : dict
            {
                'acc_rms': float (g),
                'gyro_rms': float (deg/s),
                'pitch_angle': float (deg),
                'roll_angle': float (deg),
                'yaw_rms': float (deg/s)
            }
        """
        # ジャイロマグニチュード - movement.py の関数を使用
        gyro_magnitude = compute_magnitude(gyro_x, gyro_y, gyro_z)

        # モーション指数（論文ベース）
        motion_mean, motion_max = self.compute_motion_index(acc_x, acc_y, acc_z)

        # ゼロ点補正版ジャイロRMS
        gyro_rms_corrected = self.compute_gyro_corrected(gyro_x, gyro_y, gyro_z)

        return {
            'motion_index_mean': motion_mean,
            'motion_index_max': motion_max,
            'gyro_rms': compute_rms(gyro_magnitude),
            'gyro_rms_corrected': gyro_rms_corrected,
            'pitch_angle': self.compute_pitch_angle(acc_x, acc_y, acc_z),
            'roll_angle': self.compute_roll_angle(acc_y, acc_z),
            'yaw_rms': compute_rms(gyro_z),
        }

    def compute_standard(self, acc_x: np.ndarray, acc_y: np.ndarray,
                        acc_z: np.ndarray, gyro_x: np.ndarray,
                        gyro_y: np.ndarray, gyro_z: np.ndarray) -> Dict[str, float]:
        """
        Tier 2：Standard指標を計算（週次/研究用）

        Essential指標に加えて、以下を追加：
        - acc_std: 加速度標準偏差
        - gyro_mean: ジャイロ平均
        - pitch_variance: Pitch方向の分散
        - roll_variance: Roll方向の分散

        Parameters
        ----------
        acc_x, acc_y, acc_z : np.ndarray
            加速度データ（g単位）
        gyro_x, gyro_y, gyro_z : np.ndarray
            ジャイロデータ（deg/s単位）

        Returns
        -------
        metrics : dict
            Essential指標 + 追加指標
        """
        # Essential指標を計算
        result = self.compute_essential(acc_x, acc_y, acc_z,
                                       gyro_x, gyro_y, gyro_z)

        # マグニチュード計算 - movement.py の関数を使用
        acc_magnitude = compute_magnitude(acc_x, acc_y, acc_z)
        gyro_magnitude = compute_magnitude(gyro_x, gyro_y, gyro_z)

        # 追加指標
        result.update({
            'acc_std': np.nanstd(acc_magnitude),
            'gyro_mean': np.nanmean(gyro_magnitude),
            'pitch_variance': np.nanvar(gyro_y),  # Pitch方向の回転
            'roll_variance': np.nanvar(gyro_x),   # Roll方向の回転
        })

        return result

    def analyze_intervals(self, df: pd.DataFrame, interval: str = '3min',
                         level: str = 'essential') -> pd.DataFrame:
        """
        指定間隔で坐相統計量を計算

        Parameters
        ----------
        df : pd.DataFrame
            Mind Monitorデータフレーム（TimeStamp列を含む）
        interval : str
            集計間隔（pandasのfreq文字列、例: '3min', '10s'）
        level : str
            'essential' | 'standard'

        Returns
        -------
        posture_df : pd.DataFrame
            間隔ごとの坐相統計量
            columns: ['interval', 'acc_rms', 'gyro_rms', 'pitch_angle',
                     'roll_angle', ...]
        """
        # 間隔でグループ化
        df = df.copy()
        df['interval'] = df['TimeStamp'].dt.floor(interval)

        results = []

        for interval_time, group in df.groupby('interval'):
            # データ抽出
            acc_x = group['Accelerometer_X'].values
            acc_y = group['Accelerometer_Y'].values
            acc_z = group['Accelerometer_Z'].values
            gyro_x = group['Gyro_X'].values
            gyro_y = group['Gyro_Y'].values
            gyro_z = group['Gyro_Z'].values

            # 統計量計算
            if level == 'essential':
                metrics = self.compute_essential(acc_x, acc_y, acc_z,
                                                gyro_x, gyro_y, gyro_z)
            elif level == 'standard':
                metrics = self.compute_standard(acc_x, acc_y, acc_z,
                                               gyro_x, gyro_y, gyro_z)
            else:
                raise ValueError(f"Unknown level: {level}")

            # 結果に追加
            metrics['interval'] = interval_time
            results.append(metrics)

        # DataFrameに変換
        posture_df = pd.DataFrame(results)

        # 列順を整理
        cols = ['interval', 'motion_index_mean', 'motion_index_max',
                'gyro_rms', 'gyro_rms_corrected', 'pitch_angle', 'roll_angle', 'yaw_rms']
        if level == 'standard':
            cols.extend(['acc_std', 'gyro_mean', 'pitch_variance', 'roll_variance'])

        # 存在する列のみ選択
        posture_df = posture_df[[col for col in cols if col in posture_df.columns]]

        return posture_df

    def compute_summary(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        全期間の坐相統計量サマリーを計算

        Parameters
        ----------
        df : pd.DataFrame
            Mind Monitorデータフレーム

        Returns
        -------
        summary : dict
            {
                'acc_rms': {'mean': float, 'min': float, 'max': float},
                'gyro_rms': {'mean': float, 'min': float, 'max': float},
                ...
            }
        """
        # 全期間での統計量
        acc_x = df['Accelerometer_X'].values
        acc_y = df['Accelerometer_Y'].values
        acc_z = df['Accelerometer_Z'].values
        gyro_x = df['Gyro_X'].values
        gyro_y = df['Gyro_Y'].values
        gyro_z = df['Gyro_Z'].values

        metrics = self.compute_essential(acc_x, acc_y, acc_z,
                                        gyro_x, gyro_y, gyro_z)

        # 時系列での変動を計算するため、短い間隔で分析
        interval_df = self.analyze_intervals(df, interval='10s', level='essential')

        summary = {}
        for key in ['motion_index_mean', 'motion_index_max',
                    'gyro_rms', 'gyro_rms_corrected', 'pitch_angle', 'roll_angle', 'yaw_rms']:
            summary[key] = {
                'mean': interval_df[key].mean(),
                'min': interval_df[key].min(),
                'max': interval_df[key].max(),
                'std': interval_df[key].std(),
            }

        return summary


def evaluate_posture_stability(gyro_rms: float) -> str:
    """
    姿勢安定性を評価

    ジャイロRMSに基づいて姿勢の安定性を判定します。
    脳波との相関が最も強い指標（r=0.60）を使用。

    Parameters
    ----------
    gyro_rms : float
        ジャイロRMS（deg/s）

    Returns
    -------
    evaluation : str
        '良好 ✓✓' | '安定 ✓' | 'やや不安定' | '要注意'
    """
    if gyro_rms < 1.2:
        return '良好 ✓✓'
    elif gyro_rms < 1.5:
        return '安定 ✓'
    elif gyro_rms < 2.0:
        return 'やや不安定'
    else:
        return '要注意'
