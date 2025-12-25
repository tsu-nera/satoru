"""
IMU (Inertial Measurement Unit) センサー解析ライブラリ

Muse S の加速度計・ジャイロスコープを統合的に扱います。
- モーション検出：EEG アーティファクト識別用の10秒間隔検出
- 姿勢統計量：坐禅中の姿勢安定性評価用の統計量計算
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


# ============================================================
# 定数定義
# ============================================================

# センサー定数
DEFAULT_SAMPLING_RATE = 52.0  # Hz (Muse S)
ACC_COLUMNS = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
GYRO_COLUMNS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']

# 動作検出閾値
MOTION_THRESHOLDS = {
    'acc_std': 0.02,      # 加速度標準偏差の閾値 (g)
    'gyro_mean': 3.0,     # ジャイロ平均の閾値 (deg/s)
    'gyro_max': 10.0,     # ジャイロ最大値の閾値 (deg/s)
}


# ============================================================
# 共通ユーティリティ関数
# ============================================================

def compute_magnitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
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

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 0.0, 0.0])
    >>> y = np.array([0.0, 1.0, 0.0])
    >>> z = np.array([0.0, 0.0, 1.0])
    >>> magnitude = compute_magnitude(x, y, z)
    >>> np.allclose(magnitude, [1.0, 1.0, 1.0])
    True
    """
    return np.sqrt(x**2 + y**2 + z**2)


def remove_dc_offset(data: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    DCオフセット除去（重力成分・バイアス除去）

    加速度データから重力成分を除去し、純粋な動き成分のみを抽出します。
    ジャイロデータからゼロ点ドリフト（バイアス）を除去します。

    Parameters
    ----------
    data : np.ndarray
        入力データ
    method : str, default 'median'
        除去方法
        - 'median': メディアン値を減算（外れ値に頑健、推奨）
        - 'mean': 平均値を減算

    Returns
    -------
    data_corrected : np.ndarray
        オフセット除去後のデータ

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
    >>> corrected = remove_dc_offset(data, method='median')
    >>> np.allclose(corrected.mean(), 0.0, atol=0.1)
    True
    """
    if method == 'median':
        return data - np.nanmedian(data)
    elif method == 'mean':
        return data - np.nanmean(data)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median' or 'mean'.")


def compute_rms(data: np.ndarray) -> float:
    """
    RMS (Root Mean Square) を計算

    Parameters
    ----------
    data : np.ndarray
        入力データ

    Returns
    -------
    rms : float
        RMS値 sqrt(mean(data^2))

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, -1.0, 1.0, -1.0])
    >>> rms = compute_rms(data)
    >>> np.isclose(rms, 1.0)
    True
    """
    return np.sqrt(np.nanmean(data**2))


def extract_sensor_data(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    DataFrameからセンサーデータを抽出

    Parameters
    ----------
    df : pd.DataFrame
        Mind Monitor形式のデータフレーム
        必須カラム: Accelerometer_X/Y/Z, Gyro_X/Y/Z
        オプションカラム: TimeStamp

    Returns
    -------
    sensor_data : dict
        {
            'acc_x': np.ndarray,
            'acc_y': np.ndarray,
            'acc_z': np.ndarray,
            'gyro_x': np.ndarray,
            'gyro_y': np.ndarray,
            'gyro_z': np.ndarray,
            'timestamps': pd.DatetimeIndex or None
        }

    Raises
    ------
    KeyError
        必須カラムが存在しない場合

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame({
    ...     'Accelerometer_X': [0.1, 0.2],
    ...     'Accelerometer_Y': [0.3, 0.4],
    ...     'Accelerometer_Z': [0.5, 0.6],
    ...     'Gyro_X': [1.0, 2.0],
    ...     'Gyro_Y': [3.0, 4.0],
    ...     'Gyro_Z': [5.0, 6.0],
    ... })
    >>> data = extract_sensor_data(df)
    >>> data['acc_x'].shape
    (2,)
    """
    # 必須カラムの存在確認
    required_columns = ACC_COLUMNS + GYRO_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Required columns missing: {missing_columns}")

    return {
        'acc_x': df['Accelerometer_X'].values,
        'acc_y': df['Accelerometer_Y'].values,
        'acc_z': df['Accelerometer_Z'].values,
        'gyro_x': df['Gyro_X'].values,
        'gyro_y': df['Gyro_Y'].values,
        'gyro_z': df['Gyro_Z'].values,
        'timestamps': df['TimeStamp'] if 'TimeStamp' in df.columns else None
    }


# ============================================================
# モーション検出（EEG アーティファクト検出用）
# ============================================================

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


# ============================================================
# 姿勢統計量（坐禅中の姿勢安定性評価用）
# ============================================================

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
        # Step 2: DCオフセット除去（重力成分を除去）
        acc_x_dc = remove_dc_offset(acc_x)
        acc_y_dc = remove_dc_offset(acc_y)
        acc_z_dc = remove_dc_offset(acc_z)

        # Step 3: マグニチュード計算（純粋な動き成分のみ）
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
        # ゼロ点補正（median除去）
        gyro_x_dc = remove_dc_offset(gyro_x)
        gyro_y_dc = remove_dc_offset(gyro_y)
        gyro_z_dc = remove_dc_offset(gyro_z)

        # マグニチュード計算（純粋な回転成分のみ）
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
        # ジャイロマグニチュード
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

        # マグニチュード計算
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
