"""
身体運動センサー解析の共通ユーティリティ

加速度計・ジャイロスコープの基本処理を提供します。
motion.py と posture.py の共通ロジックを抽出。
"""

import numpy as np
import pandas as pd
from typing import Dict


# センサー定数
DEFAULT_SAMPLING_RATE = 52.0  # Hz (Muse S)
ACC_COLUMNS = ['Accelerometer_X', 'Accelerometer_Y', 'Accelerometer_Z']
GYRO_COLUMNS = ['Gyro_X', 'Gyro_Y', 'Gyro_Z']


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
