"""
センサー解析モジュール
各種生体センサーの信号処理と解析

- fNIRS: 脳血流計測
- EEG: 脳波解析
- IMU: 加速度・ジャイロによる動作検出と姿勢評価
"""

# fNIRSセンサー（脳血流）
from .fnirs import (
    calculate_hbo_hbr,
    analyze_fnirs
)

# EEGセンサー（脳波）
from .eeg import (
    FREQ_BANDS,
    DEFAULT_SFREQ,
    calculate_band_statistics,
    prepare_mne_raw,
    filter_eeg_quality,
    calculate_psd,
    calculate_spectrogram,
    calculate_paf,
    get_psd_peak_frequencies
)

# IMUセンサー（加速度・ジャイロ）
from .imu import (
    # Motion detection (artifact removal)
    MOTION_THRESHOLDS,
    compute_magnitude,
    detect_motion,
    compute_motion_score,
    analyze_motion_intervals,
    get_motion_epochs,
    analyze_motion,
    # Posture analysis
    compute_posture_statistics,
    PostureAnalyzer,
    # Common utilities
    remove_dc_offset,
    compute_rms,
    extract_sensor_data,
)

__all__ = [
    # fNIRS
    'calculate_hbo_hbr',
    'analyze_fnirs',
    # EEG
    'FREQ_BANDS',
    'DEFAULT_SFREQ',
    'calculate_band_statistics',
    'prepare_mne_raw',
    'filter_eeg_quality',
    'calculate_psd',
    'calculate_spectrogram',
    'calculate_paf',
    'get_psd_peak_frequencies',
    # IMU - Motion detection
    'MOTION_THRESHOLDS',
    'compute_magnitude',
    'detect_motion',
    'compute_motion_score',
    'analyze_motion_intervals',
    'get_motion_epochs',
    'analyze_motion',
    # IMU - Posture analysis
    'compute_posture_statistics',
    'PostureAnalyzer',
    # IMU - Common utilities
    'remove_dc_offset',
    'compute_rms',
    'extract_sensor_data',
]
