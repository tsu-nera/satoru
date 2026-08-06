"""
センサー解析モジュール
各種生体センサーの信号処理と解析

- fNIRS: 脳血流計測
- EEG: 脳波解析
- IMU: 加速度・ジャイロによる動作検出と姿勢評価
"""

# fNIRSセンサー（脳血流）
# EEGセンサー（脳波）
from .eeg import (
    DEFAULT_SFREQ,
    FREQ_BANDS,
    calculate_band_statistics,
    calculate_paf,
    calculate_psd,
    calculate_spectrogram,
    filter_eeg_quality,
    get_psd_peak_frequencies,
    prepare_mne_raw,
)
from .fnirs import analyze_fnirs, calculate_hbo_hbr

# IMUセンサー（加速度・ジャイロ）
from .imu import (
    # Motion detection (artifact removal)
    MOTION_THRESHOLDS,
    PostureAnalyzer,
    analyze_motion,
    analyze_motion_intervals,
    compute_magnitude,
    compute_motion_score,
    # Posture analysis
    compute_posture_statistics,
    compute_rms,
    detect_motion,
    extract_sensor_data,
    get_motion_epochs,
    # Common utilities
    remove_dc_offset,
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
