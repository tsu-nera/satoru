"""
センサー解析モジュール
各種生体センサーの信号処理と解析

- PPG: 心拍・呼吸数推定
- fNIRS: 脳血流計測
- EEG: 脳波解析
- Motion: 加速度・ジャイロによる動作検出
"""

# PPGセンサー（心拍・呼吸）
from .ppg import (
    estimate_rr_intervals,
    estimate_respiratory_rate_welch,
    estimate_respiratory_rate_fft,
    analyze_respiratory
)

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

# モーションセンサー（加速度・ジャイロ）
from .motion import (
    MOTION_THRESHOLDS,
    get_motion_data,
    compute_magnitude,
    detect_motion,
    compute_motion_score,
    analyze_motion_intervals,
    get_motion_epochs,
    analyze_motion
)

__all__ = [
    # PPG
    'estimate_rr_intervals',
    'estimate_respiratory_rate_welch',
    'estimate_respiratory_rate_fft',
    'analyze_respiratory',
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
    # Motion
    'MOTION_THRESHOLDS',
    'get_motion_data',
    'compute_magnitude',
    'detect_motion',
    'compute_motion_score',
    'analyze_motion_intervals',
    'get_motion_epochs',
    'analyze_motion',
]
