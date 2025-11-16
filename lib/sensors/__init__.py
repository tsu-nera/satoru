"""
センサー解析モジュール
各種生体センサーの信号処理と解析

- PPG: 心拍・呼吸数推定
- fNIRS: 脳血流計測
- EEG: 脳波解析
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
    calculate_paf_time_evolution,
    plot_band_power_time_series,
    plot_psd,
    plot_psd_time_series,
    plot_spectrogram,
    plot_band_ratios,
    plot_paf,
    plot_paf_time_evolution,
    get_psd_peak_frequencies
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
    'calculate_paf_time_evolution',
    'plot_band_power_time_series',
    'plot_psd',
    'plot_psd_time_series',
    'plot_spectrogram',
    'plot_band_ratios',
    'plot_paf',
    'plot_paf_time_evolution',
    'get_psd_peak_frequencies',
]
