"""
Mind Monitor Analysis Library
"""

from .loaders import get_data_summary, get_eeg_data, get_heart_rate_data, get_optics_data, load_mind_monitor_csv
from .segment_analysis import (
    MEDITATION_SCORE_WEIGHTS,
    SegmentAnalysisResult,
    calculate_best_metrics,
    calculate_meditation_score,
    calculate_segment_analysis,
)
from .sensors.eeg import (
    DETAILED_FREQ_BANDS,
    FREQ_BANDS,
    SMR_BAND,
    AlphaPowerMethod,
    AlphaPowerResult,
    FrontalAsymmetryResult,
    FrontalThetaResult,
    HarmonicsResult,  # 後方互換性
    PeakInfo,
    PeakType,
    PsdPeaksResult,
    SMRResult,
    SpectralEntropyResult,
    analyze_harmonics,  # 後方互換性
    analyze_psd_peaks,
    calculate_alpha_power,
    calculate_alpha_power_from_raw,
    calculate_amplitude_statistics,
    calculate_band_statistics,
    calculate_frontal_asymmetry,
    calculate_frontal_theta,
    calculate_hsi_statistics,
    calculate_itf,
    calculate_paf,
    calculate_psd,
    calculate_smr,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
    calculate_spectrogram,
    calculate_spectrogram_all_channels,
    detect_artifact_windows,
    filter_eeg_quality,
    get_psd_peak_frequencies,
    prepare_mne_raw,
    summarize_artifacts,
)
from .sensors.fnirs import analyze_fnirs, calculate_hbo_hbr
from .sensors.imu import (
    MOTION_THRESHOLDS,
    analyze_motion,
)
from .session_summary import (
    SessionSummaryResult,
    extract_metric_value,
    generate_session_summary,
)

__all__ = [
    # loaders
    'load_mind_monitor_csv',
    'get_eeg_data',
    'get_optics_data',
    'get_heart_rate_data',
    'get_data_summary',
    # fnirs
    'calculate_hbo_hbr',
    'analyze_fnirs',
    # motion
    'analyze_motion',
    'MOTION_THRESHOLDS',
    # eeg
    'get_eeg_data',
    'calculate_band_statistics',
    'calculate_hsi_statistics',
    'summarize_artifacts',
    'calculate_amplitude_statistics',
    'detect_artifact_windows',
    'prepare_mne_raw',
    'filter_eeg_quality',
    'calculate_psd',
    'calculate_spectrogram',
    'calculate_spectrogram_all_channels',
    'calculate_paf',
    'calculate_itf',
    'get_psd_peak_frequencies',
    'FREQ_BANDS',
    'FrontalThetaResult',
    'calculate_frontal_theta',
    'FrontalAsymmetryResult',
    'calculate_frontal_asymmetry',
    'AlphaPowerResult',
    'AlphaPowerMethod',
    'calculate_alpha_power',
    'calculate_alpha_power_from_raw',
    'SpectralEntropyResult',
    'calculate_spectral_entropy',
    'calculate_spectral_entropy_time_series',
    'PsdPeaksResult',
    'HarmonicsResult',  # 後方互換性
    'PeakInfo',
    'PeakType',
    'analyze_psd_peaks',
    'analyze_harmonics',  # 後方互換性
    'DETAILED_FREQ_BANDS',
    'SMRResult',
    'calculate_smr',
    'SMR_BAND',
    # high-level eeg utilities
    'SegmentAnalysisResult',
    'calculate_segment_analysis',
    'MEDITATION_SCORE_WEIGHTS',
    'calculate_meditation_score',
    'calculate_best_metrics',
    # session summary
    'SessionSummaryResult',
    'generate_session_summary',
    'extract_metric_value',
]
