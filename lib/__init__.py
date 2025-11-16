"""
Mind Monitor Analysis Library
"""

from .loaders import (
    load_mind_monitor_csv,
    get_eeg_data,
    get_optics_data,
    get_heart_rate_data,
    get_data_summary
)

from .sensors.fnirs import (
    calculate_hbo_hbr,
    analyze_fnirs
)

from .sensors.ppg import (
    estimate_rr_intervals,
    estimate_respiratory_rate_welch,
    estimate_respiratory_rate_fft,
    analyze_respiratory
)

from .visualization import (
    plot_fnirs,
    plot_fnirs_muse_style,
    plot_respiratory,
    plot_frequency_spectrum,
    plot_integrated_dashboard
)

from .sensors.eeg import (
    calculate_band_statistics,
    calculate_hsi_statistics,
    prepare_mne_raw,
    filter_eeg_quality,
    calculate_psd,
    calculate_spectrogram,
    calculate_spectrogram_all_channels,
    calculate_paf,
    calculate_paf_time_evolution,
    plot_band_power_time_series,
    plot_psd,
    plot_psd_time_series,
    plot_spectrogram,
    plot_spectrogram_grid,
    plot_band_ratios,
    plot_paf,
    plot_paf_time_evolution,
    plot_raw_preview,
    get_psd_peak_frequencies,
    FREQ_BANDS,
    FrontalThetaResult,
    calculate_frontal_theta,
    plot_frontal_theta,
    FrontalAsymmetryResult,
    calculate_frontal_asymmetry,
    plot_frontal_asymmetry,
    SpectralEntropyResult,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
    plot_spectral_entropy,
)

from .segment_analysis import (
    SegmentAnalysisResult,
    calculate_segment_analysis,
    plot_segment_comparison,
    MEDITATION_SCORE_WEIGHTS,
    calculate_meditation_score,
)

from .session_summary import (
    SessionSummaryResult,
    generate_session_summary,
    extract_metric_value,
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
    # respiratory
    'estimate_rr_intervals',
    'estimate_respiratory_rate_welch',
    'estimate_respiratory_rate_fft',
    'analyze_respiratory',
    # visualization
    'plot_fnirs',
    'plot_fnirs_muse_style',
    'plot_respiratory',
    'plot_frequency_spectrum',
    'plot_integrated_dashboard',
    # eeg
    'get_eeg_data',
    'calculate_band_statistics',
    'calculate_hsi_statistics',
    'prepare_mne_raw',
    'filter_eeg_quality',
    'calculate_psd',
    'calculate_spectrogram',
    'calculate_spectrogram_all_channels',
    'calculate_paf',
    'calculate_paf_time_evolution',
    'plot_band_power_time_series',
    'plot_psd',
    'plot_psd_time_series',
    'plot_spectrogram',
    'plot_spectrogram_grid',
    'plot_band_ratios',
    'plot_paf',
    'plot_paf_time_evolution',
    'plot_raw_preview',
    'get_psd_peak_frequencies',
    'FREQ_BANDS',
    'FrontalThetaResult',
    'calculate_frontal_theta',
    'plot_frontal_theta',
    'FrontalAsymmetryResult',
    'calculate_frontal_asymmetry',
    'plot_frontal_asymmetry',
    'SpectralEntropyResult',
    'calculate_spectral_entropy',
    'calculate_spectral_entropy_time_series',
    'plot_spectral_entropy',
    # high-level eeg utilities
    'SegmentAnalysisResult',
    'calculate_segment_analysis',
    'plot_segment_comparison',
    'MEDITATION_SCORE_WEIGHTS',
    'calculate_meditation_score',
    # session summary
    'SessionSummaryResult',
    'generate_session_summary',
    'extract_metric_value',
]
