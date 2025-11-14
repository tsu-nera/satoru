"""
EEG解析ライブラリ
Muse脳波データの周波数バンド解析、PSD、PAF計算、可視化
"""

# 定数
from .constants import FREQ_BANDS, DEFAULT_SFREQ

# 前処理
from .preprocessing import prepare_mne_raw, filter_eeg_quality

# 周波数解析
from .frequency import calculate_psd, calculate_spectrogram, calculate_spectrogram_all_channels

# 統計
from .statistics import calculate_band_statistics, calculate_hsi_statistics

# バンド比率
from .ratios import calculate_band_ratios

# PAF解析
from .paf import calculate_paf, calculate_paf_time_evolution

# 可視化
from .visualization import (
    plot_band_power_time_series,
    plot_psd,
    plot_psd_time_series,
    plot_spectrogram,
    plot_spectrogram_grid,
    plot_band_ratios,
    plot_paf,
    plot_paf_time_evolution,
    plot_raw_preview,
)

# ユーティリティ
from .utils import get_psd_peak_frequencies

# Fmθ解析
from .frontal_theta import (
    FrontalThetaResult,
    calculate_frontal_theta,
    plot_frontal_theta,
)

# FAA解析
from .frontal_asymmetry import (
    FrontalAsymmetryResult,
    calculate_frontal_asymmetry,
    plot_frontal_asymmetry,
)

# Spectral Entropy解析
from .spectral_entropy import (
    SpectralEntropyResult,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
    plot_spectral_entropy,
)

__all__ = [
    # 定数
    'FREQ_BANDS',
    'DEFAULT_SFREQ',
    # 前処理
    'prepare_mne_raw',
    'filter_eeg_quality',
    # 周波数解析
    'calculate_psd',
    'calculate_spectrogram',
    'calculate_spectrogram_all_channels',
    # 統計
    'calculate_band_statistics',
    'calculate_hsi_statistics',
    # バンド比率
    'calculate_band_ratios',
    # PAF解析
    'calculate_paf',
    'calculate_paf_time_evolution',
    # 可視化
    'plot_band_power_time_series',
    'plot_psd',
    'plot_psd_time_series',
    'plot_spectrogram',
    'plot_spectrogram_grid',
    'plot_band_ratios',
    'plot_paf',
    'plot_paf_time_evolution',
    'plot_raw_preview',
    # ユーティリティ
    'get_psd_peak_frequencies',
    # Fmθ解析
    'FrontalThetaResult',
    'calculate_frontal_theta',
    'plot_frontal_theta',
    # FAA解析
    'FrontalAsymmetryResult',
    'calculate_frontal_asymmetry',
    'plot_frontal_asymmetry',
    # Spectral Entropy解析
    'SpectralEntropyResult',
    'calculate_spectral_entropy',
    'calculate_spectral_entropy_time_series',
    'plot_spectral_entropy',
]
