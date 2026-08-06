"""
EEG解析ライブラリ
Muse脳波データの周波数バンド解析、PSD、PAF計算、可視化
"""

# 定数
# Alpha Power解析
from .alpha_power import (
    AlphaPowerMethod,
    AlphaPowerResult,
    calculate_alpha_power,
    calculate_alpha_power_from_raw,
)

# アーチファクト検出（振幅ベース）
from .artifact import (
    ARTIFACT_P2P_THRESHOLD_UV,
    calculate_amplitude_statistics,
    detect_artifact_windows,
    summarize_artifacts,
)
from .constants import DEFAULT_SFREQ, FREQ_BANDS

# 周波数解析
from .frequency import calculate_psd, calculate_spectrogram, calculate_spectrogram_all_channels

# FAA解析
from .frontal_asymmetry import (
    FrontalAsymmetryResult,
    calculate_frontal_asymmetry,
)

# Fmθ解析
from .frontal_theta import (
    FrontalThetaResult,
    calculate_frontal_theta,
)

# ITF解析
from .itf import calculate_itf

# PAF解析
from .paf import calculate_paf

# 前処理
from .preprocessing import filter_eeg_quality, prepare_mne_raw

# PSDピーク解析
from .psd_peaks import (
    DETAILED_FREQ_BANDS,
    HarmonicsResult,  # 後方互換性
    PeakInfo,
    PeakType,
    PsdPeaksResult,
    analyze_harmonics,  # 後方互換性
    analyze_psd_peaks,
)

# SMR解析
from .smr import (
    SMR_BAND,
    SMRResult,
    calculate_smr,
)

# Spectral Entropy解析
from .spectral_entropy import (
    SpectralEntropyResult,
    calculate_spectral_entropy,
    calculate_spectral_entropy_time_series,
)

# 統計
from .statistics import calculate_band_statistics, calculate_hsi_statistics

# ユーティリティ
from .utils import get_psd_peak_frequencies

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
    'summarize_artifacts',
    'calculate_amplitude_statistics',
    'detect_artifact_windows',
    'ARTIFACT_P2P_THRESHOLD_UV',
    # PAF解析
    'calculate_paf',
    # ITF解析
    'calculate_itf',
    # ユーティリティ
    'get_psd_peak_frequencies',
    # Fmθ解析
    'FrontalThetaResult',
    'calculate_frontal_theta',
    # FAA解析
    'FrontalAsymmetryResult',
    'calculate_frontal_asymmetry',
    # Spectral Entropy解析
    'SpectralEntropyResult',
    'calculate_spectral_entropy',
    'calculate_spectral_entropy_time_series',
    # Alpha Power解析
    'AlphaPowerResult',
    'AlphaPowerMethod',
    'calculate_alpha_power',
    'calculate_alpha_power_from_raw',
    # PSDピーク解析
    'PsdPeaksResult',
    'HarmonicsResult',  # 後方互換性
    'PeakInfo',
    'PeakType',
    'analyze_psd_peaks',
    'analyze_harmonics',  # 後方互換性
    'DETAILED_FREQ_BANDS',
    # SMR解析
    'SMRResult',
    'calculate_smr',
    'SMR_BAND',
]
