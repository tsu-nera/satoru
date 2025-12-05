"""
EEG可視化モジュール

このパッケージには、EEG解析結果の可視化関数が含まれています。
計算ロジックは親ディレクトリ（lib/sensors/eeg/）にあります。
"""

# 基本EEG可視化
from .eeg_plots import (
    plot_raw_preview,
    plot_band_power_time_series,
    plot_psd,
    plot_spectrogram,
    plot_spectrogram_grid,
    plot_band_ratios,
    plot_paf,
)

# 指標別可視化
from .frontal_theta_plot import plot_frontal_theta
from .frontal_asymmetry_plot import plot_frontal_asymmetry
from .spectral_entropy_plot import plot_spectral_entropy

__all__ = [
    # 基本EEG可視化
    'plot_raw_preview',
    'plot_band_power_time_series',
    'plot_psd',
    'plot_spectrogram',
    'plot_spectrogram_grid',
    'plot_band_ratios',
    'plot_paf',
    # 指標別可視化
    'plot_frontal_theta',
    'plot_frontal_asymmetry',
    'plot_spectral_entropy',
]
