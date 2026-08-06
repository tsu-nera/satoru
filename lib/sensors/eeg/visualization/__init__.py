"""
EEG可視化モジュール

このパッケージには、EEG解析結果の可視化関数が含まれています。
計算ロジックは親ディレクトリ（lib/sensors/eeg/）にあります。
"""

# 基本EEG可視化
from .eeg_plots import (
    plot_band_power_time_series,
    plot_band_ratios,
    plot_paf,
    plot_psd,
    plot_raw_preview,
    plot_spectrogram,
    plot_spectrogram_grid,
)

# 指標別可視化
from .psd_peaks_plot import plot_harmonics, plot_psd_peaks  # plot_harmonicsは後方互換性

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
    'plot_psd_peaks',
    'plot_harmonics',  # 後方互換性
]
