"""
統合可視化モジュール

このパッケージには、各種センサー（EEG、fNIRS、呼吸・心拍）およびセグメント分析の
可視化関数が含まれています。
"""

# セグメント分析可視化
from .segment_plot import plot_segment_comparison

# fNIRS可視化
from .fnirs import plot_fnirs, plot_fnirs_muse_style

# 呼吸・心拍可視化
from .respiratory import plot_respiratory, plot_frequency_spectrum

# 統合ダッシュボード
from .dashboard import plot_integrated_dashboard

__all__ = [
    # セグメント分析
    'plot_segment_comparison',
    # fNIRS
    'plot_fnirs',
    'plot_fnirs_muse_style',
    # 呼吸・心拍
    'plot_respiratory',
    'plot_frequency_spectrum',
    # 統合ダッシュボード
    'plot_integrated_dashboard',
]
