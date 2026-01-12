"""
統合可視化モジュール

このパッケージには、各種センサー（EEG、fNIRS、動作検出）およびセグメント分析の
可視化関数が含まれています。
"""

# セグメント分析可視化
from .segment_plot import plot_segment_comparison

# fNIRS可視化
from .fnirs import plot_fnirs, plot_fnirs_muse_style

# 動作検出可視化
from .motion import plot_motion_heart_rate, create_motion_stats_table

# 共通ユーティリティ
from .utils import format_time_axis

__all__ = [
    # セグメント分析
    'plot_segment_comparison',
    # fNIRS
    'plot_fnirs',
    'plot_fnirs_muse_style',
    # 動作検出
    'plot_motion_heart_rate',
    'create_motion_stats_table',
    # 共通ユーティリティ
    'format_time_axis',
]
