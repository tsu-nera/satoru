"""
`scripts/generate_report.py` の `run_full_analysis` を構成する解析ステップ群。

各ステップ関数は `results` dict を引数で受け取って直接 mutate し、後続ステップが
必要とする中間オブジェクトを return する。orchestrator（run_full_analysis）は
これらの中間オブジェクトを明示的なローカル変数として保持し、必要なステップへ
引数で渡す。
"""

from .step import analysis_step

from .steps_physio import (
    analyze_fnirs,
    analyze_motion_and_hr,
    analyze_hrv,
    analyze_respiration,
)

from .steps_eeg import (
    plot_band_power_series,
    prepare_mne_and_spectral,
    analyze_frontal_theta_step,
    analyze_smr_step,
)

from .steps_summary import (
    build_statistical_dataframe,
    analyze_segments,
    plot_band_ratios_step,
    calculate_session_score,
    save_session_log,
)

__all__ = [
    'analysis_step',
    'analyze_fnirs',
    'analyze_motion_and_hr',
    'analyze_hrv',
    'analyze_respiration',
    'plot_band_power_series',
    'prepare_mne_and_spectral',
    'analyze_frontal_theta_step',
    'analyze_smr_step',
    'build_statistical_dataframe',
    'analyze_segments',
    'plot_band_ratios_step',
    'calculate_session_score',
    'save_session_log',
]
