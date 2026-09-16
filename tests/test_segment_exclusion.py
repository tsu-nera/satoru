"""
exclude_last_segment 削除後のセグメント除外ロジックのテスト

Issue #42: 最終セグメントを無条件で post meditation 扱いにする除外を削除した。
以降、除外されるのは以下の2系統のみ:
- exclude_first_segment（固定、先頭のみ、relaxing phase）
- データ由来の artifact_indices / noisy_indices（excluded_ratio, usable）

合成データで calculate_segment_analysis / calculate_best_metrics を直接検証する。
実データ（data/sample.csv）には依存しない。
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.segment_analysis import (  # noqa: E402
    SegmentAnalysisResult,
    calculate_best_metrics,
    calculate_segment_analysis,
)

SESSION_START = pd.Timestamp('2026-09-16 06:00:00')
SEGMENT_MINUTES = 3
N_SEGMENTS = 4


def _segment_starts(n_segments=N_SEGMENTS, session_start=SESSION_START):
    return [
        session_start + pd.Timedelta(minutes=SEGMENT_MINUTES * i)
        for i in range(n_segments)
    ]


def make_statistical_df(
    alpha_values,
    excluded_ratios=None,
    usable=None,
    segment_starts=None,
):
    """calculate_segment_analysis に渡す最小限の statistical_df を合成する。"""
    n = len(alpha_values)
    starts = segment_starts or _segment_starts(n)
    if excluded_ratios is None:
        excluded_ratios = [0.05] * n
    if usable is None:
        usable = [True] * n

    band_powers_df = pd.DataFrame(
        {
            'Delta': [-5.0] * n,
            'Theta': [-6.0] * n,
            'Alpha': alpha_values,
            'Beta': [-8.0] * n,
            'Gamma': [-10.0] * n,
        },
        index=starts,
    )
    band_ratios_df = pd.DataFrame(
        {
            'theta_alpha_db': [1.0] * n,
            'beta_alpha_db': [-2.0] * n,
            'beta_theta_db': [-1.0] * n,
            'theta_alpha': [0.8] * n,
            'beta_alpha': [0.5] * n,
            'beta_theta': [0.6] * n,
        },
        index=starts,
    )
    se_df = pd.DataFrame({'spectral_entropy': [0.7] * n}, index=starts)
    quality_df = pd.DataFrame(
        {
            'valid_ratio': [1.0 - r for r in excluded_ratios],
            'excluded_ratio': excluded_ratios,
            'usable': usable,
        },
        index=starts,
    )
    iaf_series = pd.Series([10.0] * n, index=starts)
    itf_series = pd.Series([6.0] * n, index=starts)

    return {
        'band_powers': band_powers_df,
        'band_ratios': band_ratios_df,
        'spectral_entropy': se_df,
        'iaf': iaf_series,
        'itf': itf_series,
        'quality': quality_df,
    }


def make_df_clean(segment_starts=None, n_segments=N_SEGMENTS):
    """TimeStamp列だけを持つ最小限のdf_cleanを合成する（セッション全体をカバー）。"""
    starts = segment_starts or _segment_starts(n_segments)
    end = starts[-1] + pd.Timedelta(minutes=SEGMENT_MINUTES)
    timestamps = pd.date_range(starts[0], end, freq='1s', inclusive='left')
    return pd.DataFrame({'TimeStamp': timestamps})


def make_fmtheta_series(segment_starts=None, n_segments=N_SEGMENTS):
    starts = segment_starts or _segment_starts(n_segments)
    end = starts[-1] + pd.Timedelta(minutes=SEGMENT_MINUTES)
    timestamps = pd.date_range(starts[0], end, freq='1s', inclusive='left')
    return pd.Series(0.5, index=timestamps)


class TestCalculateSegmentAnalysis:
    def test_last_segment_is_included_in_scoring_indices_by_default(self):
        """exclude_last_segment 削除後、最終セグメントは scoring_indices に残る。"""
        alpha_values = [1.0, 1.5, 2.0, 3.0]  # 最終セグメントが最高alpha
        statistical_df = make_statistical_df(alpha_values)

        result = calculate_segment_analysis(
            make_df_clean(),
            make_fmtheta_series(),
            statistical_df,
            segment_minutes=SEGMENT_MINUTES,
        )

        scoring_indices = result.metadata['scoring_indices']
        all_indices = result.segments['segment_index'].tolist()
        assert all_indices[-1] in scoring_indices, '最終セグメントが scoring_indices から外れている'

    def test_exclude_first_segment_still_excludes_first_only(self):
        """exclude_first_segment=True は先頭セグメントのみをscoring_indicesから外し、relaxingラベルを付ける。"""
        alpha_values = [1.0, 1.5, 2.0, 3.0]
        statistical_df = make_statistical_df(alpha_values)

        result = calculate_segment_analysis(
            make_df_clean(),
            make_fmtheta_series(),
            statistical_df,
            segment_minutes=SEGMENT_MINUTES,
            exclude_first_segment=True,
        )

        all_indices = result.segments['segment_index'].tolist()
        scoring_indices = result.metadata['scoring_indices']

        assert all_indices[0] not in scoring_indices, '先頭セグメントが scoring_indices から除外されていない'
        assert all_indices[-1] in scoring_indices, '最終セグメントが scoring_indices から外れている'

        first_row = result.table.iloc[0]
        assert first_row['備考'] == 'relaxing'
        assert 'post meditation' not in result.table['備考'].tolist()

    def test_noisy_last_segment_is_excluded_by_excluded_ratio(self):
        """データ由来(excluded_ratio超過)の除外は、最終セグメントでも従来どおり働く。"""
        alpha_values = [1.0, 1.5, 2.0, 3.0]
        # 最終セグメントの除外率がしきい値(0.2)を超える
        excluded_ratios = [0.05, 0.05, 0.05, 0.5]
        statistical_df = make_statistical_df(alpha_values, excluded_ratios=excluded_ratios)

        result = calculate_segment_analysis(
            make_df_clean(),
            make_fmtheta_series(),
            statistical_df,
            segment_minutes=SEGMENT_MINUTES,
        )

        all_indices = result.segments['segment_index'].tolist()
        scoring_indices = result.metadata['scoring_indices']

        assert all_indices[-1] not in scoring_indices, 'ノイズ超過の最終セグメントが除外されていない'
        last_row = result.table.iloc[-1]
        assert last_row['備考'] == 'noisy'
        assert 'post meditation' not in result.table['備考'].tolist()

    def test_artifact_last_segment_is_excluded_by_usable_flag(self):
        """データ由来(usable=False)の除外は、最終セグメントでも従来どおり働く。"""
        alpha_values = [1.0, 1.5, 2.0, 3.0]
        usable = [True, True, True, False]
        statistical_df = make_statistical_df(alpha_values, usable=usable)

        result = calculate_segment_analysis(
            make_df_clean(),
            make_fmtheta_series(),
            statistical_df,
            segment_minutes=SEGMENT_MINUTES,
        )

        all_indices = result.segments['segment_index'].tolist()
        scoring_indices = result.metadata['scoring_indices']

        assert all_indices[-1] not in scoring_indices, 'artifact判定の最終セグメントが除外されていない'
        last_row = result.table.iloc[-1]
        assert last_row['備考'] == 'artifact'


class TestCalculateBestMetrics:
    def test_best_metrics_can_come_from_last_segment(self):
        """最終セグメントがscoring対象なら、best値の候補として採用されうる。"""
        alpha_values = [1.0, 1.5, 2.0, 3.0]  # 最終セグメントが最高alpha
        statistical_df = make_statistical_df(alpha_values)

        result = calculate_segment_analysis(
            make_df_clean(),
            make_fmtheta_series(),
            statistical_df,
            segment_minutes=SEGMENT_MINUTES,
        )

        best_metrics = calculate_best_metrics(result)

        last_alpha_db = result.segments.set_index('segment_index').loc[
            result.segments['segment_index'].max(), 'alpha_mean'
        ]
        assert best_metrics['alpha_best'] == pytest.approx(last_alpha_db)

    def test_scoring_indices_include_last_segment_hand_built(self):
        """SegmentAnalysisResultを手で構築し、metadataレベルでも最終セグメントが候補に入ることを確認する。"""
        segments = pd.DataFrame(
            {
                'segment_index': [1, 2, 3, 4],
                'fmtheta_mean': [0.3, 0.4, 0.5, 0.6],
                'iaf_mean': [9.5, 9.8, 10.0, 10.2],
                'alpha_mean': [1.0, 1.5, 2.0, 2.7],
                'beta_mean': [-8.0, -8.2, -8.5, -8.8],
                'theta_alpha_ratio': [0.7, 0.75, 0.8, 0.9],
            }
        )
        # exclude_first_segment のみが効いている状態を模す（先頭を除外、最終は残す）
        scoring_indices = [2, 3, 4]

        result = SegmentAnalysisResult(
            segments=segments,
            table=pd.DataFrame(),
            normalized=pd.DataFrame(),
            metadata={'scoring_indices': scoring_indices},
            band_power_table=pd.DataFrame(),
            metrics_table=pd.DataFrame(),
        )

        best_metrics = calculate_best_metrics(result)

        # 最終セグメント(index=4)がalpha_best/iaf_best/theta_alpha_bestの最大値
        assert best_metrics['alpha_best'] == pytest.approx(2.7)
        assert best_metrics['iaf_best'] == pytest.approx(10.2)
        assert best_metrics['theta_alpha_best'] == pytest.approx(0.9)
