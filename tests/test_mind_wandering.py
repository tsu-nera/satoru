"""
マインドワンダリング（タップ打刻ログ由来の主観指標）のテスト

`tests/test_tap_log.py` のスタイル（tmp_path、from __future__ import annotations）
に倣う。実 data/ は汚さない。
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import numpy as np
import pandas as pd

from lib.loaders.tap_log import find_tap_log_for_session, load_tap_log_csv
from lib.mind_wandering import calculate_mind_wandering_stats, calculate_segment_tap_counts
from lib.report.steps_physio import analyze_mind_wandering
from lib.sensors.eeg.visualization.eeg_plots import plot_band_power_time_series
from lib.templates import MeditationReportRenderer

TAP_LOG_COLUMNS = ['seq', 'event', 'client_ts', 'server_ts', 'elapsed_s']


def _write_tap_csv(path: Path, rows: list[list]) -> None:
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(TAP_LOG_COLUMNS)
        writer.writerows(rows)


class TestLoadTapLogCsv:
    """load_tap_log_csv() のテスト"""

    def test_naive_local_timestamp_preserves_wall_clock(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'taps_2026-01-10--16-08-53.csv'
        _write_tap_csv(csv_path, [
            [0, 'start', '2026-01-10T07:08:53.000Z', '2026-01-10T16:08:53.000+09:00', 0.0],
            [1, 'tap', '2026-01-10T07:09:00.000Z', '2026-01-10T16:09:00.000+09:00', 7.0],
        ])

        df = load_tap_log_csv(csv_path)

        assert 'TimeStamp' in df.columns
        # +09:00の壁時計をそのまま保持していること（UTC変換で9時間ずれないこと）
        assert df['TimeStamp'].iloc[0] == pd.Timestamp('2026-01-10 16:08:53')
        assert df['TimeStamp'].iloc[1] == pd.Timestamp('2026-01-10 16:09:00')
        assert df['TimeStamp'].dt.tz is None

        assert df.attrs['session_start'] == pd.Timestamp('2026-01-10 16:08:53')

    def test_missing_required_column_raises(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'taps_bad.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seq', 'event'])
            writer.writerow([0, 'start'])

        import pytest
        with pytest.raises(ValueError):
            load_tap_log_csv(csv_path)


class TestCalculateMindWanderingStats:
    """calculate_mind_wandering_stats() のテスト"""

    def test_stats_computed_as_specified(self) -> None:
        df = pd.DataFrame([
            {'seq': 0, 'event': 'start', 'elapsed_s': 0.0},
            {'seq': 1, 'event': 'tap', 'elapsed_s': 30.0},
            {'seq': 2, 'event': 'tap', 'elapsed_s': 90.0},
            {'seq': 3, 'event': 'tap', 'elapsed_s': 120.0},
            {'seq': 4, 'event': 'stop', 'elapsed_s': 600.0},
        ])

        stats = calculate_mind_wandering_stats(df)

        assert stats['tap_count'] == 3
        assert stats['duration_min'] == 10.0
        assert stats['tap_rate_per_min'] == 0.3
        assert stats['time_to_first_tap_s'] == 30.0
        # ITI: 90-30=60, 120-90=30 → median=45
        assert stats['median_iti_s'] == 45.0
        itis = np.array([60.0, 30.0])
        expected_cv = float(np.std(itis) / np.mean(itis))
        assert stats['iti_cv'] == expected_cv

    def test_zero_taps_returns_zero_not_none(self) -> None:
        df = pd.DataFrame([
            {'seq': 0, 'event': 'start', 'elapsed_s': 0.0},
            {'seq': 1, 'event': 'stop', 'elapsed_s': 300.0},
        ])

        stats = calculate_mind_wandering_stats(df)

        assert stats['tap_count'] == 0
        assert stats['tap_count'] is not None
        assert stats['time_to_first_tap_s'] is None
        assert stats['median_iti_s'] is None
        assert stats['iti_cv'] is None

    def test_single_tap_has_no_iti(self) -> None:
        df = pd.DataFrame([
            {'seq': 0, 'event': 'start', 'elapsed_s': 0.0},
            {'seq': 1, 'event': 'tap', 'elapsed_s': 10.0},
            {'seq': 2, 'event': 'stop', 'elapsed_s': 100.0},
        ])

        stats = calculate_mind_wandering_stats(df)

        assert stats['tap_count'] == 1
        assert stats['time_to_first_tap_s'] == 10.0
        assert stats['median_iti_s'] is None
        assert stats['iti_cv'] is None


class TestCalculateSegmentTapCounts:
    """calculate_segment_tap_counts() のテスト"""

    def test_counts_taps_within_half_open_segment_bounds(self) -> None:
        tap_df = pd.DataFrame([
            {'event': 'start', 'TimeStamp': pd.Timestamp('2026-01-10 10:00:00')},
            # segment0にちょうど乗る（開始境界含む）
            {'event': 'tap', 'TimeStamp': pd.Timestamp('2026-01-10 10:00:00')},
            # segment0内
            {'event': 'tap', 'TimeStamp': pd.Timestamp('2026-01-10 10:01:00')},
            # segment0の終端境界（次segmentへ、[start, end)なのでsegment1側）
            {'event': 'tap', 'TimeStamp': pd.Timestamp('2026-01-10 10:03:00')},
            # segment1内
            {'event': 'tap', 'TimeStamp': pd.Timestamp('2026-01-10 10:04:00')},
        ])

        segments = pd.DataFrame([
            {'segment_start': pd.Timestamp('2026-01-10 10:00:00'), 'segment_end': pd.Timestamp('2026-01-10 10:03:00')},
            {'segment_start': pd.Timestamp('2026-01-10 10:03:00'), 'segment_end': pd.Timestamp('2026-01-10 10:06:00')},
        ])

        result = calculate_segment_tap_counts(tap_df, segments)

        assert list(result['tap_count']) == [2, 2]
        assert list(result.columns) == ['segment_index', 'segment_start', 'segment_end', 'tap_count']


class TestFindTapLogForSession:
    """find_tap_log_for_session() のテスト"""

    def test_returns_nearest_within_tolerance(self, tmp_path: Path) -> None:
        (tmp_path / 'taps_2026-01-10--16-08-00.csv').touch()
        (tmp_path / 'taps_2026-01-10--16-20-00.csv').touch()

        session_start = pd.Timestamp('2026-01-10 16:10:00')
        result = find_tap_log_for_session(tmp_path, session_start, tolerance_minutes=5.0)

        assert result is not None
        assert result.name == 'taps_2026-01-10--16-08-00.csv'

    def test_returns_none_when_out_of_tolerance(self, tmp_path: Path) -> None:
        (tmp_path / 'taps_2026-01-10--16-00-00.csv').touch()

        session_start = pd.Timestamp('2026-01-10 16:30:00')
        result = find_tap_log_for_session(tmp_path, session_start, tolerance_minutes=5.0)

        assert result is None

    def test_returns_none_when_no_files(self, tmp_path: Path) -> None:
        session_start = pd.Timestamp('2026-01-10 16:00:00')
        result = find_tap_log_for_session(tmp_path, session_start, tolerance_minutes=5.0)
        assert result is None


class TestAnalyzeMindWandering:
    """analyze_mind_wandering() のテスト"""

    def test_none_tap_df_returns_none_and_does_not_add_key(self) -> None:
        results: dict = {}
        result = analyze_mind_wandering(None, None, results)

        assert result is None
        assert 'mind_wandering' not in results


class TestPlotBandPowerTimeSeriesEventTimesDefault:
    """plot_band_power_time_series() の event_times=None が従来と同一挙動であること"""

    def test_default_event_times_none_produces_image_without_error(self, tmp_path: Path) -> None:
        n = 200
        timestamps = pd.date_range('2026-01-10 10:00:00', periods=n, freq='1s')
        df = pd.DataFrame({
            'TimeStamp': timestamps,
            'Alpha_TP9': np.random.rand(n),
            'Theta_TP9': np.random.rand(n),
        })

        img_path = tmp_path / 'band_power.png'
        fig = plot_band_power_time_series(df, img_path=img_path)

        assert img_path.exists()
        import matplotlib.pyplot as plt
        plt.close(fig)


class TestRendererMindWanderingContext:
    """build_context() が mind_wandering 関連キー無しの results で context に mind_wandering を作らないこと"""

    def test_no_mind_wandering_key_when_absent_from_results(self) -> None:
        renderer = MeditationReportRenderer()
        results: dict = {
            'data_info': {
                'start_time': pd.Timestamp('2026-01-10 16:00:00'),
                'end_time': pd.Timestamp('2026-01-10 16:30:00'),
                'duration_sec': 1800.0,
            },
        }
        context = renderer.build_context(results, Path('dummy.csv'))

        assert 'mind_wandering' not in context
