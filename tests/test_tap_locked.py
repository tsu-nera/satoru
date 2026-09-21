"""タップ連動解析（lib.tap_locked）とスライディング窓バンドパワーのテスト"""

import numpy as np
import pandas as pd
import pytest

from lib.sensors.eeg.sliding_power import compute_sliding_band_power
from lib.tap_locked import (
    contrast_mw_vs_control,
    detrend_and_zscore,
    label_tap_context,
    peri_event_average,
    required_sessions,
)

SFREQ = 256.0


def _sine_eeg(n_samples: int, freq_hz: float, amplitude_uv: float = 20.0, n_channels: int = 4):
    """指定周波数の正弦波からなるダミーEEG（チャネル×サンプル、μV）"""
    t = np.arange(n_samples) / SFREQ
    wave = amplitude_uv * np.sin(2 * np.pi * freq_hz * t)
    return np.tile(wave, (n_channels, 1))


def _window_frame(n_windows: int, start='2026-09-21 07:41:00'):
    """1秒刻みの窓DataFrame（t_sec / center_ts / バンド列）"""
    return pd.DataFrame({
        't_sec': np.arange(n_windows, dtype=float),
        'center_ts': pd.date_range(start, periods=n_windows, freq='1s'),
        'alpha': np.zeros(n_windows),
    })


class TestComputeSlidingBandPower:
    def test_alpha_peak_is_strongest_band(self):
        """10Hzの正弦波では alpha が最大になる"""
        data = _sine_eeg(int(SFREQ * 20), freq_hz=10.0)
        out = compute_sliding_band_power(data, SFREQ)

        first = out.iloc[0]
        assert first['alpha'] > first['theta']
        assert first['alpha'] > first['beta']

    def test_window_count_and_columns(self):
        data = _sine_eeg(int(SFREQ * 10), freq_hz=10.0)
        out = compute_sliding_band_power(data, SFREQ, window_samples=1024, step_samples=256)

        expected = len(np.arange(0, data.shape[1] - 1024 + 1, 256))
        assert len(out) == expected
        assert 'center_ts' not in out.columns
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            assert band in out.columns

    def test_timestamps_are_attached_at_window_center(self):
        n = int(SFREQ * 10)
        stamps = pd.date_range('2026-09-21 07:40:00', periods=n, freq=pd.Timedelta(seconds=1 / SFREQ))
        out = compute_sliding_band_power(_sine_eeg(n, 10.0), SFREQ, timestamps=stamps)

        assert out['center_ts'].iloc[0] == stamps[512]

    def test_artifact_channel_is_excluded(self):
        """1チャネルだけ巨大振幅にすると、その窓の採用チャネル数が減る"""
        data = _sine_eeg(int(SFREQ * 10), freq_hz=10.0)
        data[0, 1024:2048] += 5000.0
        out = compute_sliding_band_power(data, SFREQ)

        contaminated = out[(out['t_sec'] >= 4.0) & (out['t_sec'] < 8.0)]
        assert (contaminated['n_clean_ch'] < 4).any()

    def test_rejects_short_input(self):
        with pytest.raises(ValueError, match='窓長'):
            compute_sliding_band_power(_sine_eeg(100, 10.0), SFREQ)

    def test_rejects_timestamp_length_mismatch(self):
        n = int(SFREQ * 10)
        with pytest.raises(ValueError, match='timestamps'):
            compute_sliding_band_power(_sine_eeg(n, 10.0), SFREQ, timestamps=pd.date_range('2026-09-21', periods=3))


class TestDetrendAndZscore:
    def test_linear_trend_is_removed(self):
        frame = _window_frame(60)
        frame['alpha'] = np.arange(60, dtype=float) * 0.5 + 3.0

        out = detrend_and_zscore(frame, ['alpha'])

        # 完全な直線は残差が浮動小数点誤差だけになるため、z化せず NaN を返す
        assert out['alpha_z'].isna().all()

    def test_deviation_survives_detrending(self):
        frame = _window_frame(60)
        frame['alpha'] = np.arange(60, dtype=float) * 0.5
        frame.loc[30, 'alpha'] += 10.0

        out = detrend_and_zscore(frame, ['alpha'])

        assert out.loc[30, 'alpha_z'] == pytest.approx(out['alpha_z'].max())

    def test_constant_band_becomes_nan(self):
        frame = _window_frame(30)
        out = detrend_and_zscore(frame, ['alpha'])

        assert out['alpha_z'].isna().all()


class TestLabelTapContext:
    def test_mw_and_control_are_assigned(self):
        frame = _window_frame(200)
        taps = [pd.Timestamp('2026-09-21 07:42:00')]

        out = label_tap_context(frame, taps, mw_window_s=(1.0, 15.0), control_min_gap_s=40.0)

        at_tap = out['center_ts'] == taps[0]
        five_before = out['center_ts'] == taps[0] - pd.Timedelta(seconds=5)
        far = out['center_ts'] == taps[0] - pd.Timedelta(seconds=50)

        assert out.loc[five_before, 'tap_context'].iloc[0] == 'mw'
        assert out.loc[far, 'tap_context'].iloc[0] == 'control'
        assert out.loc[at_tap, 'tap_context'].iloc[0] == 'other'

    def test_window_just_after_tap_is_not_mw(self):
        """タップ直後は『次のタップまで』が遠いので MW にならない"""
        frame = _window_frame(200)
        taps = [pd.Timestamp('2026-09-21 07:42:00')]

        out = label_tap_context(frame, taps)
        after = out['center_ts'] == taps[0] + pd.Timedelta(seconds=5)

        assert out.loc[after, 'tap_context'].iloc[0] == 'other'

    def test_empty_taps_gives_all_other(self):
        out = label_tap_context(_window_frame(30), [])

        assert (out['tap_context'] == 'other').all()
        assert np.isinf(out['to_next_tap']).all()


class TestContrastMwVsControl:
    def test_positive_when_mw_windows_are_higher(self):
        frame = _window_frame(200)
        taps = [pd.Timestamp('2026-09-21 07:42:00')]
        labeled = label_tap_context(frame, taps)
        labeled['alpha_z'] = np.where(labeled['tap_context'] == 'mw', 1.0, 0.0)

        assert contrast_mw_vs_control(labeled, ['alpha'])['alpha'] == pytest.approx(1.0)

    def test_nan_when_a_group_is_missing(self):
        labeled = label_tap_context(_window_frame(30), [])
        labeled['alpha_z'] = 1.0

        assert np.isnan(contrast_mw_vs_control(labeled, ['alpha'])['alpha'])


class TestPeriEventAverage:
    def test_picks_the_window_at_each_lag(self):
        frame = _window_frame(200)
        frame['alpha_z'] = np.arange(200, dtype=float)
        taps = [frame['center_ts'].iloc[100]]

        out = peri_event_average(frame, taps, 'alpha', [-10.0, 0.0, 10.0])

        assert out.tolist() == [90.0, 100.0, 110.0]

    def test_out_of_range_lag_is_nan(self):
        frame = _window_frame(20)
        frame['alpha_z'] = 1.0
        taps = [frame['center_ts'].iloc[0]]

        assert np.isnan(peri_event_average(frame, taps, 'alpha', [-500.0])[0])


class TestRequiredSessions:
    def test_smaller_target_needs_more_sessions(self):
        effects = [0.1, -0.4, 0.4, 0.5, 0.05]

        strict = required_sessions(effects, 0.3)
        loose = required_sessions(effects, 0.5)

        assert strict is not None and loose is not None
        assert strict > loose

    def test_none_when_too_few_or_no_spread(self):
        assert required_sessions([0.2], 0.3) is None
        assert required_sessions([0.2, 0.2, 0.2], 0.3) is None

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError, match='target_effect_sd'):
            required_sessions([0.1, 0.2], 0.0)
