"""
振幅ベースのアーチファクト除去のテスト

HSIがGoodのままアーチファクトが混入するケースを合成データで再現し、
チャネル×窓単位の除外が意図通り効くことを検証する。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.sensors.eeg.artifact import (  # noqa: E402
    ARTIFACT_RELATIVE_FLOOR_UV,
    ARTIFACT_WINDOW_SAMPLES,
    _window_p2p,
    calculate_amplitude_statistics,
    channel_thresholds,
    detect_artifact_windows,
    detect_hsi_windows,
    detect_valid_windows,
    segment_valid_ratio,
    summarize_artifacts,
)

SFREQ = 256.0
CHANNELS = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

# 統合テストは segment_minutes=1 で回すので、窓数は窓長から導出する。
# ARTIFACT_WINDOW_SAMPLES を変えてもセグメント数が3のまま保たれるようにするため、
# 定数をベタ書きしない（窓長は docs/adr/001 の通り調整対象）
WINDOWS_PER_SEGMENT = int(round(60.0 * SFREQ / ARTIFACT_WINDOW_SAMPLES))
N_SEGMENTS = 3
N_WINDOWS = N_SEGMENTS * WINDOWS_PER_SEGMENT


def make_clean_data(amplitude_uv=10.0, seed=0):
    """振幅が揃ったクリーンな合成EEG（n_channels, n_samples）を作る。"""
    rng = np.random.default_rng(seed)
    n_samples = N_WINDOWS * ARTIFACT_WINDOW_SAMPLES
    return rng.normal(0.0, amplitude_uv, size=(len(CHANNELS), n_samples))


def inject_burst(data, channel_idx, window_idx, amplitude_uv=500.0, freq_hz=2.0):
    """
    指定チャネルの指定窓に大振幅バーストを注入する。

    既定の2Hzはδ帯（1-4Hz）に収まる。PSDの下限は1Hzなので、
    それより低い周波数で注入するとバンドパワーに現れず、
    除去の有無を検証できないテストになる。
    """
    data = data.copy()
    start = window_idx * ARTIFACT_WINDOW_SAMPLES
    end = start + ARTIFACT_WINDOW_SAMPLES
    t = np.arange(ARTIFACT_WINDOW_SAMPLES) / SFREQ
    data[channel_idx, start:end] += amplitude_uv * np.sin(2 * np.pi * freq_hz * t)
    return data


class TestDetectArtifactWindows:
    def test_clean_data_is_fully_valid(self):
        valid = detect_artifact_windows(make_clean_data())
        assert valid.shape == (len(CHANNELS), N_WINDOWS)
        assert valid.all()

    def test_burst_rejects_only_that_channel_and_window(self):
        data = inject_burst(make_clean_data(), channel_idx=1, window_idx=5)
        valid = detect_artifact_windows(data)

        assert not valid[1, 5], 'バーストを注入した窓が除外されていない'
        assert valid[1, :5].all() and valid[1, 6:].all(), '同一チャネルの他窓まで除外している'
        assert valid[[0, 2, 3]].all(), '他チャネルまで除外している'

    def test_noisy_channel_does_not_kill_clean_channels(self):
        """耳側だけが恒常的に汚いケース。前頭チャネルは保持されること。"""
        data = make_clean_data()
        for w in range(N_WINDOWS):
            data = inject_burst(data, channel_idx=0, window_idx=w, amplitude_uv=800.0)

        valid = detect_artifact_windows(data)
        assert valid[[1, 2, 3]].all(), 'クリーンなチャネルが巻き添えで除外されている'

    def test_returns_empty_mask_when_data_shorter_than_one_window(self):
        data = np.zeros((len(CHANNELS), ARTIFACT_WINDOW_SAMPLES - 1))
        valid = detect_artifact_windows(data)
        assert valid.shape == (len(CHANNELS), 0)


class TestChannelThresholds:
    def test_absolute_cap_applies_to_high_amplitude_channel(self):
        """p2p中央値が大きいチャネルでは絶対閾値が効く。"""
        data = make_clean_data(amplitude_uv=60.0)
        thresholds = channel_thresholds(_window_p2p(data, ARTIFACT_WINDOW_SAMPLES))
        assert np.allclose(thresholds, 150.0)

    def test_relative_threshold_applies_to_low_amplitude_channel(self):
        """振幅が小さいチャネルでは絶対150μVより厳しい相対閾値が採用される。"""
        data = make_clean_data(amplitude_uv=5.0)
        p2p = _window_p2p(data, ARTIFACT_WINDOW_SAMPLES)
        thresholds = channel_thresholds(p2p)

        assert (thresholds < 150.0).all(), '相対閾値が効いていない'
        assert (thresholds >= ARTIFACT_RELATIVE_FLOOR_UV).all(), '下限を割り込んでいる'

    def test_floor_prevents_over_rejection_on_very_clean_channel(self):
        """極端にクリーンなチャネルで正常な変動まで落とさないこと。"""
        data = make_clean_data(amplitude_uv=1.0)
        valid = detect_artifact_windows(data)
        assert valid.all()


class TestAmplitudeStatistics:
    def test_reports_per_channel_rejection_rate(self):
        data = make_clean_data()
        for w in range(6):
            data = inject_burst(data, channel_idx=3, window_idx=w)

        stats = calculate_amplitude_statistics(data, CHANNELS)

        assert list(stats['Channel']) == ['TP9', 'AF7', 'AF8', 'TP10']
        assert stats.loc[stats['Channel'] == 'TP10', 'Rejected'].iloc[0] == pytest.approx(
            6 / N_WINDOWS * 100
        )
        assert (stats.loc[stats['Channel'] != 'TP10', 'Rejected'] == 0.0).all()

    def test_summary_reports_overall_ratio(self):
        data = inject_burst(make_clean_data(), channel_idx=0, window_idx=0)
        summary = summarize_artifacts(data, CHANNELS)

        assert summary['rejected_ratio'] == pytest.approx(1 / (len(CHANNELS) * N_WINDOWS))
        assert summary['channels'] == ['TP9', 'AF7', 'AF8', 'TP10']


class TestStatisticalDataFrameIntegration:
    """create_statistical_dataframe() 側の除外挙動。"""

    @staticmethod
    def _make_raw(data_uv):
        mne = pytest.importorskip('mne')
        mne.set_log_level('ERROR')
        info = mne.create_info(CHANNELS, SFREQ, 'eeg')
        return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)

    def _build(self, data_uv, segment_minutes=1):
        import pandas as pd

        from lib.statistical_dataframe import create_statistical_dataframe

        return create_statistical_dataframe(
            self._make_raw(data_uv),
            segment_minutes=segment_minutes,
            session_start=pd.Timestamp('2026-01-01 00:00:00'),
        )

    def test_clean_session_reports_zero_exclusion(self):
        result = self._build(make_clean_data())
        assert (result['quality']['excluded_ratio'] == 0.0).all()
        assert result['quality']['usable'].all()
        assert result['band_powers']['Alpha'].notna().all()

    def test_contaminated_segment_matches_clean_segments_after_rejection(self):
        """バーストを注入したセグメントのδが、クリーンなセグメントと同水準に戻ること。

        除去が効いていなければ、バーストの低周波成分でδが跳ね上がる。
        """
        dirty = make_clean_data()
        # 先頭セグメントにのみバーストを集中させる（残存率は50%を下回らせない）
        for w in range(WINDOWS_PER_SEGMENT // 2):
            dirty = inject_burst(dirty, channel_idx=1, window_idx=w, amplitude_uv=800.0)

        result = self._build(dirty)
        delta = result['band_powers']['Delta']

        assert result['quality']['excluded_ratio'].iloc[0] > 0, '汚染窓が除外されていない'
        assert delta.iloc[0] == pytest.approx(delta.iloc[1:].mean(), abs=1.0), (
            'アーチファクト除去後もδがクリーン区間と乖離している'
        )

    def test_heavily_contaminated_segment_is_nan(self):
        """残存率が閾値を下回るセグメントは指標を出さない。"""
        data = make_clean_data()
        # 先頭セグメントを全チャネル・全窓で汚染する
        for ch in range(len(CHANNELS)):
            for w in range(WINDOWS_PER_SEGMENT):
                data = inject_burst(data, channel_idx=ch, window_idx=w, amplitude_uv=800.0)

        result = self._build(data)

        assert not result['quality']['usable'].iloc[0]
        assert np.isnan(result['band_powers']['Alpha'].iloc[0])
        assert np.isnan(result['spectral_entropy']['spectral_entropy'].iloc[0])
        assert np.isnan(result['iaf'].iloc[0])
        assert np.isnan(result['itf'].iloc[0])


class TestSegmentValidRatio:
    def test_ratio_counts_channel_by_window_cells(self):
        valid = np.ones((4, 10), dtype=bool)
        valid[0, 0:2] = False  # 4x4=16セル中2セルが不良
        assert segment_valid_ratio(valid, 0, 4) == pytest.approx(14 / 16)

    def test_empty_range_returns_zero(self):
        valid = np.ones((4, 10), dtype=bool)
        assert segment_valid_ratio(valid, 3, 3) == 0.0


class TestDetectHsiWindows:
    """接触不良はHSIで窓単位に外す（窓単位の振幅除去を通る汚染があるため）。"""

    def test_channel_lost_mid_session_invalidates_only_later_windows(self):
        hsi = np.ones((len(CHANNELS), N_WINDOWS * ARTIFACT_WINDOW_SAMPLES))
        lost_at = 10 * ARTIFACT_WINDOW_SAMPLES
        hsi[3, lost_at:] = 2.0  # TP10が窓10の先頭からMedium

        valid = detect_hsi_windows(hsi)

        assert valid[3, :10].all()
        assert not valid[3, 10:].any()
        assert valid[:3].all()

    def test_window_is_invalid_only_when_bad_exceeds_half(self):
        hsi = np.ones((1, 3 * ARTIFACT_WINDOW_SAMPLES))
        half = ARTIFACT_WINDOW_SAMPLES // 2
        hsi[0, :half] = 2.0  # 窓0はちょうど半分
        hsi[0, ARTIFACT_WINDOW_SAMPLES : ARTIFACT_WINDOW_SAMPLES + half + 1] = 2.0  # 窓1は半分超

        assert detect_hsi_windows(hsi)[0].tolist() == [True, False, True]

    def test_nan_hsi_counts_as_good(self):
        hsi = np.full((1, 2 * ARTIFACT_WINDOW_SAMPLES), np.nan)
        assert detect_hsi_windows(hsi).all()

    def test_whole_session_bad_channel_is_fully_invalid(self):
        """PR #56のセッション全体での除外は、全窓が無効になる形でこの判定に含まれる。"""
        hsi = np.ones((len(CHANNELS), N_WINDOWS * ARTIFACT_WINDOW_SAMPLES))
        hsi[[0, 3]] = 2.0
        valid = detect_hsi_windows(hsi)
        assert not valid[[0, 3]].any()
        assert valid[[1, 2]].all()


class TestDetectValidWindows:
    def test_all_good_hsi_matches_amplitude_mask(self):
        """回帰確認: 全チャネルGoodなら合成マスクは従来の振幅マスクと一致する。"""
        data = make_clean_data()
        for w in (0, 5, 17):
            data = inject_burst(data, channel_idx=w % len(CHANNELS), window_idx=w)
        hsi = np.ones_like(data)

        np.testing.assert_array_equal(
            detect_valid_windows(data, hsi), detect_artifact_windows(data)
        )

    def test_combines_amplitude_and_hsi_with_and(self):
        data = inject_burst(make_clean_data(), channel_idx=1, window_idx=0)
        hsi = np.ones_like(data)
        hsi[2, : 2 * ARTIFACT_WINDOW_SAMPLES] = 4.0

        valid = detect_valid_windows(data, hsi)

        assert not valid[1, 0]
        assert not valid[2, 0] and not valid[2, 1]
        assert valid.sum() == valid.size - 3
