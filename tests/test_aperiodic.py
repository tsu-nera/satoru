"""
非周期成分（1/f）分離モジュールの真値検証テスト

合成PSD（既知のexponent/offset、既知のピーク中心周波数・高さ）を使い、
fit_aperiodic / find_band_peak / oscillatory_band_power が期待値を
復元できるかを確認する。

specparam自体が持つエッジ除外・SNR閾値の挙動と、本モジュールが追加で
かけているガード（PEAK_EDGE_MARGIN_HZ / MIN_REPORT_PEAK_HEIGHT）を
区別して検証するため、ガードのテストはspecparamの出力をモックして行う
（specparamは端に近いピークやごく小さいピークをそもそも検出しないことが
多く、実データだけではこちらのガードが実際に効く場面を再現しにくいため）。
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lib.sensors.eeg.aperiodic import (
    FIT_RANGE_HZ,
    MIN_REPORT_PEAK_HEIGHT,
    PEAK_EDGE_MARGIN_HZ,
    find_band_peak,
    fit_aperiodic,
    oscillatory_band_power,
)


def _synthesize_psd(offset, exponent, freqs, peak_cf=None, peak_height=0.0, peak_bw=1.5, noise=0.01, seed=0):
    """
    1/f（＋任意でガウシアンピーク）の合成PSDを生成する。

    log10(PSD) = offset - exponent * log10(f) + peak_height * exp(-(f-cf)^2 / (2*bw^2))
    """
    rng = np.random.default_rng(seed)
    aperiodic = 10 ** offset / freqs ** exponent
    if peak_cf is not None and peak_height > 0:
        peak_log = peak_height * np.exp(-0.5 * ((freqs - peak_cf) / peak_bw) ** 2)
        psd = aperiodic * (10 ** peak_log)
    else:
        psd = aperiodic
    psd = psd * (1 + noise * rng.standard_normal(len(freqs)))
    return np.abs(psd)


class TestFitAperiodicRecoversAperiodicParams:
    """既知のexponent/offsetを復元できること"""

    def test_recovers_known_exponent_and_offset(self):
        freqs = np.linspace(0.5, 45, 800)
        true_offset, true_exponent = 1.05, 1.32
        psd = _synthesize_psd(true_offset, true_exponent, freqs, noise=0.01, seed=1)

        result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.offset == pytest.approx(true_offset, abs=0.05)
        assert result.exponent == pytest.approx(true_exponent, abs=0.05)
        assert result.r_squared > 0.98
        assert result.n_peaks == 0
        assert result.peaks.empty

    def test_higher_exponent_recovered_too(self):
        """複数の指数（フラットな系列と急峻な系列）でも復元できること"""
        freqs = np.linspace(0.5, 45, 800)
        for true_offset, true_exponent in [(0.4, 0.6), (1.5, 1.8)]:
            psd = _synthesize_psd(true_offset, true_exponent, freqs, noise=0.01, seed=2)
            result = fit_aperiodic(freqs, psd)
            assert result is not None
            assert result.offset == pytest.approx(true_offset, abs=0.1)
            assert result.exponent == pytest.approx(true_exponent, abs=0.1)


class TestFitAperiodicRecoversPeak:
    """既知のピーク（中心周波数・高さ）を復元できること"""

    def test_recovers_known_alpha_peak(self):
        freqs = np.linspace(0.5, 45, 1000)
        true_offset, true_exponent = 1.0, 1.3
        true_cf, true_height = 10.0, 0.6
        psd = _synthesize_psd(
            true_offset, true_exponent, freqs,
            peak_cf=true_cf, peak_height=true_height, peak_bw=1.5,
            noise=0.005, seed=3,
        )

        result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 1
        peak = result.peaks.iloc[0]
        assert peak['center_hz'] == pytest.approx(true_cf, abs=0.3)
        assert peak['height'] == pytest.approx(true_height, abs=0.15)

        band_peak = find_band_peak(result, (8.0, 13.0))
        assert band_peak is not None
        assert band_peak['center_hz'] == pytest.approx(true_cf, abs=0.3)


class TestFindBandPeakNoFalsePositive:
    """ピークが無い合成データで誤検出しないこと"""

    def test_no_peak_returns_none(self):
        freqs = np.linspace(0.5, 45, 800)
        psd = _synthesize_psd(1.0, 1.3, freqs, noise=0.01, seed=4)

        result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 0
        assert find_band_peak(result, (4.0, 8.0)) is None
        assert find_band_peak(result, (8.0, 13.0)) is None


def _make_mock_specparam(ap_params, peak_params, metrics=None):
    """specparam.SpectralModelの戻り値をモックする。"""
    mock_fm = MagicMock()
    mock_fm.get_params.side_effect = lambda kind: (
        np.asarray(ap_params) if kind == 'aperiodic' else np.asarray(peak_params)
    )
    mock_fm.results.metrics.results = metrics or {'error_mae': 0.01, 'gof_rsquared': 0.98}
    return mock_fm


class TestPeakEdgeGuard:
    """フィット範囲の端に立てたピークがPEAK_EDGE_MARGIN_HZガードで棄却されること"""

    def test_peak_near_low_edge_is_rejected(self):
        low, high = FIT_RANGE_HZ
        edge_cf = low + PEAK_EDGE_MARGIN_HZ - 0.1  # ガード境界のすぐ内側（棄却されるべき）
        mock_fm = _make_mock_specparam(
            ap_params=[1.0, 1.3],
            peak_params=[[edge_cf, 0.5, 1.0]],
        )
        with patch('lib.sensors.eeg.aperiodic.SpectralModel', return_value=mock_fm):
            freqs = np.linspace(0.5, 45, 100)
            psd = np.ones_like(freqs)
            result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 0
        assert result.peaks.empty

    def test_peak_near_high_edge_is_rejected(self):
        low, high = FIT_RANGE_HZ
        edge_cf = high - PEAK_EDGE_MARGIN_HZ + 0.1  # ガード境界のすぐ内側（棄却されるべき）
        mock_fm = _make_mock_specparam(
            ap_params=[1.0, 1.3],
            peak_params=[[edge_cf, 0.5, 1.0]],
        )
        with patch('lib.sensors.eeg.aperiodic.SpectralModel', return_value=mock_fm):
            freqs = np.linspace(0.5, 45, 100)
            psd = np.ones_like(freqs)
            result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 0

    def test_peak_well_inside_range_is_kept(self):
        low, high = FIT_RANGE_HZ
        inside_cf = (low + high) / 2
        mock_fm = _make_mock_specparam(
            ap_params=[1.0, 1.3],
            peak_params=[[inside_cf, 0.5, 1.0]],
        )
        with patch('lib.sensors.eeg.aperiodic.SpectralModel', return_value=mock_fm):
            freqs = np.linspace(0.5, 45, 100)
            psd = np.ones_like(freqs)
            result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 1
        assert result.peaks.iloc[0]['center_hz'] == pytest.approx(inside_cf)


class TestMinPeakHeightGuard:
    """MIN_REPORT_PEAK_HEIGHT未満の微小ピークが棄却されること"""

    def test_tiny_peak_is_rejected(self):
        tiny_height = MIN_REPORT_PEAK_HEIGHT - 0.05
        mock_fm = _make_mock_specparam(
            ap_params=[1.0, 1.3],
            peak_params=[[10.0, tiny_height, 1.0]],
        )
        with patch('lib.sensors.eeg.aperiodic.SpectralModel', return_value=mock_fm):
            freqs = np.linspace(0.5, 45, 100)
            psd = np.ones_like(freqs)
            result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 0

    def test_peak_above_threshold_is_kept(self):
        ok_height = MIN_REPORT_PEAK_HEIGHT + 0.05
        mock_fm = _make_mock_specparam(
            ap_params=[1.0, 1.3],
            peak_params=[[10.0, ok_height, 1.0]],
        )
        with patch('lib.sensors.eeg.aperiodic.SpectralModel', return_value=mock_fm):
            freqs = np.linspace(0.5, 45, 100)
            psd = np.ones_like(freqs)
            result = fit_aperiodic(freqs, psd)

        assert result is not None
        assert result.n_peaks == 1
        assert result.peaks.iloc[0]['height'] == pytest.approx(ok_height)


class TestOscillatoryBandPower:
    """振動性バンドパワーが既知の高さに対して期待通りのdBを返すこと"""

    def test_recovers_expected_db_for_known_peak(self):
        freqs = np.linspace(0.5, 45, 1000)
        true_offset, true_exponent = 1.0, 1.3
        true_cf, true_height = 10.0, 0.6
        psd = _synthesize_psd(
            true_offset, true_exponent, freqs,
            peak_cf=true_cf, peak_height=true_height, peak_bw=1.5,
            noise=0.002, seed=5,
        )

        result = fit_aperiodic(freqs, psd)
        assert result is not None

        # ピーク帯域中心付近の狭い帯域で評価し、非周期成分とのdB差が
        # ほぼpeak_height * 10（log10パワー→dB換算）に一致することを確認する。
        band = (true_cf - 0.5, true_cf + 0.5)
        db = oscillatory_band_power(freqs, psd, result, band)

        expected_db = true_height * 10  # log10パワーのpeak_heightは10倍するとdB相当
        assert db == pytest.approx(expected_db, abs=2.0)

    def test_zero_for_pure_aperiodic_signal(self):
        """ピークが無ければ振動性パワーはほぼ0dBになること"""
        freqs = np.linspace(0.5, 45, 800)
        psd = _synthesize_psd(1.0, 1.3, freqs, noise=0.005, seed=6)

        result = fit_aperiodic(freqs, psd)
        assert result is not None

        db = oscillatory_band_power(freqs, psd, result, (8.0, 13.0))
        assert db == pytest.approx(0.0, abs=1.0)
