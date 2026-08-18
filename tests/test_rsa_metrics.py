"""
RSA指標（peak-valley振幅・呼吸追従バンドパワー）の真値検証

既知の呼吸周波数・既知の振幅で合成したR-R間隔を使い、
実装が期待値を復元できるかを確認する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lib.sensors.ecg.hrv import (
    RSA_BAND_MIN_HALF_WIDTH_HZ,
    RSA_BAND_REL_HALF_WIDTH,
    calculate_rsa_band_power,
)
from lib.sensors.ecg.respiration import calculate_rsa_amplitude

# 合成信号のパラメータ
BASE_RR_MS = 900.0  # 平均R-R間隔 → 約67 bpm
SYNTH_DURATION_SEC = 600.0


def _synthesize_rr(breathing_hz: float, amplitude_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """
    正弦波のRSAを載せたR-R間隔を合成

    RR(t) = BASE_RR_MS + amplitude_ms * sin(2π * breathing_hz * t)

    Returns
    -------
    rr_intervals : np.ndarray
        R-R間隔（ms）
    rr_time_sec : np.ndarray
        各R-R間隔に対応する時刻（秒）
    """
    rr_list = []
    time_list = []
    t = 0.0
    while t < SYNTH_DURATION_SEC:
        rr = BASE_RR_MS + amplitude_ms * np.sin(2 * np.pi * breathing_hz * t)
        t += rr / 1000.0
        rr_list.append(rr)
        time_list.append(t)
    return np.array(rr_list), np.array(time_list)


class TestRsaAmplitude:
    """peak-valley法によるRSA振幅"""

    def test_recovers_known_amplitude(self):
        """正弦波RSAのpeak-to-peak振幅（= 2 * 片振幅）を復元できる"""
        breathing_hz = 0.0741  # 4.45 bpm（超低速呼吸）
        amplitude_ms = 60.0
        rr, rr_time = _synthesize_rr(breathing_hz, amplitude_ms)

        # 呼吸周期の境界（トラフ相当）を真の呼吸周波数から与える
        period = 1.0 / breathing_hz
        bounds = np.arange(0.75 * period, rr_time[-1], period)

        cycles = calculate_rsa_amplitude(rr, rr_time, bounds)

        assert len(cycles) > 5
        measured = cycles['RSA Amplitude (ms)'].mean()
        expected = 2 * amplitude_ms
        # 離散的なR-R間隔でのサンプリング誤差を考慮して5%許容
        assert measured == pytest.approx(expected, rel=0.05)

    def test_scales_with_amplitude(self):
        """振幅を倍にすればRSA振幅も倍になる"""
        breathing_hz = 0.1
        rr_a, t_a = _synthesize_rr(breathing_hz, 30.0)
        rr_b, t_b = _synthesize_rr(breathing_hz, 60.0)

        period = 1.0 / breathing_hz
        bounds_a = np.arange(0.75 * period, t_a[-1], period)
        bounds_b = np.arange(0.75 * period, t_b[-1], period)

        amp_a = calculate_rsa_amplitude(rr_a, t_a, bounds_a)['RSA Amplitude (ms)'].mean()
        amp_b = calculate_rsa_amplitude(rr_b, t_b, bounds_b)['RSA Amplitude (ms)'].mean()

        assert amp_b / amp_a == pytest.approx(2.0, rel=0.05)

    def test_independent_of_breathing_rate(self):
        """
        同じ振幅なら呼吸数が変わってもRSA振幅は変わらない

        これがpeak-valley法を導入する理由。固定HF帯（0.15-0.4Hz）では
        4bpmと12bpmで測定値が大きく食い違う。
        """
        amplitude_ms = 50.0
        results = []
        for bpm in (4.0, 6.0, 12.0):
            breathing_hz = bpm / 60.0
            rr, t = _synthesize_rr(breathing_hz, amplitude_ms)
            period = 1.0 / breathing_hz
            bounds = np.arange(0.75 * period, t[-1], period)
            results.append(calculate_rsa_amplitude(rr, t, bounds)['RSA Amplitude (ms)'].mean())

        for measured in results:
            assert measured == pytest.approx(2 * amplitude_ms, rel=0.08)

    def test_short_cycles_are_dropped(self):
        """min_beatsに満たない周期は除外される"""
        rr, t = _synthesize_rr(0.0741, 60.0)
        # 1秒刻みの境界では1周期あたり1拍程度しか入らない
        bounds = np.arange(0.0, t[-1], 1.0)

        cycles = calculate_rsa_amplitude(rr, t, bounds, min_beats=3)

        assert cycles.empty

    def test_empty_result_has_expected_columns(self):
        """周期が得られない場合も列構成は保たれる"""
        rr, t = _synthesize_rr(0.0741, 60.0)
        cycles = calculate_rsa_amplitude(rr, t, np.array([0.0]))

        assert isinstance(cycles, pd.DataFrame)
        assert list(cycles.columns) == [
            'Time (min)', 'RSA Amplitude (ms)', 'EDR Amplitude (a.u.)', 'Beats'
        ]

    def test_edr_amplitude_requires_fs(self):
        """edr_signalだけ渡してedr_fsを省くとエラー"""
        rr, t = _synthesize_rr(0.0741, 60.0)
        with pytest.raises(ValueError, match='edr_fs'):
            calculate_rsa_amplitude(rr, t, np.array([0.0, 10.0]), edr_signal=np.zeros(100))


class TestRsaBandPower:
    """呼吸追従バンドのパワー"""

    def test_band_follows_breathing_rate(self):
        """バンド中心が呼吸周波数に一致し、幅が相対幅と下限のルールに従う"""
        rr, _ = _synthesize_rr(0.0741, 60.0)

        slow = calculate_rsa_band_power(rr, 4.45)
        assert slow is not None
        assert slow['center_hz'] == pytest.approx(4.45 / 60.0)
        # 超低速呼吸では相対幅（0.25 * 0.074 = 0.019）より絶対下限が効く
        half_width = (slow['band_high_hz'] - slow['band_low_hz']) / 2
        assert half_width == pytest.approx(RSA_BAND_MIN_HALF_WIDTH_HZ)

        normal = calculate_rsa_band_power(rr, 15.0)
        assert normal is not None
        half_width = (normal['band_high_hz'] - normal['band_low_hz']) / 2
        assert half_width == pytest.approx(RSA_BAND_REL_HALF_WIDTH * 0.25)

    def test_captures_power_at_breathing_frequency(self):
        """呼吸周波数に合わせたバンドは、外したバンドより大きなパワーを拾う"""
        breathing_bpm = 4.45
        rr, _ = _synthesize_rr(breathing_bpm / 60.0, 60.0)

        on_target = calculate_rsa_band_power(rr, breathing_bpm)
        off_target = calculate_rsa_band_power(rr, 15.0)

        assert on_target is not None and off_target is not None
        assert on_target['power'] > 10 * off_target['power']

    def test_scales_with_amplitude(self):
        """RSA振幅を倍にするとバンドパワーは約4倍（パワーは振幅の2乗）"""
        breathing_bpm = 6.0
        rr_a, _ = _synthesize_rr(breathing_bpm / 60.0, 30.0)
        rr_b, _ = _synthesize_rr(breathing_bpm / 60.0, 60.0)

        result_a = calculate_rsa_band_power(rr_a, breathing_bpm)
        result_b = calculate_rsa_band_power(rr_b, breathing_bpm)

        assert result_a is not None and result_b is not None
        assert result_b['power'] / result_a['power'] == pytest.approx(4.0, rel=0.15)

    @pytest.mark.parametrize('invalid_bpm', [np.nan, 0.0, -1.0])
    def test_invalid_breathing_rate_returns_none(self, invalid_bpm):
        """呼吸数が無効ならNoneを返す（呼び出し側でスキップできる）"""
        rr, _ = _synthesize_rr(0.0741, 60.0)
        assert calculate_rsa_band_power(rr, invalid_bpm) is None

    def test_band_low_is_clipped_to_floor(self):
        """極端に遅い呼吸でもバンド下限が0以下にならない"""
        rr, _ = _synthesize_rr(0.0741, 60.0)
        result = calculate_rsa_band_power(rr, 0.6)  # 0.01 Hz

        assert result is not None
        assert result['band_low_hz'] > 0
