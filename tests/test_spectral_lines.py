"""
線スペクトル除去とTP指標（tp_iaf_hz）のテスト

- 1/fノイズ + 幅のある10Hz α + 4.5Hz間隔の鋭い線の合成データで、線だけが消えαが残ること
- 窓ごとのPSD（0.25Hz分解能）とセッションPSD（0.125Hz分解能）の両方で動くこと
- TPの有効窓が3分未満なら tp_iaf_hz が空欄、3分以上なら値が入ること
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mne = pytest.importorskip('mne')
mne.set_log_level('ERROR')

from lib.report.steps_eeg import TP_MIN_VALID_SEC, compute_tp_metrics  # noqa: E402
from lib.sensors.eeg.artifact import ARTIFACT_WINDOW_SAMPLES  # noqa: E402
from lib.sensors.eeg.frequency import calculate_psd_masked  # noqa: E402
from lib.sensors.eeg.spectral_lines import (  # noqa: E402
    detect_spectral_lines,
    suppress_spectral_lines,
)
from lib.session_log import _extract_session_data  # noqa: E402

SFREQ = 256.0
WIN = ARTIFACT_WINDOW_SAMPLES
LINE_FREQS = [4.5 * k for k in range(1, 9)]  # 4.5 ... 36.0 Hz


def make_signal(n_samples, line_amp=10.0, alpha_amp=8.0, seed=0):
    """1/fノイズ + 幅のある10Hz α（約9-11Hzの帯域ノイズ）+ 鋭い線（μV）。"""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, 1, n_samples))
    f = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    spectrum[1:] /= np.sqrt(f[1:])  # パワーが1/f
    pink = np.fft.irfft(spectrum, n_samples)
    pink *= 15.0 / pink.std()

    sos = signal.butter(4, [9.0, 11.0], btype='bandpass', fs=SFREQ, output='sos')
    alpha = signal.sosfilt(sos, rng.normal(0, 1, n_samples))
    alpha *= alpha_amp / alpha.std()

    t = np.arange(n_samples) / SFREQ
    lines = sum(line_amp * np.sin(2 * np.pi * f0 * t + k) for k, f0 in enumerate(LINE_FREQS))
    return pink + alpha + lines


def welch_psd(x, n_fft):
    freqs, psd = signal.welch(x, fs=SFREQ, nperseg=n_fft, noverlap=n_fft // 2)
    keep = (freqs >= 1.0) & (freqs <= 50.0)
    return freqs[keep], psd[keep]


def band_power(freqs, psd, lo, hi):
    return psd[(freqs >= lo) & (freqs <= hi)].sum()


def alpha_peak(freqs, psd):
    band = (freqs >= 8.0) & (freqs <= 13.0)
    return freqs[band][psd[band].argmax()]


@pytest.mark.parametrize('n_fft', [WIN, 2 * WIN], ids=['0.25Hz', '0.125Hz'])
class TestSpectralLineSuppression:
    def test_lines_are_flagged_but_alpha_is_not(self, n_fft):
        freqs, psd = welch_psd(make_signal(300 * int(SFREQ)), n_fft)

        mask = detect_spectral_lines(psd, freqs)

        for f0 in LINE_FREQS:
            assert mask[np.abs(freqs - f0) < 0.15].any(), f'{f0}Hzの線が検出されていない'
        alpha_core = (freqs >= 9.75) & (freqs <= 10.5)  # 9.0Hzの線の裾は肩の手前(9.5Hz)まで
        assert not mask[alpha_core].any(), 'αのピークが線として検出された'
        # 線以外のビンを大量に巻き込まない（1本あたり1Hz幅 + 両隣まで）
        assert mask.sum() <= len(LINE_FREQS) * (1.0 / (freqs[1] - freqs[0]) + 2)

    def test_suppression_flattens_lines_and_preserves_alpha(self, n_fft):
        freqs, psd = welch_psd(make_signal(300 * int(SFREQ)), n_fft)
        clean_freqs, clean_psd = welch_psd(make_signal(300 * int(SFREQ), line_amp=0.0), n_fft)
        assert np.array_equal(freqs, clean_freqs)

        cleaned, lines = suppress_spectral_lines(psd, freqs)

        assert len(lines) == len(LINE_FREQS)
        for f0 in LINE_FREQS:
            idx = np.abs(freqs - f0).argmin()
            before_db = 10 * np.log10(psd[idx] / clean_psd[idx])  # 線を足す前の同一信号との差
            after_db = 10 * np.log10(cleaned[idx] / clean_psd[idx])
            assert before_db > 6.0
            assert abs(after_db) < 5.0, f'{f0}Hzの線が残っている ({after_db:.1f}dB)'

        assert alpha_peak(freqs, cleaned) == pytest.approx(
            alpha_peak(clean_freqs, clean_psd), abs=0.5
        )
        assert alpha_peak(freqs, cleaned) == pytest.approx(10.0, abs=0.75)
        # α帯域(9.5-10.5Hz)のパワーは10%以内で保たれる
        np.testing.assert_allclose(
            band_power(freqs, cleaned, 9.5, 10.5), band_power(freqs, psd, 9.5, 10.5), rtol=0.1
        )

    def test_strong_lines_are_still_narrow(self, n_fft):
        """+30dB超の強い線もHann窓の裾で幅広と誤判定されない。"""
        freqs, psd = welch_psd(make_signal(300 * int(SFREQ), line_amp=40.0), n_fft)
        mask = detect_spectral_lines(psd, freqs)
        for f0 in LINE_FREQS:
            assert mask[np.abs(freqs - f0) < 0.15].any()

    def test_clean_signal_is_unchanged(self, n_fft):
        freqs, psd = welch_psd(make_signal(300 * int(SFREQ), line_amp=0.0), n_fft)
        cleaned, lines = suppress_spectral_lines(psd, freqs)
        assert lines == []
        np.testing.assert_array_equal(cleaned, psd)


class TestSuppressRows:
    def test_reference_detection_applies_to_every_row(self):
        """参照PSDで検出した線を、揺らぎのある全窓のPSDで補間する。"""
        x = make_signal(120 * int(SFREQ))
        freqs, _ = welch_psd(x[:WIN], WIN)
        rows = np.array([welch_psd(x[w * WIN : (w + 1) * WIN], WIN)[1] for w in range(len(x) // WIN)])

        cleaned, lines = suppress_spectral_lines(rows, freqs, detect_from=rows.mean(axis=0))

        assert cleaned.shape == rows.shape
        assert len(lines) == len(LINE_FREQS)
        idx = np.abs(freqs - 4.5).argmin()
        assert cleaned[:, idx].mean() < rows[:, idx].mean() / 10

    def test_multi_row_without_reference_is_an_error(self):
        with pytest.raises(ValueError):
            suppress_spectral_lines(np.ones((3, 50)), np.arange(50) * 0.25)


class TestMaskedPsdSuppressesLines:
    def test_calculate_psd_masked_removes_line_and_reports_it(self):
        n_windows = 40
        data = np.vstack([
            make_signal(n_windows * WIN),
            make_signal(n_windows * WIN, line_amp=0.0, seed=1),
        ])
        raw = mne.io.RawArray(
            data * 1e-6, mne.create_info(['RAW_AF7', 'RAW_AF8'], SFREQ, 'eeg'), verbose=False
        )

        psd_dict = calculate_psd_masked(raw, np.ones((2, n_windows), dtype=bool), WIN)

        assert len(psd_dict['line_freqs']['RAW_AF7']) == len(LINE_FREQS)
        assert psd_dict['line_freqs']['RAW_AF8'] == []
        freqs = psd_dict['freqs']
        idx = np.abs(freqs - 4.5).argmin()
        local = np.median(psd_dict['psds'][0][idx - 16 : idx + 17])
        assert abs(10 * np.log10(psd_dict['psds'][0][idx] / local)) < 3.0


def make_tp_raw(n_windows, seed=0):
    x = np.vstack([make_signal(n_windows * WIN, line_amp=0.0, seed=seed + k) for k in range(4)])
    info = mne.create_info(['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10'], SFREQ, 'eeg')
    return mne.io.RawArray(x * 1e-6, info, verbose=False)


class TestTpMetrics:
    N_WINDOWS = 120  # 8分

    def _valid_mask(self, tp_valid_windows_each):
        valid = np.ones((4, self.N_WINDOWS), dtype=bool)
        for idx in (0, 3):
            valid[idx, tp_valid_windows_each:] = False
        return valid

    def test_tp_iaf_is_blank_below_minimum_valid_time(self):
        each = 20  # 80秒 x 2ch = 160秒 < 180秒
        assert 2 * each * WIN / SFREQ < TP_MIN_VALID_SEC

        metrics = compute_tp_metrics(make_tp_raw(self.N_WINDOWS), self._valid_mask(each), WIN)

        assert np.isnan(metrics['tp_iaf_hz'])
        assert metrics['tp9_valid_pct'] == pytest.approx(each / self.N_WINDOWS * 100)
        assert metrics['tp10_valid_pct'] == pytest.approx(each / self.N_WINDOWS * 100)

    def test_tp_iaf_is_recorded_at_minimum_valid_time(self):
        each = 24  # 96秒 x 2ch = 192秒 >= 180秒
        assert 2 * each * WIN / SFREQ >= TP_MIN_VALID_SEC

        metrics = compute_tp_metrics(make_tp_raw(self.N_WINDOWS), self._valid_mask(each), WIN)

        assert metrics['tp_iaf_hz'] == pytest.approx(10.0, abs=0.75)
        assert metrics['tp9_valid_pct'] == pytest.approx(each / self.N_WINDOWS * 100)

    def test_one_tp_channel_can_carry_the_minimum(self):
        valid = np.ones((4, self.N_WINDOWS), dtype=bool)
        valid[0] = False  # TP9は全窓無効、TP10だけで180秒以上
        metrics = compute_tp_metrics(make_tp_raw(self.N_WINDOWS), valid, WIN)
        assert metrics['tp9_valid_pct'] == 0.0
        assert metrics['tp10_valid_pct'] == 100.0
        assert metrics['tp_iaf_hz'] == pytest.approx(10.0, abs=0.75)

    def test_session_log_columns(self):
        base = {
            'data_info': {'start_time': pd.Timestamp('2026-10-29 08:00:00'), 'duration_sec': 600}
        }
        blank = _extract_session_data(base)
        assert np.isnan(blank['tp9_valid_pct']) and np.isnan(blank['tp_iaf_hz'])

        metrics = compute_tp_metrics(make_tp_raw(self.N_WINDOWS), self._valid_mask(20), WIN)
        row = _extract_session_data({**base, 'tp_metrics': metrics})
        assert row['tp10_valid_pct'] == pytest.approx(20 / self.N_WINDOWS * 100)
        assert np.isnan(row['tp_iaf_hz'])


def make_custom(n_samples, comb_freqs=(), comb_amp=10.0, narrow_alpha_hz=None, spike_hz=None, seed=0):
    """1/fノイズ + 任意の線 + 幅の狭い本物のα(約0.4Hz幅)。孤立ピークと櫛を区別するテスト用。"""
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.normal(0, 1, n_samples))
    f = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    spectrum[1:] /= np.sqrt(f[1:])
    x = np.fft.irfft(spectrum, n_samples)
    x *= 15.0 / x.std()
    t = np.arange(n_samples) / SFREQ
    for k, f0 in enumerate(comb_freqs):
        x = x + comb_amp * np.sin(2 * np.pi * f0 * t + k)
    if narrow_alpha_hz is not None:
        sos = signal.butter(
            4, [narrow_alpha_hz - 0.2, narrow_alpha_hz + 0.2], btype='bandpass', fs=SFREQ, output='sos'
        )
        narrow = signal.sosfilt(sos, rng.normal(0, 1, n_samples))
        x = x + narrow * (3.0 / narrow.std())
    if spike_hz is not None:
        x = x + 6.0 * np.sin(2 * np.pi * spike_hz * t)
    return x


@pytest.mark.parametrize('n_fft', [WIN, 2 * WIN], ids=['0.25Hz', '0.125Hz'])
class TestOnlyHarmonicSeriesIsRemoved:
    """孤立した狭いピークは本物の脳波かもしれないので触らない(10-27 AF8の8.5Hz等の誤検出対策)。"""

    N = 300 * int(SFREQ)

    @staticmethod
    def _near(freqs, mask, f0, tol=0.2):
        return bool(mask[np.abs(freqs - f0) <= tol].any())

    def test_isolated_narrow_alpha_is_kept(self, n_fft):
        freqs, psd = welch_psd(make_custom(self.N, narrow_alpha_hz=8.5), n_fft)
        mask = detect_spectral_lines(psd, freqs)
        assert not mask.any()
        cleaned, lines = suppress_spectral_lines(psd, freqs)
        assert lines == []
        np.testing.assert_array_equal(cleaned, psd)

    def test_comb_is_removed_and_adjacent_narrow_alpha_is_kept(self, n_fft):
        comb = [4.5 * k for k in range(1, 9)]
        freqs, psd = welch_psd(make_custom(self.N, comb_freqs=comb, narrow_alpha_hz=8.5), n_fft)

        mask = detect_spectral_lines(psd, freqs)

        for f0 in comb:
            assert self._near(freqs, mask, f0), f'{f0}Hzの櫛が残っている'
        assert not self._near(freqs, mask, 8.5, tol=0.1), '8.5Hzの本物のαが除去された'

    def test_jittered_comb_is_removed(self, n_fft):
        rng = np.random.default_rng(3)
        comb = [7.1 * k + rng.uniform(-0.4, 0.4) for k in range(1, 7)]  # 7.1〜42.6Hz
        freqs, psd = welch_psd(make_custom(self.N, comb_freqs=comb, comb_amp=12.0), n_fft)

        mask = detect_spectral_lines(psd, freqs)

        for f0 in comb:
            assert self._near(freqs, mask, f0), f'{f0:.2f}Hzの櫛が残っている'

    def test_isolated_spike_is_kept(self, n_fft):
        freqs, psd = welch_psd(make_custom(self.N, spike_hz=39.0), n_fft)
        mask = detect_spectral_lines(psd, freqs)
        assert not mask.any()

    def test_weak_fundamental_is_recovered_once_comb_is_accepted(self, n_fft):
        comb = [4.5 * k for k in range(1, 9)]
        x = make_custom(self.N, comb_freqs=comb)
        t = np.arange(self.N) / SFREQ
        x = x - 10.0 * np.sin(2 * np.pi * 4.5 * t) + 3.5 * np.sin(2 * np.pi * 4.5 * t)  # 基本波だけ弱める
        freqs, psd = welch_psd(x, n_fft)

        mask = detect_spectral_lines(psd, freqs)

        assert self._near(freqs, mask, 4.5)
