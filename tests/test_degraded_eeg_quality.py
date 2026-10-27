"""
EEG品質が劣化したセッションでも、使える部分（チャネル×窓）から解析を続けることのテスト

- rawとHSIの窓境界が揃うこと（prepare_mne_raw は HSI Bad の行を落としてから raw を作る）
- 有効窓だけのPSDが、全窓有効なら従来のPSDと一致すること
- FAAが窓ごとに mastoid / Fpz 基準を切り替えること
- 区間の前頭有効率が50%未満なら前頭指標のセルがNAになること
- ヘッダーが欠落したMind Monitor CSVを修復して読めること
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mne = pytest.importorskip('mne')
mne.set_log_level('ERROR')

from lib.loaders.mind_monitor import MIND_MONITOR_CSV_COLUMNS, load_mind_monitor_csv  # noqa: E402
from lib.report.steps_eeg import mask_invalid_rows  # noqa: E402
from lib.segment_analysis import calculate_segment_analysis  # noqa: E402
from lib.sensors.eeg.artifact import ARTIFACT_WINDOW_SAMPLES, detect_hsi_windows  # noqa: E402
from lib.sensors.eeg.frequency import calculate_psd, calculate_psd_masked  # noqa: E402
from lib.sensors.eeg.frontal_asymmetry import (  # noqa: E402
    _apply_mastoid_reference,
    calculate_frontal_asymmetry,
)
from lib.sensors.eeg.preprocessing import prepare_mne_raw  # noqa: E402
from lib.session_log import _extract_session_data  # noqa: E402
from lib.statistical_dataframe import create_statistical_dataframe  # noqa: E402

SFREQ = 256.0
SITES = ['TP9', 'AF7', 'AF8', 'TP10']
CHANNELS = [f'RAW_{s}' for s in SITES]
WIN = ARTIFACT_WINDOW_SAMPLES
START = pd.Timestamp('2026-10-28 06:00:00')


def make_raw(data_uv, channels=CHANNELS):
    info = mne.create_info(channels, SFREQ, 'eeg')
    return mne.io.RawArray(data_uv * 1e-6, info, verbose=False)


def make_mind_monitor_df(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'TimeStamp': START + pd.to_timedelta(np.arange(n_samples) / SFREQ, unit='s'),
    })
    for site in SITES:
        df[f'RAW_{site}'] = rng.normal(800.0, 10.0, n_samples)
        df[f'HSI_{site}'] = 1.0
    return df


class TestHsiAlignment:
    def test_hsi_follows_raw_after_bad_rows_are_dropped(self):
        """HSI Bad（4）の行が落ちた後も、HSIの窓境界がrawの窓境界と一致する。"""
        df = make_mind_monitor_df(12 * WIN)
        dropped = 300
        df.loc[100 : 100 + dropped - 1, 'HSI_AF7'] = 4.0  # rawから落ちる行
        lost_at = 6 * WIN + dropped  # dfの行番号。rawでは窓6の先頭になる
        df.loc[lost_at:, 'HSI_TP10'] = 2.0

        mne_dict = prepare_mne_raw(df, apply_bandpass=False, apply_notch=False)

        assert mne_dict['hsi'].shape == (4, mne_dict['raw'].n_times)
        assert mne_dict['raw'].n_times == 12 * WIN - dropped
        valid = detect_hsi_windows(mne_dict['hsi'])
        tp10 = mne_dict['channels'].index('RAW_TP10')
        assert valid[tp10, :6].all()
        assert not valid[tp10, 6:].any()

    def test_mask_invalid_rows_blanks_only_invalid_windows(self):
        df = make_mind_monitor_df(4 * WIN)
        df['Alpha_TP10'] = 1.0
        df['Alpha_AF7'] = 0.5
        mne_dict = prepare_mne_raw(df, apply_bandpass=False, apply_notch=False)
        valid = np.ones((4, 4), dtype=bool)
        valid[3, 2:] = False

        masked = mask_invalid_rows(
            df, {'RAW_TP10': 'Alpha_TP10', 'RAW_AF7': 'Alpha_AF7'}, mne_dict, valid, WIN
        )

        assert masked['Alpha_TP10'].iloc[: 2 * WIN].notna().all()
        assert masked['Alpha_TP10'].iloc[2 * WIN :].isna().all()
        assert masked['Alpha_AF7'].notna().all()


class TestMaskedPsd:
    def test_all_valid_matches_unmasked_psd(self):
        """回帰確認: 全窓有効なら従来のWelch PSDと一致する。"""
        rng = np.random.default_rng(1)
        raw = make_raw(rng.normal(0, 10, (4, 20 * WIN)))
        valid = np.ones((4, 20), dtype=bool)

        masked = calculate_psd_masked(raw, valid, WIN)
        reference = calculate_psd(raw)

        np.testing.assert_allclose(masked['freqs'], reference['freqs'])
        np.testing.assert_allclose(masked['psds'], reference['psds'], rtol=1e-6)

    def test_contaminated_windows_do_not_reach_psd(self):
        """途中から7Hz倍音が乗ったチャネルでも、有効窓だけならクリーンと同じPSDになる。"""
        rng = np.random.default_rng(2)
        clean = rng.normal(0, 10, (4, 20 * WIN))
        dirty = clean.copy()
        t = np.arange(10 * WIN) / SFREQ
        dirty[3, 10 * WIN :] += 200 * np.sin(2 * np.pi * 7.0 * t)
        valid = np.ones((4, 20), dtype=bool)
        valid[3, 10:] = False

        masked = calculate_psd_masked(make_raw(dirty), valid, WIN)
        clean_head = calculate_psd(make_raw(clean[:, : 10 * WIN]))

        np.testing.assert_allclose(masked['psds'][3], clean_head['psds'][3], rtol=1e-6)
        assert masked['n_segments']['RAW_TP10'] == 9

    def test_fully_invalid_channel_is_dropped(self):
        rng = np.random.default_rng(3)
        raw = make_raw(rng.normal(0, 10, (4, 8 * WIN)))
        valid = np.ones((4, 8), dtype=bool)
        valid[0] = False

        masked = calculate_psd_masked(raw, valid, WIN)

        assert masked['channels'] == ['RAW_AF7', 'RAW_AF8', 'RAW_TP10']
        assert masked['psds'].shape[0] == 3


class TestMaskedSpectrogram:
    def test_invalid_windows_are_blank_and_color_scale_stays_finite(self, tmp_path):
        import matplotlib

        matplotlib.use('Agg')
        from lib.sensors.eeg.frequency import calculate_spectrogram_all_channels, mask_spectrogram
        from lib.sensors.eeg.visualization.eeg_plots import (
            plot_spectrogram_grid,
            shared_spectrogram_vlim,
        )

        rng = np.random.default_rng(7)
        raw = make_raw(rng.normal(0, 10, (4, 8 * WIN)))
        tfr_results = calculate_spectrogram_all_channels(raw.copy().resample(64, verbose=False))
        valid = np.ones(8, dtype=bool)
        valid[4:] = False
        mask_spectrogram(tfr_results['RAW_TP10'], valid, WIN, SFREQ)

        tp10 = tfr_results['RAW_TP10']
        second_half = tp10['times'] >= 4 * WIN / SFREQ
        assert np.isnan(tp10['power'][:, second_half]).all()
        assert not np.isnan(tp10['power'][:, ~second_half]).any()
        assert np.isnan(tp10['tfr'].data[0][:, second_half]).all()

        vlim = shared_spectrogram_vlim(tfr_results)
        assert vlim is not None and np.isfinite(vlim).all()
        plot_spectrogram_grid(tfr_results, img_path=tmp_path / 'spectrogram.png')
        assert (tmp_path / 'spectrogram.png').exists()


class TestFaaReferenceSwitch:
    def test_reference_is_subtracted_only_in_valid_windows(self):
        rng = np.random.default_rng(4)
        data = rng.normal(0, 10, (4, 4 * WIN))
        raw = make_raw(data)
        sample_mask = np.repeat([True, True, False, False], WIN)

        referenced = _apply_mastoid_reference(
            raw, 'RAW_AF7', 'RAW_AF8', 'RAW_TP9', 'RAW_TP10', sample_mask
        ).get_data() * 1e6

        mastoid = (data[0] + data[3]) / 2
        np.testing.assert_allclose(referenced[0, : 2 * WIN], (data[1] - mastoid)[: 2 * WIN])
        np.testing.assert_allclose(referenced[0, 2 * WIN :], data[1, 2 * WIN :])  # Fpz基準のまま
        np.testing.assert_allclose(referenced[1, 2 * WIN :], data[2, 2 * WIN :])

    def test_tp10_lost_mid_session_switches_to_fpz(self):
        n_windows = 60  # 4分
        df = make_mind_monitor_df(n_windows * WIN)
        rng = np.random.default_rng(5)
        data = rng.normal(0, 10, (4, n_windows * WIN))
        raw = make_raw(data)
        tp10_valid = np.arange(n_windows) < 30  # TP10が窓30から無効

        result = calculate_frontal_asymmetry(
            df, raw=raw, use_mastoid_reference=True, mastoid_valid=tp10_valid
        )
        assert result.metadata['reference_method'] == 'mixed'
        assert result.metadata['mastoid_reference_ratio'] == pytest.approx(0.5)

        fpz_only = calculate_frontal_asymmetry(
            df, raw=raw, use_mastoid_reference=True, mastoid_valid=np.zeros(n_windows, bool)
        )
        assert fpz_only.metadata['reference_method'] == 'default_fpz'


class TestSegmentCellNa:
    def test_frontal_cells_are_na_when_frontal_coverage_below_half(self):
        n_minutes = 3
        n_samples = int(n_minutes * 60 * SFREQ)
        rng = np.random.default_rng(6)
        raw = make_raw(rng.normal(0, 10, (4, n_samples)))
        hsi = np.ones((4, n_samples))
        first_minute = int(60 * SFREQ)
        hsi[1:3, :first_minute] = 2.0  # 最初の1分だけAF7/AF8が接触不良（全体の有効率は50%）

        stat_df = create_statistical_dataframe(
            raw, segment_minutes=1, session_start=START, hsi=hsi
        )
        quality = stat_df['quality']
        assert quality['frontal_valid_ratio'].iloc[0] == pytest.approx(0.0)
        assert quality['usable'].iloc[0]  # TPが生きているのでバンドパワーのセルは出す

        timestamps = START + pd.to_timedelta(np.arange(0, n_minutes * 60, 1.0), unit='s')
        df_clean = pd.DataFrame({'TimeStamp': timestamps})
        fmtheta = pd.Series(rng.normal(0, 1, len(timestamps)), index=timestamps)
        result = calculate_segment_analysis(
            df_clean, fmtheta, stat_df, segment_minutes=1, smr_series=fmtheta
        )
        segments = result.segments

        assert np.isnan(segments['fmtheta_mean'].iloc[0])
        assert np.isnan(segments['smr_mean'].iloc[0])
        assert segments['fmtheta_mean'].iloc[1:].notna().all()
        assert segments['alpha_mean'].notna().all()


class TestHeaderRepair:
    BROKEN_HEADER = [
        c for c in MIND_MONITOR_CSV_COLUMNS
        if not (c.startswith(('AUX_', 'Gyro_', 'Optics')) or c == 'Heart_Rate')
    ]

    def _write(self, path, header, rows):
        lines = [','.join(header)] + [','.join(str(v) for v in row) for row in rows]
        path.write_text('\n'.join(lines) + '\n')

    def _data_row(self, i):
        values = []
        for col in MIND_MONITOR_CSV_COLUMNS[:-1]:  # データ行は Elements を除く60列
            if col == 'TimeStamp':
                values.append((START + pd.Timedelta(seconds=i / SFREQ)).strftime('%Y-%m-%d %H:%M:%S.%f'))
            elif col.startswith('Optics'):
                values.append(100 + int(col[len('Optics'):]))
            elif col == 'Heart_Rate':
                values.append(62)
            elif col == 'HeadBandOn':
                values.append(1)
            elif col.startswith('HSI_'):
                values.append(1.0)
            else:
                values.append(0.5)
        return values

    def test_broken_header_is_repaired(self, tmp_path, capsys):
        assert len(self.BROKEN_HEADER) == 37
        csv_path = tmp_path / 'broken.csv'
        self._write(csv_path, self.BROKEN_HEADER, [self._data_row(i) for i in range(50)])

        df = load_mind_monitor_csv(csv_path, filter_headband=True, warmup_seconds=0)

        assert len(df) == 50
        assert (df['Optics1'] == 101).all()
        assert (df['Optics16'] == 116).all()
        assert (df['Heart_Rate'] == 62).all()
        assert '警告' in capsys.readouterr().out

    def test_normal_header_is_untouched(self, tmp_path, capsys):
        csv_path = tmp_path / 'normal.csv'
        rows = [self._data_row(i) for i in range(10)]
        event_row = [rows[-1][0]] + [''] * 59 + ['/muse/event/connected MuseS']
        self._write(csv_path, MIND_MONITOR_CSV_COLUMNS, rows + [event_row])

        df = load_mind_monitor_csv(csv_path, filter_headband=True, warmup_seconds=0)

        assert len(df) == 10
        assert (df['Heart_Rate'] == 62).all()
        assert '警告' not in capsys.readouterr().out


class TestSessionLogFrontal:
    def test_main_eeg_columns_come_from_frontal_metrics(self):
        results = {
            'data_info': {'start_time': START, 'duration_sec': 1200.0},
            'mean_metrics': {'alpha_mean': 99.0, 'hrv_mean': 40.0},
            'best_metrics': {'alpha_best': 99.0},
            'aperiodic': {'exponent': 9.9},
            'aperiodic_frontal': {'exponent': 1.2, 'alpha_osc_db': 3.0, 'alpha_peak': None},
            'frontal_metrics': {
                'mean_metrics': {'alpha_mean': 1.5, 'alpha_rel_pct': 20.0},
                'best_metrics': {'alpha_best': 2.5},
                'montage': 'AF7+AF8/Fpz',
                'valid_coverage_pct': 97.5,
            },
        }

        row = _extract_session_data(results)

        assert row['alpha_mean'] == 1.5
        assert row['alpha_best'] == 2.5
        assert row['alpha_rel_pct'] == 20.0
        assert row['aperiodic_exponent'] == 1.2
        assert row['hrv_mean'] == 40.0
        assert row['montage'] == 'AF7+AF8/Fpz'
        assert row['valid_coverage_pct'] == 97.5
