"""
TPのα・δ絶対パワー（入り/終わりの3分区間, dB）のテスト

- 前半に大きなδ、全体に一定のαを持つ合成信号で、δは入り>終わり、αはほぼ一定になること
- 区間内の有効窓が60秒未満のチャネルは除かれ、両チャネルとも不足ならNaNになること
"""

import numpy as np
import pandas as pd
import pytest
from scipy import signal

mne = pytest.importorskip('mne')
mne.set_log_level('ERROR')

from lib.report.steps_eeg import TP_BLOCK_SEC, compute_tp_metrics  # noqa: E402
from lib.sensors.eeg.artifact import ARTIFACT_WINDOW_SAMPLES  # noqa: E402
from lib.session_log import _extract_session_data  # noqa: E402

SFREQ = 256.0
WIN = ARTIFACT_WINDOW_SAMPLES
BLOCK = int(TP_BLOCK_SEC * SFREQ / WIN)  # 3分の窓数
N_WINDOWS = 3 * BLOCK + 10  # 3区間 + 端数（端数は最後の区間に入らない）
CHANNELS = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']


def bandnoise(rng, n, lo, hi, amp):
    sos = signal.butter(4, [lo, hi], btype='bandpass', fs=SFREQ, output='sos')
    x = signal.sosfilt(sos, rng.normal(0, 1, n))
    return x * amp / x.std()


def make_raw(seed=0):
    """δ（1.5-3.5Hz）は最初の区間だけ大きく、α（9-11Hz）は全体で一定の信号（μV）。"""
    n = N_WINDOWS * WIN
    rows = []
    for k in range(len(CHANNELS)):
        rng = np.random.default_rng(seed + k)
        delta_amp = np.full(n, 5.0)
        delta_amp[: BLOCK * WIN] = 20.0
        x = (
            bandnoise(rng, n, 1.5, 3.5, 1.0) * delta_amp
            + bandnoise(rng, n, 9.0, 11.0, 8.0)
            + rng.normal(0, 1.0, n)
        )
        rows.append(x)
    info = mne.create_info(CHANNELS, SFREQ, 'eeg')
    return mne.io.RawArray(np.vstack(rows) * 1e-6, info, verbose=False)


def test_delta_drops_while_alpha_stays():
    valid = np.ones((len(CHANNELS), N_WINDOWS), dtype=bool)
    m = compute_tp_metrics(make_raw(), valid, WIN)

    assert m['tp_delta_db_first'] - m['tp_delta_db_last'] > 6.0
    assert abs(m['tp_alpha_db_first'] - m['tp_alpha_db_last']) < 1.0


def test_last_block_ignores_trailing_fraction():
    """端数の窓だけが無効でも、最後の完全な区間は使える。"""
    valid = np.ones((len(CHANNELS), N_WINDOWS), dtype=bool)
    valid[:, 3 * BLOCK :] = False
    m = compute_tp_metrics(make_raw(), valid, WIN)
    assert not np.isnan(m['tp_alpha_db_last'])


def test_channel_with_too_few_valid_windows_is_excluded():
    """TP10の最初の区間を有効窓60秒未満にすると、入りの値はTP9だけで決まる。"""
    raw = make_raw()
    valid = np.ones((len(CHANNELS), N_WINDOWS), dtype=bool)
    tp9_only = valid.copy()
    tp9_only[3] = False  # TP10を全窓無効

    few = valid.copy()
    min_windows = int(60 * SFREQ / WIN)
    few[3, :BLOCK] = False
    few[3, : min_windows - 1] = True  # 60秒に1窓足りない

    expected = compute_tp_metrics(raw, tp9_only, WIN)
    m = compute_tp_metrics(raw, few, WIN)
    assert m['tp_alpha_db_first'] == pytest.approx(expected['tp_alpha_db_first'])
    assert m['tp_delta_db_first'] == pytest.approx(expected['tp_delta_db_first'])
    # 最後の区間はTP10も使われるので、TP9だけの値とは一致しない
    assert m['tp_alpha_db_last'] != pytest.approx(expected['tp_alpha_db_last'])


def test_both_channels_short_gives_nan():
    valid = np.ones((len(CHANNELS), N_WINDOWS), dtype=bool)
    min_windows = int(60 * SFREQ / WIN)
    for idx in (0, 3):
        valid[idx, min_windows - 1 : BLOCK] = False  # 最初の区間は60秒に1窓足りない
    m = compute_tp_metrics(make_raw(), valid, WIN)
    assert np.isnan(m['tp_alpha_db_first']) and np.isnan(m['tp_delta_db_first'])
    assert not np.isnan(m['tp_alpha_db_last'])


def test_short_session_gives_nan():
    raw = make_raw()
    n = BLOCK - 1
    short = mne.io.RawArray(raw.get_data()[:, : n * WIN], raw.info, verbose=False)
    m = compute_tp_metrics(short, np.ones((len(CHANNELS), n), dtype=bool), WIN)
    for key in ('tp_alpha_db_first', 'tp_alpha_db_last', 'tp_delta_db_first', 'tp_delta_db_last'):
        assert np.isnan(m[key])


def test_session_log_columns():
    base = {'data_info': {'start_time': pd.Timestamp('2026-11-04 07:00:00'), 'duration_sec': 600}}
    assert np.isnan(_extract_session_data(base)['tp_delta_db_first'])
    row = _extract_session_data({**base, 'tp_metrics': {'tp_delta_db_last': 12.1}})
    assert row['tp_delta_db_last'] == 12.1
