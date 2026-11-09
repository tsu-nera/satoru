"""
elitehrv CLI と HRV 基本統計・共鳴呼吸指標の真値検証（合成 RR）
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lib.loaders.elite_hrv import correct_rr_artifacts
from lib.sensors.ecg.hrv import calculate_hrv_basic_stats, resample_rr
from lib.sensors.ecg.respiration import analyze_resonance_breathing

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location('elitehrv', PROJECT_ROOT / 'scripts' / 'elitehrv.py')
assert _spec is not None and _spec.loader is not None
elitehrv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(elitehrv)


def _synth_rr(duration_sec: float = 600.0) -> np.ndarray:
    """RR = 800 + 75 sin(2π t / 15)。周期15秒 = 4 bpm、ピーク振幅 150 ms"""
    t = 0.0
    out = []
    while t < duration_sec:
        rr = 800.0 + 75.0 * math.sin(2 * math.pi * t / 15.0)
        t += rr / 1000.0
        out.append(rr)
    return np.array(out)


def _with_missed_beats(rr: np.ndarray, positions: list[int]) -> np.ndarray:
    """rr[i] と rr[i+1] を和で置き換え、約2倍の RR を1拍作る（後ろから処理）"""
    out = list(rr)
    for i in sorted(positions, reverse=True):
        out[i:i + 2] = [out[i] + out[i + 1]]
    return np.array(out)


def _write_txt(directory: Path, name: str, rr: np.ndarray) -> Path:
    path = directory / name
    path.write_text('\n'.join(f'{v:.1f}' for v in rr) + '\n')
    return path


def test_resonance_metrics_recover_synthetic_truth():
    rr = _synth_rr()
    stats = calculate_hrv_basic_stats(rr)
    res = analyze_resonance_breathing(rr, stats.lf_peak_hz)
    assert res.breathing_rate_bpm == pytest.approx(4.0, abs=0.1)
    assert res.rsa_amplitude_ms == pytest.approx(150.0, abs=15.0)
    assert stats.lf_peak_hz == pytest.approx(0.0667, abs=0.005)
    assert not res.segments.empty


def test_missed_beats_are_counted_and_corrected():
    clean = _synth_rr()
    broken = _with_missed_beats(clean, [100, 300, 500])
    corrected, mask = correct_rr_artifacts(broken)
    assert int(mask.sum()) == 3
    rmssd_clean = calculate_hrv_basic_stats(clean).rmssd_ms
    rmssd_fixed = calculate_hrv_basic_stats(corrected).rmssd_ms
    assert rmssd_fixed == pytest.approx(rmssd_clean, rel=0.05)


def test_time_domain_matches_hand_calculation():
    stats = calculate_hrv_basic_stats(np.array([800.0, 850.0, 790.0, 900.0, 820.0]))
    # diffs = 50, -60, 110, -80 → mean(d^2) = (2500+3600+12100+6400)/4 = 6150
    assert stats.rmssd_ms == pytest.approx(78.4219, abs=1e-3)  # sqrt(6150)
    # mean = 832、偏差^2 = 1024+324+1764+4624+144 = 7880 → /4 = 1970
    assert stats.mean_rr_ms == pytest.approx(832.0)
    assert stats.sdnn_ms == pytest.approx(44.3847, abs=1e-3)  # sqrt(1970), ddof=1
    # |diff| > 50 は 60, 110, 80 の3つ / 4 diffs = 75%（ちょうど50は含めない）
    assert stats.pnn50_pct == pytest.approx(75.0)
    assert stats.ln_rmssd == pytest.approx(math.log(78.4219), abs=1e-4)
    assert math.isnan(stats.total_power_ms2)  # 短い系列は周波数領域なし


def test_total_power_is_lf_plus_hf_and_same_order_as_variance():
    rr = _synth_rr()
    stats = calculate_hrv_basic_stats(rr)
    assert stats.total_power_ms2 == pytest.approx(stats.lf_power_ms2 + stats.hf_power_ms2)
    _, series = resample_rr(rr)
    ratio = stats.total_power_ms2 / float(np.var(series))
    assert 0.3 < ratio < 3.0


def test_short_reading_is_readiness_with_empty_frequency_columns(tmp_path):
    rr = _synth_rr(150.0)
    txt = _write_txt(tmp_path, '2026-11-10 07-22-58.txt', rr)
    log = tmp_path / 'log.csv'
    row = elitehrv.analyze_file(str(txt), log_path=log)
    assert row['kind'] == 'readiness'
    saved = pd.read_csv(log)
    assert list(saved.columns) == elitehrv.COLUMNS
    empty_cols = [
        'lf_power_ms2', 'hf_power_ms2', 'total_power_ms2', 'lf_hf_ratio', 'lf_peak_hz',
        'hf_peak_hz', 'breathing_rate_bpm', 'breath_period_sd_s', 'rsa_amplitude_ms',
    ]
    assert saved.loc[0, empty_cols].isna().all()
    assert saved.loc[0, 'rmssd_ms'] > 0


def test_analyze_twice_keeps_row_count_and_practice(tmp_path):
    txt = _write_txt(tmp_path, '2026-11-10 07-22-58.txt', _synth_rr())
    log = tmp_path / 'log.csv'
    first = elitehrv.analyze_file(str(txt), practice='rf', log_path=log)
    assert first['kind'] == 'biofeedback'
    second = elitehrv.analyze_file(str(txt), log_path=log)
    saved = pd.read_csv(log)
    assert len(saved) == 1
    assert saved.loc[0, 'practice'] == 'rf'
    assert second['practice'] == 'rf'
    assert saved.loc[0, 'version'] == elitehrv.VERSION


def test_cli_prints_report_and_comparison(tmp_path, capsys):
    log = tmp_path / 'log.csv'
    for name in ('2026-11-09 07-00-00.txt', '2026-11-10 07-00-00.txt'):
        txt = _write_txt(tmp_path, name, _synth_rr())
        assert elitehrv.main(['analyze', str(txt), '--log', str(log)]) == 0
    out = capsys.readouterr().out
    assert 'no history' in out
    assert 'Comparison (last 1 biofeedback)' in out
    assert '| Metric' in out


def test_deleted_functions_are_not_referenced():
    deleted = [
        'calculate_' + 'rmssd', 'calculate_' + 'hr_stats', 'calculate_' + 'lf_power',
        'analyze_' + 'measurement_order_effect', 'analyze_hrv_' + 'time_domain',
        'analyze_hrv_' + 'frequency_domain', 'analyze_hrv_' + 'nonlinear',
    ]
    pattern = re.compile(r'\b(' + '|'.join(deleted) + r')\b')
    hits = []
    for sub in ('lib', 'scripts', 'tests'):
        for path in (PROJECT_ROOT / sub).rglob('*.py'):
            if pattern.search(path.read_text(encoding='utf-8')):
                hits.append(str(path.relative_to(PROJECT_ROOT)))
    assert hits == []
