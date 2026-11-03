"""session_log の backfill 書き込みと pipeline_version のテスト"""

import math

import pandas as pd
import pytest

from lib.session_log import (
    CANONICAL_COLUMNS,
    _extract_session_data,
    get_pipeline_version,
    write_backfill_row,
)

TS = '2026-10-28 07:30:00'


def make_results(start='2026-10-28 07:30:00', alpha=1.5, version='abc1234'):
    return {
        'data_info': {'start_time': pd.Timestamp(start), 'duration_sec': 600.0},
        'frontal_metrics': {
            'mean_metrics': {'alpha_mean': alpha},
            'best_metrics': {'alpha_best': alpha + 1},
            'montage': 'AF7+AF8/Fpz',
        },
        'pipeline_version': version,
    }


def write_log(path):
    pd.DataFrame(
        [
            {'timestamp': '2026-10-27 07:00:00', 'alpha_mean': 9.0, 'hrv_mean': 10.0},
            {
                'timestamp': TS,
                'alpha_mean': 0.1,
                'hrv_mean': 42.0,
                'mw_count': 3.0,
                'note': 'keep me',
            },
        ]
    ).to_csv(path, index=False)


def row(path, ts=TS):
    df = pd.read_csv(path)
    return df[df['timestamp'] == ts].iloc[0]


def test_preserved_columns_keep_existing_values(tmp_path):
    p = tmp_path / 'log.csv'
    write_log(p)
    write_backfill_row(make_results(), TS, csv_path=p)
    r = row(p)
    assert r['hrv_mean'] == 42.0
    assert r['mw_count'] == 3.0
    assert r['note'] == 'keep me'


def test_non_preserved_columns_are_overwritten(tmp_path):
    p = tmp_path / 'log.csv'
    write_log(p)
    write_backfill_row(make_results(alpha=1.5), TS, csv_path=p)
    r = row(p)
    assert r['alpha_mean'] == pytest.approx(1.5)
    assert r['montage'] == 'AF7+AF8/Fpz'
    assert r['pipeline_version'] == 'abc1234'
    # 他の行は変わらない
    assert row(p, '2026-10-27 07:00:00')['alpha_mean'] == 9.0


def test_timestamp_drift_updates_existing_row(tmp_path):
    p = tmp_path / 'log.csv'
    write_log(p)
    write_backfill_row(make_results(start='2026-10-28 07:30:07'), TS, csv_path=p)
    df = pd.read_csv(p)
    assert len(df) == 2
    assert '2026-10-28 07:30:07' not in set(df['timestamp'])
    assert row(p)['pipeline_version'] == 'abc1234'


def test_missing_timestamp_raises_and_leaves_file(tmp_path):
    p = tmp_path / 'log.csv'
    write_log(p)
    before = p.read_bytes()
    with pytest.raises(ValueError):
        write_backfill_row(make_results(), '2026-01-01 00:00:00', csv_path=p)
    assert p.read_bytes() == before


def test_extract_includes_pipeline_version():
    assert 'pipeline_version' in CANONICAL_COLUMNS
    assert _extract_session_data(make_results())['pipeline_version'] == 'abc1234'
    no_version = make_results()
    del no_version['pipeline_version']
    assert _extract_session_data(no_version)['pipeline_version'] == ''
    assert math.isnan(_extract_session_data(no_version)['hrv_mean'])


def test_get_pipeline_version_empty_outside_git(tmp_path):
    assert get_pipeline_version(repo_root=tmp_path) == ''
