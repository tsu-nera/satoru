"""scripts/annotate_session.py と注釈列の保持のテスト"""

import pandas as pd
import pytest

from lib.session_log import BACKFILL_PRESERVED_COLUMNS, write_to_csv
from scripts.annotate_session import annotate_session

ROWS = [
    {'timestamp': '2026-11-03 07:00:00', 'alpha_mean': 1.25, 'montage': 'AF7+AF8/Fpz', 'note': 'x'},
    {'timestamp': '2026-11-04 07:00:00', 'alpha_mean': 2.5, 'montage': 'AF7+AF8/Fpz', 'note': ''},
]


@pytest.fixture
def log(tmp_path):
    path = tmp_path / 'session_log.csv'
    pd.DataFrame(ROWS).to_csv(path, index=False, float_format='%.3f')
    return path


def test_updates_only_given_columns_of_target_row(log):
    before = pd.read_csv(log)

    ts = annotate_session('2026-11-03 07:00:00', meditation_type='samatha', focus=4, csv_path=log)

    after = pd.read_csv(log)
    assert ts == '2026-11-03 07:00:00'
    target = after['timestamp'] == ts
    assert after.loc[target, 'meditation_type'].item() == 'samatha'
    assert after.loc[target, 'focus_rating'].item() == 4
    assert 'drowsiness_rating' not in after.columns  # 指定していない列は作らない
    # 既存の列は全行そのまま、他の行の注釈は空
    pd.testing.assert_frame_equal(after[before.columns], before)
    assert after.loc[~target, ['meditation_type', 'focus_rating']].isna().all().all()


def test_latest_and_partial_update_keeps_earlier_annotation(log):
    annotate_session('latest', meditation_type='control', focus=3, csv_path=log)
    ts = annotate_session('latest', drowsiness=5, csv_path=log)

    after = pd.read_csv(log).set_index('timestamp')
    assert ts == '2026-11-04 07:00:00'
    assert after.loc[ts, 'meditation_type'] == 'control'
    assert after.loc[ts, 'focus_rating'] == 3
    assert after.loc[ts, 'drowsiness_rating'] == 5
    assert ',3,5,' in log.read_text()  # 整数で書かれる（3.000 にならない）


@pytest.mark.parametrize('kwargs', [{'focus': 0}, {'focus': 6}, {'drowsiness': -1}])
def test_out_of_range_rating_is_error(log, kwargs):
    before = log.read_text()
    with pytest.raises(ValueError):
        annotate_session('latest', csv_path=log, **kwargs)
    assert log.read_text() == before


def test_unknown_timestamp_is_error(log):
    with pytest.raises(ValueError, match='行がありません'):
        annotate_session('2026-01-01 00:00:00', focus=3, csv_path=log)


def test_annotation_columns_are_preserved_by_backfill():
    for col in ('meditation_type', 'focus_rating', 'drowsiness_rating', 'note'):
        assert col in BACKFILL_PRESERVED_COLUMNS


def test_rerunning_analysis_keeps_annotation(log):
    annotate_session('latest', meditation_type='samatha', focus=4, drowsiness=2, csv_path=log)
    results = {
        'data_info': {'start_time': pd.Timestamp('2026-11-04 07:00:00'), 'duration_sec': 600.0}
    }
    write_to_csv(results, csv_path=log)

    row = pd.read_csv(log).set_index('timestamp').loc['2026-11-04 07:00:00']
    assert row['meditation_type'] == 'samatha'
    assert row['focus_rating'] == 4
    assert row['drowsiness_rating'] == 2
