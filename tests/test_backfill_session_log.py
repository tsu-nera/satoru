"""backfill_session_log.py のテスト（Drive・計算は一切呼ばない）"""

from datetime import date

import pandas as pd
import pytest

from scripts import backfill_session_log as bf
from scripts.backfill_session_log import match_files_to_rows, select_targets


def f(name):
    return {'id': name, 'name': name}


def mm(dt):
    return f'mindMonitor_{dt}_123.zip'


class TestMatchFilesToRows:
    T = '2026-10-28 07:30:00'

    @pytest.mark.parametrize(
        'stamp',
        ['2026-10-28--07-30-00', '2026-10-28--07-29-00', '2026-10-28--07-27-00'],
    )
    def test_within_window_matches(self, stamp):
        matched, missing = match_files_to_rows([self.T], [f(mm(stamp))])
        assert matched[self.T]['name'] == mm(stamp)
        assert missing == []

    @pytest.mark.parametrize('stamp', ['2026-10-28--07-26-59', '2026-10-28--07-30-01'])
    def test_outside_window_no_match(self, stamp):
        matched, missing = match_files_to_rows([self.T], [f(mm(stamp))])
        assert matched == {}
        assert missing == [self.T]

    def test_unrelated_and_undated_files_ignored(self):
        files = [f('readme.txt'), f(mm('2026-10-01--07-00-00'))]
        matched, missing = match_files_to_rows([self.T], files)
        assert matched == {}
        assert missing == [self.T]

    def test_picks_closest(self):
        files = [f(mm('2026-10-28--07-28-00')), f(mm('2026-10-28--07-29-30'))]
        matched, _ = match_files_to_rows([self.T], files)
        assert matched[self.T]['name'] == mm('2026-10-28--07-29-30')

    def test_unmatched_rows_listed(self):
        other = '2026-10-29 07:30:00'
        matched, missing = match_files_to_rows(
            [self.T, other], [f(mm('2026-10-28--07-29-00'))]
        )
        assert list(matched) == [self.T]
        assert missing == [other]


class TestSelectTargets:
    def df(self):
        return pd.DataFrame(
            {
                'timestamp': [
                    '2026-10-26 07:00:00',
                    '2026-10-27 07:00:00',
                    '2026-10-28 07:00:00',
                ],
                'pipeline_version': ['old', 'cur', None],
            }
        )

    def test_same_version_skipped(self):
        todo, skipped = select_targets(self.df(), 'cur', None, None, False)
        assert todo == ['2026-10-26 07:00:00', '2026-10-28 07:00:00']
        assert skipped == ['2026-10-27 07:00:00']

    def test_force_reprocesses(self):
        todo, skipped = select_targets(self.df(), 'cur', None, None, True)
        assert len(todo) == 3 and skipped == []

    def test_unknown_version_never_skipped(self):
        todo, skipped = select_targets(self.df(), '', None, None, False)
        assert len(todo) == 3 and skipped == []

    def test_missing_column_means_unprocessed(self):
        df = self.df().drop(columns=['pipeline_version'])
        todo, _ = select_targets(df, 'cur', None, None, False)
        assert len(todo) == 3

    def test_since_until_inclusive(self):
        todo, _ = select_targets(
            self.df(), 'zzz', date(2026, 10, 27), date(2026, 10, 27), False
        )
        assert todo == ['2026-10-27 07:00:00']


def test_dry_run_has_no_side_effects(tmp_path, monkeypatch):
    log = tmp_path / 'log.csv'
    log.write_text('timestamp,alpha_mean\n2026-10-28 07:30:00,1.0\n')
    before = log.read_bytes()

    monkeypatch.setattr(
        bf,
        '_list_drive_files',
        lambda cred, folder: (object(), [f(mm('2026-10-28--07-29-00'))]),
    )

    def boom(*args, **kwargs):
        raise AssertionError('dry-run must not process')

    monkeypatch.setattr(bf, '_process_one', boom)
    monkeypatch.setattr(bf, 'download_file', boom)

    rc = bf.main(['--dry-run', '--folder-id', 'x', '--log-path', str(log)])
    assert rc == 0
    assert log.read_bytes() == before
