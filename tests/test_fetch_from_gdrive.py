"""fetch_from_gdrive.py のテスト

Drive APIは一切呼ばず、ファイル一覧の dict だけで選択ロジックを検証する。
"""

from datetime import datetime

from scripts.fetch_from_gdrive import parse_filename_datetime, select_latest_file


def make_file(name: str, modified: str) -> dict:
    return {'id': name, 'name': name, 'modifiedTime': modified}


class TestParseFilenameDatetime:
    def test_parses_muse_zip_name(self):
        name = 'mindMonitor_2026-09-21--07-39-38_5022520958388465773.zip'
        assert parse_filename_datetime(name) == datetime(2026, 9, 21, 7, 39, 38)

    def test_parses_selfloops_name(self):
        assert parse_filename_datetime('selfloops_2026-09-16--08-05-20.csv') == datetime(
            2026, 9, 16, 8, 5, 20
        )

    def test_parses_taps_name(self):
        assert parse_filename_datetime('taps_2026-09-19--07-56-07.csv') == datetime(
            2026, 9, 19, 7, 56, 7
        )

    def test_returns_none_without_timestamp(self):
        assert parse_filename_datetime('SelfLoops HRV data') is None

    def test_returns_none_for_date_only_name(self):
        # 時刻部分（--HH-MM-SS）が無い名前は対象外
        assert parse_filename_datetime('taps_2026-09-19.csv') is None

    def test_returns_none_for_impossible_date(self):
        assert parse_filename_datetime('taps_2026-13-45--07-56-07.csv') is None


class TestSelectLatestFile:
    def test_ignores_modified_time_order(self):
        """modifiedTime が最新でも、ファイル名の日時が古ければ選ばれない"""
        files = [
            make_file('taps_2026-09-19--07-56-07.csv', '2026-09-21T00:20:00Z'),
            make_file('taps_2026-09-18--08-45-12.csv', '2026-09-21T00:19:00Z'),
            make_file('taps_2026-09-21--07-39-41.csv', '2026-09-21T00:10:00Z'),
        ]
        selected = select_latest_file(files)
        assert selected is not None
        assert selected['name'] == 'taps_2026-09-21--07-39-41.csv'

    def test_picks_latest_time_within_same_day(self):
        files = [
            make_file('taps_2026-09-21--07-39-41.csv', '2026-09-21T00:10:00Z'),
            make_file('taps_2026-09-21--19-02-00.csv', '2026-09-21T00:11:00Z'),
        ]
        selected = select_latest_file(files)
        assert selected is not None
        assert selected['name'] == 'taps_2026-09-21--19-02-00.csv'

    def test_excludes_unparseable_names(self):
        files = [
            make_file('SelfLoops HRV data', '2026-09-21T00:30:00Z'),
            make_file('selfloops_2026-09-16--08-05-20.csv', '2026-09-21T00:10:00Z'),
        ]
        selected = select_latest_file(files)
        assert selected is not None
        assert selected['name'] == 'selfloops_2026-09-16--08-05-20.csv'

    def test_falls_back_to_modified_time_when_no_name_parses(self):
        files = [
            make_file('SelfLoops HRV data', '2026-09-21T00:30:00Z'),
            make_file('SelfLoops HRV data (1)', '2026-09-20T00:30:00Z'),
        ]
        selected = select_latest_file(files)
        assert selected is not None
        assert selected['name'] == 'SelfLoops HRV data'

    def test_returns_none_for_empty_list(self):
        assert select_latest_file([]) is None
