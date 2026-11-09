"""fetch_from_gdrive.py のテスト

Drive APIは一切呼ばず、ファイル一覧の dict だけで選択ロジックを検証する。
"""

from datetime import datetime

from lib.loaders.selfloops import parse_selfloops_timestamp
from scripts.fetch_from_gdrive import (
    find_file_by_date,
    list_csv_files,
    parse_filename_datetime,
    select_latest_file,
)


def make_file(name: str, modified: str, file_id: str = '') -> dict:
    return {'id': file_id or name, 'name': name, 'modifiedTime': modified}


def make_resolver(first_lines: dict):
    """ファイルID → 1行目 の対応から、Driveを使わないリゾルバを作る（呼ばれたIDを記録する）。"""
    called: list = []

    def resolve(file: dict):
        called.append(file['id'])
        return parse_selfloops_timestamp(first_lines.get(file['id'], ''))

    return resolve, called


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

    def test_uses_first_line_for_undated_name(self):
        """日付なし名でも、本体1行目の日時が新しければ選ばれる"""
        files = [
            make_file('SelfLoops HRV data', '2026-11-05T00:00:00Z', file_id='undated'),
            make_file('selfloops_2026-09-16--08-05-20.csv', '2026-09-21T00:10:00Z'),
        ]
        resolver, called = make_resolver({'undated': '05 11月 2026 07:53:44'})
        selected = select_latest_file(files, resolver)
        assert selected is not None
        assert selected['id'] == 'undated'
        # 名前に日時があるファイルの中身は読まない
        assert called == ['undated']

    def test_picks_latest_among_undated_names(self):
        files = [
            make_file('SelfLoops HRV data', '2026-11-05T00:00:00Z', file_id='nov02'),
            make_file('SelfLoops HRV data', '2026-11-04T00:00:00Z', file_id='nov05'),
            make_file('SelfLoops HRV data', '2026-11-03T00:00:00Z', file_id='oct27'),
        ]
        resolver, _ = make_resolver({
            'nov02': '02 11月 2026 07:40:00',
            'nov05': '05 11月 2026 07:53:44',
            'oct27': '27 10月 2026 07:30:00',
        })
        selected = select_latest_file(files, resolver)
        assert selected is not None
        assert selected['id'] == 'nov05'

    def test_excludes_files_without_any_timestamp(self):
        """名前にも本体にも日時が無いファイルは候補から外れる"""
        files = [
            make_file('SelfLoops HRV data', '2026-09-21T00:30:00Z', file_id='broken'),
            make_file('selfloops_2026-09-16--08-05-20.csv', '2026-09-21T00:10:00Z'),
        ]
        resolver, _ = make_resolver({'broken': 'not a timestamp'})
        selected = select_latest_file(files, resolver)
        assert selected is not None
        assert selected['name'] == 'selfloops_2026-09-16--08-05-20.csv'

    def test_excludes_unparseable_names_without_resolver(self):
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

    def test_falls_back_to_modified_time_when_nothing_resolves(self):
        files = [
            make_file('SelfLoops HRV data', '2026-09-21T00:30:00Z', file_id='a'),
            make_file('SelfLoops HRV data', '2026-09-20T00:30:00Z', file_id='b'),
        ]
        resolver, _ = make_resolver({})
        selected = select_latest_file(files, resolver)
        assert selected is not None
        assert selected['id'] == 'a'

    def test_returns_none_for_empty_list(self):
        assert select_latest_file([]) is None


class TestFindFileByDate:
    def test_matches_dated_name(self):
        files = [
            make_file('taps_2026-09-19--07-56-07.csv', '2026-09-21T00:20:00Z'),
            make_file('taps_2026-09-21--07-39-41.csv', '2026-09-21T00:10:00Z'),
        ]
        selected = find_file_by_date(files, '2026-09-21')
        assert selected is not None
        assert selected['name'] == 'taps_2026-09-21--07-39-41.csv'

    def test_uses_first_line_for_undated_name(self):
        """--download 2026-11-02 で、1行目が 11/02 の日付なし名ファイルが選ばれる"""
        files = [
            make_file('SelfLoops HRV data', '2026-11-05T00:00:00Z', file_id='nov05'),
            make_file('SelfLoops HRV data', '2026-11-02T00:00:00Z', file_id='nov02'),
            make_file('SelfLoops HRV data', '2026-10-27T00:00:00Z', file_id='oct27'),
            make_file('selfloops_2026-09-16--08-05-20.csv', '2026-09-21T00:10:00Z'),
        ]
        resolver, called = make_resolver({
            'nov05': '05 11月 2026 07:53:44',
            'nov02': '02 11月 2026 07:40:00',
            'oct27': '27 10月 2026 07:30:00',
        })
        selected = find_file_by_date(files, '2026-11-02', resolver)
        assert selected is not None
        assert selected['id'] == 'nov02'
        assert 'selfloops_2026-09-16--08-05-20.csv' not in called

    def test_returns_none_when_no_file_matches(self):
        files = [make_file('SelfLoops HRV data', '2026-11-05T00:00:00Z', file_id='nov05')]
        resolver, _ = make_resolver({'nov05': '05 11月 2026 07:53:44'})
        assert find_file_by_date(files, '2026-11-02', resolver) is None

    def test_name_only_without_resolver(self):
        files = [make_file('SelfLoops HRV data', '2026-11-02T00:00:00Z')]
        assert find_file_by_date(files, '2026-11-02') is None


class _FakeRequest:
    def __init__(self, response: dict):
        self._response = response

    def execute(self) -> dict:
        return self._response


class _FakeFiles:
    """pageToken ごとに決まったページを返す Drive files() の代役。"""

    def __init__(self, pages: dict):
        self._pages = pages
        self.tokens: list = []

    def list(self, **kwargs) -> _FakeRequest:
        token = kwargs.get('pageToken')
        self.tokens.append(token)
        return _FakeRequest(self._pages[token])


class _FakeService:
    def __init__(self, pages: dict):
        self.files_api = _FakeFiles(pages)

    def files(self) -> _FakeFiles:
        return self.files_api


class TestListCsvFilesPagination:
    def test_concatenates_all_pages(self):
        page1 = {'files': [{'id': 'a', 'name': 'a.zip'}], 'nextPageToken': 'p2'}
        page2 = {'files': [{'id': 'b', 'name': 'b.zip'}]}
        service = _FakeService({None: page1, 'p2': page2})

        files = list_csv_files(service, 'folder')

        assert [f['id'] for f in files] == ['a', 'b']
        assert service.files_api.tokens == [None, 'p2']
