"""
scripts/rename_gdrive_selfloops.py のテスト

Google Drive API は FakeDriveService でモックし、実際のAPIは一切叩かない。
modifiedTime には誤った値を混ぜて、リネームがそれに依存しないことを検証する。
"""

from __future__ import annotations

from typing import Any, Dict, List

import scripts.rename_gdrive_selfloops as rename_gdrive_selfloops
from scripts.rename_gdrive_selfloops import decide_new_name, needs_rename, rename_files


class _FakeExecutable:
    def __init__(self, result: Any) -> None:
        self._result = result

    def execute(self) -> Any:
        return self._result


class FakeFilesResource:
    def __init__(self) -> None:
        self.update_calls: List[Dict[str, Any]] = []

    def update(self, **kwargs: Any) -> _FakeExecutable:
        self.update_calls.append(kwargs)
        return _FakeExecutable({'id': kwargs.get('fileId')})

    # get_media / list は fetch_first_line を monkeypatch するテストでは未使用
    def get_media(self, **kwargs: Any) -> _FakeExecutable:
        return _FakeExecutable(b'')


class FakeDriveService:
    def __init__(self) -> None:
        self._files = FakeFilesResource()

    def files(self) -> FakeFilesResource:
        return self._files


class TestNeedsRename:
    def test_raw_selfloops_name_needs_rename(self) -> None:
        assert needs_rename('SelfLoops HRV data') is True

    def test_already_renamed_name_does_not_need_rename(self) -> None:
        assert needs_rename('selfloops_2026-01-10--16-08-50.csv') is False


class TestDecideNewName:
    def test_parses_valid_timestamp_line(self) -> None:
        assert decide_new_name('10 1月 2026 16:08:50') == 'selfloops_2026-01-10--16-08-50.csv'

    def test_returns_none_for_unparsable_line(self) -> None:
        assert decide_new_name('not a timestamp') is None


class TestRenameFiles:
    def test_dry_run_does_not_call_update(self, monkeypatch: Any) -> None:
        service = FakeDriveService()
        files = [{'id': 'f1', 'name': 'SelfLoops HRV data', 'modifiedTime': '2099-01-01T00:00:00.000Z'}]

        monkeypatch.setattr(
            rename_gdrive_selfloops, 'fetch_first_line', lambda svc, fid: '10 1月 2026 16:08:50'
        )

        renamed, skipped = rename_files(service, files, set(), dry_run=True)

        assert renamed == 1
        assert skipped == 0
        assert service.files().update_calls == []

    def test_execute_calls_update_with_expected_body(self, monkeypatch: Any) -> None:
        service = FakeDriveService()
        files = [{'id': 'f1', 'name': 'SelfLoops HRV data', 'modifiedTime': '2099-01-01T00:00:00.000Z'}]

        monkeypatch.setattr(
            rename_gdrive_selfloops, 'fetch_first_line', lambda svc, fid: '10 1月 2026 16:08:50'
        )

        renamed, skipped = rename_files(service, files, set(), dry_run=False)

        assert renamed == 1
        assert skipped == 0
        assert len(service.files().update_calls) == 1
        call = service.files().update_calls[0]
        assert call['fileId'] == 'f1'
        assert call['body'] == {'name': 'selfloops_2026-01-10--16-08-50.csv'}

    def test_skips_on_collision_with_existing_name(self, monkeypatch: Any) -> None:
        service = FakeDriveService()
        files = [{'id': 'f1', 'name': 'SelfLoops HRV data', 'modifiedTime': '2099-01-01T00:00:00.000Z'}]
        existing_names = {'selfloops_2026-01-10--16-08-50.csv'}

        monkeypatch.setattr(
            rename_gdrive_selfloops, 'fetch_first_line', lambda svc, fid: '10 1月 2026 16:08:50'
        )

        renamed, skipped = rename_files(service, files, existing_names, dry_run=False)

        assert renamed == 0
        assert skipped == 1
        assert service.files().update_calls == []

    def test_unparsable_first_line_is_skipped_without_rename(self, monkeypatch: Any) -> None:
        service = FakeDriveService()
        files = [{'id': 'f1', 'name': 'SelfLoops HRV data', 'modifiedTime': '2099-01-01T00:00:00.000Z'}]

        monkeypatch.setattr(rename_gdrive_selfloops, 'fetch_first_line', lambda svc, fid: 'garbage')

        renamed, skipped = rename_files(service, files, set(), dry_run=False)

        assert renamed == 0
        assert skipped == 1
        assert service.files().update_calls == []

    def test_ignores_modified_time_and_uses_first_line_content(self, monkeypatch: Any) -> None:
        """modifiedTime に誤った日付を入れても、1行目由来の日付が使われることを検証"""
        service = FakeDriveService()
        files = [
            {
                'id': 'f1',
                'name': 'SelfLoops HRV data',
                # わざと1行目の日付と全く異なる誤ったmodifiedTimeを入れる
                'modifiedTime': '2099-12-31T23:59:59.000Z',
            }
        ]

        monkeypatch.setattr(
            rename_gdrive_selfloops, 'fetch_first_line', lambda svc, fid: '10 1月 2026 16:08:50'
        )

        renamed, skipped = rename_files(service, files, set(), dry_run=False)

        assert renamed == 1
        call = service.files().update_calls[0]
        # 2099年ではなく、1行目由来の2026-01-10になっていること
        assert call['body'] == {'name': 'selfloops_2026-01-10--16-08-50.csv'}
