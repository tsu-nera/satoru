"""
scripts/upload_to_gdrive.py のテスト

Google Drive API は FakeDriveService でモックし、実際のAPIは一切叩かない。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import scripts.upload_to_gdrive as upload_to_gdrive
from scripts.upload_to_gdrive import list_remote_names, main, upload_file


class _FakeExecutable:
    def __init__(self, result: Any) -> None:
        self._result = result

    def execute(self) -> Any:
        return self._result


class FakeFilesResource:
    """Google Drive API の files() リソースを模したフェイク"""

    def __init__(self, remote_files: Optional[List[Dict[str, str]]] = None) -> None:
        self.remote_files = remote_files or []
        self.create_calls: List[Dict[str, Any]] = []
        self.list_calls: List[Dict[str, Any]] = []

    def list(self, **kwargs: Any) -> _FakeExecutable:
        self.list_calls.append(kwargs)
        page_token = kwargs.get('pageToken')

        # 1ページ目とページングを模擬（2件ずつ返す）
        page_size = 2
        start = 0 if page_token is None else int(page_token)
        end = start + page_size
        page = self.remote_files[start:end]
        next_token = str(end) if end < len(self.remote_files) else None

        return _FakeExecutable({'files': page, 'nextPageToken': next_token})

    def create(self, **kwargs: Any) -> _FakeExecutable:
        self.create_calls.append(kwargs)
        return _FakeExecutable({'id': 'fake-id'})


class FakeDriveService:
    def __init__(self, remote_files: Optional[List[Dict[str, str]]] = None) -> None:
        self._files = FakeFilesResource(remote_files)

    def files(self) -> FakeFilesResource:
        return self._files


class TestListRemoteNames:
    """list_remote_names のページング検証"""

    def test_pages_through_all_files(self) -> None:
        remote_files = [{'id': str(i), 'name': f'file{i}.csv'} for i in range(5)]
        service = FakeDriveService(remote_files)

        names = list_remote_names(service, 'folder-id')

        assert names == {f'file{i}.csv' for i in range(5)}
        # 5件を2件ずつ取得するので3回呼ばれる
        assert len(service.files().list_calls) == 3


class TestUploadFile:
    """upload_file の冪等性検証"""

    def test_skips_when_name_already_exists(self, tmp_path: Path) -> None:
        path = tmp_path / 'taps_2026-01-10--10-00-00.csv'
        path.write_text('dummy')
        service = FakeDriveService()
        existing_names = {path.name}

        result = upload_file(service, 'folder-id', path, existing_names)

        assert result is False
        assert service.files().create_calls == []

    def test_uploads_new_file_with_expected_body(self, tmp_path: Path) -> None:
        path = tmp_path / 'taps_2026-01-10--10-00-00.csv'
        path.write_text('dummy')
        service = FakeDriveService()
        existing_names: set[str] = set()

        result = upload_file(service, 'folder-id', path, existing_names)

        assert result is True
        assert len(service.files().create_calls) == 1
        call = service.files().create_calls[0]
        assert call['body'] == {'name': path.name, 'parents': ['folder-id']}

    def test_dry_run_does_not_call_create(self, tmp_path: Path) -> None:
        path = tmp_path / 'taps_2026-01-10--10-00-00.csv'
        path.write_text('dummy')
        service = FakeDriveService()
        existing_names: set[str] = set()

        result = upload_file(service, 'folder-id', path, existing_names, dry_run=True)

        assert result is True
        assert service.files().create_calls == []

    def test_same_file_uploaded_twice_in_one_run_creates_once(self, tmp_path: Path) -> None:
        path = tmp_path / 'taps_2026-01-10--10-00-00.csv'
        path.write_text('dummy')
        service = FakeDriveService()
        existing_names: set[str] = set()

        first = upload_file(service, 'folder-id', path, existing_names)
        second = upload_file(service, 'folder-id', path, existing_names)

        assert first is True
        assert second is False
        assert len(service.files().create_calls) == 1


class TestMain:
    """main のフォルダID未設定時の挙動検証"""

    def test_returns_zero_and_skips_auth_when_folder_id_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {'auth': False}

        def _fake_authenticate(*args: Any, **kwargs: Any) -> Any:
            called['auth'] = True
            return FakeDriveService()

        monkeypatch.setattr(upload_to_gdrive, 'authenticate_gdrive', _fake_authenticate)
        monkeypatch.delenv('GDRIVE_FOLDER_ID_TAPS', raising=False)

        result = main(['--folder-id', ''])

        assert result == 0
        assert called['auth'] is False

    def test_uploads_files_in_data_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'taps_2026-01-10--10-00-00.csv').write_text('dummy')
        (tmp_path / 'taps_2026-01-11--10-00-00.csv').write_text('dummy')

        service = FakeDriveService()
        monkeypatch.setattr(upload_to_gdrive, 'authenticate_gdrive', lambda *a, **k: service)

        result = main(['--folder-id', 'folder-id', '--data-dir', str(tmp_path)])

        assert result == 0
        assert len(service.files().create_calls) == 2
