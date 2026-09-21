"""
scripts/upload_to_gdrive.py のテスト

Google Drive API・OAuthフローは FakeDriveService / フェイク Credentials でモックし、
実際のAPI・実ブラウザは一切呼ばない。
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from google.auth.exceptions import RefreshError

import scripts.upload_to_gdrive as upload_to_gdrive
from scripts.upload_to_gdrive import (
    authorize_gdrive,
    create_folder,
    list_remote_names,
    main,
    upload_file,
)


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
        return _FakeExecutable({'id': 'fake-id', 'name': kwargs.get('body', {}).get('name')})


class FakeDriveService:
    def __init__(self, remote_files: Optional[List[Dict[str, str]]] = None) -> None:
        self._files = FakeFilesResource(remote_files)

    def files(self) -> FakeFilesResource:
        return self._files


class FakeCreds:
    """google.oauth2.credentials.Credentials を模したフェイク"""

    def __init__(
        self,
        valid: bool = True,
        expired: bool = False,
        refresh_token: Optional[str] = 'refresh-token',
        raise_on_refresh: Optional[Exception] = None,
    ) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self._raise_on_refresh = raise_on_refresh
        self.refresh_called = False

    def refresh(self, request: Any) -> None:
        self.refresh_called = True
        if self._raise_on_refresh:
            raise self._raise_on_refresh
        self.valid = True
        self.expired = False

    def to_json(self) -> str:
        return json.dumps({'token': 'fake-token'})


class FakeFlow:
    def __init__(self, creds: FakeCreds) -> None:
        self._creds = creds
        self.run_local_server_called = False

    def run_local_server(self, port: int = 0) -> FakeCreds:
        self.run_local_server_called = True
        return self._creds


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


class TestCreateFolder:
    """create_folder のリクエスト内容検証"""

    def test_creates_folder_with_expected_mimetype(self) -> None:
        service = FakeDriveService()

        folder_id = create_folder(service, 'Taps')

        assert folder_id == 'fake-id'
        assert len(service.files().create_calls) == 1
        call = service.files().create_calls[0]
        assert call['body'] == {'name': 'Taps', 'mimeType': 'application/vnd.google-apps.folder'}


class TestAuthorizeGdrive:
    """authorize_gdrive の認証フロー分岐検証"""

    def test_valid_existing_token_skips_flow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token_path = tmp_path / 'gdrive.json'
        token_path.write_text('{}')
        client_secrets = tmp_path / 'oauth-client.json'
        client_secrets.write_text('{}')

        creds = FakeCreds(valid=True)
        monkeypatch.setattr(
            upload_to_gdrive.Credentials, 'from_authorized_user_file', lambda *a, **k: creds
        )

        def _fail_flow(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('InstalledAppFlow は呼ばれてはいけない')

        monkeypatch.setattr(
            upload_to_gdrive.InstalledAppFlow, 'from_client_secrets_file', _fail_flow
        )
        monkeypatch.setattr(upload_to_gdrive, 'build', lambda *a, **k: FakeDriveService())

        service = authorize_gdrive(client_secrets, token_path)

        assert isinstance(service, FakeDriveService)

    def test_expired_token_with_refresh_token_refreshes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token_path = tmp_path / 'gdrive.json'
        token_path.write_text('{}')
        client_secrets = tmp_path / 'oauth-client.json'
        client_secrets.write_text('{}')

        creds = FakeCreds(valid=False, expired=True, refresh_token='rt')
        monkeypatch.setattr(
            upload_to_gdrive.Credentials, 'from_authorized_user_file', lambda *a, **k: creds
        )

        def _fail_flow(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('InstalledAppFlow は呼ばれてはいけない')

        monkeypatch.setattr(
            upload_to_gdrive.InstalledAppFlow, 'from_client_secrets_file', _fail_flow
        )
        monkeypatch.setattr(upload_to_gdrive, 'build', lambda *a, **k: FakeDriveService())

        authorize_gdrive(client_secrets, token_path)

        assert creds.refresh_called is True

    def test_refresh_error_falls_back_to_flow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token_path = tmp_path / 'gdrive.json'
        token_path.write_text('{}')
        client_secrets = tmp_path / 'oauth-client.json'
        client_secrets.write_text('{}')

        broken_creds = FakeCreds(
            valid=False, expired=True, refresh_token='rt', raise_on_refresh=RefreshError('expired')
        )
        new_creds = FakeCreds(valid=True)
        fake_flow = FakeFlow(new_creds)

        monkeypatch.setattr(
            upload_to_gdrive.Credentials, 'from_authorized_user_file', lambda *a, **k: broken_creds
        )
        monkeypatch.setattr(
            upload_to_gdrive.InstalledAppFlow,
            'from_client_secrets_file',
            lambda *a, **k: fake_flow,
        )
        monkeypatch.setattr(upload_to_gdrive, 'build', lambda *a, **k: FakeDriveService())

        authorize_gdrive(client_secrets, token_path)

        assert broken_creds.refresh_called is True
        assert fake_flow.run_local_server_called is True

    def test_no_existing_token_runs_flow_and_writes_token(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        token_path = tmp_path / 'tokens' / 'gdrive.json'
        client_secrets = tmp_path / 'oauth-client.json'
        client_secrets.write_text('{}')

        new_creds = FakeCreds(valid=True)
        fake_flow = FakeFlow(new_creds)

        monkeypatch.setattr(
            upload_to_gdrive.InstalledAppFlow,
            'from_client_secrets_file',
            lambda *a, **k: fake_flow,
        )
        monkeypatch.setattr(upload_to_gdrive, 'build', lambda *a, **k: FakeDriveService())

        authorize_gdrive(client_secrets, token_path)

        assert fake_flow.run_local_server_called is True
        assert token_path.exists()
        mode = stat.S_IMODE(token_path.stat().st_mode)
        assert mode == 0o600


class TestMain:
    """main の認証・分岐制御の検証"""

    def test_dry_run_never_authorizes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'taps_2026-01-10--10-00-00.csv').write_text('dummy')

        def _fail_authorize(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError('dry-run では authorize_gdrive は呼ばれてはいけない')

        monkeypatch.setattr(upload_to_gdrive, 'authorize_gdrive', _fail_authorize)

        result = main(['--data-dir', str(tmp_path), '--folder-id', 'DUMMY', '--dry-run'])

        assert result == 0

    def test_returns_zero_and_skips_auth_when_folder_id_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        called = {'auth': False}

        def _fake_authorize(*args: Any, **kwargs: Any) -> Any:
            called['auth'] = True
            return FakeDriveService()

        monkeypatch.setattr(upload_to_gdrive, 'authorize_gdrive', _fake_authorize)
        monkeypatch.delenv('GDRIVE_FOLDER_ID_TAPS', raising=False)

        result = main(['--folder-id', '', '--data-dir', str(tmp_path)])

        assert result == 0
        assert called['auth'] is False

    def test_missing_oauth_client_returns_one_without_authorizing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'taps_2026-01-10--10-00-00.csv').write_text('dummy')
        called = {'auth': False}

        def _fake_authorize(*args: Any, **kwargs: Any) -> Any:
            called['auth'] = True
            return FakeDriveService()

        monkeypatch.setattr(upload_to_gdrive, 'authorize_gdrive', _fake_authorize)

        missing_client = tmp_path / 'no-such-oauth-client.json'
        result = main(
            [
                '--data-dir', str(tmp_path),
                '--folder-id', 'folder-id',
                '--oauth-client', str(missing_client),
            ]
        )

        assert result == 1
        assert called['auth'] is False

    def test_uploads_files_in_data_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / 'taps_2026-01-10--10-00-00.csv').write_text('dummy')
        (tmp_path / 'taps_2026-01-11--10-00-00.csv').write_text('dummy')
        client_secrets = tmp_path / 'oauth-client.json'
        client_secrets.write_text('{}')

        service = FakeDriveService()
        monkeypatch.setattr(upload_to_gdrive, 'authorize_gdrive', lambda *a, **k: service)

        result = main(
            [
                '--folder-id', 'folder-id',
                '--data-dir', str(tmp_path),
                '--oauth-client', str(client_secrets),
            ]
        )

        assert result == 0
        assert len(service.files().create_calls) == 2

    def test_create_folder_creates_and_skips_upload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client_secrets = tmp_path / 'oauth-client.json'
        client_secrets.write_text('{}')
        service = FakeDriveService()
        monkeypatch.setattr(upload_to_gdrive, 'authorize_gdrive', lambda *a, **k: service)

        result = main(
            [
                '--create-folder', 'Taps',
                '--oauth-client', str(client_secrets),
            ]
        )

        assert result == 0
        assert len(service.files().create_calls) == 1
        call = service.files().create_calls[0]
        assert call['body'] == {'name': 'Taps', 'mimeType': 'application/vnd.google-apps.folder'}
