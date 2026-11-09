"""
elitehrv import / analyze --drive のテスト（Drive service はフェイク）
"""

from __future__ import annotations

import importlib.util
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location('elitehrv', PROJECT_ROOT / 'scripts' / 'elitehrv.py')
assert _spec is not None and _spec.loader is not None
elitehrv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(elitehrv)

FOLDER = 'folder-id'


class _Exec:
    def __init__(self, result: Any) -> None:
        self._result = result

    def execute(self) -> Any:
        return self._result


class FakeFiles:
    """files() リソースのフェイク。name -> bytes で Drive の中身を持つ"""

    def __init__(self, store: Optional[Dict[str, bytes]] = None) -> None:
        self.store = store or {}
        self.create_calls: List[Dict[str, Any]] = []
        self.update_calls: List[Dict[str, Any]] = []

    def list(self, **kwargs: Any) -> _Exec:
        files = [{'id': name, 'name': name} for name in self.store if f"name = '{name}'" in kwargs['q']]
        return _Exec({'files': files})

    def get_media(self, fileId: str) -> _Exec:
        return _Exec(self.store[fileId])

    def create(self, **kwargs: Any) -> _Exec:
        self.create_calls.append(kwargs)
        name = kwargs['body']['name']
        self.store[name] = Path(kwargs['media_body']._filename).read_bytes()
        return _Exec({'id': name})

    def update(self, **kwargs: Any) -> _Exec:
        self.update_calls.append(kwargs)
        self.store[kwargs['fileId']] = Path(kwargs['media_body']._filename).read_bytes()
        return _Exec({'id': kwargs['fileId']})


class FakeService:
    def __init__(self, store: Optional[Dict[str, bytes]] = None) -> None:
        self._files = FakeFiles(store)

    def files(self) -> FakeFiles:
        return self._files


def _rr_text(duration_sec: float = 120.0) -> str:
    out, t = [], 0.0
    while t < duration_sec:
        rr = 800.0 + 75.0 * math.sin(2 * math.pi * t / 15.0)
        t += rr / 1000.0
        out.append(f'{rr:.1f}')
    return '\n'.join(out) + '\n'


def _make_zip(path: Path, entries: Dict[str, str]) -> Path:
    with zipfile.ZipFile(path, 'w') as zf:
        for name, text in entries.items():
            zf.writestr(name, text)
    return path


def _export(path: Path, names: List[str]) -> Path:
    return _make_zip(path, {f'user/{n}': _rr_text() for n in names})


A = '2026-11-01 07-00-00.txt'
B = '2026-11-02 07-00-00.txt'
C = '2026-11-03 07-00-00.txt'


def test_import_creates_when_drive_has_no_zip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    service = FakeService()
    code = elitehrv.import_export(str(_export(tmp_path / 'e.zip', [A, B])), service, FOLDER)
    assert code == 0
    files = service.files()
    assert len(files.create_calls) == 1
    assert files.create_calls[0]['body'] == {'name': 'elitehrv_export.zip', 'parents': [FOLDER]}
    assert files.update_calls == []
    out = capsys.readouterr().out
    assert f'new: {A}' in out and f'new: {B}' in out


def test_import_updates_when_new_zip_is_superset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old = _export(tmp_path / 'old.zip', [A, B]).read_bytes()
    service = FakeService({'elitehrv_export.zip': old})
    code = elitehrv.import_export(str(_export(tmp_path / 'new.zip', [A, B, C])), service, FOLDER)
    assert code == 0
    files = service.files()
    assert files.create_calls == []
    assert len(files.update_calls) == 1
    assert files.update_calls[0]['fileId'] == 'elitehrv_export.zip'
    out = capsys.readouterr().out
    assert f'new: {C}' in out
    assert f'new: {A}' not in out


def test_import_refuses_when_reading_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    old = _export(tmp_path / 'old.zip', [A, B]).read_bytes()
    service = FakeService({'elitehrv_export.zip': old})
    code = elitehrv.import_export(str(_export(tmp_path / 'new.zip', [A, C])), service, FOLDER)
    assert code != 0
    assert service.files().update_calls == []
    assert service.files().store['elitehrv_export.zip'] == old
    assert f'missing: {B}' in capsys.readouterr().err


def test_import_compares_names_without_directory(tmp_path: Path) -> None:
    old = _make_zip(tmp_path / 'old.zip', {f'alice/{A}': _rr_text()}).read_bytes()
    service = FakeService({'elitehrv_export.zip': old})
    new = _make_zip(tmp_path / 'new.zip', {f'bob/{A}': _rr_text(), f'bob/{B}': _rr_text()})
    assert elitehrv.import_export(str(new), service, FOLDER) == 0
    assert len(service.files().update_calls) == 1


@pytest.mark.parametrize('entries', [
    {'user/readme.txt': '800\n'},
    {'user/2026-11-01 07-00-00.csv': '800\n'},
    {'../2026-11-01 07-00-00.txt': '800\n'},
    {'user/../2026-11-01 07-00-00.txt': '800\n'},
    {'/2026-11-01 07-00-00.txt': '800\n'},
    {'a/b/2026-11-01 07-00-00.txt': '800\n'},
    {'user/2026-11-01 07-00-00.txt': '800\nabc\n'},
])
def test_import_rejects_invalid_zip(tmp_path: Path, entries: Dict[str, str]) -> None:
    service = FakeService()
    code = elitehrv.import_export(str(_make_zip(tmp_path / 'bad.zip', entries)), service, FOLDER)
    assert code != 0
    assert service.files().create_calls == []


def test_validate_accepts_flat_and_one_level(tmp_path: Path) -> None:
    z = _make_zip(tmp_path / 'ok.zip', {'user/': '', f'user/{A}': '800\n\n810\n', B: '790\n'})
    assert elitehrv.validate_export_zip(z) == {A: f'user/{A}', B: B}


def test_validate_skips_export_readme(tmp_path: Path) -> None:
    readme = 'This export is of all of your raw RR interval data.\n'
    z = _make_zip(tmp_path / 'ok.zip', {'!About This Export.txt': readme, f'user/{A}': '800\n'})
    assert elitehrv.validate_export_zip(z) == {A: f'user/{A}'}


def test_validate_rejects_readme_name_in_subdirectory(tmp_path: Path) -> None:
    z = _make_zip(tmp_path / 'bad.zip', {'user/!About This Export.txt': 'text\n', f'user/{A}': '800\n'})
    with pytest.raises(elitehrv.ExportZipError):
        elitehrv.validate_export_zip(z)


def test_import_rejects_non_zip(tmp_path: Path) -> None:
    path = tmp_path / 'page.html'
    path.write_text('<html></html>')
    assert elitehrv.import_export(str(path), FakeService(), FOLDER) != 0


def test_analyze_drive_processes_only_new_readings(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    service = FakeService({'elitehrv_export.zip': _export(tmp_path / 'e.zip', [A, B]).read_bytes()})
    log = tmp_path / 'resonance_log.csv'
    pd.DataFrame([{'timestamp': '2026-11-01 07:00:00', 'kind': 'readiness', 'practice': 'rf'}]).to_csv(
        log, index=False,
    )

    assert elitehrv.analyze_drive(service, FOLDER, log_path=log, verbose=False) == 0
    df = pd.read_csv(log, dtype={'timestamp': str})
    assert sorted(df['timestamp']) == ['2026-11-01 07:00:00', '2026-11-02 07:00:00']
    assert '1 / 2 readings' in capsys.readouterr().out
    # 既存行は再計算されない
    assert pd.isna(df.loc[df['timestamp'] == '2026-11-01 07:00:00', 'rmssd_ms']).all()

    assert elitehrv.analyze_drive(service, FOLDER, log_path=log, verbose=False) == 0
    assert '0 / 2 readings' in capsys.readouterr().out


def test_analyze_drive_all_reprocesses(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    service = FakeService({'elitehrv_export.zip': _export(tmp_path / 'e.zip', [A, B]).read_bytes()})
    log = tmp_path / 'resonance_log.csv'
    assert elitehrv.analyze_drive(service, FOLDER, log_path=log, verbose=False) == 0
    capsys.readouterr()
    assert elitehrv.analyze_drive(service, FOLDER, log_path=log, reprocess_all=True, verbose=False) == 0
    assert '2 / 2 readings' in capsys.readouterr().out
    assert len(pd.read_csv(log)) == 2


def test_analyze_drive_without_zip_fails(tmp_path: Path) -> None:
    assert elitehrv.analyze_drive(FakeService(), FOLDER, log_path=tmp_path / 'l.csv') != 0


def test_main_rejects_drive_with_txt() -> None:
    with pytest.raises(SystemExit):
        elitehrv.main(['analyze', 'x.txt', '--drive'])
