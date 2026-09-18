"""
scripts/compress_muse_data.py のテスト

すべて tmp_path を使い、実データディレクトリ (data/) には一切触れない。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import scripts.compress_muse_data as compress_muse_data
from scripts.compress_muse_data import VerificationError, compress_file, main


def _write_sample_csv(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            'TimeStamp': ['2026-01-01 00:00:00.0', '2026-01-01 00:00:01.0', '2026-01-01 00:00:02.0'],
            'Delta_TP9': [0.1, 0.2, 0.3],
            'Alpha_TP9': [1.1, 1.2, 1.3],
        }
    )
    df.to_csv(path, index=False)
    return df


class TestCompressFile:
    """compress_file のユニットテスト"""

    def test_compresses_and_content_matches(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        original = _write_sample_csv(csv_path)

        gz_path = compress_file(csv_path)

        assert gz_path.exists()
        restored = pd.read_csv(gz_path)
        pd.testing.assert_frame_equal(original, restored)

    def test_keeps_original_by_default(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        compress_file(csv_path)

        assert csv_path.exists()

    def test_delete_original_removes_source_after_success(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        gz_path = compress_file(csv_path, delete_original=True)

        assert not csv_path.exists()
        assert gz_path.exists()

    def test_verification_failure_keeps_original_and_removes_gz(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        def _fail_verify(csv_path: Path, gz_path: Path) -> None:
            raise VerificationError('検証失敗（テスト用）')

        monkeypatch.setattr(compress_muse_data, 'verify_compressed', _fail_verify)

        with pytest.raises(VerificationError):
            compress_file(csv_path, delete_original=True)

        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        assert csv_path.exists()
        assert not gz_path.exists()


class TestMain:
    """main のユニットテスト"""

    def test_processes_directory_and_returns_zero(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        result = main(['--data-dir', str(tmp_path)])

        assert result == 0
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        assert gz_path.exists()
        assert csv_path.exists()

    def test_returns_nonzero_on_verification_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        def _fail_verify(csv_path: Path, gz_path: Path) -> None:
            raise VerificationError('検証失敗（テスト用）')

        monkeypatch.setattr(compress_muse_data, 'verify_compressed', _fail_verify)

        result = main(['--data-dir', str(tmp_path)])

        assert result == 1
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        assert csv_path.exists()
        assert not gz_path.exists()


class TestMainFile:
    """main の --file 単一ファイル指定のユニットテスト"""

    def test_compresses_single_file_and_keeps_original(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        result = main(['--file', str(csv_path)])

        assert result == 0
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        assert gz_path.exists()
        assert csv_path.exists()

    def test_delete_original_removes_source_after_success(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        result = main(['--file', str(csv_path), '--delete-original'])

        assert result == 0
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        assert gz_path.exists()
        assert not csv_path.exists()

    def test_file_and_data_dir_together_is_mutually_exclusive(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        with pytest.raises(SystemExit) as excinfo:
            main(['--file', str(csv_path), '--data-dir', str(tmp_path)])

        assert excinfo.value.code != 0

    def test_nonexistent_file_returns_nonzero(self, tmp_path: Path) -> None:
        result = main(['--file', str(tmp_path / 'does-not-exist.csv')])

        assert result != 0

    def test_non_csv_extension_returns_nonzero(self, tmp_path: Path) -> None:
        txt_path = tmp_path / 'not-a-csv.txt'
        txt_path.write_text('dummy')

        result = main(['--file', str(txt_path)])

        assert result != 0

    def test_verification_failure_keeps_original_and_removes_gz(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)

        def _fail_verify(csv_path: Path, gz_path: Path) -> None:
            raise VerificationError('検証失敗（テスト用）')

        monkeypatch.setattr(compress_muse_data, 'verify_compressed', _fail_verify)

        result = main(['--file', str(csv_path)])

        assert result != 0
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        assert csv_path.exists()
        assert not gz_path.exists()

    def test_skips_when_gz_already_exists(self, tmp_path: Path) -> None:
        csv_path = tmp_path / 'mindMonitor_2026-01-01--00-00-00.csv'
        _write_sample_csv(csv_path)
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        gz_path.write_bytes(b'dummy-existing-gz')

        result = main(['--file', str(csv_path), '--delete-original'])

        assert result == 0
        assert csv_path.exists()
        assert gz_path.read_bytes() == b'dummy-existing-gz'
