"""
Tap Log Recorder / Tap Server のテスト

TapLogRecorder のユニットテストに加え、tap_server.py の create_server を
port=0 で起動して HTTP 疎通を検証する。
"""

from __future__ import annotations

import csv
import json
import threading
from datetime import datetime
from http.client import HTTPConnection
from pathlib import Path

import pytest

from lib.recorders.tap_log import TapLogRecorder
from scripts.tap_server import create_server


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


class TestTapLogRecorder:
    """TapLogRecorder のユニットテスト"""

    def test_writes_header_and_events_in_order(self, tmp_path: Path) -> None:
        recorder = TapLogRecorder(tmp_path)
        recorder.start('2026-01-01T00:00:00.000Z')
        recorder.tap('2026-01-01T00:00:01.000Z')
        recorder.stop('2026-01-01T00:00:02.000Z')

        csv_path = tmp_path / f'taps_{recorder.session_id}.csv'
        assert csv_path.exists()

        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            header = next(reader)

        assert header == ['seq', 'event', 'client_ts', 'server_ts', 'elapsed_s']

        rows = _read_csv_rows(csv_path)
        assert [row['event'] for row in rows] == ['start', 'tap', 'stop']

    def test_seq_is_zero_based_and_monotonic(self, tmp_path: Path) -> None:
        recorder = TapLogRecorder(tmp_path)
        recorder.start('t0')
        recorder.tap('t1')
        recorder.tap('t2')
        recorder.stop('t3')

        csv_path = tmp_path / f'taps_{recorder.session_id}.csv'
        rows = _read_csv_rows(csv_path)
        seqs = [int(row['seq']) for row in rows]

        assert seqs == [0, 1, 2, 3]

    def test_elapsed_s_is_nonnegative_and_nondecreasing(self, tmp_path: Path) -> None:
        recorder = TapLogRecorder(tmp_path)
        recorder.start('t0')
        recorder.tap('t1')
        recorder.tap('t2')
        recorder.stop('t3')

        csv_path = tmp_path / f'taps_{recorder.session_id}.csv'
        rows = _read_csv_rows(csv_path)
        elapsed = [float(row['elapsed_s']) for row in rows]

        assert elapsed[0] == 0.0
        assert all(e >= 0.0 for e in elapsed)
        assert elapsed == sorted(elapsed)

    def test_client_ts_and_server_ts_are_parseable_iso8601(self, tmp_path: Path) -> None:
        recorder = TapLogRecorder(tmp_path)
        recorder.start('2026-01-01T00:00:00.000Z')
        recorder.tap('2026-01-01T00:00:01.000Z')

        csv_path = tmp_path / f'taps_{recorder.session_id}.csv'
        rows = _read_csv_rows(csv_path)

        for row in rows:
            assert row['client_ts'] != ''
            assert row['server_ts'] != ''
            # client_ts は 'Z' 表記(ISO8601)なので fromisoformat が扱える形へ変換
            datetime.fromisoformat(row['client_ts'].replace('Z', '+00:00'))
            datetime.fromisoformat(row['server_ts'])

    def test_tap_before_start_raises(self, tmp_path: Path) -> None:
        recorder = TapLogRecorder(tmp_path)
        with pytest.raises(RuntimeError):
            recorder.tap('t0')

    def test_stop_before_start_raises(self, tmp_path: Path) -> None:
        recorder = TapLogRecorder(tmp_path)
        with pytest.raises(RuntimeError):
            recorder.stop('t0')


class TestTapServer:
    """tap_server.py の HTTP 疎通テスト"""

    @pytest.fixture
    def server(self, tmp_path: Path):
        srv = create_server('127.0.0.1', 0, tmp_path)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        try:
            yield srv
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=5)

    def _post(self, port: int, path: str, body: dict) -> tuple[int, dict]:
        conn = HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            payload = json.dumps(body).encode('utf-8')
            conn.request('POST', path, body=payload, headers={'Content-Type': 'application/json'})
            res = conn.getresponse()
            data = json.loads(res.read().decode('utf-8'))
            return res.status, data
        finally:
            conn.close()

    def test_full_session_flow_creates_csv(self, server, tmp_path: Path) -> None:
        port = server.server_address[1]

        status, data = self._post(port, '/api/session/start', {'client_ts': 't0'})
        assert status == 200
        session_id = data['session_id']

        status, data = self._post(port, '/api/tap', {'client_ts': 't1'})
        assert status == 200
        assert data == {'ok': True}

        status, data = self._post(port, '/api/session/stop', {'client_ts': 't2'})
        assert status == 200
        assert data == {'ok': True}

        csv_path = tmp_path / f'taps_{session_id}.csv'
        assert csv_path.exists()

        rows = _read_csv_rows(csv_path)
        assert [row['event'] for row in rows] == ['start', 'tap', 'stop']

    def test_get_root_serves_index_html(self, server) -> None:
        port = server.server_address[1]
        conn = HTTPConnection('127.0.0.1', port, timeout=5)
        try:
            conn.request('GET', '/')
            res = conn.getresponse()
            body = res.read().decode('utf-8')
            assert res.status == 200
            assert '<title>Tap Logger</title>' in body
        finally:
            conn.close()

    def test_unknown_path_returns_404(self, server) -> None:
        port = server.server_address[1]
        status, _ = self._post(port, '/api/does-not-exist', {'client_ts': 't0'})
        assert status == 404

    def test_missing_client_ts_returns_400(self, server) -> None:
        port = server.server_address[1]
        status, _ = self._post(port, '/api/session/start', {})
        assert status == 400
