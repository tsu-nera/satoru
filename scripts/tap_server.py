"""
Tap Server - 瞑想中のマインドワンダリング打刻サーバー

スマホのブラウザから LAN 経由でタップイベントを受け取り、
data/taps/ 配下にセッションごとの CSV として記録する。
標準ライブラリのみで実装しており、新規の依存追加はしていない。

使い方:
    uv run python scripts/tap_server.py
    uv run python scripts/tap_server.py --port 8765 --output-dir data/taps
"""

import argparse
import json
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from lib.recorders import TapLogRecorder  # noqa: E402

WEB_ROOT = project_root / 'src' / 'app' / 'tap_web'


def get_lan_ip() -> str:
    """LAN上でこのホストが使っているIPアドレスを推定する（実際の通信は発生しない）"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except OSError:
        return socket.gethostname()
    finally:
        sock.close()


def create_server(host: str, port: int, output_dir: Path) -> ThreadingHTTPServer:
    """タップ記録サーバーを構築する（テストから port=0 で呼び出せる）"""
    recorder_holder: dict[str, TapLogRecorder | None] = {'recorder': None}
    lock = threading.Lock()

    class TapRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            # http.server 標準の毎リクエストログは冗長なので抑制する
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict | None:
            length = int(self.headers.get('Content-Length', 0))
            if length <= 0:
                return None
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return None
            if not isinstance(data, dict):
                return None
            return data

        def do_POST(self) -> None:  # noqa: N802
            data = self._read_json_body()
            if data is None or 'client_ts' not in data:
                self._send_json(400, {'error': 'invalid request body'})
                return
            client_ts = data['client_ts']

            with lock:
                if self.path == '/api/session/start':
                    recorder = TapLogRecorder(output_dir)
                    session_id = recorder.start(client_ts)
                    recorder_holder['recorder'] = recorder
                    self._send_json(200, {'session_id': session_id})
                    return

                if self.path == '/api/tap':
                    tap_recorder = recorder_holder['recorder']
                    if tap_recorder is None:
                        self._send_json(400, {'error': 'session not started'})
                        return
                    tap_recorder.tap(client_ts)
                    self._send_json(200, {'ok': True})
                    return

                if self.path == '/api/session/stop':
                    stop_recorder = recorder_holder['recorder']
                    if stop_recorder is None:
                        self._send_json(400, {'error': 'session not started'})
                        return
                    stop_recorder.stop(client_ts)
                    self._send_json(200, {'ok': True})
                    return

            self._send_json(404, {'error': 'not found'})

    return ThreadingHTTPServer((host, port), TapRequestHandler)


def main() -> None:
    parser = argparse.ArgumentParser(description='瞑想中のマインドワンダリング打刻サーバー')
    parser.add_argument('--host', default='0.0.0.0', help='待ち受けホスト (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8765, help='待ち受けポート (default: 8765)')
    parser.add_argument('--output-dir', default='data/taps', help='CSV出力先ディレクトリ (default: data/taps)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    server = create_server(args.host, args.port, output_dir)

    lan_ip = get_lan_ip()
    print('=' * 60)
    print('Tap Server - マインドワンダリング打刻サーバー')
    print('=' * 60)
    print(f'Listening on {args.host}:{args.port}')
    print(f'出力先: {output_dir}')
    print()
    print(f'スマホのブラウザで開く: http://{lan_ip}:{args.port}/')
    print()
    print('Ctrl+C で終了')
    print('=' * 60, flush=True)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
