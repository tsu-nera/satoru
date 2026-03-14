"""
Binaural Beat WebSocket Server

MuseAppStateと連携し、Mind Stateに応じたバイノーラルビートの
パラメータをWebSocket経由でブラウザに送信する。

使い方:
    source venv/bin/activate
    python src/app/binaural_ws_server.py [options]

オプション:
    --ip IP             OSC リッスンIP (default: 0.0.0.0)
    --port PORT         OSC リッスンポート (default: 5000)
    --ws-port PORT      WebSocket ポート (default: 8765)
    --calibration SEC   キャリブレーション秒数 (default: 60)

ブラウザ:
    src/web/binaural.html を開き、WebSocket URLに ws://localhost:8765 を指定
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from typing import Optional, Set

import tornado.ioloop
import tornado.web
import tornado.websocket
from pythonosc import dispatcher, osc_server

from muse_app import (
    ALPHA_OSC_PATH,
    BETA_OSC_PATH,
    DEFAULT_CALIBRATION_SEC,
    HSI_PATH,
    AppPhase,
    MindState,
    MuseAppState,
)

# --------------------------------------------------------------------------- #
# Mind State → Binaural Beat パラメータ
# --------------------------------------------------------------------------- #

BINAURAL_MAP: dict[MindState, tuple[float, float]] = {
    MindState.CALM:    (200, 10),   # α波 (8-12Hz): リラックス維持
    MindState.NEUTRAL: (200,  7),   # θ波 (4-8Hz): 瞑想誘導
    MindState.ACTIVE:  (200,  5),   # θ波低域: 鎮静化
}

# WebSocket送信間隔（秒）
WS_SEND_INTERVAL = 0.5


# --------------------------------------------------------------------------- #
# WebSocket Handler
# --------------------------------------------------------------------------- #

class BinauralWebSocketHandler(tornado.websocket.WebSocketHandler):
    """ブラウザ向けWebSocketハンドラー。"""

    clients: Set[BinauralWebSocketHandler] = set()

    def check_origin(self, origin: str) -> bool:
        return True  # ローカル使用のためCORS無制限

    def open(self) -> None:
        BinauralWebSocketHandler.clients.add(self)
        print(f"[WS] Client connected ({len(self.clients)} total)")

    def on_close(self) -> None:
        BinauralWebSocketHandler.clients.discard(self)
        print(f"[WS] Client disconnected ({len(self.clients)} total)")

    def on_message(self, message: str) -> None:
        pass  # ブラウザからのメッセージは不要

    @classmethod
    def broadcast(cls, data: dict) -> None:
        """全クライアントにJSONメッセージを送信。"""
        msg = json.dumps(data)
        for client in list(cls.clients):
            try:
                client.write_message(msg)
            except tornado.websocket.WebSocketClosedError:
                cls.clients.discard(client)


# --------------------------------------------------------------------------- #
# State → WebSocket ブリッジ
# --------------------------------------------------------------------------- #

class BinauralBridge:
    """MuseAppStateを監視し、状態変化をWebSocketで配信する。"""

    def __init__(self, state: MuseAppState):
        self.state = state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_state: Optional[MindState] = None
        self._io_loop: Optional[tornado.ioloop.IOLoop] = None

    def start(self, io_loop: tornado.ioloop.IOLoop) -> None:
        self._io_loop = io_loop
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _loop(self) -> None:
        while self._running:
            self._send_state()
            time.sleep(WS_SEND_INTERVAL)

    def _send_state(self) -> None:
        if not BinauralWebSocketHandler.clients:
            return

        state = self.state
        mind_state = state.current_state
        carrier, beat = BINAURAL_MAP.get(mind_state, (200, 7))

        data = {
            'mind_state': mind_state.value,
            'carrier_hz': carrier,
            'beat_hz': beat,
            'z_score': round(state.current_z, 2),
            'ratio': round(state.current_ratio, 3),
            'phase': state.phase.value,
        }

        if self._io_loop is not None:
            self._io_loop.add_callback(BinauralWebSocketHandler.broadcast, data)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Binaural Beat WebSocket Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--ip', default='0.0.0.0', help='OSC listen IP (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='OSC listen port (default: 5000)')
    parser.add_argument('--ws-port', type=int, default=8765, help='WebSocket port (default: 8765)')
    parser.add_argument('--calibration', type=int, default=DEFAULT_CALIBRATION_SEC,
                        help=f'Calibration seconds (default: {DEFAULT_CALIBRATION_SEC})')
    args = parser.parse_args()

    # MuseAppState（音声フィードバックなし — ブラウザ側で再生）
    state = MuseAppState(calibration_sec=args.calibration)

    # OSC Dispatcher
    disp = dispatcher.Dispatcher()
    disp.map(ALPHA_OSC_PATH, state.handle_alpha)
    disp.map(BETA_OSC_PATH, state.handle_beta)
    disp.map(HSI_PATH, state.handle_hsi)

    osc = osc_server.ThreadingOSCUDPServer((args.ip, args.port), disp)

    # Tornado WebSocket
    app = tornado.web.Application([
        (r'/ws', BinauralWebSocketHandler),
    ])
    app.listen(args.ws_port)
    io_loop = tornado.ioloop.IOLoop.current()

    # Bridge
    bridge = BinauralBridge(state)
    bridge.start(io_loop)

    print("=" * 50)
    print("  Binaural Beat WebSocket Server")
    print("=" * 50)
    print(f"OSC Listen:   {args.ip}:{args.port}")
    print(f"WebSocket:    ws://localhost:{args.ws_port}/ws")
    print(f"Calibration:  {args.calibration}s")
    print()
    print("1. Mind Monitor → OSC streaming to this server")
    print(f"2. Open src/web/binaural.html in browser")
    print(f"   WebSocket URL: ws://localhost:{args.ws_port}/ws")
    print()
    print("Waiting for data... (Ctrl+C to exit)")
    print("=" * 50)

    # OSCサーバーを別スレッドで起動
    osc_thread = threading.Thread(target=osc.serve_forever, daemon=True)
    osc_thread.start()

    try:
        io_loop.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        bridge.stop()
        osc.shutdown()
        state.print_summary()


if __name__ == '__main__':
    main()
