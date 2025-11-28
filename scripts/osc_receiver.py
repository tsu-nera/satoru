"""
Muse App OSC Receiver - シンプルな受信テスト用スクリプト

使い方:
1. pip install python-osc
2. python osc_receiver.py
3. Muse AppでOSC出力先をこのPCのIPとポート5000に設定
"""

from pythonosc import dispatcher
from pythonosc import osc_server
import argparse
from datetime import datetime


def make_handler(name):
    """汎用ハンドラーを作成"""
    def handler(address, *args):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] {address}: {args}")
    return handler


def eeg_handler(address, *args):
    """EEGデータ用ハンドラー（高頻度なので簡略表示）"""
    # 10回に1回だけ表示（256Hzは多すぎる）
    if not hasattr(eeg_handler, 'count'):
        eeg_handler.count = 0
    eeg_handler.count += 1
    if eeg_handler.count % 50 == 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] EEG: TP9={args[0]:.1f}, AF7={args[1]:.1f}, AF8={args[2]:.1f}, TP10={args[3]:.1f}")


def default_handler(address, *args):
    """未知のアドレス用ハンドラー"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [NEW] {address}: {args}")


def main():
    parser = argparse.ArgumentParser(description='Muse OSC Receiver')
    parser.add_argument('--ip', default='0.0.0.0', help='リッスンするIP (default: 0.0.0.0 = all interfaces)')
    parser.add_argument('--port', type=int, default=5000, help='リッスンするポート (default: 5000)')
    args = parser.parse_args()

    disp = dispatcher.Dispatcher()

    # Mind Monitor / Muse Direct 標準パス
    disp.map("/muse/eeg", eeg_handler)
    disp.map("/muse/elements/delta_absolute", make_handler("Delta"))
    disp.map("/muse/elements/theta_absolute", make_handler("Theta"))
    disp.map("/muse/elements/alpha_absolute", make_handler("Alpha"))
    disp.map("/muse/elements/beta_absolute", make_handler("Beta"))
    disp.map("/muse/elements/gamma_absolute", make_handler("Gamma"))
    disp.map("/muse/elements/horseshoe", make_handler("Horseshoe"))
    disp.map("/muse/acc", make_handler("Accelerometer"))
    disp.map("/muse/gyro", make_handler("Gyro"))
    disp.map("/muse/elements/blink", make_handler("Blink"))
    disp.map("/muse/elements/jaw_clench", make_handler("JawClench"))
    disp.map("/muse/batt", make_handler("Battery"))

    # 未知のアドレスもキャッチ（Muse Appの独自フォーマットを発見するため）
    disp.set_default_handler(default_handler)

    print("=" * 60)
    print("Muse OSC Receiver")
    print("=" * 60)
    print(f"Listening on {args.ip}:{args.port}")
    print()
    print("Muse Appの設定:")
    print(f"  - IP: あなたのPCのIPアドレス")
    print(f"  - Port: {args.port}")
    print()
    print("待機中... (Ctrl+C で終了)")
    print("=" * 60)

    server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), disp)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n終了します")


if __name__ == "__main__":
    main()
