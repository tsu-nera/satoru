#!/usr/bin/env python
"""
Muse App OSC → Mind Monitor CSV 変換スクリプト

Muse AppのOSC Output機能からデータを受信し、
Mind Monitor互換のCSV形式で保存する。

使い方:
    source venv/bin/activate
    python scripts/osc_to_csv.py --port 5000 --output data

Muse Appの設定:
    - IP: このPCのIPアドレス
    - Port: 5000
    - Streaming Enabled: ON
"""

import argparse
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from lib.recorders import MuseOSCRecorder


def main():
    parser = argparse.ArgumentParser(
        description='Muse App OSC → Mind Monitor CSV 変換'
    )
    parser.add_argument(
        '--port', type=int, default=5000,
        help='リッスンするポート (default: 5000)'
    )
    parser.add_argument(
        '--ip', default='0.0.0.0',
        help='リッスンするIP (default: 0.0.0.0 = all interfaces)'
    )
    parser.add_argument(
        '--output', type=Path, default=Path('data'),
        help='CSV出力先ディレクトリ (default: data)'
    )
    args = parser.parse_args()

    recorder = MuseOSCRecorder(args.output)

    try:
        recorder.start_server(args.ip, args.port)
    except KeyboardInterrupt:
        print(f"\n記録終了: {recorder.record_count} サンプル")
        path = recorder.save()
        if path:
            print(f"保存完了: {path}")
        else:
            print("データなし: ファイルは保存されませんでした")


if __name__ == '__main__':
    main()
