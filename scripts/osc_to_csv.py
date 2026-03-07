#!/usr/bin/env python
"""
Muse OSC → Mind Monitor CSV 変換スクリプト

Muse App / Mind Monitor 両対応。
--source で切り替え。

使い方:
    source venv/bin/activate
    python scripts/osc_to_csv.py --source muse_app_osc --port 5000
    python scripts/osc_to_csv.py --source mind_monitor --port 5000
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
        description='Muse OSC → Mind Monitor CSV 変換'
    )
    parser.add_argument(
        '--source', choices=['muse_app_osc', 'mind_monitor_osc'], default='muse_app_osc',
        help='データソース (default: muse_app_osc)'
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
        '--output', type=Path, default=None,
        help='CSV出力先ディレクトリ (default: data/{source})'
    )
    args = parser.parse_args()

    output_dir = args.output if args.output else Path(f'data/{args.source}')
    recorder = MuseOSCRecorder(output_dir, source=args.source)

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
