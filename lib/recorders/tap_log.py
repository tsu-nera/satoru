"""
Tap Log Recorder - 瞑想中のマインドワンダリング打刻ロガー

スマホのブラウザから届いたタップイベントを、セッション単位のCSVに
追記していく純粋なロジック。HTTPサーバー（scripts/tap_server.py）から
切り離してあり、単体でテスト可能にしている。
"""

import csv
from datetime import datetime
from pathlib import Path

# CSVカラム順序
TAP_LOG_COLUMNS = ['seq', 'event', 'client_ts', 'server_ts', 'elapsed_s']


class TapLogRecorder:
    """
    打刻セッションのCSV記録ロジック

    1セッション = 1CSVファイル。start() でセッションを開始し、
    tap() / stop() でイベントを追記する。イベントごとに即 flush するため、
    セッション途中でプロセスが落ちてもそこまでの記録は残る。
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.csv_path: Path | None = None
        self.session_id: str | None = None
        self.start_server_time: datetime | None = None
        self._seq = 0

    def start(self, client_ts: str) -> str:
        """セッションを開始し、CSVファイルを新規作成する"""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        self.session_id = now.strftime('%Y-%m-%dT%H-%M-%S')
        self.csv_path = self.output_dir / f'{self.session_id}_taps.csv'
        self.start_server_time = now.astimezone()
        self._seq = 0

        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(TAP_LOG_COLUMNS)
            f.flush()

        self._append_row('start', client_ts, self.start_server_time)

        return self.session_id

    def tap(self, client_ts: str) -> None:
        """タップイベントを追記する"""
        if self.csv_path is None or self.start_server_time is None:
            raise RuntimeError('セッションが開始されていません（start() を先に呼ぶこと）')
        self._append_row('tap', client_ts, datetime.now().astimezone())

    def stop(self, client_ts: str) -> None:
        """セッション終了イベントを追記する"""
        if self.csv_path is None or self.start_server_time is None:
            raise RuntimeError('セッションが開始されていません（start() を先に呼ぶこと）')
        self._append_row('stop', client_ts, datetime.now().astimezone())

    def _append_row(self, event: str, client_ts: str, server_time: datetime) -> None:
        assert self.csv_path is not None
        assert self.start_server_time is not None

        elapsed_s = round((server_time - self.start_server_time).total_seconds(), 3)

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([self._seq, event, client_ts, server_time.isoformat(), elapsed_s])
            f.flush()

        self._seq += 1
