#!/usr/bin/env python3
"""
session_log を Drive 上の生データから現行パイプラインで再計算する

既存ログ行に対応する Muse ファイル（ファイル名の日時が行の timestamp の 0〜3 分前）
だけを1本ずつ一時ディレクトリに落として計算し、既存行を更新する。新しい行は作らない。
SelfLoops・タップログ・手入力の列は既存値を残す（lib/session_log.BACKFILL_PRESERVED_COLUMNS）。

使い方:
    uv run python scripts/backfill_session_log.py --dry-run
    uv run python scripts/backfill_session_log.py --since 2026-10-27
    uv run python scripts/backfill_session_log.py --force
"""

import argparse
import os
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.env import load_env
from lib.session_log import (
    default_log_path,
    get_pipeline_version,
    read_session_log,
    write_backfill_row,
)
from scripts.fetch_from_gdrive import (
    authenticate_gdrive,
    download_file,
    extract_zip,
    list_csv_files,
    parse_filename_datetime,
)

TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'
#: ログの timestamp はウォームアップ1分を除いた開始時刻なので、ファイル名の日時はその前にある
MATCH_WINDOW = timedelta(minutes=3)


def match_files_to_rows(
    row_timestamps: List[str],
    files: List[Dict],
    window: timedelta = MATCH_WINDOW,
) -> Tuple[Dict[str, Dict], List[str]]:
    """ログの timestamp と Drive ファイルを対応付ける。

    ファイル名の日時 dt が `t - window <= dt <= t` を満たすものが候補で、
    複数あれば t に最も近い（dt が最大の）ものを選ぶ。

    Returns
    -------
    (timestamp -> file の対応, 対応ファイルが無い timestamp のリスト)
    """
    dated: List[Tuple[datetime, Dict]] = []
    for f in files:
        dt = parse_filename_datetime(f.get('name', ''))
        if dt is not None:
            dated.append((dt, f))

    matched: Dict[str, Dict] = {}
    missing: List[str] = []
    for ts in row_timestamps:
        t = datetime.strptime(ts, TIMESTAMP_FORMAT)
        candidates = [(dt, f) for dt, f in dated if t - window <= dt <= t]
        if candidates:
            matched[ts] = max(candidates, key=lambda c: c[0])[1]
        else:
            missing.append(ts)
    return matched, missing


def select_targets(
    log_df: pd.DataFrame,
    current_version: str,
    since: Optional[date],
    until: Optional[date],
    force: bool,
) -> Tuple[List[str], List[str]]:
    """範囲内の行から (処理対象, 最新版で計算済みのためスキップ) の timestamp を返す。"""
    if 'timestamp' not in log_df.columns or log_df.empty:
        return [], []
    if 'pipeline_version' in log_df.columns:
        versions = log_df['pipeline_version'].fillna('').astype(str)
    else:
        versions = pd.Series([''] * len(log_df), index=log_df.index)

    to_process: List[str] = []
    skipped: List[str] = []
    for ts, version in zip(log_df['timestamp'], versions):
        d = datetime.strptime(ts, TIMESTAMP_FORMAT).date()
        if since is not None and d < since:
            continue
        if until is not None and d > until:
            continue
        # 版が不明（空）のときは最新かどうか判断できないので常に処理する
        if not force and current_version != '' and version == current_version:
            skipped.append(ts)
        else:
            to_process.append(ts)
    return to_process, skipped


def _list_drive_files(credentials: Optional[str], folder_id: str) -> Tuple[Any, List[Dict]]:
    """Drive に認証してフォルダ内のファイル一覧を返す。"""
    service = authenticate_gdrive(credentials)
    return service, list_csv_files(service, folder_id)


def _process_one(
    service: Any, file: Dict, timestamp: str, log_path: Path, warmup: float
) -> None:
    """1本をダウンロード→計算→既存行に書き戻す。一時ファイルは必ず消える。"""
    # 重いライブラリを読むので遅延 import
    from scripts.generate_report import run_full_analysis

    with tempfile.TemporaryDirectory() as tmp:
        downloaded = download_file(service, file['id'], file['name'], tmp)
        if downloaded.endswith('.zip'):
            csv_path = extract_zip(downloaded, tmp)
            os.remove(downloaded)
            if csv_path is None:
                raise RuntimeError(f"ZIP内にCSVがありません: {file['name']}")
        else:
            csv_path = downloaded

        out_dir = Path(tmp) / 'out'
        out_dir.mkdir()
        results = run_full_analysis(
            Path(csv_path), out_dir, save_to='none', warmup_minutes=warmup
        )
        write_backfill_row(results, timestamp, csv_path=log_path)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, '%Y-%m-%d').date()


def main(argv: Optional[List[str]] = None) -> int:
    load_env()
    parser = argparse.ArgumentParser(
        description='session_log を Drive の生データから現行パイプラインで再計算する'
    )
    parser.add_argument('--since', type=_parse_date, help='この日以降（YYYY-MM-DD, 含む）')
    parser.add_argument('--until', type=_parse_date, help='この日以前（YYYY-MM-DD, 含む）')
    parser.add_argument('--force', action='store_true', help='pipeline_version が一致する行も再計算')
    parser.add_argument('--dry-run', action='store_true', help='対応付け結果だけ表示して計算しない')
    parser.add_argument('--folder-id', default=os.getenv('GDRIVE_FOLDER_ID_MUSE'))
    parser.add_argument('--credentials', default=os.getenv('GDRIVE_CREDENTIALS'))
    parser.add_argument('--log-path', type=Path, default=default_log_path())
    parser.add_argument('--warmup', type=float, default=1.0, help='ウォームアップ除外（分）')
    args = parser.parse_args(argv)

    if not args.folder_id:
        print('エラー: --folder-id または GDRIVE_FOLDER_ID_MUSE が必要です', file=sys.stderr)
        return 1

    log_df = read_session_log(args.log_path)
    version = get_pipeline_version()
    targets, skipped = select_targets(log_df, version, args.since, args.until, args.force)

    service, files = _list_drive_files(args.credentials, args.folder_id)
    matched, missing = match_files_to_rows(targets, files)

    print(f'\npipeline_version: {version or "(不明)"}')
    print(f'対象: {len(matched)} 本 / 最新版のためスキップ: {len(skipped)} 行 / '
          f'Drive に対応ファイル無し: {len(missing)} 行')
    for ts in missing:
        print(f'  Drive 欠落: {ts}')

    if args.dry_run:
        for ts, f in matched.items():
            print(f'  {ts} <- {f["name"]}')
        return 0

    processed = 0
    failures: List[Tuple[str, str]] = []
    for i, (ts, f) in enumerate(matched.items(), 1):
        print(f'\n[{i}/{len(matched)}] {ts} <- {f["name"]}')
        try:
            _process_one(service, f, ts, args.log_path, args.warmup)
            processed += 1
        except Exception as e:  # 1本の失敗で全体を止めない
            failures.append((ts, f'{type(e).__name__}: {e}'))
            print(f'  失敗: {e}', file=sys.stderr)

    print(f'\n処理完了: {processed} / {len(matched)} 本')
    if missing:
        print(f'Drive 欠落 ({len(missing)} 行): ' + ', '.join(missing))
    if failures:
        print(f'失敗 ({len(failures)} 本):')
        for ts, msg in failures:
            print(f'  {ts}: {msg}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
