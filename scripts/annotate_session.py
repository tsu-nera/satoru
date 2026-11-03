#!/usr/bin/env python3
"""
session_log の既存行に手入力の注釈（瞑想の種類・主観評価）を書き込む

指定した列だけを更新し、他の列と他の行は変えない。該当行が無ければエラー。

使い方:
    uv run python scripts/annotate_session.py --timestamp latest --type samatha --focus 4 --drowsiness 2
    uv run python scripts/annotate_session.py --timestamp "2026-11-04 07:31:02" --type control
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.session_log import _save, default_log_path

RATING_MIN = 1
RATING_MAX = 5


def annotate_session(
    timestamp: str,
    meditation_type: Optional[str] = None,
    focus: Optional[int] = None,
    drowsiness: Optional[int] = None,
    csv_path: Optional[Path] = None,
) -> str:
    """
    session_log の1行に注釈を書き込み、更新した行の timestamp を返す。

    Parameters
    ----------
    timestamp : str
        'latest'（最新行）または '%Y-%m-%d %H:%M:%S'
    meditation_type : str, optional
        瞑想の種類（samatha / control など。値は検証しない）
    focus, drowsiness : int, optional
        集中の安定・眠気（1〜5）
    csv_path : Path, optional
        セッションログCSV。省略時は `logs/session_log.csv`。

    Raises
    ------
    ValueError
        rating が範囲外、該当行が無い、更新する列が1つも指定されていない場合
    """
    updates: Dict[str, object] = {}
    if meditation_type is not None:
        updates['meditation_type'] = meditation_type
    for col, value in (('focus_rating', focus), ('drowsiness_rating', drowsiness)):
        if value is None:
            continue
        if not RATING_MIN <= value <= RATING_MAX:
            raise ValueError(f'{col} は {RATING_MIN}〜{RATING_MAX} で指定してください: {value}')
        updates[col] = value
    if not updates:
        raise ValueError('--type / --focus / --drowsiness のどれかを指定してください')

    csv_path = Path(csv_path) if csv_path is not None else default_log_path()
    if not csv_path.exists():
        raise ValueError(f'session_log がありません: {csv_path}')
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError('session_log に行がありません')

    if timestamp == 'latest':
        timestamp = str(df['timestamp'].max())
    mask = df['timestamp'] == timestamp
    if not mask.any():
        raise ValueError(f'session_log に timestamp={timestamp} の行がありません')

    for col, new_value in updates.items():
        if col not in df.columns:
            df[col] = pd.Series([pd.NA] * len(df), dtype=object)
        elif isinstance(new_value, str) and df[col].dtype != object:
            df[col] = df[col].astype(object)
        df.loc[mask, col] = new_value

    _save(df, csv_path)
    return timestamp


def main() -> int:
    parser = argparse.ArgumentParser(description='session_log に瞑想の種類と主観評価を書き込む')
    parser.add_argument(
        '--timestamp', required=True, help='latest または "YYYY-MM-DD HH:MM:SS"'
    )
    parser.add_argument('--type', dest='meditation_type', help='瞑想の種類（例: samatha, control）')
    parser.add_argument('--focus', type=int, help='集中の安定（1〜5）')
    parser.add_argument('--drowsiness', type=int, help='眠気（1〜5）')
    parser.add_argument('--csv', type=Path, help='セッションログCSV（省略時 logs/session_log.csv）')
    args = parser.parse_args()

    try:
        ts = annotate_session(
            args.timestamp,
            meditation_type=args.meditation_type,
            focus=args.focus,
            drowsiness=args.drowsiness,
            csv_path=args.csv,
        )
    except ValueError as e:
        print(f'エラー: {e}', file=sys.stderr)
        return 1
    print(f'注釈を書き込みました: {ts}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
