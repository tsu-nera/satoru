#!/usr/bin/env python3
"""
サマタ評価の進み（種類ごとの回数・主観の有無）を表示する

種類と目安の回数は config/meditation_types.yaml。種類（meditation_type）が付いた行だけを数え、
回数が0の種類は出さない（目安を持つ種類は0でも出す）。

使い方:
    uv run python scripts/eval_progress.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.meditation_types import load_types
from lib.session_log import read_session_log


def main():
    df = read_session_log()
    for col in ('meditation_type', 'focus_rating'):
        if col not in df.columns:
            df[col] = None
    typed = df[df['meditation_type'].notna() & (df['meditation_type'] != '')]
    parts = []
    for key, spec in load_types().items():
        n = int((typed['meditation_type'] == key).sum())
        if spec['target'] is not None:
            parts.append(f'{spec["label"]} {n}/{spec["target"]}')
        elif n:
            parts.append(f'{spec["label"]} {n}')
    parts.append(f'主観あり {int(typed["focus_rating"].notna().sum())}')
    print('評価の進み: ' + '・'.join(parts))


if __name__ == '__main__':
    main()
