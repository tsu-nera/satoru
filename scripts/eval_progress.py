#!/usr/bin/env python3
"""
サマタ評価の進み（種類ごとの回数・主観の有無）と、次の回が対照の番かを表示する

対照は4回に1回（2026-11-14 決定）。曜日で固定すると曜日ごとの生活の差が混ざるので回数で回す。
種類（meditation_type）が付いた行だけを数える。目安はサマタ20回・対照6回で判定に入る。

使い方:
    uv run python scripts/eval_progress.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.session_log import read_session_log

CONTROL_EVERY = 4
TARGETS = {'samatha': 20, 'control': 6}


def progress(df):
    typed = df[df['meditation_type'].notna() & (df['meditation_type'] != '')]
    counts = {t: int((typed['meditation_type'] == t).sum()) for t in TARGETS}
    rated = int(typed['focus_rating'].notna().sum())
    types = list(typed['meditation_type'])
    since_control = len(types) - 1 - max((i for i, t in enumerate(types) if t == 'control'), default=-1)
    next_is_control = since_control >= CONTROL_EVERY - 1
    return counts, rated, since_control, next_is_control


def main():
    df = read_session_log()
    for col in ('meditation_type', 'focus_rating'):
        if col not in df.columns:
            df[col] = None
    counts, rated, since_control, next_is_control = progress(df)
    parts = [f'サマタ {counts["samatha"]}/{TARGETS["samatha"]}', f'対照 {counts["control"]}/{TARGETS["control"]}',
             f'主観あり {rated}']
    print('評価の進み: ' + '・'.join(parts))
    print(f'前の対照から {since_control} 回 → 次の回は' + ('対照の番' if next_is_control else 'サマタ'))


if __name__ == '__main__':
    main()
