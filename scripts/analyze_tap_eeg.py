#!/usr/bin/env python3
"""
タップ連動のEEG横断解析

複数セッションのMuse CSVとタップ打刻ログを突き合わせ、タップ直前のバンドパワーが
対照区間と違うかを調べる。検定の単位はセッションであり、窓ではない
（窓はオーバーラップしていて独立でない）。

使い方:
    uv run python scripts/analyze_tap_eeg.py data/muse/*.csv.gz
    uv run python scripts/analyze_tap_eeg.py --taps-dir data/taps data/muse/*.csv.gz

Muse CSVはローカルに無ければ先に取得すること:
    bash scripts/download_data.sh muse 2026-09-15
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.loaders.mind_monitor import load_mind_monitor_csv  # noqa: E402
from lib.loaders.tap_log import load_tap_log_csv  # noqa: E402
from lib.sensors.eeg.constants import FREQ_BANDS  # noqa: E402
from lib.sensors.eeg.preprocessing import filter_eeg_quality, prepare_mne_raw  # noqa: E402
from lib.sensors.eeg.sliding_power import compute_sliding_band_power  # noqa: E402
from lib.tap_locked import (  # noqa: E402
    TAP_ANALYSIS_WARMUP_S,
    contrast_mw_vs_control,
    detrend_and_zscore,
    label_tap_context,
    peri_event_average,
    required_sessions,
)

BANDS = [name.lower() for name in FREQ_BANDS]
PERI_LAGS_S = np.arange(-40, 21, 2)
TARGET_EFFECTS_SD = (0.5, 0.3)


def find_tap_log(taps_dir: Path, session_date: str) -> Optional[Path]:
    """セッション日付に対応するタップログを探す。無ければ None。"""
    matches = sorted(taps_dir.glob(f'taps_{session_date}--*.csv'))
    return matches[0] if matches else None


def analyze_session(muse_path: Path, taps_dir: Path) -> dict:
    """1セッションを解析し、効果量とペリイベント平均を返す。"""
    date_match = re.search(r'(\d{4}-\d{2}-\d{2})', muse_path.name)
    if date_match is None:
        raise ValueError(f'ファイル名から日付を抽出できない: {muse_path.name}')
    session_date = date_match.group(1)

    tap_path = find_tap_log(taps_dir, session_date)
    if tap_path is None:
        return {'date': session_date, 'skipped': 'タップログなし'}

    tap_df = load_tap_log_csv(str(tap_path))
    tap_times = tap_df.loc[tap_df['event'] == 'tap', 'TimeStamp']
    if tap_times.empty:
        return {'date': session_date, 'skipped': 'タップ0件'}

    df = load_mind_monitor_csv(str(muse_path), filter_headband=True, warmup_seconds=0.0)
    # prepare_mne_raw は内部で品質フィルタを掛けて行を落とす。MNE は連続サンプルと
    # して扱うため、時刻対応にはフィルタ後の TimeStamp を使う。
    df_filtered, _ = filter_eeg_quality(df)
    mne_dict = prepare_mne_raw(df_filtered)
    if mne_dict is None:
        return {'date': session_date, 'skipped': 'RAWチャネルなし'}

    power = compute_sliding_band_power(
        mne_dict['raw'].get_data() * 1e6,
        mne_dict['sfreq'],
        timestamps=pd.to_datetime(df_filtered['TimeStamp'].values),
    )
    power = power[(power['t_sec'] >= TAP_ANALYSIS_WARMUP_S) & (power['n_clean_ch'] > 0)]

    labeled = label_tap_context(detrend_and_zscore(power, BANDS), tap_times)
    counts = labeled['tap_context'].value_counts()

    return {
        'date': session_date,
        'taps': len(tap_times),
        'windows': len(labeled),
        'mw': int(counts.get('mw', 0)),
        'control': int(counts.get('control', 0)),
        'effects': contrast_mw_vs_control(labeled, BANDS),
        'peri': {band: peri_event_average(labeled, tap_times, band, PERI_LAGS_S.tolist()) for band in BANDS},
    }


def print_report(results: list) -> None:
    """セッション別の効果量、ペリイベント平均、必要セッション数を表示する。"""
    dates = [r['date'] for r in results]

    print('\n=== 窓の内訳 ===')
    print(pd.DataFrame([
        {'date': r['date'], 'taps': r['taps'], '全窓': r['windows'], 'MW窓': r['mw'], '対照窓': r['control']}
        for r in results
    ]).to_string(index=False))

    print('\n=== MW − 対照（時間トレンド除去後のz、単位=SD）===')
    rows = []
    for band in BANDS:
        values = [r['effects'][band] for r in results]
        finite = [v for v in values if np.isfinite(v)]
        row: dict = {'band': band}
        row.update({d[5:]: v for d, v in zip(dates, values)})
        row['平均'] = float(np.mean(finite)) if finite else np.nan
        if finite:
            same = sum(np.sign(v) == np.sign(np.mean(finite)) for v in finite)
            row['同符号'] = f'{same}/{len(finite)}'
        rows.append(row)
    print(pd.DataFrame(rows).set_index('band').round(3).to_string())

    print(f'\n=== ペリイベント平均（{PERI_LAGS_S[0]:+.0f}〜{PERI_LAGS_S[-1]:+.0f}秒、セッション平均）===')
    for band in BANDS:
        matrix = np.array([r['peri'][band] for r in results])
        mean = np.nanmean(matrix, axis=0)
        if np.all(np.isnan(mean)):
            continue
        peak = int(np.nanargmax(np.abs(mean)))
        print(f'  {band:6} |最大| {mean[peak]:+.2f} SD @ lag {PERI_LAGS_S[peak]:+.0f}s')

    print(f'\n=== 必要セッション数（両側5%・検出力80%、現在 n={len(results)}）===')
    for band in BANDS:
        effects = [r['effects'][band] for r in results]
        needs = [f'効果{t}SD → n={required_sessions(effects, t)}' for t in TARGET_EFFECTS_SD]
        print(f'  {band:6} {" / ".join(str(n) for n in needs)}')


def main() -> int:
    parser = argparse.ArgumentParser(description='タップ連動のEEG横断解析')
    parser.add_argument('muse_files', nargs='+', help='Muse CSV（.csv / .csv.gz）')
    parser.add_argument('--taps-dir', default='data/taps', help='タップログのディレクトリ')
    args = parser.parse_args()

    taps_dir = Path(args.taps_dir)
    results = []
    for path_str in args.muse_files:
        path = Path(path_str)
        if not path.exists():
            print(f'⚠️  見つかりません: {path}')
            continue
        result = analyze_session(path, taps_dir)
        if 'skipped' in result:
            print(f'スキップ {result["date"]}: {result["skipped"]}')
            continue
        print(f'解析 {result["date"]}: タップ{result["taps"]}件 / {result["windows"]}窓')
        results.append(result)

    if len(results) < 2:
        print('\n❌ 解析できたセッションが2件未満です。横断比較できません。')
        return 1

    print_report(results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
