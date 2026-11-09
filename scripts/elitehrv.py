#!/usr/bin/env python3
"""
Elite HRV の reading（1 txt = 1 reading）を分析して表示し、CSV に記録する

4分以上の reading は biofeedback（共鳴呼吸）、未満は readiness として扱う。
結果は logs/resonance_log.csv に timestamp をキーに upsert する。

使い方:
    uv run python scripts/elitehrv.py analyze "data/elitehrv/2026-11-10 07-22-58.txt" --practice rf
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from lib.loaders.elite_hrv import (
    get_hrv_data,
    load_elite_hrv_txt,
    parse_elite_hrv_timestamp,
)
from lib.sensors.ecg.hrv import calculate_hrv_basic_stats
from lib.sensors.ecg.respiration import analyze_resonance_breathing

VERSION = 1  # 指標の定義を変えたら上げる
BIOFEEDBACK_MIN_MIN = 4.0  # この長さ（分）以上を biofeedback とみなす
TS_FORMAT = '%Y-%m-%d %H:%M:%S'
HISTORY_N = 5

COLUMNS = [
    'timestamp', 'kind', 'practice', 'duration_min', 'artifact_count', 'artifact_pct',
    'mean_hr_bpm', 'min_hr_bpm', 'max_hr_bpm', 'mean_rr_ms',
    'rmssd_ms', 'ln_rmssd', 'sdnn_ms', 'pnn50_pct',
    'lf_power_ms2', 'hf_power_ms2', 'total_power_ms2', 'lf_hf_ratio',
    'lf_peak_hz', 'hf_peak_hz',
    'breathing_rate_bpm', 'breath_period_sd_s', 'rsa_amplitude_ms', 'version',
]

# (CSV 列, 表示名, 単位)
METRIC_ROWS = [
    ('mean_hr_bpm', 'Mean HR', 'bpm'),
    ('min_hr_bpm', 'Min HR', 'bpm'),
    ('max_hr_bpm', 'Max HR', 'bpm'),
    ('mean_rr_ms', 'Mean RR', 'ms'),
    ('rmssd_ms', 'RMSSD', 'ms'),
    ('ln_rmssd', 'ln(RMSSD)', ''),
    ('sdnn_ms', 'SDNN', 'ms'),
    ('pnn50_pct', 'pNN50', '%'),
    ('lf_power_ms2', 'LF Power', 'ms^2'),
    ('hf_power_ms2', 'HF Power', 'ms^2'),
    ('total_power_ms2', 'Total Power', 'ms^2'),
    ('lf_hf_ratio', 'LF/HF', ''),
    ('lf_peak_hz', 'LF Peak', 'Hz'),
    ('hf_peak_hz', 'HF Peak', 'Hz'),
    ('breathing_rate_bpm', 'Breathing Rate', 'breaths/min'),
    ('breath_period_sd_s', 'Breath Period SD', 's'),
    ('rsa_amplitude_ms', 'RSA Amplitude', 'ms'),
]


def default_log_path() -> Path:
    return PROJECT_ROOT / 'logs' / 'resonance_log.csv'


def _read_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(
        log_path, dtype={'timestamp': str, 'kind': str, 'practice': str},
    )


def _upsert(log_path: Path, row: Dict[str, Any]) -> None:
    """timestamp をキーに1行 upsert する。practice が None なら既存値を残す"""
    df = _read_log(log_path)
    mask = df['timestamp'] == row['timestamp']
    if row.get('practice') is None:
        existing = df.loc[mask, 'practice']
        row['practice'] = existing.iloc[0] if len(existing) and pd.notna(existing.iloc[0]) else None
    df = df[~mask]
    new = pd.DataFrame([row], columns=COLUMNS)
    df = new if df.empty else pd.concat([df, new], ignore_index=True)
    df = df.sort_values('timestamp').reset_index(drop=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path, index=False, columns=COLUMNS)


def _fmt(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return '-'
    return f'{v:.4g}' if isinstance(v, float) else str(v)


def _comparison_line(log_path: Path, row: Dict[str, Any]) -> str:
    """同じ kind の直近 HISTORY_N 行の中央値と今回値の比較行"""
    df = _read_log(log_path)
    hist = df[(df['kind'] == row['kind']) & (df['timestamp'] != row['timestamp'])]
    hist = hist.sort_values('timestamp').tail(HISTORY_N)
    if hist.empty:
        return f"Comparison: no history for kind={row['kind']} (first reading)"
    parts: List[str] = []
    for col, label in (('rsa_amplitude_ms', 'RSA amplitude'), ('rmssd_ms', 'RMSSD')):
        cur = row.get(col)
        med = pd.to_numeric(hist[col], errors='coerce').median()
        if cur is None or (isinstance(cur, float) and math.isnan(cur)) or pd.isna(med):
            continue
        parts.append(f'{label} {cur:.1f} ms vs median {med:.1f} ms')
    if not parts:
        return f"Comparison: no comparable values in last {len(hist)} {row['kind']} rows"
    return f"Comparison (last {len(hist)} {row['kind']}): " + ', '.join(parts)


def analyze_file(
    txt_path: str,
    practice: Optional[str] = None,
    log_path: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """txt 1本を分析し CSV に upsert する。CSV 行と同じ内容の dict を返す"""
    log = Path(log_path) if log_path is not None else default_log_path()
    start = parse_elite_hrv_timestamp(Path(txt_path).name)
    if start is None:
        raise ValueError(f'ファイル名から開始時刻を取得できません: {txt_path}')

    df = load_elite_hrv_txt(txt_path)
    hrv = get_hrv_data(df, clean_artifacts=True)
    raw_rr = hrv['rr_intervals']
    rr = hrv['rr_intervals_clean']
    duration_min = float(np.sum(raw_rr) / 60000.0)
    kind = 'biofeedback' if duration_min >= BIOFEEDBACK_MIN_MIN else 'readiness'

    stats = calculate_hrv_basic_stats(rr, min_freq_duration_sec=BIOFEEDBACK_MIN_MIN * 60.0)
    row: Dict[str, Any] = {c: np.nan for c in COLUMNS}
    row.update({
        'timestamp': start.strftime(TS_FORMAT),
        'kind': kind,
        'practice': practice,
        'duration_min': duration_min,
        'artifact_count': hrv['artifact_count'],
        'artifact_pct': hrv['artifact_pct'],
        'version': VERSION,
    })
    row.update(stats.to_dict())

    segments = None
    if kind == 'biofeedback' and np.isfinite(stats.lf_peak_hz):
        res = analyze_resonance_breathing(rr, stats.lf_peak_hz)
        row['breathing_rate_bpm'] = res.breathing_rate_bpm
        row['breath_period_sd_s'] = res.breath_period_sd_s
        row['rsa_amplitude_ms'] = res.rsa_amplitude_ms
        segments = res.segments

    _upsert(log, row)
    saved = _read_log(log)
    row['practice'] = saved.loc[saved['timestamp'] == row['timestamp'], 'practice'].iloc[0]

    if verbose:
        _print_report(row, segments, _comparison_line(log, row))
    return row


def _print_report(row: Dict[str, Any], segments: Optional[pd.DataFrame], comparison: str) -> None:
    print(
        f"{row['timestamp']}  {row['duration_min']:.1f} min  {row['kind']}  "
        f"artifacts: {row['artifact_count']} ({row['artifact_pct']:.1f}%)"
    )
    print()
    table = pd.DataFrame(
        [
            {'Metric': label, 'Value': row[col], 'Unit': unit}
            for col, label, unit in METRIC_ROWS
            if not (isinstance(row[col], float) and math.isnan(row[col]))
        ]
    )
    print(table.to_markdown(index=False, floatfmt='.3f'))
    if segments is not None and not segments.empty:
        print()
        print(segments.to_markdown(index=False, floatfmt='.1f'))
    print()
    print(comparison)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Elite HRV reading の分析と記録')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('analyze', help='txt 1本を分析して CSV に記録する')
    p.add_argument('txt', help='Elite HRV の txt（YYYY-MM-DD HH-MM-SS.txt）')
    p.add_argument('--practice', choices=['rf', 'rf+metta'], default=None)
    p.add_argument('--log', default=None, help='記録先 CSV（既定: logs/resonance_log.csv）')
    args = parser.parse_args(argv)

    try:
        analyze_file(
            args.txt, practice=args.practice,
            log_path=Path(args.log) if args.log else None, verbose=True,
        )
    except (ValueError, OSError) as e:
        print(f'エラー: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
