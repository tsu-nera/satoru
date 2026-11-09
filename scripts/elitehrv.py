#!/usr/bin/env python3
"""
Elite HRV の reading（1 txt = 1 reading）を分析して表示し、CSV に記録する

4分以上の reading は biofeedback（共鳴呼吸）、未満は readiness として扱う。
結果は logs/resonance_log.csv に timestamp をキーに upsert する。

Elite HRV の export zip（全期間）は Google Drive に elitehrv_export.zip の1本だけ置き、
import のたびに上書きする。既存 zip にある reading が新しい zip で1件でも欠けていれば
上書きしない。

使い方:
    uv run python scripts/elitehrv.py analyze "data/elitehrv/2026-11-10 07-22-58.txt" --practice rf
    uv run python scripts/elitehrv.py import "https://app.elitehrv.com/export/<hash>"
    uv run python scripts/elitehrv.py import export.zip
    uv run python scripts/elitehrv.py analyze --drive        # CSV に無い reading だけ
    uv run python scripts/elitehrv.py analyze --drive --all  # 全 reading を再計算

Drive の認証は OAuth（GDRIVE_OAUTH_CLIENT / GDRIVE_OAUTH_TOKEN、drive.file スコープ）。
フォルダは `upload_to_gdrive.py --create-folder <名前>` で作り、ID を
GDRIVE_FOLDER_ID_ELITEHRV に設定する。
"""

import argparse
import math
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Set, Tuple

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
from scripts.upload_to_gdrive import (
    DEFAULT_OAUTH_CLIENT,
    DEFAULT_TOKEN_PATH,
    authorize_gdrive,
    create_file,
    download_file_bytes,
    find_file_id,
    update_file_content,
)

VERSION = 1  # 指標の定義を変えたら上げる
BIOFEEDBACK_MIN_MIN = 4.0  # この長さ（分）以上を biofeedback とみなす
TS_FORMAT = '%Y-%m-%d %H:%M:%S'
HISTORY_N = 5

DRIVE_ZIP_NAME = 'elitehrv_export.zip'
ZIP_MIMETYPE = 'application/zip'
READING_NAME_RE = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}-\d{2}-\d{2}\.txt$')
# Readme that Elite HRV puts at the top level of every export
EXPORT_README_NAME = '!About This Export.txt'

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
    if kind == 'readiness':
        # practice describes a breathing session; a readiness reading has none
        practice = None

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


class ExportZipError(ValueError):
    """export zip が想定の形式でない"""


def _check_entry_name(name: str) -> Optional[str]:
    """zip エントリ名を検証し、reading ならファイル名を返す。ディレクトリと export 付属の説明ファイルなら None"""
    if '\\' in name or name.startswith('/'):
        raise ExportZipError(f'不正なエントリ名: {name}')
    parts = PurePosixPath(name).parts
    if any(p in ('..', '.') for p in name.split('/')):
        raise ExportZipError(f'不正なエントリ名（パストラバーサル）: {name}')
    if name == EXPORT_README_NAME:
        return None
    if name.endswith('/'):
        if len(parts) != 1:
            raise ExportZipError(f'ディレクトリが2段以上あります: {name}')
        return None
    if len(parts) > 2:
        raise ExportZipError(f'ディレクトリが2段以上あります: {name}')
    if not READING_NAME_RE.match(parts[-1]):
        raise ExportZipError(
            f'reading のファイル名規則（YYYY-MM-DD HH-MM-SS.txt）に合いません: {name}'
        )
    return parts[-1]


def validate_export_zip(zip_path: Path) -> Dict[str, str]:
    """
    export zip を検証し、{reading ファイル名: zip エントリ名} を返す

    エントリ名の規則（ディレクトリ1段まで）・パストラバーサル・各行が数値であることを
    確かめ、違反があれば ExportZipError を投げる。
    """
    if not zipfile.is_zipfile(zip_path):
        raise ExportZipError(f'zip ではありません: {zip_path}')
    readings: Dict[str, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = _check_entry_name(info.filename)
            if base is None:
                continue
            if base in readings:
                raise ExportZipError(f'同じ reading 名が複数あります: {base}')
            try:
                text = zf.read(info).decode('utf-8')
            except UnicodeDecodeError:
                raise ExportZipError(f'UTF-8 として読めません: {info.filename}') from None
            for lineno, line in enumerate(text.splitlines(), start=1):
                v = line.strip()
                if not v:
                    continue
                try:
                    float(v)
                except ValueError:
                    raise ExportZipError(
                        f'数値でない行があります: {info.filename}:{lineno}: {v[:40]!r}'
                    ) from None
            readings[base] = info.filename
    if not readings:
        raise ExportZipError(f'reading が1件もありません: {zip_path}')
    return readings


def _reading_names(zip_path: Path) -> Set[str]:
    """zip 内の reading 名集合（ディレクトリ部分を除く）"""
    with zipfile.ZipFile(zip_path) as zf:
        return {
            PurePosixPath(n).name for n in zf.namelist()
            if not n.endswith('/') and READING_NAME_RE.match(PurePosixPath(n).name)
        }


def _fetch_source(source: str, workdir: Path) -> Path:
    """URL なら workdir に落とし、ローカルパスならそのまま返す"""
    if source.startswith(('http://', 'https://')):
        dest = workdir / 'download.zip'
        with urllib.request.urlopen(source, timeout=120) as resp, open(dest, 'wb') as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
        return dest
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f'ファイルが見つかりません: {source}')
    return path


def import_export(source: str, service: Any, folder_id: str) -> int:
    """export zip を検証し、Drive の elitehrv_export.zip を作成または上書きする。終了コードを返す"""
    with tempfile.TemporaryDirectory(prefix='elitehrv-import-') as tmp:
        workdir = Path(tmp)
        try:
            zip_path = _fetch_source(source, workdir)
            new_names = set(validate_export_zip(zip_path))
        except (ExportZipError, OSError) as e:
            print(f'エラー: {e}', file=sys.stderr)
            return 1

        file_id = find_file_id(service, folder_id, DRIVE_ZIP_NAME)
        if file_id is None:
            create_file(service, folder_id, zip_path, DRIVE_ZIP_NAME, ZIP_MIMETYPE)
            print(f'Drive に {DRIVE_ZIP_NAME} を新規作成しました（{len(new_names)} readings）')
            added = sorted(new_names)
        else:
            old_zip = workdir / 'drive.zip'
            old_zip.write_bytes(download_file_bytes(service, file_id))
            old_names = _reading_names(old_zip)
            missing = sorted(old_names - new_names)
            if missing:
                print(
                    f'エラー: 新しい zip に既存の reading が {len(missing)} 件ありません。上書きしません',
                    file=sys.stderr,
                )
                for name in missing:
                    print(f'  missing: {name}', file=sys.stderr)
                return 1
            update_file_content(service, file_id, zip_path, ZIP_MIMETYPE)
            print(f'Drive の {DRIVE_ZIP_NAME} を上書きしました（{len(new_names)} readings）')
            added = sorted(new_names - old_names)

    print(f'新しい reading: {len(added)} 件')
    for name in added:
        print(f'  new: {name}')
    return 0


def analyze_drive(
    service: Any,
    folder_id: str,
    log_path: Optional[Path] = None,
    reprocess_all: bool = False,
    verbose: bool = True,
    practice: Optional[str] = None,
) -> int:
    """
    Drive の zip を落とし、CSV に無い reading（reprocess_all なら全部）を analyze する

    practice は処理する biofeedback の reading に付ける（readiness には付けない）。
    """
    log = Path(log_path) if log_path is not None else default_log_path()
    file_id = find_file_id(service, folder_id, DRIVE_ZIP_NAME)
    if file_id is None:
        print(f'エラー: Drive に {DRIVE_ZIP_NAME} がありません。先に import してください', file=sys.stderr)
        return 1

    failed = 0
    with tempfile.TemporaryDirectory(prefix='elitehrv-drive-') as tmp:
        workdir = Path(tmp)
        zip_path = workdir / DRIVE_ZIP_NAME
        zip_path.write_bytes(download_file_bytes(service, file_id))
        try:
            readings = validate_export_zip(zip_path)
        except ExportZipError as e:
            print(f'エラー: Drive の zip が不正です: {e}', file=sys.stderr)
            return 1

        done: Set[str] = set() if reprocess_all else set(
            _read_log(log)['timestamp'].dropna().astype(str)
        )
        targets = []
        for base in sorted(readings):
            start = parse_elite_hrv_timestamp(base)
            if start is not None and start.strftime(TS_FORMAT) not in done:
                targets.append(base)

        print(f'{len(targets)} / {len(readings)} readings を処理します')
        extract_dir = workdir / 'readings'
        extract_dir.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            for base in targets:
                txt = extract_dir / base
                txt.write_bytes(zf.read(readings[base]))
                if verbose:
                    print()
                    print('=' * 60)
                try:
                    analyze_file(str(txt), practice=practice, log_path=log, verbose=verbose)
                except (ValueError, OSError) as e:
                    failed += 1
                    print(f'エラー: {base}: {e}', file=sys.stderr)
    return 1 if failed else 0


def _drive_service() -> Tuple[Any, str]:
    """環境変数から Drive service とフォルダ ID を用意する"""
    folder_id = os.environ.get('GDRIVE_FOLDER_ID_ELITEHRV')
    if not folder_id:
        raise RuntimeError('GDRIVE_FOLDER_ID_ELITEHRV が設定されていません')
    client = Path(os.environ.get('GDRIVE_OAUTH_CLIENT', str(DEFAULT_OAUTH_CLIENT)))
    token = Path(os.environ.get('GDRIVE_OAUTH_TOKEN', str(DEFAULT_TOKEN_PATH)))
    if not client.exists():
        raise RuntimeError(f'OAuth クライアントシークレットが見つかりません: {client}')
    return authorize_gdrive(client, token), folder_id


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Elite HRV reading の分析と記録')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('analyze', help='reading を分析して CSV に記録する')
    p.add_argument('txt', nargs='?', help='Elite HRV の txt（YYYY-MM-DD HH-MM-SS.txt）')
    p.add_argument('--practice', choices=['rf', 'rf+metta'], default=None,
                   help='biofeedback の reading に付ける練習の種類（readiness には付けない）')
    p.add_argument('--log', default=None, help='記録先 CSV（既定: logs/resonance_log.csv）')
    p.add_argument('--drive', action='store_true',
                   help=f'Drive の {DRIVE_ZIP_NAME} から CSV に無い reading をすべて分析する')
    p.add_argument('--all', action='store_true', help='--drive と併用し、全 reading を再計算する')
    imp = sub.add_parser('import', help=f'export zip を検証して Drive の {DRIVE_ZIP_NAME} に保存する')
    imp.add_argument('source', help='export のリンク（https://app.elitehrv.com/export/...）または zip のパス')
    args = parser.parse_args(argv)

    if args.command == 'import':
        try:
            service, folder_id = _drive_service()
            return import_export(args.source, service, folder_id)
        except Exception as e:
            print(f'エラー: {e}', file=sys.stderr)
            return 1

    log_path = Path(args.log) if args.log else None
    if args.drive:
        if args.txt:
            parser.error('--drive は txt と同時に指定できません')
        if args.all and args.practice:
            parser.error('--all は --practice と同時に指定できません（全 reading の practice を上書きするため）')
        try:
            service, folder_id = _drive_service()
            return analyze_drive(
                service, folder_id, log_path=log_path, reprocess_all=args.all, practice=args.practice,
            )
        except Exception as e:
            print(f'エラー: {e}', file=sys.stderr)
            return 1
    if args.all:
        parser.error('--all は --drive と併用してください')
    if not args.txt:
        parser.error('txt か --drive を指定してください')

    try:
        analyze_file(args.txt, practice=args.practice, log_path=log_path, verbose=True)
    except (ValueError, OSError) as e:
        print(f'エラー: {e}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
