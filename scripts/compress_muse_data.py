"""
data/muse/ 配下の Muse CSV をその場で gzip 圧縮するスクリプト

圧縮 → 検証（行数・列数・TimeStamp 先頭末尾） → 元ファイル削除（--delete-original 指定時のみ）
の順で処理する。検証に失敗した場合は生成した .csv.gz を削除し、元ファイルを残して非ゼロ終了する。

使い方:
    # data/muse/ 配下のCSVを圧縮（元ファイルは残す）
    uv run python scripts/compress_muse_data.py

    # 検証成功後、元ファイルを削除する
    uv run python scripts/compress_muse_data.py --delete-original

    # 対象ディレクトリを変更する（テスト等）
    uv run python scripts/compress_muse_data.py --data-dir /path/to/dir
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'muse'
COMPRESS_LEVEL = 6


class VerificationError(Exception):
    """圧縮後の検証に失敗したことを表す例外"""


def verify_compressed(csv_path: Path, gz_path: Path) -> None:
    """
    圧縮結果を検証する

    元CSVと圧縮後CSVを読み直し、行数・列数・TimeStamp列の先頭末尾が一致するか確認する。
    TimeStamp列が存在しない場合は行数・列数のみ照合し、警告を表示する。

    Args:
        csv_path: 元のCSVファイルパス
        gz_path: 圧縮後の .csv.gz ファイルパス

    Raises:
        VerificationError: 行数・列数・TimeStampの不一致を検出した場合
    """
    original = pd.read_csv(csv_path)
    compressed = pd.read_csv(gz_path)

    if original.shape != compressed.shape:
        raise VerificationError(
            f'行数・列数が一致しません: 元={original.shape} 圧縮後={compressed.shape} ({csv_path.name})'
        )

    if 'TimeStamp' not in original.columns:
        print(f'  警告: TimeStamp列が存在しないため、TimeStampの照合をスキップします ({csv_path.name})')
        return

    original_ts = original['TimeStamp']
    compressed_ts = compressed['TimeStamp']

    if len(original_ts) == 0:
        return

    if original_ts.iloc[0] != compressed_ts.iloc[0] or original_ts.iloc[-1] != compressed_ts.iloc[-1]:
        raise VerificationError(
            f'TimeStampの先頭・末尾が一致しません: '
            f'元=({original_ts.iloc[0]}, {original_ts.iloc[-1]}) '
            f'圧縮後=({compressed_ts.iloc[0]}, {compressed_ts.iloc[-1]}) ({csv_path.name})'
        )


def compress_file(csv_path: Path, delete_original: bool = False) -> Path:
    """
    1つのCSVファイルを .csv.gz に圧縮し、検証してから必要に応じて元ファイルを削除する

    Args:
        csv_path: 圧縮対象のCSVファイルパス
        delete_original: 検証成功後に元ファイルを削除するかどうか

    Returns:
        生成した .csv.gz のパス

    Raises:
        VerificationError: 検証に失敗した場合（生成した .csv.gz は削除済み）
    """
    gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')

    with open(csv_path, 'rb') as f_in, gzip.open(gz_path, 'wb', compresslevel=COMPRESS_LEVEL) as f_out:
        shutil.copyfileobj(f_in, f_out)

    try:
        # モジュールグローバル経由で呼ぶ（テストが monkeypatch で差し替えるため）
        verify_compressed(csv_path, gz_path)
    except VerificationError:
        gz_path.unlink(missing_ok=True)
        raise

    if delete_original:
        csv_path.unlink()

    return gz_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='data/muse/ 配下の Muse CSV を gzip 圧縮する')
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f'対象ディレクトリ（デフォルト: {DEFAULT_DATA_DIR}）',
    )
    parser.add_argument(
        '--delete-original',
        action='store_true',
        help='検証成功後に元の .csv ファイルを削除する（デフォルトでは残す）',
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        print(f'エラー: ディレクトリが見つかりません: {data_dir}', file=sys.stderr)
        return 1

    csv_files = sorted(data_dir.glob('*.csv'))
    if not csv_files:
        print(f'圧縮対象のCSVファイルが見つかりません: {data_dir}')
        return 0

    failure_count = 0

    for csv_path in csv_files:
        gz_path = csv_path.with_suffix(csv_path.suffix + '.gz')
        if gz_path.exists():
            print(f'スキップ（既に圧縮済み）: {csv_path.name}')
            continue

        original_size = csv_path.stat().st_size
        print(f'圧縮中: {csv_path.name} ...')

        try:
            compress_file(csv_path, delete_original=args.delete_original)
        except VerificationError as e:
            print(f'❌ 検証失敗: {e}', file=sys.stderr)
            failure_count += 1
            continue

        compressed_size = gz_path.stat().st_size
        ratio = original_size / compressed_size if compressed_size else float('inf')
        print(
            f'✅ {csv_path.name} -> {gz_path.name} '
            f'({original_size / 1024 / 1024:.1f}MB -> {compressed_size / 1024 / 1024:.1f}MB, {ratio:.1f}x)'
        )

    if failure_count > 0:
        print(f'\n{failure_count}件のファイルで検証に失敗しました', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
