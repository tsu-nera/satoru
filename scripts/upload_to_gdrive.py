#!/usr/bin/env python3
"""
ローカルファイルをGoogle Driveの指定フォルダにアップロードするスクリプト

同名ファイルが既にDrive上に存在する場合はスキップし、重複を作らない（冪等）。
アップロードは追加のみで、Drive上のファイルを削除・上書きすることはない。

使い方:
    # data/taps/ 配下のCSVをすべてアップロード
    uv run python scripts/upload_to_gdrive.py --folder-id <FOLDER_ID>

    # 単一ファイルのみアップロード
    uv run python scripts/upload_to_gdrive.py --folder-id <FOLDER_ID> --file data/taps/taps_2026-01-10--10-00-00.csv

    # dry-run（アップロードせず予定のみ表示）
    uv run python scripts/upload_to_gdrive.py --folder-id <FOLDER_ID> --dry-run

    # 環境変数 GDRIVE_FOLDER_ID_TAPS / GDRIVE_CREDENTIALS を使う場合はオプション省略可
    uv run python scripts/upload_to_gdrive.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Set

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'taps'

# APIスコープ（アップロードのために書き込み権限が必要）
SCOPES = ['https://www.googleapis.com/auth/drive']


def authenticate_gdrive(credentials_path: Optional[str] = None) -> Any:
    """
    Google Drive APIに認証する

    Args:
        credentials_path: サービスアカウントJSONファイルのパス
                          Noneの場合は環境変数 GOOGLE_APPLICATION_CREDENTIALS を使用

    Returns:
        Google Drive APIサービスオブジェクト
    """
    if credentials_path is None:
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError(
                "認証情報が見つかりません。\n"
                "--credentials オプションで指定するか、\n"
                "環境変数 GOOGLE_APPLICATION_CREDENTIALS を設定してください。"
            )

    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"認証情報ファイルが見つかりません: {credentials_path}")

    print(f"認証情報: {credentials_path}")

    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=credentials)

    print("✅ Google Drive API 認証成功")
    return service


def list_remote_names(service: Any, folder_id: str) -> Set[str]:
    """
    指定フォルダ内の既存ファイル名一覧を取得する

    Args:
        service: Google Drive APIサービス
        folder_id: フォルダID

    Returns:
        ファイル名の集合
    """
    names: Set[str] = set()
    page_token: Optional[str] = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields='nextPageToken, files(id, name)',
            pageSize=1000,
            pageToken=page_token,
        ).execute()

        for file in results.get('files', []):
            names.add(file['name'])

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    return names


def upload_file(
    service: Any,
    folder_id: str,
    path: Path,
    existing_names: Set[str],
    dry_run: bool = False,
) -> bool:
    """
    1ファイルをDriveにアップロードする（既存ファイルはスキップ）

    Args:
        service: Google Drive APIサービス
        folder_id: アップロード先フォルダID
        path: アップロード対象のローカルファイルパス
        existing_names: Drive上の既存ファイル名集合（アップロード成功時に更新される）
        dry_run: True の場合はアップロードせず予定のみ表示

    Returns:
        アップロードを実行した場合True、スキップした場合False
    """
    name = path.name

    if name in existing_names:
        print(f"⏭️  スキップ (既に存在): {name}")
        return False

    if dry_run:
        print(f"🔍 [DRY RUN] アップロード予定: {name}")
        return True

    media = MediaFileUpload(str(path), mimetype='text/csv', resumable=False)
    service.files().create(
        body={'name': name, 'parents': [folder_id]},
        media_body=media,
        fields='id',
    ).execute()

    existing_names.add(name)
    print(f"✅ アップロード完了: {name}")
    return True


def collect_files(data_dir: Path, pattern: str = '*.csv') -> List[Path]:
    """
    アップロード対象ファイルを収集する

    Args:
        data_dir: 対象ディレクトリ
        pattern: glob パターン

    Returns:
        ソート済みファイルパスのリスト
    """
    return sorted(data_dir.glob(pattern))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='ローカルファイルをGoogle Driveの指定フォルダにアップロードする',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # data/taps/ 配下のCSVをすべてアップロード
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5

  # 単一ファイルのみアップロード
  python %(prog)s --folder-id XXX --file data/taps/taps_2026-01-10--10-00-00.csv

  # dry-run
  python %(prog)s --folder-id XXX --dry-run
        """
    )

    parser.add_argument(
        '--credentials',
        type=str,
        default=os.environ.get('GDRIVE_CREDENTIALS'),
        help='サービスアカウントJSONファイルのパス（省略時は環境変数 GDRIVE_CREDENTIALS を使用）'
    )

    parser.add_argument(
        '--folder-id',
        type=str,
        default=os.environ.get('GDRIVE_FOLDER_ID_TAPS'),
        help='アップロード先のGoogle DriveフォルダID（省略時は環境変数 GDRIVE_FOLDER_ID_TAPS を使用）'
    )

    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        '--file',
        type=Path,
        help='単一ファイルのみアップロードする（--data-dir と同時指定不可）'
    )
    target_group.add_argument(
        '--data-dir',
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f'アップロード対象ディレクトリ（デフォルト: {DEFAULT_DATA_DIR}）'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.csv',
        help='--data-dir 使用時の glob パターン（デフォルト: *.csv）'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='アップロードせず、アップロード予定のみ表示'
    )

    args = parser.parse_args(argv)

    if not args.folder_id:
        print(
            "⚠️  警告: GDRIVE_FOLDER_ID_TAPS が設定されていません。"
            "タップログのアップロードをスキップします"
        )
        return 0

    if args.file is not None:
        if not args.file.is_file():
            print(f'エラー: ファイルが見つかりません: {args.file}', file=sys.stderr)
            return 1
        files = [args.file]
    else:
        if not args.data_dir.is_dir():
            print(f'エラー: ディレクトリが見つかりません: {args.data_dir}', file=sys.stderr)
            return 1
        files = collect_files(args.data_dir, args.pattern)
        if not files:
            print(f'アップロード対象のファイルが見つかりません: {args.data_dir}')
            return 0

    try:
        service = authenticate_gdrive(args.credentials)
        existing_names = list_remote_names(service, args.folder_id)

        uploaded_count = 0
        skipped_count = 0

        for path in files:
            if upload_file(service, args.folder_id, path, existing_names, dry_run=args.dry_run):
                uploaded_count += 1
            else:
                skipped_count += 1

        print(f"\n📦 完了: アップロード {uploaded_count} 件 / スキップ {skipped_count} 件")
        return 0

    except Exception as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
