#!/usr/bin/env python3
"""
Google Drive上のSelfLoopsファイル（同名 "SelfLoops HRV data"）を日付入りにリネームするスクリプト

リネームの日付はDriveの modifiedTime（UTC・コピーや移動で変わりうる）ではなく、
ファイル内容の1行目に記録されたタイムスタンプから決定する（lib.loaders.selfloops を再利用）。

デフォルトはdry-runで、実際のリネームは行わない。--execute を指定した場合のみ実行する。

使い方:
    # dry-run（リネーム予定を表示するのみ）
    python scripts/rename_gdrive_selfloops.py --folder-id <FOLDER_ID>

    # 実際にリネームを実行
    python scripts/rename_gdrive_selfloops.py --folder-id <FOLDER_ID> --execute
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.loaders.selfloops import generate_selfloops_filename, parse_selfloops_timestamp

# APIスコープ（リネームのために書き込み権限が必要）
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


def list_files(service: Any, folder_id: str) -> List[Dict]:
    """
    指定フォルダ内のファイル一覧を取得する（modifiedTime は取得しない）

    Args:
        service: Google Drive APIサービス
        folder_id: フォルダID

    Returns:
        ファイル情報（id, name）のリスト
    """
    files: List[Dict] = []
    page_token: Optional[str] = None

    while True:
        results = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields='nextPageToken, files(id, name)',
            pageSize=1000,
            pageToken=page_token,
        ).execute()

        files.extend(results.get('files', []))

        page_token = results.get('nextPageToken')
        if not page_token:
            break

    return files


def needs_rename(name: str) -> bool:
    """
    ファイル名がリネーム対象かどうかを判定する

    既に "selfloops_" で始まり "--" を含む場合はリネーム不要

    Args:
        name: ファイル名

    Returns:
        リネームが必要ならTrue
    """
    return not (name.startswith('selfloops_') and '--' in name)


def fetch_first_line(service: Any, file_id: str) -> str:
    """
    ファイルの1行目をDriveからダウンロードして取得する

    テストからmonkeypatchで差し替えられるよう、モジュールグローバル経由で呼ぶこと。

    Args:
        service: Google Drive APIサービス
        file_id: ファイルID

    Returns:
        1行目の文字列（デコードエラーは置換）
    """
    request = service.files().get_media(fileId=file_id)
    buffer = io.BytesIO()
    downloader = MediaIoBaseDownload(buffer, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()

    content = buffer.getvalue().decode('utf-8', errors='replace')
    return content.splitlines()[0] if content else ''


def decide_new_name(first_line: str) -> Optional[str]:
    """
    ファイル1行目のタイムスタンプから新しいファイル名を決定する

    Args:
        first_line: ファイルの1行目

    Returns:
        新しいファイル名、パース不能な場合はNone
    """
    timestamp = parse_selfloops_timestamp(first_line)
    if timestamp is None:
        return None
    return generate_selfloops_filename(timestamp)


def rename_files(
    service: Any,
    files: List[Dict],
    existing_names: Set[str],
    dry_run: bool = True,
) -> Tuple[int, int]:
    """
    ファイル一覧をリネームする（対象外・パース失敗・衝突はスキップ）

    Args:
        service: Google Drive APIサービス
        files: リネーム対象候補のファイル情報リスト（id, name）
        existing_names: Drive上の既存ファイル名集合（リネーム成功時に更新される）
        dry_run: True の場合はリネームせず予定のみ表示

    Returns:
        (リネーム対象/実施数, スキップ数)
    """
    renamed_count = 0
    skipped_count = 0

    for file in files:
        name = file['name']

        if not needs_rename(name):
            skipped_count += 1
            continue

        first_line = fetch_first_line(service, file['id'])
        new_name = decide_new_name(first_line)

        if new_name is None:
            print(f"⚠️  スキップ (タイムスタンプのパースに失敗): {name}")
            skipped_count += 1
            continue

        if new_name in existing_names:
            print(f"⚠️  スキップ (同名が既に存在): {name} -> {new_name}")
            skipped_count += 1
            continue

        if dry_run:
            print(f"🔍 [DRY RUN] {name} → {new_name}")
            renamed_count += 1
            continue

        service.files().update(fileId=file['id'], body={'name': new_name}).execute()
        existing_names.add(new_name)
        print(f"✅ リネーム完了: {name} → {new_name}")
        renamed_count += 1

    return renamed_count, skipped_count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Google Drive上のSelfLoopsファイルを日付入りにリネームする',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # dry-run（リネーム予定を表示するのみ）
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5

  # 実際にリネームを実行
  python %(prog)s --folder-id XXX --execute
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
        default=os.environ.get('GDRIVE_FOLDER_ID_SELFLOOPS'),
        help='対象のGoogle DriveフォルダID（省略時は環境変数 GDRIVE_FOLDER_ID_SELFLOOPS を使用）'
    )

    parser.add_argument(
        '--execute',
        action='store_true',
        help='実際にリネームを実行する（省略時はdry-run）'
    )

    args = parser.parse_args(argv)

    if not args.folder_id:
        print(
            "⚠️  警告: GDRIVE_FOLDER_ID_SELFLOOPS が設定されていません。"
            "リネームをスキップします"
        )
        return 0

    try:
        service = authenticate_gdrive(args.credentials)
        files = list_files(service, args.folder_id)
        existing_names = {f['name'] for f in files}

        dry_run = not args.execute
        renamed_count, skipped_count = rename_files(service, files, existing_names, dry_run=dry_run)

        action = "予定" if dry_run else "実施"
        print(f"\n📦 完了: リネーム{action} {renamed_count} 件 / スキップ {skipped_count} 件")

        if dry_run:
            print("\n実際に適用するには --execute を付けてください")

        return 0

    except Exception as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
