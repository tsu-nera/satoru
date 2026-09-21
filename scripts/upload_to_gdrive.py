#!/usr/bin/env python3
"""
ローカルファイルをGoogle Driveの指定フォルダにアップロードするスクリプト

同名ファイルが既にDrive上に存在する場合はスキップし、重複を作らない（冪等）。
アップロードは追加のみで、Drive上のファイルを削除・上書きすることはない。

認証は OAuth ユーザー認証（drive.file スコープ）を使う。サービスアカウントは
保存容量枠を持たないため、ユーザー所有フォルダへの新規ファイル作成が
`storageQuotaExceeded` で拒否される（実測済み）。初回実行時はブラウザが開き、
Google アカウントでの同意操作が必要になる。以降はトークンファイルに保存された
認証情報で自動的に認証される（OAuth同意画面が「テスト」状態の場合、
リフレッシュトークンは約7日で失効し、再度ブラウザ同意が必要になる）。

drive.file スコープの制約: `files().list` は「このアプリが作成・オープンした
ファイル」しか返さない。Drive UI からの手動アップロード等、別経路で置かれた
同名ファイルは検出できず、重複を作りうる。

また drive.file スコープでは、アプリが作成していないフォルダ（ユーザーが
Drive UI で手動作成したフォルダ等）を parents に指定した新規作成が
403/404 で拒否されることがある。その場合は `--create-folder <名前>` で
アプリ自身にフォルダを作らせ、出力されたフォルダIDを設定し直すこと。

使い方:
    # data/taps/ 配下のCSVをすべてアップロード（初回はブラウザで同意が必要）
    uv run python scripts/upload_to_gdrive.py --folder-id <FOLDER_ID>

    # 単一ファイルのみアップロード
    uv run python scripts/upload_to_gdrive.py --folder-id <FOLDER_ID> --file data/taps/taps_2026-01-10--10-00-00.csv

    # dry-run（認証せず、アップロード予定のみ表示）
    uv run python scripts/upload_to_gdrive.py --folder-id <FOLDER_ID> --dry-run

    # アプリ所有のアップロード先フォルダを新規作成する（drive.file の制約を回避）
    uv run python scripts/upload_to_gdrive.py --create-folder Taps

    # 環境変数 GDRIVE_FOLDER_ID_TAPS / GDRIVE_OAUTH_CLIENT / GDRIVE_OAUTH_TOKEN を使う場合はオプション省略可
    uv run python scripts/upload_to_gdrive.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Set

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data' / 'taps'

DEFAULT_OAUTH_CLIENT = Path.home() / '.config/gcp/oauth-client.json'
DEFAULT_TOKEN_PATH = Path.home() / '.config/gcp/tokens/gdrive.json'

# アップロードにはファイル作成が必要だが、非センシティブスコープの drive.file で足りる
# （このアプリが作成したファイルのみにアクセス。Google審査不要）
UPLOAD_SCOPES = ['https://www.googleapis.com/auth/drive.file']


def authorize_gdrive(client_secrets: Path, token_path: Path) -> Any:
    """
    OAuthユーザー認証でGoogle Drive APIに認証する

    既存の有効なトークンがあればそれを使う。期限切れでリフレッシュトークンが
    あればリフレッシュする。どちらも無い/失敗した場合はブラウザを開いて
    ユーザーの同意を得る（InstalledAppFlow）。取得したトークンは token_path に
    保存する（パーミッション 0o600）。

    Args:
        client_secrets: OAuthクライアントシークレットJSONのパス
        token_path: トークン保存先のパス

    Returns:
        Google Drive APIサービスオブジェクト
    """
    creds: Optional[Credentials] = None

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), UPLOAD_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                creds = None

        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), UPLOAD_SCOPES)
            creds = flow.run_local_server(port=0)

        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())
        token_path.chmod(0o600)

    service = build('drive', 'v3', credentials=creds)
    print("✅ Google Drive API 認証成功（OAuth）")
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
    try:
        service.files().create(
            body={'name': name, 'parents': [folder_id]},
            media_body=media,
            fields='id',
        ).execute()
    except HttpError as e:
        if e.resp is not None and e.resp.status in (403, 404):
            print(
                f"❌ アップロード失敗 ({e.resp.status}): {name}\n"
                "   drive.file スコープではアプリが作成していないフォルダへ書き込めない場合があります。\n"
                "   --create-folder <名前> でアプリ自身にフォルダを作らせ、\n"
                "   出力されたフォルダIDを .env の GDRIVE_FOLDER_ID_TAPS に設定し直してください。"
            )
        raise

    existing_names.add(name)
    print(f"✅ アップロード完了: {name}")
    return True


def create_folder(service: Any, name: str) -> str:
    """
    アプリ所有（OAuthユーザー所有）のフォルダをDrive上に新規作成する

    drive.file スコープの制約（アプリが作成していないフォルダへの書き込み拒否）を
    回避するために使う。作成後、Drive UI で任意の場所に移動してもIDは変わらない。

    Args:
        service: Google Drive APIサービス
        name: 作成するフォルダ名

    Returns:
        作成したフォルダのID
    """
    folder = service.files().create(
        body={'name': name, 'mimeType': 'application/vnd.google-apps.folder'},
        fields='id, name',
    ).execute()
    return str(folder['id'])


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
  # data/taps/ 配下のCSVをすべてアップロード（初回はブラウザで同意が必要）
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5

  # 単一ファイルのみアップロード
  python %(prog)s --folder-id XXX --file data/taps/taps_2026-01-10--10-00-00.csv

  # dry-run（認証しない）
  python %(prog)s --folder-id XXX --dry-run

  # アプリ所有のアップロード先フォルダを新規作成する
  python %(prog)s --create-folder Taps
        """
    )

    parser.add_argument(
        '--oauth-client',
        type=Path,
        default=Path(os.environ.get('GDRIVE_OAUTH_CLIENT', str(DEFAULT_OAUTH_CLIENT))),
        help=f'OAuthクライアントシークレットJSONのパス（デフォルト: {DEFAULT_OAUTH_CLIENT}）'
    )

    parser.add_argument(
        '--token',
        type=Path,
        default=Path(os.environ.get('GDRIVE_OAUTH_TOKEN', str(DEFAULT_TOKEN_PATH))),
        help=f'OAuthトークン保存先のパス（デフォルト: {DEFAULT_TOKEN_PATH}）'
    )

    parser.add_argument(
        '--folder-id',
        type=str,
        default=os.environ.get('GDRIVE_FOLDER_ID_TAPS'),
        help='アップロード先のGoogle DriveフォルダID（省略時は環境変数 GDRIVE_FOLDER_ID_TAPS を使用）'
    )

    parser.add_argument(
        '--create-folder',
        type=str,
        metavar='NAME',
        help='アプリ所有のフォルダを新規作成し、そのIDを表示して終了する（アップロードは行わない）'
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
        help='認証を行わず、アップロード予定のみ表示する（Drive上の既存ファイルは分からないためスキップ判定はしない）'
    )

    args = parser.parse_args(argv)

    if args.create_folder:
        if not args.oauth_client.exists():
            print(
                f"❌ OAuthクライアントシークレットが見つかりません: {args.oauth_client}\n"
                "   Google Cloud Console でOAuthクライアント（デスクトップアプリ）を作成し、\n"
                "   ダウンロードしたJSONを配置するか --oauth-client で指定してください。",
                file=sys.stderr,
            )
            return 1
        service = authorize_gdrive(args.oauth_client, args.token)
        folder_id = create_folder(service, args.create_folder)
        print(f"✅ フォルダを作成しました: {args.create_folder} (ID: {folder_id})")
        print("   Drive UI で任意の場所に移動してもIDは変わりません。")
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

    if args.dry_run:
        print("🔍 [DRY RUN] 認証はスキップします（Drive上の既存ファイルは確認できないため、スキップ判定は行いません）")
        for path in files:
            print(f"🔍 [DRY RUN] アップロード予定: {path.name}")
        print(f"\n📦 完了: アップロード予定 {len(files)} 件")
        return 0

    if not args.folder_id:
        print(
            "⚠️  警告: GDRIVE_FOLDER_ID_TAPS が設定されていません。"
            "タップログのアップロードをスキップします"
        )
        return 0

    if not args.oauth_client.exists():
        print(
            f"❌ OAuthクライアントシークレットが見つかりません: {args.oauth_client}\n"
            "   Google Cloud Console でOAuthクライアント（デスクトップアプリ）を作成し、\n"
            "   ダウンロードしたJSONを配置するか --oauth-client で指定してください。",
            file=sys.stderr,
        )
        return 1

    try:
        service = authorize_gdrive(args.oauth_client, args.token)
        existing_names = list_remote_names(service, args.folder_id)

        uploaded_count = 0
        skipped_count = 0

        for path in files:
            if upload_file(service, args.folder_id, path, existing_names):
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
