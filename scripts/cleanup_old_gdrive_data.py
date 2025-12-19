#!/usr/bin/env python3
"""
Google Driveから1週間以上前のMind Monitor CSVファイルを削除またはアーカイブするスクリプト

使い方:
    # 削除対象ファイルを確認（dry-run）
    python scripts/cleanup_old_gdrive_data.py --folder-id <FOLDER_ID> --dry-run

    # アーカイブフォルダに移動（推奨）
    python scripts/cleanup_old_gdrive_data.py --folder-id <FOLDER_ID> --archive

    # 実際に削除を実行（削除権限が必要）
    python scripts/cleanup_old_gdrive_data.py --folder-id <FOLDER_ID>

    # 保持期間を変更（例: 14日）
    python scripts/cleanup_old_gdrive_data.py --folder-id <FOLDER_ID> --days 14 --archive
"""

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build


# APIスコープ（削除のために書き込み権限が必要）
SCOPES = ['https://www.googleapis.com/auth/drive']


def authenticate_gdrive(credentials_path: Optional[str] = None) -> any:
    """
    Google Drive APIに認証する

    Args:
        credentials_path: サービスアカウントJSONファイルのパス
                          Noneの場合は環境変数 GOOGLE_APPLICATION_CREDENTIALS を使用

    Returns:
        Google Drive APIサービスオブジェクト
    """
    # 認証情報のパスを決定
    if credentials_path is None:
        credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if not credentials_path:
            raise ValueError(
                "認証情報が見つかりません。\n"
                "--credentials オプションで指定するか、\n"
                "環境変数 GOOGLE_APPLICATION_CREDENTIALS を設定してください。"
            )

    # ファイル存在チェック
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"認証情報ファイルが見つかりません: {credentials_path}")

    print(f"認証情報: {credentials_path}")

    # サービスアカウント認証
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=SCOPES
    )

    # Drive APIサービス構築
    service = build('drive', 'v3', credentials=credentials)

    print("✅ Google Drive API 認証成功")
    return service


def list_old_files(service: any, folder_id: str, days: int) -> List[Dict]:
    """
    指定フォルダ内の古いファイル一覧を取得

    Args:
        service: Google Drive APIサービス
        folder_id: フォルダID
        days: 保持期間（日数）

    Returns:
        削除対象ファイル情報のリスト
    """
    print(f"\nフォルダID: {folder_id} から{days}日以上前のファイルを検索中...")

    # 基準日時を計算（現在時刻 - 指定日数）
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    cutoff_str = cutoff_date.isoformat()

    print(f"削除対象: {cutoff_date.strftime('%Y-%m-%d %H:%M:%S')} より前のファイル")

    # クエリ: CSVまたはZIPファイル、削除されていないもの
    query = (
        f"'{folder_id}' in parents "
        "and (mimeType='text/csv' or mimeType='application/zip' "
        "or name contains '.csv' or name contains '.zip') "
        "and trashed=false "
        f"and modifiedTime < '{cutoff_str}'"
    )

    # API呼び出し
    results = service.files().list(
        q=query,
        orderBy='modifiedTime desc',
        pageSize=100,
        fields='files(id, name, modifiedTime, size, mimeType, parents)'
    ).execute()

    files = results.get('files', [])

    if not files:
        print(f"✅ 削除対象のファイルはありません（{days}日以内のファイルのみ存在）")
        return []

    print(f"⚠️  {len(files)} 個の削除対象ファイルを発見")
    return files


def display_files(files: List[Dict]):
    """
    ファイル一覧を見やすく表示

    Args:
        files: ファイル情報のリスト
    """
    print("\n" + "=" * 80)
    print("削除対象ファイル一覧（古い順）")
    print("=" * 80)

    for i, file in enumerate(reversed(files), 1):
        name = file['name']
        modified = file['modifiedTime']
        size_mb = int(file.get('size', 0)) / (1024 * 1024)

        # 日時をパース
        dt = datetime.fromisoformat(modified.replace('Z', '+00:00'))
        modified_str = dt.strftime('%Y-%m-%d %H:%M:%S')

        # 経過日数を計算
        age_days = (datetime.now(timezone.utc) - dt).days

        print(f"{i:2d}. {name}")
        print(f"    更新日時: {modified_str}  |  経過: {age_days}日前  |  サイズ: {size_mb:.1f} MB")
        print()


def create_archive_folder(service: any, parent_folder_id: str) -> str:
    """
    アーカイブフォルダを作成（既に存在する場合は取得）

    Args:
        service: Google Drive APIサービス
        parent_folder_id: 親フォルダID

    Returns:
        アーカイブフォルダのID
    """
    archive_folder_name = "_archive_old_files"

    # 既存のアーカイブフォルダを検索
    query = (
        f"'{parent_folder_id}' in parents "
        f"and name='{archive_folder_name}' "
        "and mimeType='application/vnd.google-apps.folder' "
        "and trashed=false"
    )

    results = service.files().list(
        q=query,
        fields='files(id, name)'
    ).execute()

    folders = results.get('files', [])

    if folders:
        folder_id = folders[0]['id']
        print(f"\n既存のアーカイブフォルダを使用: {archive_folder_name} (ID: {folder_id})")
        return folder_id

    # 新規作成
    folder_metadata = {
        'name': archive_folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]
    }

    folder = service.files().create(
        body=folder_metadata,
        fields='id'
    ).execute()

    folder_id = folder.get('id')
    print(f"\nアーカイブフォルダを作成: {archive_folder_name} (ID: {folder_id})")
    return folder_id


def delete_files(service: any, files: List[Dict], dry_run: bool = False,
                 move_to_archive: bool = False, parent_folder_id: str = None) -> int:
    """
    ファイルを削除またはアーカイブフォルダに移動

    Args:
        service: Google Drive APIサービス
        files: 削除するファイル情報のリスト
        dry_run: True の場合は削除せず表示のみ
        move_to_archive: True の場合は削除ではなくアーカイブフォルダに移動
        parent_folder_id: 親フォルダID（アーカイブフォルダ作成用）

    Returns:
        削除/移動したファイル数
    """
    if not files:
        return 0

    action = "移動" if move_to_archive else "削除"

    if dry_run:
        print("\n" + "=" * 80)
        print(f"⚠️  DRY RUN モード: 実際には{action}されません")
        print("=" * 80)
        display_files(files)
        print(f"\n{action}対象: {len(files)} ファイル")
        print(f"\n実際に{action}するには --dry-run オプションを外して再実行してください")
        return 0

    print("\n" + "=" * 80)
    print(f"⚠️  以下のファイルを{action}します")
    print("=" * 80)
    display_files(files)

    # アーカイブフォルダを作成
    archive_folder_id = None
    if move_to_archive:
        if not parent_folder_id:
            print("\n❌ エラー: アーカイブ先の親フォルダIDが指定されていません")
            return 0
        archive_folder_id = create_archive_folder(service, parent_folder_id)

    # 確認プロンプト
    print(f"\n{len(files)} ファイルを{action}します。よろしいですか？ (yes/no): ", end='')
    response = input().strip().lower()

    if response not in ['yes', 'y']:
        print(f"\n❌ {action}をキャンセルしました")
        return 0

    # 削除/移動実行
    print(f"\n{action}を開始します...")
    processed_count = 0

    for i, file in enumerate(files, 1):
        try:
            print(f"[{i}/{len(files)}] {action}中: {file['name']}...", end=' ')

            if move_to_archive:
                # ファイルをアーカイブフォルダに移動
                previous_parents = ",".join(file.get('parents', []))
                service.files().update(
                    fileId=file['id'],
                    addParents=archive_folder_id,
                    removeParents=previous_parents,
                    fields='id, parents'
                ).execute()
            else:
                # ファイルを削除
                service.files().delete(fileId=file['id']).execute()

            print("✅")
            processed_count += 1
        except Exception as e:
            print(f"❌ エラー: {e}")

    print(f"\n✅ {action}完了: {processed_count}/{len(files)} ファイル")
    return processed_count


def main():
    parser = argparse.ArgumentParser(
        description='Google Driveから古いMind Monitor CSVファイルを削除',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 削除対象ファイルを確認（dry-run）
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5 --dry-run

  # 7日以上前のファイルをアーカイブフォルダに移動（推奨）
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5 --archive

  # 7日以上前のファイルを削除（削除権限が必要）
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5

  # 14日以上前のファイルをアーカイブ
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5 --days 14 --archive

  # 認証情報を明示的に指定
  python %(prog)s --credentials private/credentials.json --folder-id XXX --dry-run
        """
    )

    parser.add_argument(
        '--credentials',
        type=str,
        help='サービスアカウントJSONファイルのパス（省略時は環境変数 GOOGLE_APPLICATION_CREDENTIALS を使用）'
    )

    parser.add_argument(
        '--folder-id',
        type=str,
        required=True,
        help='Google DriveフォルダID（必須）'
    )

    parser.add_argument(
        '--days',
        type=int,
        default=7,
        help='保持期間（日数、デフォルト: 7）'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='削除せず、削除対象ファイルの一覧のみ表示'
    )

    parser.add_argument(
        '--archive',
        action='store_true',
        help='削除する代わりにアーカイブフォルダ（_archive_old_files）に移動'
    )

    args = parser.parse_args()

    try:
        # 認証
        service = authenticate_gdrive(args.credentials)

        # 古いファイル一覧取得
        old_files = list_old_files(service, args.folder_id, args.days)

        # ファイル削除またはアーカイブ
        action = "移動" if args.archive else "削除"
        processed_count = delete_files(
            service,
            old_files,
            args.dry_run,
            move_to_archive=args.archive,
            parent_folder_id=args.folder_id
        )

        if not args.dry_run and processed_count > 0:
            print(f"\n✅ すべての処理が完了しました（{processed_count}ファイル{action}）")

    except KeyboardInterrupt:
        print("\n\n❌ ユーザーによって中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
