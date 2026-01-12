#!/usr/bin/env python3
"""
Google DriveからMind Monitor CSVファイルを取得するスクリプト

使い方:
    # ファイル一覧表示
    python scripts/fetch_from_gdrive.py --folder-id <FOLDER_ID> --list

    # 最新ファイルをダウンロード
    python scripts/fetch_from_gdrive.py --folder-id <FOLDER_ID> --download latest --output ./data

    # 特定日付のファイルをダウンロード
    python scripts/fetch_from_gdrive.py --folder-id <FOLDER_ID> --download 2025-11-04 --output ./data
"""

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.loaders.selfloops import rename_selfloops_file


# APIスコープ（読み取り専用）
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


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


def list_csv_files(service: any, folder_id: str) -> List[Dict]:
    """
    指定フォルダ内のCSVファイル・ZIPファイル一覧を取得

    Args:
        service: Google Drive APIサービス
        folder_id: フォルダID

    Returns:
        ファイル情報のリスト（更新日時降順）
    """
    print(f"\nフォルダID: {folder_id} からファイルを検索中...")

    # クエリ: CSVまたはZIPファイル、削除されていないもの
    # text/plain も含める（拡張子なしのCSVファイル対応）
    query = (
        f"'{folder_id}' in parents "
        "and (mimeType='text/csv' or mimeType='application/zip' "
        "or mimeType='text/plain' "
        "or name contains '.csv' or name contains '.zip') "
        "and trashed=false"
    )

    # API呼び出し
    results = service.files().list(
        q=query,
        orderBy='modifiedTime desc',  # 更新日時降順
        pageSize=100,
        fields='files(id, name, modifiedTime, size, mimeType)'
    ).execute()

    files = results.get('files', [])

    if not files:
        print("⚠️  ファイルが見つかりませんでした")
        return []

    print(f"✅ {len(files)} 個のファイルを発見")
    return files


def display_files(files: List[Dict]):
    """
    ファイル一覧を見やすく表示

    Args:
        files: ファイル情報のリスト
    """
    print("\n" + "=" * 80)
    print("ファイル一覧（新しい順）")
    print("=" * 80)

    for i, file in enumerate(files, 1):
        name = file['name']
        modified = file['modifiedTime']
        size_mb = int(file.get('size', 0)) / (1024 * 1024)
        mime_type = file.get('mimeType', 'unknown')

        # ファイルタイプ判定
        file_type = "ZIP" if 'zip' in mime_type or name.endswith('.zip') else "CSV"

        # 日時をパース
        dt = datetime.fromisoformat(modified.replace('Z', '+00:00'))
        modified_str = dt.strftime('%Y-%m-%d %H:%M:%S')

        print(f"{i:2d}. {name}")
        print(f"    タイプ: {file_type}  |  更新日時: {modified_str}  |  サイズ: {size_mb:.1f} MB")
        print(f"    ID: {file['id']}")
        print()


def download_file(service: any, file_id: str, file_name: str, output_dir: str) -> str:
    """
    ファイルをダウンロード

    Args:
        service: Google Drive APIサービス
        file_id: ファイルID
        file_name: ファイル名
        output_dir: 出力ディレクトリ

    Returns:
        ダウンロードしたファイルのパス
    """
    # 拡張子がない場合は .csv を追加（CSV形式のファイル対応）
    if not Path(file_name).suffix and not file_name.endswith('.zip'):
        file_name = f"{file_name}.csv"

    output_path = Path(output_dir) / file_name

    # 出力ディレクトリ作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nダウンロード開始: {file_name}")
    print(f"保存先: {output_path}")

    # ファイルダウンロードリクエスト
    request = service.files().get_media(fileId=file_id)

    with open(output_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False

        while not done:
            status, done = downloader.next_chunk()
            if status:
                progress = int(status.progress() * 100)
                print(f"  進捗: {progress}%", end='\r')

        print(f"  進捗: 100% ✅")

    print(f"✅ ダウンロード完了: {output_path}")
    return str(output_path)


def extract_zip(zip_path: str, output_dir: str) -> Optional[str]:
    """
    ZIPファイルを解凍してCSVファイルを取り出す

    Args:
        zip_path: ZIPファイルのパス
        output_dir: 出力ディレクトリ

    Returns:
        抽出されたCSVファイルのパス、見つからない場合はNone
    """
    print(f"\nZIPファイルを解凍中: {zip_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # ZIP内のファイル一覧
        files = zip_ref.namelist()
        csv_files = [f for f in files if f.endswith('.csv')]

        if not csv_files:
            print("⚠️  ZIP内にCSVファイルが見つかりませんでした")
            return None

        # 最初のCSVファイルを解凍
        csv_file = csv_files[0]
        print(f"  CSVファイルを抽出: {csv_file}")

        # 解凍
        zip_ref.extract(csv_file, output_dir)
        extracted_path = Path(output_dir) / csv_file

        # ファイル名がディレクトリ構造を含む場合、ルートに移動
        if '/' in csv_file or '\\' in csv_file:
            final_path = Path(output_dir) / Path(csv_file).name
            extracted_path.rename(final_path)
            # 空のディレクトリを削除
            try:
                extracted_path.parent.rmdir()
            except:
                pass
            extracted_path = final_path

        print(f"✅ 解凍完了: {extracted_path}")
        return str(extracted_path)


def find_file_by_date(files: List[Dict], date_str: str) -> Optional[Dict]:
    """
    日付文字列を含むファイル名を検索

    Args:
        files: ファイル一覧
        date_str: 日付文字列（例: "2025-11-04"）

    Returns:
        該当ファイル、見つからない場合はNone
    """
    for file in files:
        if date_str in file['name']:
            return file
    return None


def main():
    parser = argparse.ArgumentParser(
        description='Google DriveからMind Monitor CSVファイルを取得',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ファイル一覧表示
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5 --list

  # 最新ファイルをダウンロード
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5 --download latest --output ./data

  # 特定日付のファイルをダウンロード
  python %(prog)s --folder-id 1Yo4QRa8sP16zRJ9ky-vPHzJ8zEBQ85C5 --download 2025-11-04 --output ./data

  # 認証情報を明示的に指定
  python %(prog)s --credentials private/credentials.json --folder-id XXX --list
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
        '--list',
        action='store_true',
        help='ファイル一覧を表示'
    )

    parser.add_argument(
        '--download',
        type=str,
        metavar='DATE_OR_LATEST',
        help='ダウンロードするファイル（"latest" または日付文字列 "2025-11-04"）'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./data',
        help='ダウンロード先ディレクトリ（デフォルト: ./data）'
    )

    args = parser.parse_args()

    # --list も --download も指定されていない場合はエラー
    if not args.list and not args.download:
        parser.error("--list または --download のいずれかを指定してください")

    try:
        # 認証
        service = authenticate_gdrive(args.credentials)

        # ファイル一覧取得
        files = list_csv_files(service, args.folder_id)

        if not files:
            sys.exit(1)

        # --list: 一覧表示
        if args.list:
            display_files(files)

        # --download: ダウンロード
        if args.download:
            if args.download.lower() == 'latest':
                # 最新ファイル
                target_file = files[0]
                print(f"\n最新ファイルを選択: {target_file['name']}")
            else:
                # 日付指定
                target_file = find_file_by_date(files, args.download)
                if not target_file:
                    print(f"❌ 日付 '{args.download}' を含むファイルが見つかりませんでした")
                    sys.exit(1)
                print(f"\n日付 '{args.download}' のファイルを選択: {target_file['name']}")

            # ダウンロード実行
            downloaded_path = download_file(
                service,
                target_file['id'],
                target_file['name'],
                args.output
            )

            # ZIPファイルの場合は自動解凍
            final_path = downloaded_path
            if downloaded_path.endswith('.zip'):
                csv_path = extract_zip(downloaded_path, args.output)
                if csv_path:
                    final_path = csv_path
                    # ZIPファイルを削除（オプション）
                    # Path(downloaded_path).unlink()
                else:
                    print("⚠️  ZIPファイルの解凍に失敗しましたが、ZIPファイルは保存されています")

            # Selfloopsファイルの場合はタイムスタンプからリネーム
            if final_path.endswith('.csv'):
                final_path = rename_selfloops_file(final_path)

            print(f"\n✅ すべての処理が完了しました")
            print(f"ファイルパス: {final_path}")

    except Exception as e:
        print(f"\n❌ エラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
