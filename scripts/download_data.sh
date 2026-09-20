#!/bin/bash
# データをGoogle Driveからダウンロードして解凍するスクリプト
#
# Usage:
#   ./scripts/download_data.sh <source> [date]
#
# Arguments:
#   source  : データソース (muse, selfloops, taps, all)
#   date    : ダウンロード対象日付 (latest または YYYY-MM-DD、デフォルト: latest)
#
# Examples:
#   ./scripts/download_data.sh muse latest
#   ./scripts/download_data.sh selfloops 2025-01-10
#   ./scripts/download_data.sh taps latest
#   ./scripts/download_data.sh all latest

set -e  # エラー時に停止

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ヘルプメッセージ
show_help() {
    echo "Usage: $0 <source> [date]"
    echo ""
    echo "Arguments:"
    echo "  source  : データソース (muse, selfloops, taps, all)"
    echo "  date    : ダウンロード対象日付 (latest または YYYY-MM-DD、デフォルト: latest)"
    echo ""
    echo "Examples:"
    echo "  $0 muse latest"
    echo "  $0 selfloops 2025-01-10"
    echo "  $0 taps latest"
    echo "  $0 all latest"
    exit 0
}

# 引数チェック
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_help
fi

SOURCE="$1"
DATE_OPTION="${2:-latest}"

# データソースの検証
if [ "$SOURCE" != "muse" ] && [ "$SOURCE" != "selfloops" ] && [ "$SOURCE" != "taps" ] && [ "$SOURCE" != "all" ]; then
    echo "❌ エラー: 無効なデータソース: $SOURCE"
    echo "   有効な値: muse, selfloops, taps, all"
    echo ""
    show_help
fi

# .envファイルを読み込む
if [ ! -f ".env" ]; then
    echo "❌ エラー: .envファイルが見つかりません"
    echo "   .env.exampleをコピーして.envを作成し、必要な環境変数を設定してください"
    echo ""
    echo "   cp .env.example .env"
    echo "   # .envを編集して環境変数を設定"
    exit 1
fi

# .envから環境変数を読み込む
export $(grep -v '^#' .env | xargs)

# miniconda環境のSSL証明書問題を回避
if [ -f "/etc/ssl/certs/ca-certificates.crt" ]; then
    export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
fi

# 認証情報のチェック
if [ -z "$GDRIVE_CREDENTIALS" ]; then
    echo "❌ エラー: GDRIVE_CREDENTIALSが設定されていません"
    echo "   .envファイルでGDRIVE_CREDENTIALSを設定してください"
    exit 1
fi

if [ ! -f "$GDRIVE_CREDENTIALS" ]; then
    echo "❌ エラー: 認証ファイルが見つかりません: $GDRIVE_CREDENTIALS"
    exit 1
fi

# データダウンロード関数
download_source_data() {
    local source_name=$1
    local folder_id_var=$2
    local data_dir=$3

    # フォルダIDの取得
    local folder_id="${!folder_id_var}"

    if [ -z "$folder_id" ]; then
        echo "⚠️  警告: ${folder_id_var}が設定されていません。${source_name}のダウンロードをスキップします"
        return 1
    fi

    echo ""
    echo "========================================="
    echo "${source_name}データをダウンロード中..."
    echo "========================================="

    # データディレクトリの作成
    mkdir -p "$data_dir"

    # CSVファイルをダウンロード
    if [ "$DATE_OPTION" = "latest" ]; then
        echo "最新ファイルをダウンロード中..."
        uv run python scripts/fetch_from_gdrive.py \
            --credentials "$GDRIVE_CREDENTIALS" \
            --folder-id "$folder_id" \
            --download latest \
            --output "$data_dir"
    else
        echo "日付指定でダウンロード中: $DATE_OPTION"
        uv run python scripts/fetch_from_gdrive.py \
            --credentials "$GDRIVE_CREDENTIALS" \
            --folder-id "$folder_id" \
            --download "$DATE_OPTION" \
            --output "$data_dir"
    fi

    # ZIPファイルを削除
    if ls "$data_dir"/*.zip 1> /dev/null 2>&1; then
        echo "ZIPファイルを削除中..."
        rm -f "$data_dir"/*.zip
        echo "✅ ZIPファイルを削除しました"
    fi

    echo ""
    echo "✅ ${source_name}のダウンロードが完了しました"
    echo "データディレクトリ: $data_dir"
    echo ""
    echo "ダウンロードされたファイル:"
    ls -lh "$data_dir"/*.csv 2>/dev/null || echo "(CSVファイルなし)"

    return 0
}

# ダウンロード実行
case "$SOURCE" in
    "muse")
        download_source_data "Muse" "GDRIVE_FOLDER_ID_MUSE" "$PROJECT_ROOT/data/muse" || true
        ;;
    "selfloops")
        download_source_data "Selfloops" "GDRIVE_FOLDER_ID_SELFLOOPS" "$PROJECT_ROOT/data/selfloops" || true
        ;;
    "taps")
        download_source_data "Taps" "GDRIVE_FOLDER_ID_TAPS" "$PROJECT_ROOT/data/taps" || true
        ;;
    "all")
        download_source_data "Muse" "GDRIVE_FOLDER_ID_MUSE" "$PROJECT_ROOT/data/muse" || true
        download_source_data "Selfloops" "GDRIVE_FOLDER_ID_SELFLOOPS" "$PROJECT_ROOT/data/selfloops" || true
        download_source_data "Taps" "GDRIVE_FOLDER_ID_TAPS" "$PROJECT_ROOT/data/taps" || true
        ;;
esac

echo ""
echo "========================================="
echo "✅ すべてのダウンロードが完了しました"
echo "========================================="
