#!/bin/bash
# 1週間以上前のMind Monitor EEGデータをGoogle Driveから削除またはアーカイブするスクリプト
#
# Usage:
#   ./scripts/cleanup_old_gdrive_data.sh [--dry-run] [--days N] [--archive]
#
# Examples:
#   ./scripts/cleanup_old_gdrive_data.sh --dry-run
#   ./scripts/cleanup_old_gdrive_data.sh --archive
#   ./scripts/cleanup_old_gdrive_data.sh --days 14 --archive

set -e  # エラー時に停止

# プロジェクトルートに移動
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# .envファイルを読み込む
if [ ! -f ".env" ]; then
    echo "❌ エラー: .envファイルが見つかりません"
    echo "   .env.exampleをコピーして.envを作成し、GDRIVE_FOLDER_IDを設定してください"
    echo ""
    echo "   cp .env.example .env"
    echo "   # .envを編集してGDRIVE_FOLDER_IDを設定"
    exit 1
fi

# .envから環境変数を読み込む
export $(grep -v '^#' .env | xargs)

# miniconda環境のSSL証明書問題を回避
if [ -f "/etc/ssl/certs/ca-certificates.crt" ]; then
    export SSL_CERT_FILE="/etc/ssl/certs/ca-certificates.crt"
fi

# 必須環境変数のチェック
if [ -z "$GDRIVE_FOLDER_ID" ]; then
    echo "❌ エラー: GDRIVE_FOLDER_IDが設定されていません"
    echo "   .envファイルでGDRIVE_FOLDER_IDを設定してください"
    exit 1
fi

if [ -z "$GDRIVE_CREDENTIALS" ]; then
    echo "❌ エラー: GDRIVE_CREDENTIALSが設定されていません"
    echo "   .envファイルでGDRIVE_CREDENTIALSを設定してください"
    exit 1
fi

# 認証ファイルの存在チェック
if [ ! -f "$GDRIVE_CREDENTIALS" ]; then
    echo "❌ エラー: 認証ファイルが見つかりません: $GDRIVE_CREDENTIALS"
    exit 1
fi

# 仮想環境の確認
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  警告: 仮想環境が有効になっていません"
    echo "   仮想環境をアクティベートしますか？ (y/n)"
    read -r answer
    if [ "$answer" = "y" ] || [ "$answer" = "Y" ]; then
        if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
            source "$PROJECT_ROOT/venv/bin/activate"
            echo "✅ 仮想環境をアクティベートしました"
        else
            echo "❌ エラー: venv/bin/activate が見つかりません"
            exit 1
        fi
    else
        echo "仮想環境なしで続行します..."
    fi
fi

# コマンドライン引数を解析
DRY_RUN=""
DAYS=""
ARCHIVE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --days)
            DAYS="--days $2"
            shift 2
            ;;
        --archive)
            ARCHIVE="--archive"
            shift
            ;;
        *)
            echo "❌ エラー: 不明なオプション: $1"
            echo ""
            echo "Usage: $0 [--dry-run] [--days N] [--archive]"
            echo ""
            echo "Options:"
            echo "  --dry-run    削除せず、削除対象ファイルの一覧のみ表示"
            echo "  --days N     保持期間（日数、デフォルト: 7）"
            echo "  --archive    削除する代わりにアーカイブフォルダに移動（推奨）"
            echo ""
            echo "Examples:"
            echo "  $0 --dry-run"
            echo "  $0 --archive"
            echo "  $0 --days 14 --archive"
            echo "  $0"
            exit 1
            ;;
    esac
done

# 古いファイルを削除またはアーカイブ
if [ -n "$ARCHIVE" ]; then
    echo "古いファイルをアーカイブフォルダに移動中..."
else
    echo "古いファイルを削除中..."
fi

python scripts/cleanup_old_gdrive_data.py \
    --credentials "$GDRIVE_CREDENTIALS" \
    --folder-id "$GDRIVE_FOLDER_ID" \
    $DAYS \
    $ARCHIVE \
    $DRY_RUN

echo ""
echo "✅ クリーンアップ処理が完了しました"
