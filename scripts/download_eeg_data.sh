#!/bin/bash
# Mind Monitor EEGデータをGoogle Driveからダウンロードして解凍するスクリプト
#
# Usage:
#   ./scripts/download_eeg_data.sh [latest|YYYY-MM-DD]
#
# Examples:
#   ./scripts/download_eeg_data.sh latest
#   ./scripts/download_eeg_data.sh 2025-11-04

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

# ダウンロード対象の決定
DATE_OPTION="${1:-latest}"

# データディレクトリの作成
DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DATA_DIR"

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

# CSVファイルをダウンロード
if [ "$DATE_OPTION" = "latest" ]; then
    echo "最新ファイルをダウンロード中..."
    python scripts/fetch_from_gdrive.py \
        --credentials "$GDRIVE_CREDENTIALS" \
        --folder-id "$GDRIVE_FOLDER_ID" \
        --download latest \
        --output "$DATA_DIR"
else
    echo "日付指定でダウンロード中: $DATE_OPTION"
    python scripts/fetch_from_gdrive.py \
        --credentials "$GDRIVE_CREDENTIALS" \
        --folder-id "$GDRIVE_FOLDER_ID" \
        --download "$DATE_OPTION" \
        --output "$DATA_DIR"
fi

echo ""

# ZIPファイルを削除
if ls "$DATA_DIR"/*.zip 1> /dev/null 2>&1; then
    echo "ZIPファイルを削除中..."
    rm -f "$DATA_DIR"/*.zip
    echo "✅ ZIPファイルを削除しました"
fi

echo ""
echo "✅ ダウンロードが完了しました"
echo "データディレクトリ: $DATA_DIR"
echo ""
echo "ダウンロードされたファイル:"
ls -lh "$DATA_DIR"/*.csv 2>/dev/null || echo "(CSVファイルなし)"
