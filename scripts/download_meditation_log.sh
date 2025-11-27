#!/bin/bash
#
# 瞑想記録をGoogle SpreadsheetからCSVとしてダウンロード
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 設定
SHEET_ID="1S0SwyRbM2cAATv_IOpkOspor-_Ov49rTygyTFr2gkwA"
CREDENTIALS="$PROJECT_ROOT/private/gdrive-creds.json"
OUTPUT_DIR="$PROJECT_ROOT/tmp"
OUTPUT_FILE="$OUTPUT_DIR/meditation_log.csv"

# 出力ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

# 仮想環境をアクティブ化
source "$PROJECT_ROOT/venv/bin/activate"

# ダウンロード実行
python "$SCRIPT_DIR/download_meditation_log.py" \
    --credentials "$CREDENTIALS" \
    --sheet-id "$SHEET_ID" \
    --output "$OUTPUT_FILE"

echo ""
echo "ダウンロード完了: $OUTPUT_FILE"
