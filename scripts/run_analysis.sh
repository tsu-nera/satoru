#!/bin/bash
#
# Muse脳波データ基本分析 実行スクリプト
#
# Usage: ./run_analysis.sh [CSV_FILE_PATH] [OUTPUT_DIR] [SAVE_TO]
#        SAVE_TO=sheets ./run_analysis.sh
#
# Arguments:
#   CSV_FILE_PATH  : 分析するCSVファイルのパス（省略時は最新ファイル）
#   OUTPUT_DIR     : 出力先ディレクトリ（デフォルト: tmp/）
#   SAVE_TO        : セッションログ保存先（none/csv/sheets、デフォルト: none）
#                    環境変数 SAVE_TO でも指定可能
#

set -e

# プロジェクトルートへの相対パス
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# .envファイルを読み込む
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# 仮想環境のチェック
if [ ! -d "$PROJECT_ROOT/venv" ]; then
    echo "エラー: 仮想環境 'venv' が見つかりません"
    echo "以下のコマンドでセットアップしてください:"
    echo "  cd $PROJECT_ROOT"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# 仮想環境の有効化
source "$PROJECT_ROOT/venv/bin/activate"

# CSVファイルパスの取得
if [ $# -eq 0 ]; then
    # 引数なしの場合、dataディレクトリから最新のCSVを使用
    CSV_FILE=$(find "$PROJECT_ROOT/data" -name "*.csv" -type f -printf "%T@ %p\n" | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$CSV_FILE" ]; then
        echo "エラー: data/ ディレクトリにCSVファイルが見つかりません"
        echo ""
        echo "Usage: $0 [CSV_FILE_PATH] [OUTPUT_DIR] [SAVE_TO]"
        echo ""
        echo "例："
        echo "  $0                                                              # 最新CSVをtmp/に出力（セッションログ保存なし）"
        echo "  SAVE_TO=csv $0                                                  # 最新CSVをtmp/に出力＆ローカルCSVに保存"
        echo "  SAVE_TO=sheets $0                                               # 最新CSVをtmp/に出力＆Google Sheetsに保存"
        echo "  $0 data/mindMonitor_2025-11-04--16-59-52.csv                   # 指定CSVをtmp/に出力"
        echo "  $0 data/mindMonitor_2025-11-04--16-59-52.csv tmp sheets        # Google Sheetsに保存（本番用）"
        exit 1
    fi

    echo "使用するCSVファイル: $CSV_FILE"
else
    CSV_FILE="$1"

    if [ ! -f "$CSV_FILE" ]; then
        echo "エラー: ファイルが見つかりません: $CSV_FILE"
        exit 1
    fi
fi

# 出力先ディレクトリ（デフォルト: tmp）
OUTPUT_DIR="${2:-$PROJECT_ROOT/tmp}"

# セッションログ保存先（環境変数または引数、デフォルト: none）
SAVE_TO="${3:-${SAVE_TO:-none}}"

# tmpディレクトリが存在しない場合は作成
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "出力ディレクトリを作成しました: $OUTPUT_DIR"
fi

# 分析実行
echo "============================================================"
echo "Muse脳波データ基本分析を実行します"
echo "============================================================"
echo ""
echo "データファイル: $(basename "$CSV_FILE")"
echo "出力先: $OUTPUT_DIR"
echo "セッションログ保存先: $SAVE_TO"
echo ""

# プロジェクトルートの共通スクリプトを使用
python "$PROJECT_ROOT/scripts/generate_report.py" --data "$CSV_FILE" --output "$OUTPUT_DIR" --save-to "$SAVE_TO"

echo ""
echo "============================================================"
echo "分析完了!"
echo "============================================================"
echo ""
echo "生成されたファイル:"
echo "  - REPORT.md (マークダウンレポート)"
echo "  - img/*.png (グラフ画像)"
echo ""
echo "レポートを確認:"
echo "  cat $OUTPUT_DIR/REPORT.md"
echo "  または"
echo "  マークダウンビューアで $OUTPUT_DIR/REPORT.md を開く"
echo ""
