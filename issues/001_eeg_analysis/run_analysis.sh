#!/bin/bash
#
# Muse脳波データ基本分析 実行スクリプト
#
# Usage: ./run_analysis.sh [CSV_FILE_PATH]
#

set -e

# プロジェクトルートへの相対パス
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

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
        echo "Usage: $0 [CSV_FILE_PATH]"
        echo ""
        echo "例："
        echo "  $0 data/mindMonitor_2025-10-15--17-08-15.csv"
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

# 分析実行
echo "============================================================"
echo "Muse脳波データ基本分析を実行します"
echo "============================================================"
echo ""
echo "データファイル: $(basename "$CSV_FILE")"
echo "出力先: $SCRIPT_DIR"
echo ""

cd "$SCRIPT_DIR"
python generate_report.py --data "$CSV_FILE" --output .

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
echo "  cat REPORT.md"
echo "  または"
echo "  マークダウンビューアで REPORT.md を開く"
echo ""
