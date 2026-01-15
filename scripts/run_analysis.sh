#!/bin/bash
#
# Muse脳波データ基本分析 実行スクリプト
#
# Usage: ./run_analysis.sh [--fetch] [CSV_FILE_PATH] [OUTPUT_DIR] [SAVE_TO]
#        SAVE_TO=sheets ./run_analysis.sh
#
# Arguments:
#   --fetch     : Google Driveから最新データをダウンロード（オプション）
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

# --fetchオプションの処理
FETCH_DATA=false
if [ "$1" = "--fetch" ]; then
    FETCH_DATA=true
    shift  # --fetchを引数リストから削除
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

# データダウンロード処理
if [ "$FETCH_DATA" = true ]; then
    echo "============================================================"
    echo "Google Driveから最新データをダウンロードします"
    echo "============================================================"
    echo ""

    # download_data.sh を all latest で実行
    bash "$SCRIPT_DIR/download_data.sh" all latest

    echo ""
    echo "データダウンロード完了。分析を開始します..."
    echo ""
fi

# CSVファイルパスの取得
if [ $# -eq 0 ]; then
    # 引数なしの場合、data/museディレクトリから最新のCSVを使用
    CSV_FILE=$(find "$PROJECT_ROOT/data/muse" -name "*.csv" -type f -printf "%T@ %p\n" | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -z "$CSV_FILE" ]; then
        echo "エラー: data/muse/ ディレクトリにCSVファイルが見つかりません"
        echo ""
        echo "Usage: $0 [--fetch] [CSV_FILE_PATH] [OUTPUT_DIR] [SAVE_TO]"
        echo ""
        echo "例："
        echo "  $0                                                              # 最新CSVをtmp/に出力（セッションログ保存なし）"
        echo "  $0 --fetch                                                   # Google Driveから最新データをダウンロードして分析"
        echo "  SAVE_TO=csv $0                                                  # 最新CSVをtmp/に出力＆ローカルCSVに保存"
        echo "  SAVE_TO=sheets $0 --fetch                                    # ダウンロード＆Google Sheetsに保存"
        echo "  $0 data/muse/mindMonitor_2025-11-04--16-59-52.csv                   # 指定CSVをtmp/に出力"
        echo "  $0 data/muse/mindMonitor_2025-11-04--16-59-52.csv tmp sheets        # Google Sheetsに保存（本番用）"
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

# Selfloopsファイルの検出（タイムスタンプマッチング方式）
SELFLOOPS_FILE=""
if [ -d "$PROJECT_ROOT/data/selfloops" ]; then
    # Museファイル名からタイムスタンプを抽出
    # 例: mindMonitor_2026-01-10--16-08-53_1294036912907381397.csv
    #     → 2026-01-10--16-08-53
    CSV_BASENAME=$(basename "$CSV_FILE")
    TIMESTAMP=$(echo "$CSV_BASENAME" | grep -oP 'mindMonitor_\K[0-9]{4}-[0-9]{2}-[0-9]{2}--[0-9]{2}-[0-9]{2}-[0-9]{2}')

    if [ -n "$TIMESTAMP" ]; then
        # 同じタイムスタンプのSelfloopsファイルを検索
        # 例: selfloops_2026-01-10--16-08-50.csv（数秒のズレは許容）
        # 完全一致を優先、なければ同じ分（HH-MM）で検索
        SELFLOOPS_EXACT="${PROJECT_ROOT}/data/selfloops/selfloops_${TIMESTAMP}.csv"

        if [ -f "$SELFLOOPS_EXACT" ]; then
            SELFLOOPS_FILE="$SELFLOOPS_EXACT"
            echo "検出されたSelfloopsファイル（完全一致）: $SELFLOOPS_FILE"
        else
            # 完全一致がない場合、同じ分（秒は無視）で検索
            TIMESTAMP_PREFIX=$(echo "$TIMESTAMP" | cut -d'-' -f1-5)  # YYYY-MM-DD--HH-MM
            SELFLOOPS_FILE=$(find "$PROJECT_ROOT/data/selfloops" -name "selfloops_${TIMESTAMP_PREFIX}-*.csv" -type f | head -1)

            if [ -n "$SELFLOOPS_FILE" ]; then
                echo "検出されたSelfloopsファイル（同一分）: $SELFLOOPS_FILE"
            else
                echo "Selfloopsファイルが見つかりませんでした（Muse心拍数を使用）"
            fi
        fi
    else
        echo "警告: Museファイル名からタイムスタンプを抽出できませんでした"
    fi
fi

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
# Selfloopsファイルがあれば渡す
if [ -n "$SELFLOOPS_FILE" ]; then
    python "$PROJECT_ROOT/scripts/generate_report.py" --data "$CSV_FILE" --output "$OUTPUT_DIR" --save-to "$SAVE_TO" --selfloops-data "$SELFLOOPS_FILE"
else
    python "$PROJECT_ROOT/scripts/generate_report.py" --data "$CSV_FILE" --output "$OUTPUT_DIR" --save-to "$SAVE_TO"
fi

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
