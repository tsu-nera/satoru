#!/bin/bash
#
# GitHub Actions の Meditation Analysis をリモート起動するスクリプト
#
# ローカルに Docker や venv を用意せず、GitHub ランナー上で分析を実行する。
# 事前に gh CLI の認証が必要（gh auth login）。
#
# Usage: ./trigger_analysis.sh [DATE] [--watch]
#
# Arguments:
#   DATE     : 分析する日付（例: 2025-11-04）省略時は latest（最新ファイル）
#   --watch  : 起動後、実行完了までログをライブ表示する（オプション）
#
# 例:
#   ./trigger_analysis.sh                     # 最新ファイルで起動
#   ./trigger_analysis.sh 2025-11-04          # 日付指定で起動
#   ./trigger_analysis.sh --watch             # 起動して完了まで見張る
#   ./trigger_analysis.sh 2025-11-04 --watch  # 日付指定＋見張り
#

set -e

WORKFLOW="meditation_analysis.yml"

# gh CLI の存在チェック
if ! command -v gh >/dev/null 2>&1; then
    echo "エラー: gh CLI が見つかりません" >&2
    echo "  インストール: https://cli.github.com/" >&2
    exit 1
fi

# 認証チェック
if ! gh auth status >/dev/null 2>&1; then
    echo "エラー: gh が未認証です。以下を実行してください:" >&2
    echo "  gh auth login" >&2
    exit 1
fi

# 引数パース（DATE と --watch を順不同で受け取る）
DATE="latest"
WATCH=false
for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=true ;;
        *)       DATE="$arg" ;;
    esac
done

echo "============================================================"
echo "Meditation Analysis をリモート起動します"
echo "============================================================"
echo "  日付: $DATE"
echo ""

gh workflow run "$WORKFLOW" -f date="$DATE"

# run が登録されるまで少し待つ
sleep 3

RUN_ID=$(gh run list --workflow="$WORKFLOW" --limit 1 --json databaseId --jq '.[0].databaseId')
RUN_URL=$(gh run list --workflow="$WORKFLOW" --limit 1 --json url --jq '.[0].url')

echo "✅ 起動しました"
echo "  Run: $RUN_URL"
echo ""

if [ "$WATCH" = true ]; then
    echo "完了まで見張ります（Ctrl-C で中断）..."
    gh run watch "$RUN_ID"
else
    echo "進行を確認するには:"
    echo "  gh run watch $RUN_ID"
    echo "  gh run view $RUN_ID --log"
fi
