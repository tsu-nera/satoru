#!/bin/bash
#
# Lint / 型チェック / テストをまとめて実行する
#
# Usage:
#   bash scripts/check.sh          # 全部実行
#   bash scripts/check.sh lint     # ruff のみ
#   bash scripts/check.sh types    # mypy のみ
#   bash scripts/check.sh test     # pytest のみ
#   bash scripts/check.sh fix      # ruff の自動修正を適用
#
# 全ステージを実行し、途中で失敗しても最後まで回してから
# まとめて失敗を報告する（1回の実行で全ての問題を把握できるようにする）。

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

TARGET="${1:-all}"
FAILED=()

run_stage() {
    local name="$1"; shift
    echo ""
    echo "============================================================"
    echo "  $name"
    echo "============================================================"
    if "$@"; then
        echo "✓ $name OK"
    else
        echo "✗ $name FAILED"
        FAILED+=("$name")
    fi
}

case "$TARGET" in
    fix)
        exec uv run ruff check . --fix
        ;;
    lint)
        run_stage "ruff (lint)" uv run ruff check .
        ;;
    types)
        run_stage "mypy (型チェック)" uv run mypy
        ;;
    test)
        run_stage "pytest" uv run python -m pytest
        ;;
    all)
        run_stage "ruff (lint)" uv run ruff check .
        run_stage "mypy (型チェック)" uv run mypy
        run_stage "pytest" uv run python -m pytest
        ;;
    *)
        echo "❌ 不明な引数: $TARGET (使えるのは all|lint|types|test|fix)"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✅ すべて通過"
    exit 0
else
    echo "❌ 失敗: ${FAILED[*]}"
    exit 1
fi
