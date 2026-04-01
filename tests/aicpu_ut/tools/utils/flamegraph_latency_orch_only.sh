#!/usr/bin/env bash
# 使用 FlameGraph 为 test_latency_orch_only_1 生成 CPU 火焰图
#
# 用法：
#   bash tools/flamegraph_latency_orch_only.sh              # 单次运行采集（样本较少）
#   REPEAT=100 bash tools/flamegraph_latency_orch_only.sh   # 重复 100 次以增加采样
#
# 依赖：perf，FlameGraph 位于项目根目录 FlameGraph/
# 输出：outputs/flamegraph/test_latency_orch_only_1.perf, .folded, .svg

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_DIR="$SCRIPT_DIR"
while [[ "$SEARCH_DIR" != "/" && ! -f "$SEARCH_DIR/run_tests.sh" ]]; do
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done
AICPU_UT_DIR="$SEARCH_DIR"
PROJECT_ROOT="$(cd "${AICPU_UT_DIR}/../.." && pwd)"
BIN="${AICPU_UT_DIR}/build/bin/test_latency_orch_only_1"
OUT_DIR="${AICPU_UT_DIR}/outputs/flamegraph"
FLAMEGRAPH_DIR="${PROJECT_ROOT}"
REPEAT=${REPEAT:-1}

mkdir -p "$OUT_DIR"
cd "$AICPU_UT_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "Error: binary not found or not executable: $BIN" >&2
    exit 1
fi
if [[ ! -f "${FLAMEGRAPH_DIR}/FlameGraph/stackcollapse-perf.pl" ]]; then
    echo "Error: FlameGraph not found at ${FLAMEGRAPH_DIR}/FlameGraph/" >&2
    exit 1
fi

echo "Recording with perf (REPEAT=${REPEAT})..."
if [[ "$REPEAT" -eq 1 ]]; then
    perf record -F 99 -g -- "$BIN" 2>&1
else
    perf record -F 99 -g -- bash -c "for i in \$(seq 1 $REPEAT); do '$BIN'; done" 2>&1
fi

echo "Exporting perf script..."
perf script > "${OUT_DIR}/test_latency_orch_only_1.perf"

echo "Folding stacks..."
"${FLAMEGRAPH_DIR}/FlameGraph/stackcollapse-perf.pl" "${OUT_DIR}/test_latency_orch_only_1.perf" \
    > "${OUT_DIR}/test_latency_orch_only_1.folded"

echo "Generating flame graph SVG..."
"${FLAMEGRAPH_DIR}/FlameGraph/flamegraph.pl" \
    --title "test_latency_orch_only_1 (CPU Flame Graph)" \
    "${OUT_DIR}/test_latency_orch_only_1.folded" \
    > "${OUT_DIR}/test_latency_orch_only_1.svg"

echo "Done. Output: ${OUT_DIR}/test_latency_orch_only_1.svg"
echo "  (perf data: ${OUT_DIR}/test_latency_orch_only_1.perf, folded: test_latency_orch_only_1.folded)"
