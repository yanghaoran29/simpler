#!/usr/bin/env bash
# 整理 run_tests.sh --profiling 的输出为参考格式（不修改 main 脚本，仅后处理）。
# 用法（推荐：直接得到格式化后的命令行输出）:
#   bash run_tests.sh --profiling 2>&1 | bash tools/format_profiling_output.sh
# 或先跑测试保存再整理:
#   bash run_tests.sh --profiling 2>&1 | tee raw.log
#   bash tools/format_profiling_output.sh < raw.log
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/format_profiling_output.py" "$@"
