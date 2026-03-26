# tools 分类说明

已将 `~/Desktop/simpler/tools` 并入 `tests/aicpu_ut/tools`，并按文件内容分类如下：

- `benchmark/`：参数扫描与压测入口（`sweep_latency.sh`、`sweep_throughput.sh`）
- `profiling/`：profiling 数据采集、格式化、汇总分析与可视化转换
- `analysis/`：参数组合校验、日志区间统计等分析辅助脚本
- `utils/`：perf/flamegraph 等通用辅助脚本
- `dev/`：开发辅助脚本（编译数据库、测试目录发现）

本次重组后，脚本路径推导已统一修复为“从当前脚本向上自动定位工程根或 `run_tests.sh`”，避免目录层级变化导致路径失效。
