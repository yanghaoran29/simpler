# BGEMM Case0：AUTO Die-affine group实验

## 结论

最终实现使用`AUTO_DIE_AFFINE`而非`DIE_AFFINE`：每个独立group在两个Die间轮转，group内两个GEMM
及对同一C_view的两个串行ADD固定在同一Die，同时保留TensorMap自动生成的ADD依赖。

卡2四级泳道中GEMM和ADD均严格按Die0/Die1=`250/250`分配；卡1关闭泳道100轮Device平均
`1,419.827 us`，相对历史Main `1,458.300 us`改善`2.64%`，相对旧Mode9 `1,642.829 us`
改善`13.57%`。

| 指标 | 历史Main | 当前 | 变化 |
|---|---:|---:|---:|
| Device | 1,458.300 us | 1,419.827 us | -2.64% |
| Effective | 1,427.400 us | 1,390.049 us | -2.62% |
| Orchestrator | 1,336.800 us | 1,302.228 us | -2.59% |
| Scheduler | 1,421.750 us | 1,383.106 us | -2.72% |
| Device p50 | — | 1,404.500 us | — |
| Device p95 | — | 1,544.400 us | — |

## 泳道任务分布

| Kernel | Die0 | Die1 |
|---|---:|---:|
| GEMM | 250 | 250 |
| ADD | 250 | 250 |

| Scheduler | GEMM | ADD | 合计 |
|---|---:|---:|---:|
| S0 | 115 | 111 | 226 |
| S1 | 135 | 139 | 274 |
| S2 | 120 | 120 | 240 |
| S3 | 130 | 130 | 260 |

## 正确性说明

`DIE_AFFINE`采用MANUAL依赖语义，若直接替换原BGEMM的AUTO scope，会关闭TensorMap对两个ADD的
自动串行化并造成C_view写竞争。提交前golden检查发现并修复了这一问题。最终代码使用
`AUTO_DIE_AFFINE`，golden Case0通过；提交中不包含错误的MANUAL语义版本。

## 文件

- `swimlane/merged_swimlane.json`：最终正确版本的4级泳道；
- `swimlane/chip_swimlane_records.json`：原始泳道记录；
- `task.log`：泳道运行日志；
- 完整100轮日志位于`outputs/mode9-current-card1-100r-qwen5-20260826/`。
