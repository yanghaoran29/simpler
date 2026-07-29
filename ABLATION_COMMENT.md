# 上板实测评论 — F9

> 私仓分支备注（不开 PR）。对照基线：`pr906-main-baseline` / B_main（main + OCCUPY topo fallback）。
> 主测：`paged_attention_unroll` Case1，device 0，STRACE Orch/Sched via `tools/benchmark_a5_case1.sh`（100 rounds × 3）。

## 改动
two-pass orch：先预发射全部 QK，再 SF/PV/UP。

## 实测 vs B_main

| | Orch | Sched | Device | Orch Δ% | Sched Δ% |
|--|------|-------|--------|---------|----------|
| F9 | 2123.4 | 2150.5 | 2188.2 | **+7.87%** | +1.68% |

## 结论
orch **明显回归**，**不要单独合入**。本分支仅作消融存档，**不提上游 PR**。

## 基线 B_main（对照）

| | Orch (us) | Sched (us) | Device (us) |
|--|-----------|------------|-------------|
| mean | 1968.4 | 2114.9 | 2152.7 |

完整消融表见同工作树 `outputs/pr906_ablation/REPORT.md`。推荐合入的是 **F3**（已开上游 PR）。
