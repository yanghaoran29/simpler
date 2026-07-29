# 上板实测评论 — F2

> 私仓分支备注（不开 PR）。对照基线：`pr906-main-baseline` / B_main（main + OCCUPY topo fallback）。
> 主测：`paged_attention_unroll` Case1，device 0，STRACE Orch/Sched via `tools/benchmark_a5_case1.sh`（100 rounds × 3）。

## 改动
mailbox poll 门控：`thread_idx >= 2`（仅 sched 线程轮询 aicore mailbox）。

## 实测 vs B_main

| | Orch | Sched | Device | Orch Δ% | Sched Δ% |
|--|------|-------|--------|---------|----------|
| F2 | 1957.2 | 2100.9 | 2139.6 | **−0.57%** | −0.66% |

## 结论
收益在噪声内，**不显著**；可择机合入，非 headline。本分支仅作消融存档，**不提上游 PR**。

## 基线 B_main（对照）

| | Orch (us) | Sched (us) | Device (us) |
|--|-----------|------------|-------------|
| mean | 1968.4 | 2114.9 | 2152.7 |

完整消融表见同工作树 `outputs/pr906_ablation/REPORT.md`。推荐合入的是 **F3**（已开上游 PR）。
