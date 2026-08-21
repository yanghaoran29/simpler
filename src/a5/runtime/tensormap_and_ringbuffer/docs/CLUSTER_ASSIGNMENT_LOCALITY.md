# A5 Scheduler Cluster-Assignment Locality

**Date:** 2026-08-20
**Status:** preliminary improvement; retain the branch for broader validation

## Question

The A5 TMR runtime originally assigned cluster `ci` to scheduler
`ci % scheduler_count`. On the tested A5 device, one runtime launch discovered
28 clusters across two dies and used four scheduler threads. Round-robin
ownership therefore made every scheduler communicate with clusters on both
dies.

This experiment asks whether balanced contiguous ownership reduces scheduler
communication cost and the effective latency of a multi-kernel decode
workload.

## Assignment Policies

Each policy assigns seven clusters to every scheduler, so the comparison does
not change the number of clusters scanned by a thread.

| Policy | S0 | S1 | S2 | S3 |
| ------ | -- | -- | -- | -- |
| Round-robin | 0, 4, 8, 12, 16, 20, 24 | 1, 5, 9, 13, 17, 21, 25 | 2, 6, 10, 14, 18, 22, 26 | 3, 7, 11, 15, 19, 23, 27 |
| Contiguous | 0–6 | 7–13 | 14–20 | 21–27 |
| Upper-tail isolated | 0, 3, 6, 9, 12, 15, 18 | 1, 4, 7, 10, 13, 16, 19 | 2, 5, 8, 11, 14, 17, 20 | 21–27 |

The production implementation is not hard-coded to 28 clusters. For `N`
detected clusters and `S` scheduler threads, scheduler `t` owns
`[floor(N*t/S), floor(N*(t+1)/S))`. Both the barrier-free
`assign_own_clusters()` path and the serial `assign_cores_to_threads()` path
use the same partition.

## AICPU-to-AICore Communication Probe

The standalone probe made each scheduler communicate with every logical core
it owned 100 consecutive times after 10 warm-up operations. It tested AIC,
AIV0, and AIV1 for all 28 clusters in forward and reverse traversal order,
giving 168 records per policy. Both policies ran on device 1 through
`task-submit`.

The probe classified clusters 0–13 as die 0 and clusters 14–27 as die 1. The
four schedulers ran on AICPU CPU IDs 1–4.

| Scheduler | CPU | Round-robin die records | Round-robin mean p50 | Contiguous die records | Contiguous mean p50 |
| --------- | --: | ----------------------: | -------------------: | ---------------------: | ------------------: |
| S0 | 1 | 24 / 18 | 0.4823 µs | 42 / 0 | 0.3615 µs |
| S1 | 2 | 24 / 18 | 0.4283 µs | 42 / 0 | 0.3758 µs |
| S2 | 3 | 18 / 24 | 0.4729 µs | 0 / 42 | 0.5294 µs |
| S3 | 4 | 18 / 24 | 0.4841 µs | 0 / 42 | 0.3873 µs |

The die-record columns are `die 0 / die 1` counts. Aggregating all records,
contiguous ownership reduced the mean of per-record p50 latency from
0.4669 µs to 0.4135 µs, an 11.4% reduction. However, the maximum per-scheduler
sum of the 100-operation windows increased from 2190.181 µs to 2399.466 µs,
or 9.6%, because S2 became slower when all of its records came from die 1.

This probe therefore shows asymmetric scheduler-to-die access, but it does
not predict the workload result by itself: the aggregate improves while its
concurrent makespan proxy regresses.

| Policy | Task |
| ------ | ---- |
| Round-robin | `task_20260820_225012_219589413906` |
| Contiguous | `task_20260820_225032_219738314302` |

## `decode_attention_csa` Workload

The workload comparison used:

- Simpler base `1f27a157828d021638531fe33fe194fb2632f4ff`;
- PyPTO-Lib `2cd5f828bd3a54e2d073a7950b5fcfc0806b8bae`;
- PTO-ISA pin `f51c92f610827daad0ddfb383072e03d514b4ae9`;
- PTOAS v0.57;
- `models/deepseek_v4_pro/decode_attention_csa.py` on A5 device 1;
- `tensormap_and_ringbuffer` runtime;
- `PYPTO_BENCH=1`, with 5 warm-up rounds and 100 measured rounds.

All runs used `task-submit`, ran sequentially on the same device, and passed
the complete output validation. Only scheduler cluster ownership changed.
`effective_us` is the merged device-domain orchestrator/scheduler window.

| Order | Policy | Min | Median | Mean | Max | Task |
| ----: | ------ | --: | -----: | ---: | --: | ---- |
| 1 | Round-robin A1 | 710.9 µs | 746.0 µs | 744.2 µs | 779.5 µs | `task_20260820_230401_224764124330` |
| 2 | Contiguous B1 | 689.8 µs | 730.6 µs | 731.6 µs | 791.2 µs | `task_20260820_230706_22626969437` |
| 3 | Round-robin A2 | 690.8 µs | 741.9 µs | 743.6 µs | 788.6 µs | `task_20260820_230814_22735073166` |
| 4 | Upper-tail isolated | 704.6 µs | 743.2 µs | 742.1 µs | 809.4 µs | `task_20260820_231801_229950617243` |
| 5 | Contiguous B2 | 701.4 µs | 737.3 µs | 737.2 µs | 780.1 µs | `task_20260820_232133_231560811342` |

The two repeated policies aggregate as follows:

| Policy | Runs | Average median | Average mean | Median delta vs. round-robin | Mean delta vs. round-robin |
| ------ | ---: | -------------: | -----------: | ---------------------------: | -------------------------: |
| Round-robin | 2 | 744.0 µs | 743.9 µs | — | — |
| Contiguous | 2 | 734.0 µs | 734.4 µs | -10.0 µs (-1.34%) | -9.5 µs (-1.28%) |

The round-robin average also agrees with the previously recorded 743.2 µs
document value. The two contiguous runs both improved, but their individual
median reductions were 13.4 µs and 6.7 µs. The upper-tail-isolated policy was
within 0.8 µs of the round-robin median despite retaining the same seven
clusters per scheduler.

## Interpretation and Decision

The upper-tail-isolated result rules out cluster-count balance as the source
of the observed improvement. It is consistent with the idea that leaving S0,
S1, and S2 interleaved across both dies retains most of the communication
penalty, while fully contiguous ownership removes more remote accesses.

The evidence is not yet strong enough to attribute the full operator change
to NoC locality:

- the two contiguous operator measurements differ by 6.7 µs;
- the average operator improvement is only 1.3%, inside the repository's
  ±2% benchmark noise band;
- the communication probe's worst-scheduler window moves in the opposite
  direction from the operator result;
- only one A5 device and one production workload were used for this policy
  comparison.

The contiguous implementation is retained on branch
`perf/a5-contiguous-cluster-assignment` as a candidate, not as a generally
proven optimization. Before merging, run an interleaved multi-pair A/B test on
the same device, add other multi-task kernels, and record per-scheduler phase
timings. A scheduler-to-die affinity map would also distinguish a true NoC
locality effect from other thread-specific latency differences.

## References

- Implementation commit: `5d8b103bd4b462a9e81bd7cc6b55eb817966a137`
- Scheduler implementation:
  `runtime/scheduler/scheduler_cold_path.cpp`
- A5 MMIO constraints: [`docs/hardware/mmio-performance.md`](../../../../../docs/hardware/mmio-performance.md)
