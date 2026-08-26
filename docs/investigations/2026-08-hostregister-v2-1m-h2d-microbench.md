# HostRegisterV2 pinned host memory for 1 MiB sync H2D

**Date**: 2026-08-25
**Verdict**: dropped for bandwidth; register only when semantics require it or the same buffer is reused many times

## Question

After measuring `aclrtHostRegisterV2` on DeepSeek L3 (~45 GB per bind, parent-process
first-touch SHM), registered memory was **slower** and did not reduce per-run H2D jitter.
Does the same hold for a small, isolated 1 MiB buffer — the size class where pinning is
often assumed to help?

Intuition: lock pageable memory once, skip the driver's bounce buffer, and get stable
direct DMA. This microbench isolates that claim from HBG bind logic, NUMA placement of
multi-GB SHM, and multi-device contention.

## What was tried

### Benchmark script

`tools/bench_hostregister_v2_1m.py` — direct CANN ACL calls, no sim, no HBG.

```
mmap(MAP_SHARED) 1 MiB host buffer, memset fault-in
aclrtMalloc 1 MiB device buffer
for each trial (5 total, alternating order):
  pageable:   warmup 100 × aclrtMemcpy(H2D) + measure 2000 ×
  registered: aclrtHostRegisterV2(ptr, 1 MiB, RT_MEM_HOST_REGISTER_PINNED)
              warmup 100 × aclrtMemcpy(H2D) + measure 2000 ×
              aclrtHostUnregister(ptr)
```

- Device from `TASK_DEVICE` or `NPU_LOCKED_DEVICE` (first id if comma-separated).
- `h2d_ns` is **only** synchronous `aclrtMemcpy` latency; register/unregister are timed
  separately for amortization math.
- Trial order alternates P→V2 and V2→P to reduce ordering bias.
- One process run = 5 × 2,000 = 10,000 samples per mode.

### How to run

From the repository root, on a shared onboard host that uses `task-submit` for device
locking:

```bash
task-submit --timeout 1800 --max-time 1800 --device auto --device-num 1 \
  --run 'set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for run in 1 2 3; do
  echo "RUN=${run}"
  python3 tools/bench_hostregister_v2_1m.py
done' 2>&1 | tee hostregister_v2_1m.log
```

Single process (already holding a device):

```bash
cd "$(git rev-parse --show-toplevel)"
TASK_DEVICE=0 python3 tools/bench_hostregister_v2_1m.py
```

Requirements:

- `libascendcl.so` on `LD_LIBRARY_PATH` (normal CANN onboard environment).
- Python 3 stdlib only; no venv or torch needed.
- Real a2a3 hardware; do not run on sim.

Each stdout line is one JSON object. Key fields:

| Field | Meaning |
|-------|---------|
| `pageable` / `registered` | Aggregated H2D latency stats (`mean_us`, `p50_us`, `cv_pct`, …) |
| `register` / `unregister` | One-shot register/unregister cost per trial |
| `mean_latency_change_pct` | Registered mean vs pageable mean |
| `break_even_copies` | Copies needed to amortize register+unregister if registered is faster; `null` if never |

### Recorded run (2026-08-25)

- Platform: a2a3 onboard, device 13, `task-submit` exclusive lock.
- Three independent processes (each 10,000 samples/mode).

## Result

### Steady-state H2D (p50 / mean)

| Metric | Pageable | HostRegisterV2 |
|--------|----------|----------------|
| Mean latency (3-run avg) | 106.68 µs | 107.13 µs (+0.42%) |
| p50 latency (3-run avg) | 101.98 µs | 102.02 µs (+0.04%) |
| Implied p50 bandwidth | ~10.28 GB/s | ~10.28 GB/s |

Per-process mean change: **+3.22%**, **−1.41%**, **−0.78%** — no consistent speedup.

### Register lifecycle cost

| Call | p50 |
|------|-----|
| `aclrtHostRegisterV2` | ~22.4 µs |
| `aclrtHostUnregister` | ~15.2 µs |
| Combined | ~37.6 µs |

For a single copy, register+unregister adds ~37% on top of ~102 µs H2D — not worthwhile.

`break_even_copies` was **null** in one run (registered slower); in the other two,
~57 and ~85 reuses were needed to amortize registration.

### Tail latency (CV)

Registration reduced coefficient of variation in all three runs, driven by fewer 2–3 ms
outliers; p95 did not improve in every run.

| Process | Pageable CV | V2 CV |
|---------|-------------|-------|
| 1 | 36.78% | 25.01% |
| 2 | 35.25% | 7.77% |
| 3 | 25.07% | 8.09% |

### Relation to DeepSeek L3 measurement

The same session also ran DeepSeek L3 (3 independent launches × 11 rounds, drop cold
round per launch, 30 steady samples per mode, devices 1 and 5):

| Metric | Pageable | HostRegisterV2 |
|--------|----------|----------------|
| Per-launch normalized CV (avg) | 6.03% | 5.93% |
| Critical-path H2D p50 | 2.401 s | 3.632 s |
| Aggregate bandwidth p50 | 38.16 GB/s | 25.23 GB/s |

At 45 GB/bind, V2 was slower because registration **locks existing NUMA placement** from
parent first-touch SHM; pageable bounce may land buffers on a nearer node. The 1 MiB
microbench removes that layout but still shows **no median bandwidth win**.

## Why not (now)

- **p50 / mean**: registering pageable memory does not improve 1 MiB sync H2D throughput
  on the measured a2a3 box.
- **Single or few copies**: register+unregister cost dominates; break-even is dozens to
  hundreds of reuses even when registered is slightly faster.
- **DeepSeek-scale args**: large-buffer regression is worse than microbench neutrality;
  pinning without NUMA-aware first-touch can permanently lock suboptimal pages.
- **Jitter inside one worker**: DeepSeek per-launch CV barely moved (6.03% → 5.93%); the
  pooled cross-launch CV drop was not statistically confirmed.

HostRegisterV2 remains useful for **semantics** (stable DMA target, async copy eligibility),
not as a default performance knob.

## When to reconsider

- Host memory is first-touched on the NUMA node local to the target device **before**
  `aclrtHostRegisterV2`, and the buffer is reused across many H2D copies in the same
  process lifetime.
- Workload is latency-tail sensitive (p99 / max) on small repeated copies and can absorb
  one-time register cost.
- An HBG L3 case with only 1 MiB tensors exercised through the real child SHM register
  path — this microbench does not substitute for that integration test.

## References

- Benchmark: `tools/bench_hostregister_v2_1m.py`
- Related platform API: `src/a2a3/platform/onboard/host/device_runner.cpp` (`halHostRegister`)
- DeepSeek L3 jitter session logs (local, not in repo): `deepseek_l3_jitter_full_*.log`
