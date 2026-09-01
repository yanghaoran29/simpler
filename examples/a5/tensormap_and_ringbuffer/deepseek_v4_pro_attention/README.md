# DeepSeek-V4 Pro attention benchmark

This is a self-contained A5 `tensormap_and_ringbuffer` benchmark for the six
whole-attention programs in DeepSeek-V4 Pro:

- decode: SWA, CSA, HCA at batch 4 × 2 token rows and `start_pos=8192`;
- prefill: SWA, CSA, HCA at one 128-token chunk and `start_pos=0` (pypto-lib
  default / image alignment; not the correctness worst-case `896`).

Each case runs the harvested orchestration and all of its generated AIC/AIV
kernels directly through simpler. It does not import PyPTO or pypto-lib at test
time. The fixture uses zero weights plus valid, in-range position/cache
metadata because this is an orchestration/scheduler performance guard; numeric
correctness remains covered by the source pypto-lib programs. Each case sets
`runtime_env.ring_heap` to 4 GiB (with matching task/dep windows) so prefill
graphs match pypto-lib's `PREFILL_ATTN_RING_HEAP` and do not hit
`HEAP_RING_DEADLOCK`.

## Reference measurements

These are the source measurements that motivated the guard. They are recorded
for context, not enforced as absolute thresholds by this repository.

| Program | Layers | recipes | pypto-lib | Relative perf |
| --- | ---: | ---: | ---: | ---: |
| Decode SWA | 2 | 180.4 us/layer | 346.9 us/layer | 52.0% |
| Decode CSA | 21 | 274.2 us/layer | 476.3 us/layer | 57.6% |
| Decode HCA | 20 | 197.1 us/layer | 382.8 us/layer | 51.5% |
| Prefill SWA | 2 | 0.683 us/token | 1.350 us/token | 50.6% |
| Prefill CSA | 21 | 1.092 us/token | 10.771 us/token | 10.1% |
| Prefill HCA | 20 | 0.717 us/token | 1.419 us/token | 50.5% |

## Provenance

| Component | Revision |
| --- | --- |
| pypto-lib source | `d51480c` |
| PyPTO generator | `62d75b4475fa6ada399c2712514278c8a4871ac3` |
| PyPTO runtime submodule | `dbdd041e957420ea15b03e878400dd4de5e9c34c` |
| PTOAS | `v0.60` |
| PTO ISA | `be5ccb765a4ce5d14ca5da8b0e2f182d7f003369` |

`62d75b44` is the first PyPTO revision used here that supports PTOAS v0.60
`tsort32` codegen for the CSA indexer. The earlier same-day generator revision
fails that one function before producing a complete six-program harvest.

The contents under `kernels/vendor/` are copied verbatim from codegen:
`kernel_config.py`, `compiled_meta.json`, orchestration C++, and the AIC/AIV
C++. `fixture_meta.json` records the concrete tensor shapes and scalar values
used for specialization. Pass dumps, `.pto` files, reports, and binaries are
deliberately excluded.

## Run

On a shared A5 host, wrap onboard work in `task-submit`. For **pypto-lib**
performance alignment, use the two-round protocol (both required):

1. **第一轮 — 正常跑法**（不采泳道图）:

```bash
python examples/a5/tensormap_and_ringbuffer/deepseek_v4_pro_attention/test_deepseek_v4_pro_attention.py \
  -p a5 -d "$DEVICE" --case DecodeCSA --manual include --rounds 1 --skip-golden
```

2. **第二轮 — benchmark 模式并采集泳道图**（对标指标取本轮）:

```bash
python examples/a5/tensormap_and_ringbuffer/deepseek_v4_pro_attention/test_deepseek_v4_pro_attention.py \
  -p a5 -d "$DEVICE" --case DecodeCSA --manual include --rounds 1 --skip-golden \
  --enable-chip-swimlane 4
```

Compare the **第二轮** swimlane AICore task window
(`max(aicore_tasks.end) - min(aicore_tasks.start)`) to the **pypto-lib** column
above. Round-1 STRACE is diagnostic only.

Suite registration still uses five precise rounds via `benchmark_rounds.sh`
for runtime regression tracking; that path is separate from the two-round
swimlane alignment against pypto-lib.
