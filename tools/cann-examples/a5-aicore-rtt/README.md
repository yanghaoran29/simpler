# A5 AICore RTT

This standalone microbenchmark measures the full scheduler communication round trip for every runtime-visible A5
AICore:

```text
AICPU cntvct -> DATA_MAIN_BASE store -> AICore observes token
             -> AICore set_cond(token) -> AICPU observes COND -> AICPU cntvct
```

The launcher obtains the active cube and vector core counts with `aclrtGetStreamResLimit`, just as Simpler does. It
does not assume a fixed cluster count. Like Simpler, it selects the minimum of the reported cube count, half the
reported vector count, and the platform capacity. The shared ABI has capacity for up to 36 clusters, while loop
bounds, launch block count, record count, tables, and JSON all use that selected runtime result.

The AICPU affinity shape is supplied as `[S0, S1, S2, S3, orchestrator]` with `--allowed-cpus`. Scheduler threads S0
through S3 execute strictly in turn. During its turn, a scheduler visits every runtime-visible cluster in
cluster-major `AIC, AIV0, AIV1` order and takes 10 warmups followed immediately by 50 measured RTTs per logical
core. The sequence `S0, S1, S2, S3` repeats for five rounds. Samples for one core remain consecutive.

The timed loop contains only two counter reads, one volatile MMIO store, AICore polling plus `set_cond`, and AICPU
COND polling. Readiness publication, core mapping, barriers, validation, result writes, aggregation, sorting, and
logging are outside the loop. The result includes scheduler-side MMIO write/read costs and the hardware crossing,
but excludes production task construction, payload cache maintenance, kernel work, completion scanning across other
cores, and dispatch policy.

## Output

The launcher prints:

1. Each scheduler's per-cluster mean, with AIC and AIV shown separately. AIV is the arithmetic mean of both vector
   lanes and all five rounds.
2. Each scheduler's AIC, AIV, and all-core mean on die0 and die1.
3. A derived modulo-four assignment (`cluster % 4`) using the scheduler/core measurements from the same run.
4. A derived balanced contiguous assignment. Its boundaries are calculated from the runtime cluster count.
5. The raw JSON path. The JSON contains every RTT sample plus topology, placement, diagnostics, per-cluster means,
   per-die means, and both assignment estimates.

With `--plot <path>`, the launcher also invokes `host/plot_results.py`. It creates four Matplotlib subplots, one per
scheduler. Every logical AICore appears separately: circles are AIC, squares are AIV, red is die0, and blue is die1.
If Matplotlib is not installed, the renderer prints a warning and the benchmark still succeeds with its JSON output.

## Build

Set `ASCEND_HOME_PATH` to the active CANN toolkit, then build the three components:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
ROOT=$PWD

cmake -S tools/cann-examples/a5-aicore-rtt/device-aicore \
      -B tools/cann-examples/a5-aicore-rtt/device-aicore/build
cmake --build tools/cann-examples/a5-aicore-rtt/device-aicore/build -j

cmake -S tools/cann-examples/a5-aicore-rtt/device-aicpu \
      -B tools/cann-examples/a5-aicore-rtt/device-aicpu/build \
      -DCMAKE_C_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-gcc \
      -DCMAKE_CXX_COMPILER=${ASCEND_HOME_PATH}/tools/hcc/bin/aarch64-target-linux-gnu-g++
cmake --build tools/cann-examples/a5-aicore-rtt/device-aicpu/build -j

cmake -S tools/cann-examples/a5-aicore-rtt/host \
      -B tools/cann-examples/a5-aicore-rtt/host/build
cmake --build tools/cann-examples/a5-aicore-rtt/host/build -j
ctest --test-dir tools/cann-examples/a5-aicore-rtt/host/build --output-on-failure
```

The AICore build compiles the same source for `dav-c310-cube` and `dav-c310-vec`, then links both entry points into
one mixed relocatable object. The AICPU build must use CANN's AArch64 cross compiler.

## Run

Build the normal A5 runtime first so its dispatcher SO is available. Submit one card at a time through the repository
workflow:

```bash
export SIMPLER_DISPATCHER_SO=$ROOT/build/lib/a5/dispatcher/libsimpler_aicpu_dispatcher.so
export A5_RTT_CONSUMER_SO=$ROOT/tools/cann-examples/a5-aicore-rtt/device-aicpu/build/liba5_rtt_consumer.so
export A5_RTT_PRODUCER_O=$ROOT/tools/cann-examples/a5-aicore-rtt/device-aicore/build/a5_rtt_producer.o

task-submit --device 1 --run \
  "$ROOT/tools/cann-examples/a5-aicore-rtt/host/build/launch_a5_aicore_rtt \
     \$TASK_DEVICE --samples 50 --warmup 10 \
     --allowed-cpus 1,2,3,4,5 --json $ROOT/a5-aicore-rtt.json --plot $ROOT/a5-aicore-rtt.png"
```

`--plot` is optional. Exit status is nonzero if any core times out, returns fewer samples, or if scheduler-turn and
topology metadata are inconsistent. Plotting failure only emits a warning because the JSON remains the source data.

## Generate a packaged runtime assignment

This is an offline maintenance workflow. It runs the topology query and the full five-round RTT test, then writes a
JSON fragment for `src/a5/platform/onboard/host/aicpu_cpu_topo_fallback.json`:

```bash
task-submit --device 1 --run \
  "$ROOT/tools/cann-examples/a5-aicore-rtt/generate_runtime_assignment.sh \
     \$TASK_DEVICE $ROOT/outputs/a5-rtt-card1"
```

The generator ranks each scheduler by `die0_mean - die1_mean` independently in every round. It normally requires the
same two-plus-two scheduler partition in at least four of five rounds. If the near-die0 and near-die1 anchors are
stable in all rounds while only the middle pair swaps, a median gap of at most 5 ns is treated as indistinguishable;
the lower CPU ID is assigned to die0 as a deterministic tie-break. Run it on multiple cards of the same SoC and
require the generated `scheduler_cpu_order` to agree before merging the entry, committing it, and rebuilding Simpler.

Production launches only read the packaged JSON. They never run this benchmark and expose no RTT-preflight switch.
When an exact SoC/host/OCCUPY/scheduler-set entry is absent, the runtime uses balanced contiguous cluster ranges based
on the discovered core count without attempting to infer die affinity.
