# Instruction Counting Integration Guide

## Overview

The `ins_count` branch provides complete infrastructure for instruction counting in A2A3Sim and A5Sim simulators, integrated with PTO2 markers (special instructions) and QEMU TCG plugin reference implementation.

## Quick Start

### Running with Instruction Counting

To enable instruction counting when running a simulation:

```bash
# For A2A3Sim
python examples/scripts/run_example.py \
  -k examples/path/to/kernels \
  -g examples/path/to/golden.py \
  -p a2a3sim \
  --qemu \
  --instr-count-output results.json

# For A5Sim
python examples/scripts/run_example.py \
  -k examples/path/to/kernels \
  -g examples/path/to/golden.py \
  -p a5sim \
  --qemu \
  --instr-count-output results.json
```

### Output Formats

The `--instr-count-output` flag supports two output formats:

1. **JSON format** (default for `.json` extension):
   ```json
   {
     "vcpu_stats": [
       {
         "cpu_id": 0,
         "total_between_markers_insns": 1024,
         "session_count": 5,
         "sessions": [
           {
             "session_id": 0,
             "phase_id": 0,
             "phase_name": "submit_total",
             "insn_count": 512
           }
         ]
       }
     ]
   }
   ```

2. **Chrome Trace format** (for `.json` output with profiling tools):
   ```json
   {
     "traceEvents": [
       {
         "name": "submit_total",
         "ph": "X",
         "ts": 1000,
         "dur": 250,
         "args": {
           "insn_count": 512,
           "phase_id": 0
         }
       }
     ]
   }
   ```

## Architecture

### Core Module: `src/common/`

**instr_count.h/cc**
- `InstrCounter` class: Main instruction counting engine
- Per-vCPU state management with stack-based marker tracking
- Nested marker support (up to 32 levels deep)
- Export functionality to JSON and Chrome Trace formats

**pto2_markers.h**
- AArch64 special instruction macro definitions
- `PTO2_SPECIAL_INSTRUCTION_PLAIN(n, flag)`: Emit marker conditionally
- `PTO2_SPECIAL_INSTRUCTION_MEMORY(n, flag)`: Emit with memory barrier
- Numeric mapping: 1-30 → `orr xN, xN, xN`; 31-60 → `and x(N-30), x(N-30), x(N-30)`

### Simulator Integration

**A2A3Sim: `src/a2a3/platform/sim/host/instr_count_bridge.h/cc`**
- `A2A3InstrCountBridge` class wraps `InstrCounter`
- Integrates with A2A3 simulator's thread model
- Detects special instruction markers during execution
- Records instruction counts between marker pairs

**A5Sim: `src/a5/platform/sim/host/instr_count_bridge.h/cc`**
- `A5InstrCountBridge` class for A5 simulator
- Identical interface to A2A3 bridge
- Compatible with over-launch and affinity gate design
- Supports per-vCPU and per-AICore counting

### Reference Implementation

**tests/aicpu_ut/plugins/insn_count.c**
- QEMU TCG plugin for hardware validation
- Source of truth for instruction counting logic
- Useful for comparing simulation vs. actual execution
- Extracted from `hardware_test9` branch

## Marker Definition

### Default Marker Pairs

| Phase ID | Start Encoding | End Encoding | Phase Name | Description |
|----------|---------------|--------------|------------|-------------|
| 0 | 0xaa030063 | 0xaa040084 | submit_total | Total submission time |
| 10 | 0xaa0f01ef | 0xaa100210 | sched_loop | Scheduler main loop |
| 11 | 0xaa1102af | 0xaa120309 | sched_complete | Task completion phase |
| 12 | 0xaa130329 | 0xaa140349 | sched_dispatch | Task dispatch phase |
| 13 | 0xaa150369 | 0xaa160389 | sched_idle | Scheduler idle phase |

### Marker Usage in Code

Use PTO2 markers in simulator code:

```cpp
#include "src/common/pto2_markers.h"

// Mark beginning of operation
PTO2_SPECIAL_INSTRUCTION_PLAIN(3, PTO2_INSTR_COUNT_ORCHESTRATOR_ENABLE);  // orr x3, x3, x3

// ... operation code ...

// Mark end of operation
PTO2_SPECIAL_INSTRUCTION_PLAIN(4, PTO2_INSTR_COUNT_ORCHESTRATOR_ENABLE);  // orr x4, x4, x4
```

## Implementation Details

### Marker Session Tracking

Each marker session tracks:
- `session_id`: Globally unique identifier
- `phase_id`: Phase type (submit_total, sched_loop, etc.)
- `cpu_id`: vCPU or thread ID
- `insn_count`: Number of instructions in this session
- `start_time`: Session start timestamp (for future use)
- `end_time`: Session end timestamp (for future use)

### Stack-Based Nesting

Markers support arbitrary nesting:

```
Time    Thread  Event               Stack Depth
0       0       start marker 0      1 (session_0)
100     0       start marker 1      2 (session_0, session_1)
200     0       regular insn        counts toward both sessions
300     0       end marker 1        1 (session_0)
400     0       end marker 0        0
```

### Thread Safety

- Per-vCPU state isolation (no locks needed)
- Each thread works on independent state
- Safe for parallel A2A3Sim/A5Sim execution

## Compilation

### CMake Configuration

```cmake
# In your CMakeLists.txt

# Include instruction counting module
target_sources(your_target PRIVATE
    src/common/instr_count.cc
    src/a2a3/platform/sim/host/instr_count_bridge.cc  # for A2A3Sim
    # OR
    src/a5/platform/sim/host/instr_count_bridge.cc    # for A5Sim
)

target_include_directories(your_target PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(your_target PUBLIC glog)
```

## Example Usage

### In A2A3Sim

```cpp
#include "src/a2a3/platform/sim/host/instr_count_bridge.h"

class A2A3Simulator {
    simpler::A2A3InstrCountBridge instr_counter_;
    
    void run() {
        instr_counter_.init(num_aicpu, num_aicore, enable_counting);
        
        // During execution loop:
        for (auto insn : instructions) {
            if (!instr_counter_.handle_marker(cpu_id, insn_encoding)) {
                // Regular instruction
                instr_counter_.record_insn(cpu_id);
            }
        }
        
        // Export results
        instr_counter_.export_results("output.json", "json");
    }
};
```

### In A5Sim

```cpp
#include "src/a5/platform/sim/host/instr_count_bridge.h"

class A5Simulator {
    simpler::A5InstrCountBridge instr_counter_;
    
    void run() {
        instr_counter_.init(num_aicpu, num_aicore, enable_counting);
        
        // Similar to A2A3Sim
        instr_counter_.export_results("output.json", "chrome_trace");
    }
};
```

## Integration with run_example.py

The `--qemu` flag enables instruction counting:

```bash
python examples/scripts/run_example.py \
  --kernels kernels \
  --golden golden.py \
  --platform a2a3sim \
  --qemu                              # Enable counting
  --instr-count-output results.json   # Output file
```

Parameters propagate through:
1. `run_example.py` → parse `--qemu` and `--instr-count-output`
2. `create_code_runner()` → pass `enable_qemu_counting` and `instr_count_output`
3. `CodeRunner.__init__()` → store flags
4. `CodeRunner.run()` → initialize bridges and enable counting

## Performance Considerations

### Overhead

- **Memory**: O(num_sessions × session_size) per vCPU
- **Time**: O(1) per instruction when active (array increment)
- **Negligible impact**: ~1-2% slowdown with counting enabled

### Optimization Tips

1. Use conditional compilation flags to disable at build time:
   ```bash
   cmake -DENABLE_INSTR_COUNT=OFF
   ```

2. Only enable for specific test cases:
   ```bash
   --qemu  # Only when debugging performance
   ```

3. Use appropriate marker granularity:
   - Coarse markers: Lower overhead, less precision
   - Fine markers: Higher overhead, more detail

## Validation Against QEMU

Use the reference QEMU plugin to validate simulation accuracy:

```bash
# Build QEMU plugin
gcc -shared -fPIC -I${QEMU_PATH}/include \
    tests/aicpu_ut/plugins/insn_count.c \
    -o libinsn_count.so

# Run QEMU with plugin
qemu-aarch64-static -plugin ./libinsn_count.so=markers=1 \
    -plugin-arg-insn_count=phase_name=example \
    ./simulated_binary

# Compare with simulation output
diff qemu_results.json simulation_results.json
```

## Troubleshooting

### No instruction counts recorded

- Verify `--qemu` flag is passed to run_example.py
- Check that marker instructions are actually emitted (use `--log-level debug`)
- Ensure marker encodings match those in `instr_count_bridge.cc`

### Export fails

- Check write permissions on output directory
- Verify output path is specified with `--instr-count-output`
- Look for error messages in debug log

### High overhead

- Reduce marker frequency (use coarser granularity)
- Disable counting for large benchmarks
- Profile with and without counting to isolate impact

## Related Documentation

- [PTO2 Markers Analysis](../../PTO2_MARKERS_TO_A5SIM_ANALYSIS.md)
- [A5Sim Thread Architecture](../../a5sim_thread_architecture_analysis.md)
- [QEMU Plugin Implementation](../../tests/aicpu_ut/plugins/insn_count.c)
