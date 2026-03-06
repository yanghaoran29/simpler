# Platform Configuration Testing

This directory contains tests for validating platform configuration parameters with different compile-time settings.

## New Test: Platform Configuration Validation

**File**: `tests/functional/test_platform_config.cpp`

Validates that platform configuration parameters are correctly set through compile-time macros and that derived values are calculated properly.

### Test Coverage

1. **Base Parameters**: Validates `PLATFORM_MAX_BLOCKDIM`, `PLATFORM_AIC_CORES_PER_BLOCKDIM`, `PLATFORM_AIV_CORES_PER_BLOCKDIM`, `PLATFORM_MAX_AICPU_THREADS`
2. **Derived Limits**: Validates calculated values like `PLATFORM_MAX_AIC_PER_THREAD`, `PLATFORM_MAX_CORES`, etc.
3. **Consistency Checks**: Ensures relationships between base and derived values are correct
4. **Configuration Summary**: Displays complete platform configuration in a formatted table

## Updated Test: CPU Affinity Tests

**File**: `tests/functional/test_cpu_affinity.cpp`

Now dynamically adapts to `PLATFORM_MAX_AICPU_THREADS`:
- **Total AICPU threads**: `PLATFORM_MAX_AICPU_THREADS`
- **Scheduler threads**: `PLATFORM_MAX_AICPU_THREADS - 1`
- **Orchestrator thread**: 1 (the last thread)

This means the CPU affinity tests will automatically scale with different platform configurations.

## Multi-Configuration Testing

The Makefile provides targets to test different platform configurations:

### Available Configurations

```bash
# Configuration 1: Default (24 blocks, 1 AIC + 2 AIV per block, 4 AICPU threads)
make test-config1
# Output: Launching 4 AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)

# Configuration 2: Large Scale (32 blocks, 16 AIC + 16 AIV per block, 8 AICPU threads)
make test-config2
# Output: Launching 8 AICPU threads (Thread 0-6: scheduler, Thread 7: orchestrator)

# Configuration 3: Small Scale (8 blocks, 2 AIC + 4 AIV per block, 2 AICPU threads)
make test-config3
# Output: Launching 2 AICPU threads (Thread 0-0: scheduler, Thread 1: orchestrator)

# Configuration 4: Balanced (16 blocks, 8 AIC + 8 AIV per block, 4 AICPU threads)
make test-config4
# Output: Launching 4 AICPU threads (Thread 0-2: scheduler, Thread 3: orchestrator)

# Run all configuration tests
make test-all-configs
```

### Performance Test Configurations

The same platform configurations can be used for performance testing:

```bash
# Performance test with default configuration (4 threads)
make perf-config1

# Performance test with medium configuration (4 threads, more cores)
make perf-config2

# Performance test with large configuration (8 threads)
make perf-config3

# Performance test with balanced configuration (8 threads)
make perf-config4

# Run all performance configuration tests
make perf-all-configs
```

### Custom Configuration

You can also test with custom parameters:

```bash
make clean
make PLATFORM_MAX_BLOCKDIM=64 \
     PLATFORM_AIC_CORES_PER_BLOCKDIM=8 \
     PLATFORM_AIV_CORES_PER_BLOCKDIM=8 \
     PLATFORM_MAX_AICPU_THREADS=16 \
     build
./build/test_orchestration --func
# Output: Launching 16 AICPU threads (Thread 0-14: scheduler, Thread 15: orchestrator)
```

## CPU Affinity Support

The Makefile now supports up to 8 scheduler threads with CPU affinity binding:

- `SCHED_CPU0` through `SCHED_CPU7`: CPU cores for scheduler threads (default: 1-8)
- `ORCH_CPU`: CPU core for orchestrator thread (default: 0)

These can be overridden when building:

```bash
make ORCH_CPU=10 SCHED_CPU0=11 SCHED_CPU1=12 SCHED_CPU2=13 \
     SCHED_CPU3=14 SCHED_CPU4=15 SCHED_CPU5=16 SCHED_CPU6=17 \
     SCHED_CPU7=18 PLATFORM_MAX_AICPU_THREADS=8 build
```

## How It Works

1. **Makefile Variables**: Platform parameters are defined as Makefile variables with default values
2. **Compile-Time Macros**: These variables are passed as `-D` flags to the compiler
3. **Conditional Compilation**: The header file `platform_config.h` uses `#ifndef` guards to allow macro overrides
4. **Dynamic Test Adaptation**: Test code uses `PLATFORM_MAX_AICPU_THREADS` to determine thread counts at compile time
5. **Test Validation**: The test suite verifies that all values are correctly set and consistent

## Example Output

### Platform Configuration Summary

```
=== Platform Configuration - Summary ===

  ┌─────────────────────────────────────────────────────┐
  │         Platform Configuration Summary             │
  ├─────────────────────────────────────────────────────┤
  │ Base Configuration:                                 │
  │   MAX_BLOCKDIM              : 32                   │
  │   AIC_CORES_PER_BLOCKDIM    : 16                   │
  │   AIV_CORES_PER_BLOCKDIM    : 16                   │
  │   CORES_PER_BLOCKDIM        : 32                   │
  │   MAX_AICPU_THREADS         : 8                    │
  ├─────────────────────────────────────────────────────┤
  │ Derived Limits:                                     │
  │   MAX_AIC_PER_THREAD        : 512                  │
  │   MAX_AIV_PER_THREAD        : 512                  │
  │   MAX_CORES_PER_THREAD      : 1024                 │
  │   MAX_CORES                 : 1024                 │
  ├─────────────────────────────────────────────────────┤
  │ Calculated Ratios:                                  │
  │   AIC/AIV ratio per block   : 16:16                 │
  │   Total AIC cores           : 512                  │
  │   Total AIV cores           : 512                  │
  └─────────────────────────────────────────────────────┘

  PASS: 13, FAIL: 0
```

### CPU Affinity Test (adapts to thread count)

```
=== CPU Affinity - Without Binding (Observe OS Free Scheduling) ===
  System CPU cores: 320
  Launching 8 AICPU threads (Thread 0-6: scheduler, Thread 7: orchestrator)

  Thread ID    Role            Bound Core     Actual Core
  ----------------------------------------------------------
  Thread 0     scheduler       -1             211
  Thread 1     scheduler       -1             221
  Thread 2     scheduler       -1             231
  Thread 3     scheduler       -1             224
  Thread 4     scheduler       -1             236
  Thread 5     scheduler       -1             179
  Thread 6     scheduler       -1             188
  Thread 7     orchestrator    -1             197
```

## Modified Files

1. **`Makefile`**: Added platform configuration variables, extended SCHED_CPU0-7 support, and multi-config test targets for both functional and performance tests
2. **`main.cpp`**: Integrated platform config tests into functional test suite
3. **`tests/functional/test_platform_config.cpp`**: New test file for platform configuration validation
4. **`tests/functional/test_cpu_affinity.cpp`**: Updated to use `PLATFORM_MAX_AICPU_THREADS` dynamically
5. **`src/platform/include/common/platform_config.h`**: Added conditional compilation guards to allow macro overrides

## Key Benefits

- **Scalability**: Tests automatically adapt to different platform configurations
- **Validation**: Ensures platform parameters are correctly propagated through the build system
- **Flexibility**: Easy to test custom configurations without modifying source code
- **Consistency**: Verifies that derived values are correctly calculated from base parameters
- **Performance Comparison**: Enables benchmarking across different hardware configurations to identify optimal settings

