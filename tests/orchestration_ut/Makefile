# ─────────────────────────────────────────────────────────────────────────────
# Makefile — PTO Runtime2 Orchestrator-Scheduler simulation tests
#
# Usage:
#   make                         # build + run all tests (functional + performance)
#   make build                   # build only
#   make run                     # run all tests (functional + performance)
#   make run-func                # run functional tests only
#   make run-perf                # run performance tests only
#   make clean                   # clean build artifacts
#
# Optional overrides:
#   make TIMEOUT=300             # custom timeout in seconds (default: 600)
#   make ORCH_CPU=4 SCHED_CPU0=5 SCHED_CPU1=6 SCHED_CPU2=7
# ─────────────────────────────────────────────────────────────────────────────

# ─── CPU affinity parameters ──────────────────────────────────────────────────
ORCH_CPU   ?= 0
SCHED_CPU0 ?= 1
SCHED_CPU1 ?= 2
SCHED_CPU2 ?= 3
SCHED_CPU3 ?= 4
SCHED_CPU4 ?= 5
SCHED_CPU5 ?= 6
SCHED_CPU6 ?= 7
SCHED_CPU7 ?= 8

# ─── Platform configuration parameters ────────────────────────────────────────
PLATFORM_MAX_BLOCKDIM ?= 24
PLATFORM_AIC_CORES_PER_BLOCKDIM ?= 1
PLATFORM_AIV_CORES_PER_BLOCKDIM ?= 2
PLATFORM_MAX_AICPU_THREADS ?= 4

# Timeout in seconds for run targets (uses system 'timeout' command)
TIMEOUT ?= 600

# Project root (relative to this Makefile)
ROOT := ../..

# Runtime source directories
RUNTIME_DIR := $(ROOT)/src/runtime/tensormap_and_ringbuffer/runtime
ORCH_DIR    := $(ROOT)/src/runtime/tensormap_and_ringbuffer/orchestration

# Test source directories
COMMON_DIR   := common
FUNC_DIR     := tests/functional
PERF_DIR     := tests/perf

# Build output directory
BUILD_DIR := build

# ─── Compiler ────────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++17 -O0 -ggdb3 -fno-omit-frame-pointer \
            -Wall -Wextra -Wno-unused-parameter -Wno-class-memaccess \
            -DPTO2_PROFILING=1 \
            -DORCH_CPU=$(ORCH_CPU) \
            -DSCHED_CPU0=$(SCHED_CPU0) \
            -DSCHED_CPU1=$(SCHED_CPU1) \
            -DSCHED_CPU2=$(SCHED_CPU2) \
            -DSCHED_CPU3=$(SCHED_CPU3) \
            -DSCHED_CPU4=$(SCHED_CPU4) \
            -DSCHED_CPU5=$(SCHED_CPU5) \
            -DSCHED_CPU6=$(SCHED_CPU6) \
            -DSCHED_CPU7=$(SCHED_CPU7) \
            -DPLATFORM_MAX_BLOCKDIM=$(PLATFORM_MAX_BLOCKDIM) \
            -DPLATFORM_AIC_CORES_PER_BLOCKDIM=$(PLATFORM_AIC_CORES_PER_BLOCKDIM) \
            -DPLATFORM_AIV_CORES_PER_BLOCKDIM=$(PLATFORM_AIV_CORES_PER_BLOCKDIM) \
            -DPLATFORM_MAX_AICPU_THREADS=$(PLATFORM_MAX_AICPU_THREADS)

# ─── Static library paths ─────────────────────────────────────────────────────
STATIC_LIBSTDCXX := $(HOME)/Ascend/cann-8.5.0/tools/hcc/aarch64-target-linux-gnu/lib64/libstdc++.a
STATIC_LIBGCC := $(shell g++ -print-file-name=libgcc.a)
STATIC_LIBGCC_EH := $(shell g++ -print-file-name=libgcc_eh.a)
STATIC_LIBM := /usr/lib64/libm.a
STATIC_LIBC := /usr/lib64/libc.a
STATIC_LIBDL := /usr/lib64/libdl.a
STATIC_LIBPTHREAD := /usr/lib64/libpthread.a

# ─── Include paths ────────────────────────────────────────────────────────────
# 1. Runtime headers (pto_runtime2.h, tensor.h, pto_types.h, etc.)
# 2. Platform unified log interface (common/unified_log.h)
# 3. Orchestration directory (orchestration/common.h for assert_impl)
# 4. Test common headers
INCLUDES := \
    -I$(RUNTIME_DIR) \
    -I$(ROOT)/src/platform/include \
    -I$(ORCH_DIR) \
    -I$(COMMON_DIR)

# ─── Runtime .cpp sources ─────────────────────────────────────────────────────
# Note:
#   - runtime.cpp has device-only dependencies (HandShake etc.), excluded
#   - unified_log_host.cpp depends on full HostLogger stack, replaced by stubs
RUNTIME_SRCS := \
    $(RUNTIME_DIR)/pto_runtime2.cpp        \
    $(RUNTIME_DIR)/pto_orchestrator.cpp    \
    $(RUNTIME_DIR)/pto_scheduler.cpp       \
    $(RUNTIME_DIR)/pto_ring_buffer.cpp     \
    $(RUNTIME_DIR)/pto_shared_memory.cpp   \
    $(RUNTIME_DIR)/pto_tensormap.cpp       \
    $(ORCH_DIR)/common.cpp                 \
    $(ROOT)/src/platform/a2a3sim/aicpu/device_time.cpp

# Common test sources (shared by both functional and performance tests)
COMMON_TEST_SRCS := \
    $(COMMON_DIR)/test_common.cpp \
    $(COMMON_DIR)/test_log_stubs.cpp \
    $(COMMON_DIR)/cpu_affinity.cpp

# Functional test sources
FUNC_TEST_SRCS := \
    $(FUNC_DIR)/test_cpu_affinity.cpp \
    $(FUNC_DIR)/test_platform_config.cpp

# Performance test sources
PERF_TEST_SRCS := \
    $(PERF_DIR)/test_paged_attention.cpp \
    $(PERF_DIR)/test_batch_paged_attention.cpp

# Main entry
MAIN_SRC := main.cpp

# ─── Targets ──────────────────────────────────────────────────────────────────
TARGET := $(BUILD_DIR)/test_orchestration

ALL_SRCS := $(RUNTIME_SRCS) $(COMMON_TEST_SRCS) \
            $(FUNC_TEST_SRCS) $(PERF_TEST_SRCS) $(MAIN_SRC)

LINK_FLAGS := \
    -nodefaultlibs \
    -Wl,--start-group \
    $(STATIC_LIBSTDCXX) $(STATIC_LIBM) $(STATIC_LIBPTHREAD) $(STATIC_LIBDL) \
    $(STATIC_LIBC) $(STATIC_LIBGCC) $(STATIC_LIBGCC_EH) \
    -Wl,--end-group

.PHONY: all build run run-func run-perf clean

all: build run

build: $(TARGET)

$(TARGET): $(ALL_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LINK_FLAGS)
	@echo "Build complete: $(TARGET)"

run: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "  Running orchestration_ut (all)"
	@echo "========================================"
	@timeout $(TIMEOUT) $(TARGET); RC=$$?; \
	 if [ $$RC -eq 124 ]; then \
	     echo ""; \
	     echo "TIMEOUT: exceeded $(TIMEOUT)s limit"; \
	     exit 1; \
	 fi; \
	 exit $$RC

run-func: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "  Running orchestration_ut (functional)"
	@echo "========================================"
	@timeout $(TIMEOUT) $(TARGET) --func; RC=$$?; \
	 if [ $$RC -eq 124 ]; then \
	     echo ""; \
	     echo "TIMEOUT: exceeded $(TIMEOUT)s limit"; \
	     exit 1; \
	 fi; \
	 exit $$RC

run-perf: $(TARGET)
	@echo ""
	@echo "========================================"
	@echo "  Running orchestration_ut (performance)"
	@echo "========================================"
	@timeout $(TIMEOUT) $(TARGET) --perf; RC=$$?; \
	 if [ $$RC -eq 124 ]; then \
	     echo ""; \
	     echo "TIMEOUT: exceeded $(TIMEOUT)s limit"; \
	     exit 1; \
	 fi; \
	 exit $$RC

clean:
	rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# ─── Multi-configuration testing ──────────────────────────────────────────────
# Test platform configuration with different compile-time parameters

test-config1:
	@echo ""
	@echo "========================================"
	@echo "  Testing Configuration 1: Default"
	@echo "  BLOCKDIM=24, AIC=1, AIV=2, THREADS=4"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=24 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=2 \
	         PLATFORM_MAX_AICPU_THREADS=4 \
	         build
	@$(TARGET) --func

test-config2:
	@echo ""
	@echo "========================================"
	@echo "  Testing Configuration 2: Large Scale"
	@echo "  BLOCKDIM=24, AIC=1, AIV=2, THREADS=4"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=24 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=3 \
	         PLATFORM_MAX_AICPU_THREADS=4 \
	         build
	@$(TARGET) --func

test-config3:
	@echo ""
	@echo "========================================"
	@echo "  Testing Configuration 3: Small Scale"
	@echo "  BLOCKDIM=32, AIC=1, AIV=2, THREADS=8"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=32 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=2 \
	         PLATFORM_MAX_AICPU_THREADS=8 \
	         build
	@$(TARGET) --func

test-config4:
	@echo ""
	@echo "========================================"
	@echo "  Testing Configuration 4: Balanced"
	@echo "  BLOCKDIM=32, AIC=1, AIV=3, THREADS=8"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=32 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=3 \
	         PLATFORM_MAX_AICPU_THREADS=8 \
	         build
	@$(TARGET) --func

# Run all configuration tests
test-all-configs: test-config1 test-config2 test-config3 test-config4
	@echo ""
	@echo "========================================"
	@echo "  All Configuration Tests Complete"
	@echo "========================================"

# ─── Performance test configurations ──────────────────────────────────────────
perf-config1:
	@echo ""
	@echo "========================================"
	@echo "  Performance Test - Config 1"
	@echo "  BLOCKDIM=24, AIC=1, AIV=2, THREADS=4"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=24 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=2 \
	         PLATFORM_MAX_AICPU_THREADS=4 \
	         build
	@$(TARGET) --perf

perf-config2:
	@echo ""
	@echo "========================================"
	@echo "  Performance Test - Config 2"
	@echo "  BLOCKDIM=32, AIC=1, AIV=3, THREADS=4"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=32 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=3 \
	         PLATFORM_MAX_AICPU_THREADS=4 \
	         build
	@$(TARGET) --perf

perf-config3:
	@echo ""
	@echo "========================================"
	@echo "  Performance Test - Config 3"
	@echo "  BLOCKDIM=32, AIC=1, AIV=2, THREADS=8"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=32 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=2 \
	         PLATFORM_MAX_AICPU_THREADS=8 \
	         build
	@$(TARGET) --perf

perf-config4:
	@echo ""
	@echo "========================================"
	@echo "  Performance Test - Config 4"
	@echo "  BLOCKDIM=32, AIC=1, AIV=3, THREADS=8"
	@echo "========================================"
	@$(MAKE) clean
	@$(MAKE) PLATFORM_MAX_BLOCKDIM=32 \
	         PLATFORM_AIC_CORES_PER_BLOCKDIM=1 \
	         PLATFORM_AIV_CORES_PER_BLOCKDIM=3 \
	         PLATFORM_MAX_AICPU_THREADS=8 \
	         build
	@$(TARGET) --perf

# Run all performance configuration tests
perf-all-configs: perf-config1 perf-config2 perf-config3 perf-config4
	@echo ""
	@echo "========================================"
	@echo "  All Performance Tests Complete"
	@echo "========================================"

.PHONY: test-config1 test-config2 test-config3 test-config4 test-all-configs
.PHONY: perf-config1 perf-config2 perf-config3 perf-config4 perf-all-configs
