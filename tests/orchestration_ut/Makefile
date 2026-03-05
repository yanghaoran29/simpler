# ─────────────────────────────────────────────────────────────────────────────
# Makefile — PTO Runtime2 Orchestrator-Scheduler 模拟测试
#
# 用法：
#   cd simpler/tests/unit
#   make          # 编译并运行
#   make build    # 只编译
#   make run      # 只运行
#   make clean    # 清理
# ─────────────────────────────────────────────────────────────────────────────

# 项目根目录（相对于本 Makefile 所在目录）
ROOT := ../..

# 运行时源码目录
RUNTIME_DIR := $(ROOT)/src/runtime/tensormap_and_ringbuffer/runtime
ORCH_DIR    := $(ROOT)/src/runtime/tensormap_and_ringbuffer/orchestration

# ─── 编译器 ─────────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++17 -O0 -g \
            -Wall -Wextra -Wno-unused-parameter \
            -DPTO2_PROFILING=0

# ─── Include 路径 ────────────────────────────────────────────────────────────
# 1. 运行时头文件（pto_runtime2.h, tensor.h, pto_types.h 等）
# 2. 平台统一日志接口（common/unified_log.h）
# 3. 编排目录（orchestration/common.h 供 assert_impl 使用）
INCLUDES := \
    -I$(RUNTIME_DIR) \
    -I$(ROOT)/src/platform/include \
    -I$(ORCH_DIR)

# ─── 待编译的运行时 .cpp 文件 ────────────────────────────────────────────────
# 注意：
#   - runtime.cpp 含有 device-only 依赖（HandShake 等），不编译
#   - unified_log_host.cpp 依赖完整 HostLogger 栈，由测试文件内的桩函数替代
RUNTIME_SRCS := \
    $(RUNTIME_DIR)/pto_runtime2.cpp        \
    $(RUNTIME_DIR)/pto_orchestrator.cpp    \
    $(RUNTIME_DIR)/pto_scheduler.cpp       \
    $(RUNTIME_DIR)/pto_ring_buffer.cpp     \
    $(RUNTIME_DIR)/pto_shared_memory.cpp   \
    $(RUNTIME_DIR)/pto_tensormap.cpp       \
    $(ORCH_DIR)/common.cpp

# 测试源文件（含 unified_log 桩）
TEST_PA_SRCS := test_paged_attention.cpp

ALL_SRCS := $(RUNTIME_SRCS)

# ─── 目标 ────────────────────────────────────────────────────────────────────
TARGET_PA := test_paged_attention

.PHONY: all build run clean build-pa run-pa

all: build run

build: build-pa

build-pa: $(TARGET_PA)

$(TARGET_PA): $(ALL_SRCS) $(TEST_PA_SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ -ldl
	@echo "编译完成: $(TARGET_PA)"

run: run-pa

run-pa: $(TARGET_PA)
	@echo ""
	@./$(TARGET_PA)

clean:
	rm -f $(TARGET_PA) *.o
	@echo "清理完成"
