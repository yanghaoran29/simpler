# ─────────────────────────────────────────────────────────────────────────────
# Makefile — PTO Runtime2 Orchestrator-Scheduler 模拟测试
#
# 用法：
#   cd simpler/tests/orchestration_ut
#   make          # 编译并运行所有测试
#   make build    # 只编译
#   make run      # 只运行
#   make clean    # 清理
# ─────────────────────────────────────────────────────────────────────────────

# 项目根目录（相对于本 Makefile 所在目录）
ROOT := ../..

# 运行时源码目录
RUNTIME_DIR := $(ROOT)/src/runtime/tensormap_and_ringbuffer/runtime
ORCH_DIR    := $(ROOT)/src/runtime/tensormap_and_ringbuffer/orchestration

# 测试源码目录
COMMON_DIR := common
TESTS_DIR  := tests

# 构建输出目录
BUILD_DIR := build

# ─── 编译器 ─────────────────────────────────────────────────────────────────
CXX      := g++
CXXFLAGS := -std=c++17 -O0 -ggdb3 -fno-omit-frame-pointer \
            -Wall -Wextra -Wno-unused-parameter \
            -DPTO2_PROFILING=1

# ─── 静态链接库路径 ──────────────────────────────────────────────────────────
STATIC_LIBSTDCXX := $(HOME)/Ascend/cann-8.5.0/tools/hcc/aarch64-target-linux-gnu/lib64/libstdc++.a
STATIC_LIBGCC := $(shell g++ -print-file-name=libgcc.a)
STATIC_LIBGCC_EH := $(shell g++ -print-file-name=libgcc_eh.a)
STATIC_LIBM := /usr/lib64/libm.a
STATIC_LIBC := /usr/lib64/libc.a
STATIC_LIBDL := /usr/lib64/libdl.a
STATIC_LIBPTHREAD := /usr/lib64/libpthread.a

# ─── Include 路径 ────────────────────────────────────────────────────────────
# 1. 运行时头文件（pto_runtime2.h, tensor.h, pto_types.h 等）
# 2. 平台统一日志接口（common/unified_log.h）
# 3. 编排目录（orchestration/common.h 供 assert_impl 使用）
# 4. 测试公共头文件目录
INCLUDES := \
    -I$(RUNTIME_DIR) \
    -I$(ROOT)/src/platform/include \
    -I$(ORCH_DIR) \
    -I$(COMMON_DIR)

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
    $(ORCH_DIR)/common.cpp                 \
    $(ROOT)/src/platform/a2a3sim/aicpu/device_time.cpp

# 公共测试源文件
COMMON_TEST_SRCS := \
    $(COMMON_DIR)/test_common.cpp \
    $(COMMON_DIR)/test_log_stubs.cpp

# 测试源文件
TEST_SRCS := \
    $(TESTS_DIR)/test_paged_attention.cpp \
    $(TESTS_DIR)/test_batch_paged_attention.cpp

# 主程序源文件
MAIN_SRC := main.cpp

# 所有测试源文件
ALL_TEST_SRCS := $(COMMON_TEST_SRCS) $(TEST_SRCS) $(MAIN_SRC)

ALL_SRCS := $(RUNTIME_SRCS)

# ─── 目标 ────────────────────────────────────────────────────────────────────
TARGET := $(BUILD_DIR)/test_orchestration

.PHONY: all build run clean

all: build run

build: $(TARGET)

$(TARGET): $(ALL_SRCS) $(ALL_TEST_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ \
		-nodefaultlibs \
		-Wl,--start-group \
		$(STATIC_LIBSTDCXX) $(STATIC_LIBM) $(STATIC_LIBPTHREAD) $(STATIC_LIBDL) \
		$(STATIC_LIBC) $(STATIC_LIBGCC) $(STATIC_LIBGCC_EH) \
		-Wl,--end-group
	@echo "编译完成: $(TARGET) (完全静态链接)"

run: $(TARGET)
	@echo ""
	@$(TARGET)

clean:
	rm -rf $(BUILD_DIR)
	@echo "清理完成"
