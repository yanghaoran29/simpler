# TensorMap 使用说明（`tm_tensormap_c.h`）

生产者查找引擎，用于自动依赖发现：给定一个新任务要访问的内存区域，回答「该区域由哪个（些）已提交任务产出」，调用方据此连出任务依赖边。纯 C 实现（C99+，带 `extern "C"`，可同时被 C 与 C++ 编译），不依赖任何项目结构体/类，只用 `<stdint.h>` / `<string.h>` / `<assert.h>` / `<stdbool.h>`。

模块自身不分配内存：调用方提供一块连续 buffer，全部状态（头部、哈希桶、entry 池、空闲表、各 ring 的任务链头）排布在其中，entry 间链接用池下标而非指针，因此镜像位置无关——可 `memcpy` 搬运后 `tm_attach` 直接复用，无需指针修复。

## 1 输入与输出（总览）

抛开细节，TensorMap 只做两个动作，输入输出都很简单：

| 动作 | 你喂进去什么（输入） | 你拿到什么（输出） |
| --- | --- | --- |
| **登记** `tm_insert` | 一个 `TmRegion`（某任务产出的内存区域：基址、偏移、形状）+ `producer_id`（哪个任务产的） | 无（区域被记入表内） |
| **查询** `tm_lookup` | 一个 `TmRegion`（某新任务要访问的区域）+ 一个回调 | 经回调**逐个**返回每个与之重叠的生产者：`TmEntry`（含 `producer_id`）+ `TmOverlap`（重叠级别） |

一句话：登记时告诉它「这块区域是谁产的」，查询时问它「这块区域和之前哪些任务产的数据重叠」，它把命中的 `producer_id` 和重叠程度回吐给你——你据此连出依赖边。

- **输入的核心类型**永远是 `TmRegion`（见 3.2）；`producer_id` 由 `tm_make_id(ring, local)` 生成。
- **输出**不是返回值，而是 `tm_lookup` 对每个命中调用一次你的回调 `bool on_match(TmEntry *e, TmOverlap st, void *ctx)`；从 `e->producer_id` 读生产者，从 `st` 读重叠级别，返回 `true` 继续、`false` 提前停。
- 不相交（`TM_OVERLAP_NONE`）的区域不会触发回调；已退休的生产者会被自动跳过。

## 2 简单使用教程

**场景**：任务 A（id：ring 0 / local 1）往一块 buffer 写了 128 个 float；随后任务 B（ring 0 / local 2）要读同一块。我们希望自动发现「B 依赖 A」。

```c
#include "tensormap/tm_tensormap_c.h"
#include <stdlib.h>
#include <stdio.h>

// 描述一块「从 off 开始、len 个元素、共 storage 个元素」的连续 1D 区域
static TmRegion region_1d(uint64_t base, uint64_t off, uint64_t len, uint64_t storage) {
    TmRegion r = {0};
    r.base_addr = base; r.start_offset = off; r.extent_elem = len;
    r.storage_numel = storage; r.elem_size = 4; r.ndims = 1;
    r.is_contiguous = 1; r.shapes[0] = (uint32_t)len; r.strides[0] = 1;
    return r;
}

// 查询回调：每命中一个生产者就调用一次。ctx 这里透传消费者任务 id。
static bool on_match(TmEntry *e, TmOverlap st, void *ctx) {
    printf("B(%llu) 依赖 producer=%llu, 重叠=%d\n",
           (unsigned long long)(uintptr_t)ctx,
           (unsigned long long)e->producer_id, (int)st);
    return true;  // 继续遍历后续命中
}

int main(void) {
    // 第 1 步：配置（num_buckets 和每个 task_window 必须是 2 的幂）
    TmConfig cfg = {0};
    cfg.num_buckets = 16; cfg.pool_size = 64; cfg.num_rings = 1;
    cfg.task_window[0] = 16;

    // 第 2 步：按 tm_bytes_required 申请 64 字节对齐 buffer，初始化
    void *buf = aligned_alloc(64, tm_bytes_required(&cfg));
    TmTensorMap map;
    tm_init(&map, buf, &cfg);

    // 第 3 步：登记生产者 A —— 输入「区域 + A 的 id」，无输出
    uint64_t BUF = 0x1000;
    TmRegion a_out = region_1d(BUF, 0, 128, 128);
    tm_insert(&map, &a_out, tm_make_id(/*ring=*/0, /*local=*/1));

    // 第 4 步：以 B 的区域作探针查询 —— 输出经 on_match 回调给出
    TmRegion b_in = region_1d(BUF, 0, 128, 128);
    tm_lookup(&map, &b_in, on_match, (void *)(uintptr_t)tm_make_id(0, 2));
    // 预期输出： B(...) 依赖 producer=1, 重叠=1   (1 = TM_OVERLAP_COVERED)

    // 第 5 步：A 退休后推进水位并回收它的 entry（提交热路径每次调用）
    tm_sync_tensormap(&map, /*ring=*/0, /*last_alive=*/2);  // local 1 < 2 → 退休
    printf("有效 entry 数 = %d\n", tm_valid_count(&map));    // 预期：0

    free(buf);
    return 0;
}
```

**这段教程对应的输入/输出**：

1. 给 `tm_insert` 的输入是 A 的区域 `a_out` 和 `producer_id = tm_make_id(0,1)`，没有返回值。
2. 给 `tm_lookup` 的输入是 B 的区域 `b_in`；它的输出是对命中的一次回调，`e->producer_id == tm_make_id(0,1)`、`st == TM_OVERLAP_COVERED`（B 完整覆盖 A 的区域）。
3. 若 B 只与 A 部分重叠，`st` 会是 `TM_OVERLAP_OTHER`；若完全不沾边，回调根本不触发。
4. `tm_sync_tensormap(...,2)` 后 A（local 1）低于水位被退休，`tm_valid_count` 归 0，后续查询不再命中 A。

## 3 数据类型

### 3.1 `TmConfig`（配置，输入）

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `num_buckets` | `uint32_t` | 哈希桶数，**必须是 2 的幂** |
| `pool_size` | `uint32_t` | entry 池容量（最多同时存活的生产者区域数） |
| `num_rings` | `uint32_t` | 任务 id 的 ring 层数，`<= TM_MAX_RINGS`(8) |
| `task_window[r]` | `uint32_t` | 第 r 个 ring 的任务窗口大小，**每个必须是 2 的幂** |

### 3.2 `TmRegion`（区域描述，唯一输入类型）

描述一个张量视图。`tm_insert` 用它登记生产者，`tm_lookup` 用它做探针。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `base_addr` | `uint64_t` | 后备 buffer 基址；既是哈希键也是匹配条件（仅同基址才比较） |
| `start_offset` | `uint64_t` | 视图在后备 buffer 内的起始元素偏移 |
| `extent_elem` | `uint64_t` | 视图跨度（元素数）；L1 字节区间为 `[start_offset, start_offset+extent_elem)` |
| `storage_numel` | `uint64_t` | 后备 buffer 总元素数；用于 L2 参考形状推导 |
| `elem_size` | `uint32_t` | 每元素字节数（替代 dtype） |
| `ndims` | `uint32_t` | 维数，`<= TM_MAX_DIMS`(5) |
| `version` | `int32_t` | 存储代号；探针 `version` 更大表示整 buffer 被改写，必依赖更旧生产者 |
| `is_contiguous` | `uint8_t` | 是否连续视图 |
| `shapes[i]` | `uint32_t` | 各维形状 |
| `strides[i]` | `uint32_t` | 各维步长（元素粒度，`> 0`） |

### 3.3 `TmOverlap`（重叠级别，输出）

`tm_lookup` 回调与 `tm_overlap` 的返回值。

| 取值 | 含义 |
| --- | --- |
| `TM_OVERLAP_NONE` | 不相交（`tm_lookup` 不会对这种 entry 回调） |
| `TM_OVERLAP_COVERED` | 探针完全包含该生产者区域；调用方可据此退休一个已冗余的生产者 entry |
| `TM_OVERLAP_OTHER` | 相交但非完全包含（含保守判定） |

### 3.4 `TmEntry`（池中条目，回调里输出给调用方）

`tm_lookup` 回调收到的 `TmEntry *`，常用只读字段：`producer_id`（生产者任务 id）、`base_addr`、`start_offset`、`extent_elem`、`version`、`ndims`、`shapes[]`、`strides[]`。其余为内部链接字段，调用方不应改动。

### 3.5 `TmTensorMap`（句柄）与 `TmMatchFn`（回调类型）

`TmTensorMap` 只是一个 `base` 指针的薄包装；所有 API 以 `TmTensorMap *self` 操作它。
`typedef bool (*TmMatchFn)(TmEntry *entry, TmOverlap status, void *ctx);`——查找回调。

## 4 API 参考

下表「输出」指返回值；通过指针参数写出的结果在「说明」里标注。

### 4.1 容量与初始化

| 函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `uint64_t tm_bytes_required(const TmConfig *cfg)` | `cfg`：配置 | buffer 所需字节数 | 纯计算，不触碰内存 |
| `void tm_init(TmTensorMap *self, void *base, const TmConfig *cfg)` | `self`：句柄；`base`：`>= tm_bytes_required(cfg)` 且 **64 字节对齐** 的 buffer；`cfg`：配置 | 无 | 在 `base` 内排布并清零，绑定 `self->base`。唯一内存依赖点 |
| `void tm_attach(TmTensorMap *self, void *base)` | `self`；`base`：已初始化好的镜像（可由别处 `memcpy` 而来），或 `NULL` 表示解绑 | 无 | 仅重绑指针，不改内容；无需指针修复 |

### 4.2 登记与查询

| 函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `void tm_insert(TmTensorMap *self, const TmRegion *r, uint64_t producer_id)` | `self`；`r`：被产出的区域；`producer_id`：生产者任务 id（见 `tm_make_id`） | 无 | 从池中取一个 entry 登记。池满时触发断言 |
| `void tm_lookup(TmTensorMap *self, const TmRegion *r, TmMatchFn on_match, void *ctx)` | `self`；`r`：消费者探针；`on_match`：回调；`ctx`：透传给回调的上下文 | 无（结果经回调逐个返回） | 对每个**有效且与 `r` 字节范围重叠**的同基址 entry 调一次回调。失效（已退休）entry 自动跳过 |

`on_match(entry, status, ctx)` 的契约——**输入**：`entry`（命中的生产者条目，可读 `producer_id` 等）、`status`（`COVERED`/`OTHER`，绝不会是 `NONE`）、`ctx`（透传）；**输出**：返回 `true` 继续遍历，`false` 提前停止。回调内可安全 `tm_remove(self, entry)`（下一个链接已提前锁存）。

### 4.3 失效与回收（惰性失效）

| 函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `void tm_sync(TmTensorMap *self, uint32_t ring, int32_t last_alive)` | `self`；`ring`：环号；`last_alive`：该 ring 上最早仍存活的 local id | 无 | 仅推进水位；`local < last_alive` 的 entry 此后在查询中被跳过 |
| `void tm_sync_tensormap(TmTensorMap *self, uint32_t ring, int32_t last_alive)` | 同上 | 无 | `tm_sync` + 立即回收已退休 entry。提交热路径推荐用这个 |
| `void tm_cleanup_retired(TmTensorMap *self, uint32_t ring, int32_t old_alive, int32_t new_alive)` | `self`；`ring`；区间 `[old_alive, new_alive)` | 无 | 回收该区间内已退休任务的 entry 到空闲表 |
| `void tm_remove(TmTensorMap *self, TmEntry *e)` | `self`；`e`：要删除的 entry | 无 | 从桶链与任务链摘除并归还池。通常在 `tm_lookup` 回调内调用 |

### 4.4 生产者 id 编码与统计

| 函数 | 输入 | 输出 | 说明 |
| --- | --- | --- | --- |
| `uint64_t tm_make_id(uint32_t ring, uint32_t local)` | `ring`、`local` | 不透明 `producer_id`（`(ring<<32)\|local`） | id 编码归本模块所有 |
| `uint32_t tm_ring_of(uint64_t id)` | `id` | ring 分量 | |
| `uint32_t tm_local_of(uint64_t id)` | `id` | local 分量 | |
| `int32_t tm_valid_count(const TmTensorMap *self)` | `self` | 当前有效（未退休）entry 数 | 调试/测试用 |
| `TmOverlap tm_overlap(const TmRegion *in, const TmEntry *e)` | `in`：探针；`e`：同基址的已存 entry | 重叠级别 | 底层判定函数，一般经 `tm_lookup` 间接使用 |

## 5 重叠判定（三级级联）

`tm_overlap` 决定一个生产者区域与消费者探针的关系：

- **L1 字节区间**：`O(1)` 比较 `[start_offset, +extent_elem)`，不相交直接 `NONE`。
- **L2 超矩形**：当双方 `elem_size`/`ndims`/`strides` 一致且为规范行主序时，做 `O(ndims)` 逐维区间相交——每一维都被探针包含则 `COVERED`，否则 `OTHER`，某维不相交则 `NONE`。
- **L3 保守**：非超矩形组合一律返回 `OTHER`（宁可多连一条边，不漏依赖）。
- 此外，探针 `version` 大于 entry `version` 时直接 `OTHER`（整 buffer 改写依赖旧生产者）。

## 6 生命周期与约束

- `num_buckets` 与每个 `task_window[r]` 必须是 2 的幂；`base` 必须 64 字节对齐且 `>= tm_bytes_required(cfg)`。
- `ndims <= TM_MAX_DIMS`(5)，`num_rings <= TM_MAX_RINGS`(8)。
- 稳态下无 `malloc`/`free`：entry 在固定池里取用与回收。`tm_insert` 在池耗尽时断言失败——按峰值并发区域数设置 `pool_size`。
- 失效是惰性的：`tm_sync*` 推进水位即可让旧 entry 在查询中“消失”，物理回收由 `tm_cleanup_retired` / `tm_sync_tensormap` 完成。

## 7 a2a3 如何使用 TensorMap

a2a3 的 `tensormap_and_ringbuffer` 运行时（编排器 orchestrator）用这张表做自动依赖发现：每提交一个任务，就用任务的输入区域查出生产者、连出依赖边，再把任务的输出区域登记进表，供后续任务查询。相关代码在 `src/a2a3/runtime/tensormap_and_ringbuffer/`。

### 7.1 集成位置与生命周期

#### 7.1.1 谁持有它

编排器状态 `PTO2OrchestratorState`（`pto_orchestrator.h`）内嵌一个成员 `TmTensorMap tensor_map`，作为该编排器私有的生产者查找表。

#### 7.1.2 buffer 从哪来、何时 init / attach

TensorMap 自己不分配内存，其 buffer 是运行时 `DeviceArena` 里 `reserve` 出的一段 region（大小由 `tm_bytes_required` 算出）。host 侧在 `init_data_from_layout` 中对该 region 调 `tm_init`；AICPU 侧在 `wire_arena_pointers` 中对**同一份字节**调 `tm_attach`（镜像位置无关，无需指针修复）。编排器销毁时 `destroy` 调 `tm_attach(&tensor_map, nullptr)` 解绑，buffer 由 arena 拥有。

#### 7.1.3 配置取值

`reserve_layout` 填充 `TmConfig`：`num_buckets = PTO2_TENSORMAP_NUM_BUCKETS`(4096)、`pool_size = PTO2_TENSORMAP_POOL_SIZE`(65536)、`num_rings = PTO2_MAX_RING_DEPTH`、每个 `task_window[r]` 取自运行时的 per-ring 任务窗口大小（均为 2 的幂）。

### 7.2 数据如何映射

#### 7.2.1 Tensor → TmRegion

运行时的 `Tensor` 经 `tensor_tm_adapter.h::to_tm_region` 转成 TensorMap 的输入类型 `TmRegion`：`buffer.addr → base_addr`、`start_offset`、`extent_elem()`、由 `dtype` 推出的 `elem_size`、`buffer.size/elem_size → storage_numel`、以及 `ndims/version/is_contiguous/shapes/strides`。

#### 7.2.2 PTO2TaskId → producer_id

任务 id `PTO2TaskId` 的内部编码是 `(ring<<32)|local`，与本模块 `tm_make_id(ring, local)` 完全一致，因此 `PTO2TaskId.raw` 可直接当 `producer_id` 传给 `tm_insert`；回调里拿到的 `entry.producer_id` 也可直接构造回 `PTO2TaskId`。

#### 7.2.3 C++ 回调如何接到 C 接口

C 的 `tm_lookup` 用「函数指针 + `void *ctx`」传闭包，而运行时是 C++ 且回调要捕获局部状态。`tensor_tm_adapter.h` 提供薄封装 `tm_lookup_each(map, region, callable)`，把可调用对象 `(TmEntry&, TmOverlap)->bool` 转接到 C 回调；核心实现保持纯 C，调用点仍可写 lambda。

### 7.3 提交一个任务时的调用流程

`submit_task` 路径（`pto_orchestrator.cpp` + `pto_dep_compute.h`）按下面顺序用到这张表：

#### 7.3.1 推进水位并回收

先读共享内存里本 ring 的 `last_task_alive`，调 `tm_sync_tensormap(&tensor_map, ring, last_task_alive)`：把失效水位推进到该值，并回收已退休任务的 entry，避免池被旧条目占满。

#### 7.3.2 推导 fan-in（查询）

`compute_task_fanin` 遍历任务的非 OUTPUT 张量：对 `INPUT`/`INOUT`（且非 `manual_dep`）的张量，用 `tm_lookup_each(tensor_map, to_tm_region(*tensor), …)` 查询，回调里对每个命中的 `entry.producer_id` 调用 `emit` 连出一条依赖边。

#### 7.3.3 登记输出（写入）

`register_task_outputs` 对 `INOUT`/`OUTPUT_EXISTING`（且非 `manual_dep`）的张量，用 `tm_insert(&tensor_map, &region, task_id.raw)` 把本任务登记为这些区域的新生产者，供后续任务查询。

#### 7.3.4 INOUT + COVERED 时退休旧生产者

在 7.3.2 的查询回调里，若张量是 `INOUT` 且重叠级别为 `TM_OVERLAP_COVERED`（新任务完整覆盖旧生产者区域），立即 `tm_remove(&tensor_map, &entry)`：旧生产者已被整块覆盖、不会再被后续任务依赖，删掉可省查询开销与池空间。

### 7.4 host 侧 dep-gen 重放校验

`host/dep_gen_replay.cpp` 用**两张** `TmTensorMap`（`tm_oracle` 与 `tm_annot`，由一个 libc arena 提供 buffer）对同一批提交记录做差分校验：oracle 直接驱动 `compute_task_fanin`，annot 用一份镜像逻辑（`annot_pass`）边查边记录每条边的生产者/重叠信息（`to_overlap_status` 把 `TmOverlap` 转成注解枚举），最后比对两者得到的生产者 id 集合是否一致，以此保证 host 重放与运行时在线推导出的依赖完全相同。

## 8 测试

`tests/ut/cpp/common/test_tm_tensormap.cpp`（`ctest -R test_tm_tensormap`）覆盖精确命中、不相交、部分重叠、惰性失效、清理与池复用、回调内删除、以及 `memcpy` 后 `tm_attach` 的位置无关性。运行时层面由 `examples/a2a3/tensormap_and_ringbuffer/` 下的场景测试在 a2a3sim 与 a2a3 硬件上验证。
