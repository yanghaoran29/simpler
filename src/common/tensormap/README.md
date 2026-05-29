# TensorMap (`tm_tensormap.h`)

`tensormap_and_ringbuffer` 运行时的生产者查找引擎，用于自动依赖发现。它在 task-submit 热路径上回答一个问题：*当前新任务读取的内存区域，是由哪个（些）已提交任务产出的？* 编排器（orchestrator）把每个答案转换成一条任务依赖边。

本模块是一个**最小依赖的独立实现**：不依赖任何项目结构体/类，编译期只需 `<cstdint>` / `<cstring>` / `<cassert>`，全部类型放在 `namespace tmap` 下，可在运行时之外复用。a2a3 运行时通过一个 Tensor→TmRegion 适配层使用它（见下文「在运行时中的使用」）。

## 文件

| 文件               | 内容                                              |
| ------------------ | ------------------------------------------------- |
| `tm_tensormap.h`   | header-only：`TmConfig` / `TmRegion` / `TmOverlap` / `TmEntry` / `TensorMap` + `tm_overlap` |

## 依赖

只依赖 C++ 标准库 `<cstdint>` / `<cstring>` / `<cassert>`。没有项目内的结构体、类或分配器依赖；唯一的外部耦合是「内存从哪来」，被收敛到一个接口点（见下文）。

## 内存模型（依赖收敛到一个点）

模块**自己不分配、不释放、不持有分配器**，只接受调用方递进来的一块连续 buffer，并在内部排布子区域。整个内存依赖面就是三个函数：

```cpp
static uint64_t bytes_required(const TmConfig &cfg);  // 纯尺寸计算
void init(void *base, const TmConfig &cfg);           // 在 base 内排布并清零（唯一内存依赖点）
void attach(void *base);                              // 绑定到已初始化好的镜像
```

调用方爱用 `malloc`、device 显存、还是某个 arena 的一段，模块一概不关心。

**位置无关镜像**：对象里只存一个 `base_` 指针；buckets / entry_pool / free_list / task_heads 全部以「相对 base 的偏移」寻址，entry 间链接用 `int32_t` 池下标（而非指针）。因此整块镜像可 `memcpy` 搬运——host 构建一次后，`memcpy` 到他处再 `attach(base)` 即可，无需任何指针修复。

buffer 内部布局：

```
[ TmHeader | buckets[num_buckets] | entry_pool[pool_size]
           | free_list[pool_size] | task_heads[num_rings][task_window] ]
```

`TmHeader` 把游标（`next_entry_idx` / `free_num` / `last_alive[]` / `last_cleanup[]`）也放进 buffer，保证全部状态随镜像搬运。

## 数据类型

```cpp
constexpr uint32_t TM_MAX_DIMS = 5;
constexpr uint32_t TM_MAX_RINGS = 8;

struct TmConfig {           // num_buckets 与每个 task_window[r] 必须是 2 的幂
    uint32_t num_buckets;
    uint32_t pool_size;
    uint32_t num_rings;
    uint32_t task_window[TM_MAX_RINGS];
};

struct TmRegion {           // 唯一输入类型
    uint64_t base_addr;     // 哈希键 + 匹配
    uint64_t start_offset;  // 元素偏移
    uint64_t extent_elem;   // 视图跨度（元素数）→ L1 区间
    uint64_t storage_numel; // 后备 buffer 总元素数 → L2 参考形状推导
    uint32_t elem_size;     // 每元素字节数（替代 dtype）
    uint32_t ndims;
    int32_t  version;
    uint8_t  is_contiguous;
    uint32_t shapes[TM_MAX_DIMS];
    uint32_t strides[TM_MAX_DIMS];
};

enum class TmOverlap { None, Covered, Other };
```

生产者身份是不透明的 `uint64_t`，ring/local 编码归本模块（`make_id` / `ring_of` / `local_of`），不依赖外部任务 id 类型。

## 设计

- **仅按 base 指针哈希。** 共享同一基址的所有视图都落在同一个桶里，因此一次 lookup 可以跨所有可能重叠的子区域比较字节范围。
- **环形 entry 池。** entry 从固定池中取用（下标分配 + 空闲表），稳态下没有 `malloc`/`free`。
- **惰性失效。** 只要生产者任务尚未退休，entry 即有效。`sync` 推进每个 ring 的 `last_alive` 水位；lookup 时跳过失效 entry，由 `cleanup_retired` 回收。
- **三级重叠检测**（`tm_overlap`）：L1 字节区间相交（快速拒绝 → `None`）；L2 当双方共享同一规范行主序轴布局时做 O(ndims) 超矩形精确判断（`None` / `Covered` / `Other`）；L3 对非超矩形组合保守返回 `Other`。`Covered` 表示探针完全包含该 entry，调用方据此可退休一个已冗余的生产者 entry。

## API

```cpp
static uint64_t bytes_required(const TmConfig &);
void init(void *base, const TmConfig &);
void attach(void *base);

void insert(const TmRegion &r, uint64_t producer_id);
template <class Fn> void lookup(const TmRegion &r, Fn &&on_match);  // (TmEntry&, TmOverlap)->bool

void sync(uint32_t ring, int32_t last_alive);
void sync_tensormap(uint32_t ring, int32_t last_alive);  // sync + 回收到水位（提交热路径用）
void cleanup_retired(uint32_t ring, int32_t old_alive, int32_t new_alive);
void remove(TmEntry &e);                                 // 回调内可对当前 entry 调用
int32_t valid_count() const;

static uint64_t make_id(uint32_t ring, uint32_t local);
static uint32_t ring_of(uint64_t id);
static uint32_t local_of(uint64_t id);
```

`lookup` 对每个字节范围与 `r` 重叠的有效 entry 调用 `on_match`；回调返回 `true` 继续、`false` 提前停止，且可安全地在回调内 `remove(entry)`（下一个链接在调用前已锁存）。失效 entry 会被跳过而非截断（不同 ring 的 entry 在同一桶链中交错）。

## 在运行时中的使用

a2a3 `tensormap_and_ringbuffer` 编排器持有一个 `tmap::TensorMap`（`pto_orchestrator.h`），其 buffer 是运行时 `DeviceArena` 里 `reserve` 出的单个 region：host 侧 `init`，AICPU 侧 `attach` 同一份字节。运行时的 `Tensor` 经 `tensor_tm_adapter.h::to_tm_region` 转成 `TmRegion`；`PTO2TaskId.raw`（`(ring<<32)|local`）与 `make_id` 编码一致，直接作为 `producer_id`。`pto_dep_compute.h` 用 `lookup` 从 `INPUT` / `INOUT` 推导 fan-in、用 `insert` 登记 `OUTPUT` / `INOUT` 生产者；编排器每次 submit 调用一次 `sync_tensormap`。端到端依赖发现流程见 `src/a2a3/runtime/tensormap_and_ringbuffer/docs/RUNTIME_LOGIC.md`。

## 测试

`tests/ut/cpp/common/test_tm_tensormap.cpp`（`ctest -R test_tm_tensormap`）覆盖精确命中/不相交/部分重叠、惰性失效、清理与池复用、回调内删除、以及 `memcpy` 后 `attach` 的位置无关性。运行时层面的功能正确性由 `examples/a2a3/tensormap_and_ringbuffer/` 下的场景测试在 a2a3sim 与 a2a3 硬件上验证。
