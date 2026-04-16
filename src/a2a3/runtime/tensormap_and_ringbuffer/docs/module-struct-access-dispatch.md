# ④任务Dispatch 详细说明（基于 module-struct-access.csv）

本文聚焦 `module-struct-access.csv` 的 **④任务Dispatch**，逐个数据结构解释：
- 在 Dispatch 阶段做了什么
- 五列统计（读/写/atomic/锁/CAS）为什么是当前公式
- 对应的核心代码位置

---

## 1. Dispatch 主流程与统计变量

Dispatch 阶段总体流程：从 ready queue 取任务 -> 构建 per-core `PTO2DispatchPayload` -> 写寄存器下发 -> 推进 `next_block_idx` 并按需回队。

```744:763:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
// pop_batch 内部原子/等待写入 g_sched_pop_*；用于 CSV ④ ReadyQueue 的 atomic 近似
extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[], g_sched_ready_queue_pop_count[];
uint64_t t_pop_start = get_sys_cnt_aicpu();
int count = rt->scheduler.get_ready_tasks_batch(
    shape, local_buf, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx],
    local_dispatch_count
);
// 每轮调度循环一次 pop 尝试：CSV ④「全局就绪队列 pop」读次数 rqp 的数据源
g_sched_ready_queue_pop_count[thread_idx]++;
```

Dispatch 统计核心变量由 `pto2_scheduler_get_profiling()` 汇总：

```100:118:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
const uint64_t dsub = std::exchange(g_sched_dispatch_subtask_count[thread_idx], 0);
const uint64_t dtask = std::exchange(g_sched_dispatch_task_count[thread_idx], 0);
const uint64_t rqp = std::exchange(g_sched_ready_queue_pop_count[thread_idx], 0);
const uint64_t m4_tn = std::exchange(g_sched_m4_payload_tensor_lane_reads[thread_idx], 0);
const uint64_t m4_sc = std::exchange(g_sched_m4_payload_scalar_lane_reads[thread_idx], 0);
const uint64_t pop_at = d.pop_atomic_count;
```

---

## 2. `PTO2TaskSlotState`（CSV: ④ `S / S / atomic=S`）

### 2.1 在做什么

- Dispatch 时从 `slot_state` 读取 subtask 所需状态（`task/payload/active_mask/next_block_idx/logical_block_num`）。
- 每成功下发一个 block/subtask 后推进 `next_block_idx++`。

```776:797:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
void build_payload(PTO2DispatchPayload &dispatch_payload, PTO2TaskSlotState &slot_state, PTO2SubtaskSlot subslot) {
    int32_t slot_idx = static_cast<int32_t>(subslot);
    uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
    const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
    dispatch_payload.function_bin_addr = callable->resolved_addr();
    auto &payload = *slot_state.payload;
    ...
    dispatch_payload.local_context.block_idx = slot_state.next_block_idx;
    dispatch_payload.local_context.block_num = slot_state.logical_block_num;
}
```

```1033:1045:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
for (int32_t t = 0; t < active_sched_threads_ && slot_state->next_block_idx < block_num; t++) {
    ...
    dispatch_block_to_cluster(...);
    slot_state->next_block_idx++;
    if (slot_state->next_block_idx < block_num)
        valid = core_trackers_[t].get_valid_cluster_offset_states(shape);
}
```

### 2.2 统计公式

- `read_events = dsub`
- `write_events = dsub`
- `atomic_ops = dsub`

```150:155:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_task_slot_state.read_events = dsub;
d.csv_m4_pto2_task_slot_state.write_events = dsub;
d.csv_m4_pto2_task_slot_state.atomic_ops = dsub;
d.csv_m4_pto2_task_slot_state.lock_ops = 0;
d.csv_m4_pto2_task_slot_state.cas_ops = 0;
```

---

## 3. `PTO2TaskPayload(meta)`（CSV: ④ CL0~CL2，读 `3 × task`）

### 3.1 在做什么

元数据区主要用于告诉 dispatch 如何遍历 payload：`tensor_count/scalar_count`、上下文布局等。

```781:788:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
auto &payload = *slot_state.payload;
int n = 0;
for (int32_t i = 0; i < payload.tensor_count; i++) {
    dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
}
for (int32_t i = 0; i < payload.scalar_count; i++) {
    dispatch_payload.args[n++] = payload.scalars[i];
}
```

### 3.2 统计公式

- `read_events = dtask * 3`
- 其余为 0

```157:161:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_task_payload_meta.read_events = dtask * 3;
d.csv_m4_pto2_task_payload_meta.write_events = 0;
d.csv_m4_pto2_task_payload_meta.atomic_ops = 0;
d.csv_m4_pto2_task_payload_meta.lock_ops = 0;
d.csv_m4_pto2_task_payload_meta.cas_ops = 0;
```

---

## 4. `PTO2TaskPayload(tensors)`（CSV: ④ CL3~CL34）

### 4.1 在做什么

遍历 `payload.tensors[]`，把每个 tensor 槽位地址写到 `dispatch_payload.args[]`。

```783:785:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
for (int32_t i = 0; i < payload.tensor_count; i++) {
    dispatch_payload.args[n++] = reinterpret_cast<uint64_t>(&payload.tensors[i]);
}
```

并在 dispatch 时累计 `m4_tn`：

```838:846:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
extern uint64_t g_sched_m4_payload_tensor_lane_reads[];
extern uint64_t g_sched_m4_payload_scalar_lane_reads[];
if (slot_state.payload != nullptr) {
    g_sched_m4_payload_tensor_lane_reads[thread_idx] +=
        static_cast<uint64_t>(slot_state.payload->tensor_count);
    g_sched_m4_payload_scalar_lane_reads[thread_idx] +=
        static_cast<uint64_t>(slot_state.payload->scalar_count);
}
```

### 4.2 统计公式

```163:167:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_task_payload_tensors.read_events = m4_tn;
d.csv_m4_pto2_task_payload_tensors.write_events = 0;
d.csv_m4_pto2_task_payload_tensors.atomic_ops = 0;
d.csv_m4_pto2_task_payload_tensors.lock_ops = 0;
d.csv_m4_pto2_task_payload_tensors.cas_ops = 0;
```

---

## 5. `PTO2TaskPayload(scalars)`（CSV: ④ CL35~CL50）

### 5.1 在做什么

遍历 `payload.scalars[]` 并写入 `dispatch_payload.args[]`。

```786:788:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
for (int32_t i = 0; i < payload.scalar_count; i++) {
    dispatch_payload.args[n++] = payload.scalars[i];
}
```

### 5.2 统计公式

```169:173:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_task_payload_scalars.read_events = m4_sc;
d.csv_m4_pto2_task_payload_scalars.write_events = 0;
d.csv_m4_pto2_task_payload_scalars.atomic_ops = 0;
d.csv_m4_pto2_task_payload_scalars.lock_ops = 0;
d.csv_m4_pto2_task_payload_scalars.cas_ops = 0;
```

---

## 6. `PTO2TaskDescriptor`（CSV: ④ 读 kernel_id）

### 6.1 在做什么

从 descriptor 读取当前 subslot 的 `kernel_id`，解析函数地址，写入 dispatch payload。

```777:781:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
int32_t slot_idx = static_cast<int32_t>(subslot);
uint64_t callable_addr = get_function_bin_addr(slot_state.task->kernel_id[slot_idx]);
const CoreCallable *callable = reinterpret_cast<const CoreCallable *>(callable_addr);
dispatch_payload.function_bin_addr = callable->resolved_addr();
```

### 6.2 统计公式

```175:179:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_task_descriptor.read_events = dtask;
d.csv_m4_pto2_task_descriptor.write_events = 0;
d.csv_m4_pto2_task_descriptor.atomic_ops = 0;
d.csv_m4_pto2_task_descriptor.lock_ops = 0;
d.csv_m4_pto2_task_descriptor.cas_ops = 0;
```

---

## 7. `PTO2DispatchPayload`（CSV: ④ `write = 19 × S`）

### 7.1 在做什么

每次 dispatch 都会重建 per-core 双缓冲 payload：
- `function_bin_addr`
- `args[]`
- `local_context`
- `args` 里两条 context 指针位

```833:836:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
uint32_t buf_idx = reg_task_id & 1u;
PTO2DispatchPayload &payload = s_pto2_payload_per_core[core_id][buf_idx];
build_payload(payload, slot_state, subslot);
```

`PTO2DispatchPayload` 结构（按 cacheline 对齐）：

```70:83:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto2_dispatch_payload.h
struct alignas(64) PTO2DispatchPayload {
    uint64_t function_bin_addr;
    uint64_t args[PTO2_DISPATCH_MAX_ARGS];
    LocalContext local_context;
    GlobalContext global_context;
};
```

### 7.2 统计公式

```181:185:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_dispatch_payload.read_events = 0;
d.csv_m4_pto2_dispatch_payload.write_events = dsub * 19;
d.csv_m4_pto2_dispatch_payload.atomic_ops = 0;
d.csv_m4_pto2_dispatch_payload.lock_ops = 0;
d.csv_m4_pto2_dispatch_payload.cas_ops = 0;
```

---

## 8. `PTO2ReadyQueue(pop)`（CSV: ④ pop 消费侧）

### 8.1 在做什么

- 调度线程对 `ready_queues[shape]` 做 pop/get_batch。
- `rqp` 记录“尝试轮次”，`pop_at` 记录内部原子路径累计。

```744:753:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
extern uint64_t g_sched_pop_atomic_count[], g_sched_pop_wait_cycle[], g_sched_ready_queue_pop_count[];
int count = rt->scheduler.get_ready_tasks_batch(
    shape, local_buf, out, max_count, g_sched_pop_atomic_count[thread_idx], g_sched_pop_wait_cycle[thread_idx],
    local_dispatch_count
);
g_sched_ready_queue_pop_count[thread_idx]++;
```

### 8.2 统计公式

```187:191:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
d.csv_m4_pto2_ready_queue.read_events = rqp;
d.csv_m4_pto2_ready_queue.write_events = 0;
d.csv_m4_pto2_ready_queue.atomic_ops = pop_at;
d.csv_m4_pto2_ready_queue.lock_ops = 0;
d.csv_m4_pto2_ready_queue.cas_ops = rqp;
```

---

## 9. ④模块公式总览（实现口径）

```150:191:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.cpp
// ===== CSV ④ Dispatch：dsub=子任务下发；dtask=取出的任务数；m4_tn/m4_sc=build_payload 张量/标量槽读；rqp=pop 轮次；pop_at=pop 路径原子 =====
d.csv_m4_pto2_task_slot_state.read_events = dsub;
d.csv_m4_pto2_task_slot_state.write_events = dsub;
d.csv_m4_pto2_task_slot_state.atomic_ops = dsub;
...
d.csv_m4_pto2_task_payload_meta.read_events = dtask * 3;
...
d.csv_m4_pto2_task_payload_tensors.read_events = m4_tn;
...
d.csv_m4_pto2_task_payload_scalars.read_events = m4_sc;
...
d.csv_m4_pto2_task_descriptor.read_events = dtask;
...
d.csv_m4_pto2_dispatch_payload.write_events = dsub * 19;
...
d.csv_m4_pto2_ready_queue.read_events = rqp;
d.csv_m4_pto2_ready_queue.atomic_ops = pop_at;
d.csv_m4_pto2_ready_queue.cas_ops = rqp;
```

---

## 10. 输出展示位置（你看到的 ④日志）

`aicpu_executor.cpp` 打印 `sp.csv_m4_*`，对应 CSV 的 ④行。

```2289:2329:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp
DEV_ALWAYS("Thread %d: --- ④任务Dispatch ---", thread_idx);
DEV_ALWAYS("Thread %d:   [④任务Dispatch] PTO2TaskSlotState      r=%" PRIu64 " w=%" PRIu64 " a=%" PRIu64 " L=%" PRIu64 " cas=%" PRIu64,
          thread_idx, sp.csv_m4_pto2_task_slot_state.read_events, ...);
...
DEV_ALWAYS("Thread %d:   [④任务Dispatch] PTO2ReadyQueue(pop)    r=%" PRIu64 " w=%" PRIu64 " a=%" PRIu64 " L=%" PRIu64 " cas=%" PRIu64,
          thread_idx, sp.csv_m4_pto2_ready_queue.read_events, ...);
```

---

## 11. 补充：字段定义注释（结构级约束）

`PTO2SchedProfilingData` 中已把 ④模块每个子结构的含义固定下来。

```999:1012:simpler/src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_scheduler.h
/** CSV ④ 行「PTO2TaskSlotState」五列（dispatch 子任务 RMW 等） */
PTO2CsvAccessCounters csv_m4_pto2_task_slot_state;
/** CSV ④ 行「PTO2TaskPayload」CL0~2 元数据读五列 */
PTO2CsvAccessCounters csv_m4_pto2_task_payload_meta;
/** CSV ④ 行「PTO2TaskPayload」CL3~34 tensors[] 参与 build_payload 的读次数五列 */
PTO2CsvAccessCounters csv_m4_pto2_task_payload_tensors;
/** CSV ④ 行「PTO2TaskPayload」CL35~50 scalars[] 参与 build_payload 的读次数五列 */
PTO2CsvAccessCounters csv_m4_pto2_task_payload_scalars;
/** CSV ④ 行「PTO2TaskDescriptor」五列 */
PTO2CsvAccessCounters csv_m4_pto2_task_descriptor;
/** CSV ④ 行「PTO2DispatchPayload」五列（按子任务写满 dispatch 缓冲的写事件近似） */
PTO2CsvAccessCounters csv_m4_pto2_dispatch_payload;
/** CSV ④ 行「PTO2ReadyQueue」全局 pop 五列 */
PTO2CsvAccessCounters csv_m4_pto2_ready_queue;
```
