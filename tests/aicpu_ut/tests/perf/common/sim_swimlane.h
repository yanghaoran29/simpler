/**
 * sim_swimlane.h
 *
 * Export a synthetic perf_swimlane_*.json from post-simulation runtime state,
 * compatible with tools/swimlane_converter.py and tools/perf_to_mermaid.py.
 *
 * Traverses PTO2TaskPayload.fanin_tasks (in shared memory) to reconstruct the
 * task dependency graph without relying on dep-list pointer validity.  Timing
 * values are synthetic: layer_index ± 0.x µs, giving swimlane_converter.py a
 * valid timeline that reflects the DAG's critical-path depth.
 *
 * Call after the scheduler has finished (aicpu_sim_run_pto2 / sim_run_*),
 * before pto2_runtime_destroy.
 *
 * Output directory is read from $AICPU_UT_SWIMLANE_DIR (set by run_tests.sh),
 * falling back to "outputs" relative to the process CWD.
 */

#pragma once

#include "pto_runtime2.h"
#include "pto_shared_memory.h"
#include "pto_runtime2_types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <sys/stat.h>
#include <vector>
#include <string>

/**
 * Export a perf_swimlane_*.json for the given runtime.
 *
 * @param rt         Completed runtime (sm_handle still valid).
 * @param output_dir Directory to write the JSON file (created if missing).
 *                   Pass nullptr to read from $AICPU_UT_SWIMLANE_DIR or use "outputs".
 * @return 0 on success, -1 on error.
 */
inline int export_sim_swimlane(PTO2Runtime* rt, const char* output_dir = nullptr) {
    if (!rt || !rt->sm_handle) return -1;

    // Resolve output directory.
    std::string dir;
    if (output_dir && *output_dir) {
        dir = output_dir;
    } else {
        const char* env = std::getenv("AICPU_UT_SWIMLANE_DIR");
        dir = (env && *env) ? env : "outputs";
    }

    PTO2SharedMemoryHandle* sm = rt->sm_handle;
    int32_t total_tasks = sm->header->current_task_index.load(std::memory_order_relaxed);
    if (total_tasks <= 0) {
        printf("  [swimlane] No tasks in runtime; skipping export.\n");
        return -1;
    }

    // ── Step 1: collect per-task info from shared memory ─────────────────────
    struct TaskInfo {
        int32_t task_id;
        int32_t kernel_id;
        int32_t worker_type;
        int32_t fanin_count;
        int32_t fanin_tasks[PTO2_MAX_INPUTS];
    };

    std::vector<TaskInfo> infos(static_cast<size_t>(total_tasks));
    uint64_t window_mask = sm->header->task_window_size - 1;

    for (int32_t tid = 0; tid < total_tasks; tid++) {
        int32_t slot = tid & static_cast<int32_t>(window_mask);
        const PTO2TaskDescriptor& desc    = sm->task_descriptors[slot];
        const PTO2TaskPayload&    payload = sm->task_payloads[slot];

        TaskInfo& info     = infos[static_cast<size_t>(tid)];
        info.task_id       = tid;
        { int32_t kid = 0;
          for (int s = 0; s < PTO2_SUBTASK_SLOT_COUNT; s++)
              if (desc.kernel_id[s] != INVALID_KERNEL_ID) { kid = desc.kernel_id[s]; break; }
          info.kernel_id = kid; }
        info.worker_type   = (desc.active_mask & PTO2_SUBTASK_MASK_AIC)
                                 ? PTO2_WORKER_CUBE : PTO2_WORKER_VECTOR;
        info.fanin_count   = payload.fanin_actual_count;
        int n = (payload.fanin_actual_count < PTO2_MAX_INPUTS)
                    ? payload.fanin_actual_count : PTO2_MAX_INPUTS;
        for (int k = 0; k < n; k++)
            info.fanin_tasks[k] = payload.fanin_tasks[k];
    }

    // ── Step 2: build fanout adjacency from fanin_tasks ──────────────────────
    // fanout_adj[i] = list of task IDs that depend on task i.
    // in_degree[i]  = number of task predecessors (not counting external inputs).
    std::vector<std::vector<int32_t>> fanout_adj(static_cast<size_t>(total_tasks));
    std::vector<int32_t> in_degree(static_cast<size_t>(total_tasks), 0);

    for (int32_t tid = 0; tid < total_tasks; tid++) {
        const TaskInfo& info = infos[static_cast<size_t>(tid)];
        for (int k = 0; k < info.fanin_count; k++) {
            int32_t pred = info.fanin_tasks[k];
            if (pred >= 0 && pred < total_tasks) {
                fanout_adj[static_cast<size_t>(pred)].push_back(tid);
                in_degree[static_cast<size_t>(tid)]++;
            }
        }
    }

    // ── Step 3: BFS topological layering ─────────────────────────────────────
    // layer[i] = topological depth (0 = no task predecessors).
    std::vector<int32_t> layer(static_cast<size_t>(total_tasks), 0);
    std::vector<int32_t> rem_in_deg = in_degree;
    std::vector<int32_t> queue;
    queue.reserve(static_cast<size_t>(total_tasks));

    for (int32_t tid = 0; tid < total_tasks; tid++) {
        if (rem_in_deg[static_cast<size_t>(tid)] == 0)
            queue.push_back(tid);
    }
    for (size_t qi = 0; qi < queue.size(); qi++) {
        int32_t cur = queue[qi];
        for (int32_t succ : fanout_adj[static_cast<size_t>(cur)]) {
            int32_t new_layer = layer[static_cast<size_t>(cur)] + 1;
            if (layer[static_cast<size_t>(succ)] < new_layer)
                layer[static_cast<size_t>(succ)] = new_layer;
            if (--rem_in_deg[static_cast<size_t>(succ)] == 0)
                queue.push_back(succ);
        }
    }

    // ── Step 4: create output directory ──────────────────────────────────────
    struct stat st{};
    if (stat(dir.c_str(), &st) != 0) {
        if (mkdir(dir.c_str(), 0755) != 0) {
            printf("  [swimlane] Cannot create output dir: %s\n", dir.c_str());
            return -1;
        }
    }

    // ── Step 5: build filename ────────────────────────────────────────────────
    std::time_t now = std::time(nullptr);
    std::tm* tmi = std::localtime(&now);
    char time_buf[32];
    std::strftime(time_buf, sizeof(time_buf), "%Y%m%d_%H%M%S", tmi);
    std::string filepath = dir + "/perf_swimlane_" + time_buf + ".json";

    // ── Step 6: open file ─────────────────────────────────────────────────────
    FILE* fp = fopen(filepath.c_str(), "w");
    if (!fp) {
        printf("  [swimlane] Cannot open file: %s\n", filepath.c_str());
        return -1;
    }

    // ── Step 7: write version-1 JSON ─────────────────────────────────────────
    // Timing is synthetic: task at layer L spans [L, L+1) µs.
    // Timestamp order mirrors the real hardware sequence:
    //   dispatch_time_us (= L) ≤ start_time_us (= L) ≤ end_time_us (= L+1) ≤ finish_time_us (= L+1)
    // This gives Head OH = 0 and Tail OH = 0, which is honest for synthetic data.
    // core_id is a round-robin across 8 synthetic cores so the AICore View
    // shows multiple lanes even with no real hardware assignment.
    fprintf(fp, "{\n  \"version\": 1,\n  \"tasks\": [\n");

    for (int32_t tid = 0; tid < total_tasks; tid++) {
        const TaskInfo& info   = infos[static_cast<size_t>(tid)];
        double t_layer         = static_cast<double>(layer[static_cast<size_t>(tid)]);
        double t_dispatch      = t_layer;        // AICPU dispatches to AICore
        double t_ready         = t_layer;        // kernel_ready ≈ dispatch (synthetic)
        double t_start         = t_layer;        // AICore starts execution
        double t_end           = t_layer + 1.0;  // AICore finishes execution
        double t_finish        = t_layer + 1.0;  // AICPU detects finish (≥ end)
        const char* core_type  = (info.worker_type == PTO2_WORKER_CUBE) ? "aic" : "aiv";
        int core_id            = tid % 8;

        const auto& fanout     = fanout_adj[static_cast<size_t>(tid)];
        int fanout_count       = static_cast<int>(fanout.size());

        fprintf(fp, "    {\n");
        fprintf(fp, "      \"task_id\": %d,\n",      info.task_id);
        fprintf(fp, "      \"func_id\": %d,\n",      info.kernel_id);
        fprintf(fp, "      \"core_id\": %d,\n",      core_id);
        fprintf(fp, "      \"core_type\": \"%s\",\n", core_type);
        fprintf(fp, "      \"start_time_us\": %.3f,\n",        t_start);
        fprintf(fp, "      \"end_time_us\": %.3f,\n",          t_end);
        fprintf(fp, "      \"duration_us\": 1.000,\n");
        fprintf(fp, "      \"kernel_ready_time_us\": %.3f,\n", t_ready);
        fprintf(fp, "      \"dispatch_time_us\": %.3f,\n",     t_dispatch);
        fprintf(fp, "      \"finish_time_us\": %.3f,\n",       t_finish);
        fprintf(fp, "      \"fanout\": [");
        for (int k = 0; k < fanout_count; k++) {
            fprintf(fp, "%d", fanout[k]);
            if (k < fanout_count - 1) fprintf(fp, ", ");
        }
        fprintf(fp, "],\n");
        fprintf(fp, "      \"fanout_count\": %d\n", fanout_count);
        fprintf(fp, "    }");
        if (tid < total_tasks - 1) fprintf(fp, ",");
        fprintf(fp, "\n");
    }

    fprintf(fp, "  ]\n}\n");
    fclose(fp);

    printf("  [swimlane] Exported %d tasks → %s\n", total_tasks, filepath.c_str());
    return 0;
}
