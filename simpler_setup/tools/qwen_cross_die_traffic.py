#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Estimate Qwen3 dependency bytes whose producer and consumer run on different Dies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from simpler_setup.tools.swimlane_converter import read_perf_data

GATE_FUNCS = (21, 23, 25, 27, 29)
UP_FUNCS = (22, 24, 26, 28, 30)
GATE_LATE_FUNC = 31
UP_LATE_FUNC = 32
SILU_FUNC = 33
DOWN_FUNC = 34
DCR_FUNC = 35
LAYERS = 40
MLP_SHARDS = 17
DOWN_GROUPS = 5
GATE_SPLITS = 5
GATE_ACCUMULATOR_BYTES = 16 * 1024 * 4
SILU_SHARD_BYTES = 16 * 1024 * 2
DOWN_ACCUMULATOR_BYTES = 16 * 1024 * 4

DTYPE_BYTES = {
    "BOOL": 1,
    "INT8": 1,
    "UINT8": 1,
    "INT16": 2,
    "UINT16": 2,
    "FLOAT16": 2,
    "BFLOAT16": 2,
    "INT32": 4,
    "UINT32": 4,
    "FLOAT32": 4,
    "INT64": 8,
    "UINT64": 8,
    "FLOAT64": 8,
}


def _task_id(task: dict) -> int:
    return int(task["task_id"])


def _kernel_id(task: dict) -> int:
    ids = task.get("kernel_ids", [])
    return next((int(value) for value in ids if int(value) >= 0), -1)


def _require_task_count(by_func: dict[int, list[dict]], func: int, expected: int) -> None:
    actual = len(by_func[func])
    if actual != expected:
        raise ValueError(f"func {func} expected {expected} tasks, found {actual}")


def _index_initial_gate_up(by_func: dict[int, list[dict]], index: dict[tuple, int]) -> None:
    for split, func in enumerate(GATE_FUNCS):
        _require_task_count(by_func, func, LAYERS)
        for layer, task in enumerate(by_func[func]):
            for shard in range(6):
                index[("gate", layer, shard, split)] = _task_id(task)
    for split, func in enumerate(UP_FUNCS):
        _require_task_count(by_func, func, LAYERS)
        for layer, task in enumerate(by_func[func]):
            for shard in range(6):
                index[("up", layer, shard, split)] = _task_id(task)


def _index_late_gate_up(by_func: dict[int, list[dict]], index: dict[tuple, int]) -> None:
    for role, func in (("gate", GATE_LATE_FUNC), ("up", UP_LATE_FUNC)):
        expected = LAYERS * (MLP_SHARDS - 6) * GATE_SPLITS
        _require_task_count(by_func, func, expected)
        for ordinal, task in enumerate(by_func[func]):
            layer, within = divmod(ordinal, (MLP_SHARDS - 6) * GATE_SPLITS)
            shard_delta, split = divmod(within, GATE_SPLITS)
            index[(role, layer, shard_delta + 6, split)] = _task_id(task)


def _index_mlp_outputs(by_func: dict[int, list[dict]], index: dict[tuple, int]) -> None:
    for func, role, per_layer in (
        (SILU_FUNC, "silu", MLP_SHARDS),
        (DOWN_FUNC, "down", DOWN_GROUPS * MLP_SHARDS),
        (DCR_FUNC, "dcr", 1),
    ):
        expected = LAYERS * per_layer
        _require_task_count(by_func, func, expected)
        for ordinal, task in enumerate(by_func[func]):
            layer, within = divmod(ordinal, per_layer)
            if role == "silu":
                index[(role, layer, within)] = _task_id(task)
            elif role == "down":
                output_group, input_shard = divmod(within, MLP_SHARDS)
                index[(role, layer, output_group, input_shard)] = _task_id(task)
            else:
                for output_group in range(DOWN_GROUPS):
                    index[(role, layer, output_group)] = _task_id(task)


def index_qwen_tasks(deps: dict) -> dict[tuple, int]:
    """Map semantic MLP coordinates to task ids using the generated graph's stable order."""
    by_func: dict[int, list[dict]] = defaultdict(list)
    for task in deps.get("tasks", []):
        by_func[_kernel_id(task)].append(task)
    for tasks in by_func.values():
        tasks.sort(key=_task_id)

    index: dict[tuple, int] = {}
    _index_initial_gate_up(by_func, index)
    _index_late_gate_up(by_func, index)
    _index_mlp_outputs(by_func, index)
    return index


def data_aware_placement(index: dict[tuple, int]) -> dict[tuple[int, int], int]:
    """Return the deterministic mode6 MLP task/block placement."""
    placement: dict[tuple[int, int], int] = {}
    for key, task_id in index.items():
        role, layer = key[0], key[1]
        if role in ("gate", "up"):
            shard = key[2]
            block = shard if shard < 6 else 0
            placement[(task_id, block)] = (layer + shard) & 1
        elif role == "silu":
            placement[(task_id, 0)] = (layer + key[2]) & 1
        elif role == "down":
            placement[(task_id, 0)] = (layer + key[2] + 1) & 1
        elif role == "dcr":
            output_group = key[2]
            placement[(task_id, output_group)] = (layer + output_group + 1) & 1
    return placement


def placement_from_swimlane(path: Path) -> dict[tuple[int, int], int]:
    """Recover logical block placement by dispatch order within each task id."""
    perf = read_perf_data(path)
    core_types = perf.get("core_types")
    if core_types is None:
        with path.open() as handle:
            core_types = json.load(handle).get("metadata", {}).get("core_types", [])
    aic_count = sum(core_type == "aic" for core_type in core_types)
    if aic_count <= 0:
        raise ValueError(f"{path}: missing AIC core metadata")

    occurrences: dict[int, list[dict]] = defaultdict(list)
    for task in perf["tasks"]:
        occurrences[int(task["task_id"])].append(task)
    placement: dict[tuple[int, int], int] = {}
    for task_id, rows in occurrences.items():
        rows.sort(key=lambda row: (row.get("dispatch_time_us", 0.0), row["core_id"]))
        for block_idx, row in enumerate(rows):
            core_id = int(row["core_id"])
            cluster = core_id if row["core_type"] == "aic" else (core_id - aic_count) // 2
            placement[(task_id, block_idx)] = int(cluster >= math.ceil(aic_count / 2))
    return placement


def dependency_view_upper_bound(deps: dict) -> tuple[int, int]:
    """Return deduplicated data-view bytes and the count lacking tensor metadata."""
    views: dict[tuple, int] = {}
    edges_without_tensor_view = 0
    for edge in deps.get("edges", []):
        dtype = edge.get("consumer_dtype")
        shape = edge.get("consumer_shape")
        if not dtype or shape is None or edge.get("tensor_id") is None:
            edges_without_tensor_view += 1
            continue
        key = (
            str(edge.get("succ")),
            edge.get("arg"),
            str(edge.get("tensor_id")),
            str(edge.get("consumer_start_offset", "0")),
            tuple(shape),
            tuple(edge.get("consumer_strides", [])),
        )
        views[key] = math.prod(int(dim) for dim in shape) * DTYPE_BYTES.get(str(dtype), 0)
    return sum(views.values()), edges_without_tensor_view


def calculate_mlp_traffic(index: dict[tuple, int], placement: dict[tuple[int, int], int]) -> dict:
    categories = defaultdict(int)
    directions = defaultdict(int)
    by_layer = []

    def die(key: tuple, block: int = 0) -> int:
        task_id = index[key]
        try:
            return placement[(task_id, block)]
        except KeyError as exc:
            raise ValueError(f"placement missing task={task_id} block={block} semantic={key}") from exc

    def add(layer_counts: dict[str, int], category: str, amount: int, source_die: int, target_die: int) -> None:
        if source_die == target_die:
            return
        categories[category] += amount
        directions[f"die{source_die}_to_die{target_die}"] += amount
        layer_counts[category] += amount

    for layer in range(LAYERS):
        layer_counts: dict[str, int] = defaultdict(int)
        for shard in range(MLP_SHARDS):
            silu_die = die(("silu", layer, shard))
            for role in ("gate", "up"):
                for split in range(GATE_SPLITS):
                    block = shard if shard < 6 else 0
                    producer_die = die((role, layer, shard, split), block)
                    add(layer_counts, f"{role}_atomic_to_silu_home", GATE_ACCUMULATOR_BYTES, producer_die, silu_die)

        for output_group in range(DOWN_GROUPS):
            dcr_die = die(("dcr", layer, output_group), output_group)
            for input_shard in range(MLP_SHARDS):
                silu_die = die(("silu", layer, input_shard))
                down_die = die(("down", layer, output_group, input_shard))
                add(layer_counts, "silu_read_by_down", SILU_SHARD_BYTES, silu_die, down_die)
                add(layer_counts, "down_atomic_to_dcr_home", DOWN_ACCUMULATOR_BYTES, down_die, dcr_die)
        by_layer.append({"layer": layer, "bytes": sum(layer_counts.values()), "categories": dict(layer_counts)})

    return {
        "logical_cross_die_bytes": sum(categories.values()),
        "categories": dict(sorted(categories.items())),
        "directions": dict(sorted(directions.items())),
        "by_layer": by_layer,
    }


def _write_outputs(output_dir: Path, report: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "cross_die_traffic.json").write_text(json.dumps(report, indent=2) + "\n")
    with (output_dir / "cross_die_traffic.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("capture", "logical_cross_die_bytes", "logical_cross_die_mib"))
        for capture in report["captures"]:
            value = capture["logical_cross_die_bytes"]
            writer.writerow((capture["name"], value, f"{value / (1024 * 1024):.6f}"))
    lines = [
        "# Qwen3 logical cross-Die dependency traffic",
        "",
        "This is a logical data-footprint estimate, not a physical NoC transaction count.",
        "",
        f"- Dependency-view upper bound: `{report['dependency_view_upper_bound_bytes'] / (1024 * 1024):.3f} MiB`",
        f"- Edges without a tensor view: `{report['edges_without_tensor_view']}`",
        f"- MLP cross-Die median: `{report['median_logical_cross_die_bytes'] / (1024 * 1024):.3f} MiB`",
        "",
        "| Capture | Logical cross-Die MiB |",
        "|---|---:|",
    ]
    for capture in report["captures"]:
        lines.append(f"| {capture['name']} | {capture['logical_cross_die_bytes'] / (1024 * 1024):.3f} |")
    (output_dir / "cross_die_traffic.md").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deps-json", type=Path, required=True)
    parser.add_argument("--placement-json", type=Path, action="append", default=[])
    parser.add_argument("--policy", choices=("data-aware",), action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    deps = json.loads(args.deps_json.read_text())
    index = index_qwen_tasks(deps)
    captures = []
    for path in args.placement_json:
        captures.append({"name": path.parent.name, **calculate_mlp_traffic(index, placement_from_swimlane(path))})
    for policy in args.policy:
        captures.append({"name": policy, **calculate_mlp_traffic(index, data_aware_placement(index))})
    if not captures:
        parser.error("at least one --placement-json or --policy is required")

    upper_bound, edges_without_tensor_view = dependency_view_upper_bound(deps)
    report = {
        "schema_version": 1,
        "metric": "logical_dependency_bytes",
        "deps_json": str(args.deps_json),
        "dependency_view_upper_bound_bytes": upper_bound,
        "edges_without_tensor_view": edges_without_tensor_view,
        "median_logical_cross_die_bytes": int(
            statistics.median(capture["logical_cross_die_bytes"] for capture in captures)
        ),
        "captures": captures,
    }
    _write_outputs(args.output_dir, report)
    print(json.dumps({key: value for key, value in report.items() if key != "captures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
