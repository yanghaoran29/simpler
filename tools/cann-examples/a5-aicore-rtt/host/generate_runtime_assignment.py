#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Convert an offline A5 RTT run into a packaged runtime JSON fragment."""

from __future__ import annotations

import argparse
import collections
import json
import platform
import statistics
import sys
from pathlib import Path

SCHEDULER_COUNT = 4
EXPECTED_DIE_ORDER = [0, 0, 1, 1]
AMBIGUOUS_BOUNDARY_US = 0.005


def _load_json(path: Path) -> dict:
    """Load one JSON object and reject non-object roots."""
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _parse_mask(value: object) -> int:
    """Accept the topology tool's hexadecimal string or an integer mask."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 0)
    raise ValueError("topology occupy mask is neither an integer nor a string")


def _round_die_means(raw: dict) -> dict[tuple[int, int, int], float]:
    """Aggregate every sample by (round, scheduler, physical die)."""
    frequency = int(raw.get("counter_frequency_hz", 0))
    if frequency <= 0:
        raise ValueError("raw RTT JSON has no valid counter_frequency_hz")
    grouped: dict[tuple[int, int, int], list[int]] = collections.defaultdict(list)
    for record in raw.get("records", []):
        scheduler = int(record["scheduler_index"])
        round_index = int(record["round"])
        die = int(record["die"])
        if scheduler not in range(SCHEDULER_COUNT) or die not in (0, 1):
            raise ValueError("raw RTT record contains an invalid scheduler or die")
        samples = record.get("samples_ticks", [])
        if not samples:
            raise ValueError("raw RTT record contains no completed samples")
        grouped[(round_index, scheduler, die)].extend(int(sample) for sample in samples)

    means: dict[tuple[int, int, int], float] = {}
    scale = 1_000_000.0 / frequency
    for key, samples in grouped.items():
        means[key] = statistics.fmean(samples) * scale
    return means


def derive_assignment(raw: dict) -> tuple[list[int], dict]:
    """Derive a stable two-schedulers-per-die CPU order from five rounds."""
    allowed = [int(cpu) for cpu in raw.get("allowed_cpus", [])]
    if len(allowed) != 5 or len(set(allowed)) != 5:
        raise ValueError("raw RTT JSON must contain five unique [S0,S1,S2,S3,O] CPUs")
    scheduler_cpus = allowed[:SCHEDULER_COUNT]
    round_count = int(raw.get("round_count", 0))
    if round_count != 5:
        raise ValueError(f"expected five calibration rounds, got {round_count}")

    means = _round_die_means(raw)
    partitions: list[tuple[int, int]] = []
    deltas_by_cpu: dict[int, list[float]] = {cpu: [] for cpu in scheduler_cpus}
    round_details = []
    for round_index in range(round_count):
        ranked = []
        for scheduler, cpu in enumerate(scheduler_cpus):
            try:
                die0 = means[(round_index, scheduler, 0)]
                die1 = means[(round_index, scheduler, 1)]
            except KeyError as error:
                raise ValueError(f"missing round/scheduler/die samples: {error.args[0]}") from error
            delta = die0 - die1
            deltas_by_cpu[cpu].append(delta)
            ranked.append((delta, cpu))
        ranked.sort()
        die0_cpus = tuple(sorted(cpu for _, cpu in ranked[:2]))
        partitions.append(die0_cpus)
        round_details.append(
            {
                "round": round_index,
                "die0_scheduler_cpus": list(die0_cpus),
                "deltas_us_by_cpu": {str(cpu): delta for delta, cpu in ranked},
            }
        )

    median_deltas = {cpu: statistics.median(deltas_by_cpu[cpu]) for cpu in scheduler_cpus}
    counts = collections.Counter(partitions)
    winning_partition, agreement = counts.most_common(1)[0]
    die0_cpus = (winning_partition[0], winning_partition[1])
    tie_break = None
    if agreement < 4:
        ranked_medians = sorted((delta, cpu) for cpu, delta in median_deltas.items())
        die0_anchor = ranked_medians[0][1]
        die1_anchor = ranked_medians[-1][1]
        boundary = [ranked_medians[1][1], ranked_medians[2][1]]
        boundary_gap = abs(ranked_medians[2][0] - ranked_medians[1][0])
        anchors_stable = all(die0_anchor in partition and die1_anchor not in partition for partition in partitions)
        if not anchors_stable or boundary_gap > AMBIGUOUS_BOUNDARY_US:
            raise ValueError("unstable scheduler/die affinity: the changing boundary is not an isolated <=5 ns tie")
        # The two boundary schedulers are equivalent at RTT resolution. Use
        # CPU ID as a deterministic tie-break so every card emits one order.
        selected_die0_cpus = sorted((die0_anchor, min(boundary)))
        die0_cpus = (selected_die0_cpus[0], selected_die0_cpus[1])
        agreement = counts[die0_cpus]
        tie_break = {
            "reason": "boundary schedulers differ by at most 5 ns",
            "boundary_scheduler_cpus": sorted(boundary),
            "median_delta_gap_us": boundary_gap,
            "rule": "lower CPU ID assigned to die0",
        }
    die1_cpus = tuple(sorted(set(scheduler_cpus) - set(die0_cpus)))
    order = list(die0_cpus + die1_cpus)
    metadata = {
        "method": "a5-aicore-rtt-v2",
        "rounds": round_count,
        "samples_per_core": int(raw.get("samples_requested", 0)),
        "warmup_per_core": int(raw.get("warmup_requested", 0)),
        "partition_agreement_rounds": agreement,
        "median_die0_minus_die1_us_by_cpu": {str(cpu): median_deltas[cpu] for cpu in sorted(scheduler_cpus)},
        "round_results": round_details,
    }
    if tie_break is not None:
        metadata["deterministic_tie_break"] = tie_break
    return order, metadata


def build_fragment(raw: dict, topology: dict) -> dict:
    """Build one object suitable for merging under the packaged `socs` map."""
    raw_soc = str(raw.get("soc_name", ""))
    topo_soc = str(topology.get("soc_name", ""))
    if not raw_soc or raw_soc != topo_soc:
        raise ValueError(f"SoC mismatch between RTT ({raw_soc!r}) and topology ({topo_soc!r})")
    topology_allowed = [int(cpu) for cpu in topology.get("launch_plan", {}).get("allowed_cpus", [])]
    raw_allowed = [int(cpu) for cpu in raw.get("allowed_cpus", [])]
    if len(topology_allowed) != 5 or set(topology_allowed[:4]) != set(raw_allowed[:4]):
        raise ValueError("topology and RTT scheduler CPU sets differ")
    occupy = _parse_mask(topology["device_masks"]["occupy"]["value"])
    order, calibration = derive_assignment(raw)
    return {
        raw_soc: {
            "host_arch": platform.machine(),
            "occupy_mask": occupy,
            "rtt_scheduler_assignment": {
                "scheduler_count": SCHEDULER_COUNT,
                "scheduler_cpu_order": order,
                "aicore_die_order": EXPECTED_DIE_ORDER,
                "calibration": calibration,
            },
        }
    }


def main() -> int:
    """Parse command-line paths, validate the run, and atomically emit JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_json", type=Path, help="raw RTT JSON written by launch_a5_aicore_rtt")
    parser.add_argument("--topology", required=True, type=Path, help="aicpu-device-query --json output")
    parser.add_argument("--output", required=True, type=Path, help="runtime JSON fragment to create")
    args = parser.parse_args()
    try:
        fragment = build_fragment(_load_json(args.raw_json), _load_json(args.topology))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(fragment, stream, indent=2, sort_keys=False)
            stream.write("\n")
        temporary.replace(args.output)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"runtime assignment fragment: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
