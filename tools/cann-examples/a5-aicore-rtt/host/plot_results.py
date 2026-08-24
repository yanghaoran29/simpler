#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Plot per-scheduler, per-AICore RTT means from benchmark JSON."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot
    from matplotlib.lines import Line2D
except ImportError as error:
    pyplot = None
    Line2D = None
    MATPLOTLIB_ERROR = error
else:
    MATPLOTLIB_ERROR = None


def parse_args():
    """Parse the raw JSON input and destination image path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path, help="Raw a5-aicore-rtt JSON file")
    parser.add_argument("--output", required=True, type=Path, help="Destination PNG, PDF, or SVG file")
    return parser.parse_args()


def load_core_means(path):
    """Aggregate all rounds and samples into one mean per scheduler and logical core."""
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    frequency = payload.get("counter_frequency_hz", 0)
    if frequency <= 0:
        raise ValueError("counter_frequency_hz must be positive")

    samples = defaultdict(list)
    metadata = {}
    # Preserve the measured physical identity while combining raw ticks across rounds.
    for record in payload.get("records", []):
        key = (record["scheduler_index"], record["logical_core_id"])
        samples[key].extend(record.get("samples_ticks", []))
        metadata[key] = {
            "kind": record["kind"],
            "die": record["die"],
            "cluster": record["cluster"],
            "lane": record["lane"],
        }

    means = []
    # Convert the raw counter-tick arithmetic mean into microseconds.
    for key, ticks in samples.items():
        if not ticks:
            continue
        scheduler, logical_core = key
        means.append(
            {
                "scheduler": scheduler,
                "logical_core": logical_core,
                "mean_us": sum(ticks) * 1.0e6 / (len(ticks) * frequency),
                **metadata[key],
            }
        )
    return payload, means


def render_plot(payload, means, output_path, pyplot, line2d):
    """Render four scheduler panels using kind markers and die colors."""
    scheduler_count = payload.get("scheduler_count", 4)
    if scheduler_count != 4:
        raise ValueError(f"expected four schedulers, found {scheduler_count}")

    fig, axes = pyplot.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True, constrained_layout=True)
    marker_by_kind = {"AIC": "o", "AIV": "s"}
    color_by_die = {0: "red", 1: "blue"}

    # Draw every logical core independently so the two AIV lanes remain visible.
    for scheduler, axis in enumerate(axes.flat):
        scheduler_points = [point for point in means if point["scheduler"] == scheduler]
        for kind, marker in marker_by_kind.items():
            for die, color in color_by_die.items():
                selected = [point for point in scheduler_points if point["kind"] == kind and point["die"] == die]
                axis.scatter(
                    [point["logical_core"] for point in selected],
                    [point["mean_us"] for point in selected],
                    marker=marker,
                    c=color,
                    s=24,
                    alpha=0.8,
                )
        axis.set_title(f"Scheduler S{scheduler}")
        axis.set_xlabel("Logical AICore ID")
        axis.set_ylabel("Mean RTT (us)")
        axis.grid(True, alpha=0.25)

    # Use independent legends so marker shape and die color have unambiguous meanings.
    kind_handles = [
        line2d([], [], color="black", marker="o", linestyle="None", label="AIC"),
        line2d([], [], color="black", marker="s", linestyle="None", label="AIV"),
    ]
    die_handles = [
        line2d([], [], color="red", marker="o", linestyle="None", label="die0"),
        line2d([], [], color="blue", marker="o", linestyle="None", label="die1"),
    ]
    fig.legend(handles=kind_handles + die_handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4)
    fig.suptitle(f"A5 scheduler-to-AICore RTT ({payload.get('cluster_count', '?')} runtime clusters)", y=1.04)
    # Create parent directories only after input parsing and aggregation succeed.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    pyplot.close(fig)


def main():
    """Load benchmark data and skip plotting cleanly when Matplotlib is unavailable."""
    args = parse_args()
    if pyplot is None or Line2D is None:
        print(f"warning: matplotlib is unavailable; skipping RTT plot ({MATPLOTLIB_ERROR})", file=sys.stderr)
        return 0

    try:
        payload, means = load_core_means(args.json_path)
        render_plot(payload, means, args.output, pyplot, Line2D)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"warning: cannot render RTT plot: {error}", file=sys.stderr)
        return 1
    print(f"plot: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
