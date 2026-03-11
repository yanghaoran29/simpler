#!/usr/bin/env python3
"""Scheduler overhead analysis for PTO2.

Analyzes scheduling overhead from two sources:
  1. Per-task perf profiling data (perf_swimlane_*.json)
  2. AICPU scheduler loop breakdown (device log)

Usage:
    python sched_overhead_analysis.py                          # auto-select latest files
    python sched_overhead_analysis.py --perf-json <path>       # specify perf data
    python sched_overhead_analysis.py --device-log <path>      # specify device log
    python sched_overhead_analysis.py --perf-json <path> -d 0  # resolve from device-0
"""
import argparse
import json
import re
import sys
from pathlib import Path

try:
    from device_log_resolver import infer_device_id_from_log_path, resolve_device_log_path
except ImportError:
    from tools.device_log_resolver import infer_device_id_from_log_path, resolve_device_log_path


def auto_select_perf_json():
    """Find the latest perf_swimlane_*.json in outputs/ directory."""
    outputs_dir = Path(__file__).parent.parent / 'outputs'
    files = sorted(outputs_dir.glob('perf_swimlane_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No perf_swimlane_*.json files found in {outputs_dir}")
    return files[0]


def parse_scheduler_threads(log_path):
    """Parse device log for PTO2 scheduler stats per thread.

    Supports three formats:
    1. New two-level tree (PTO2_SCHED_PROFILING=1):
        Thread N: === Scheduler Phase Breakdown: total=Xus, Y tasks ===
        Thread N:   complete       : Xus (Y%)  [fanout: edges=A, max_degree=B, avg=C]  [fanin: edges=D, max_degree=E, avg=F]
        Thread N:   dispatch       : Xus (Y%)  [pop: hit=A, miss=B, hit_rate=C%]
        Thread N:   scan           : Xus (Y%)
        Thread N:   idle           : Xus (Y%)

    2. Summary (PTO2_SCHED_PROFILING=0):
        Thread N: Scheduler summary: total_time=Xus, loops=Y, tasks_scheduled=Z
    """
    threads = {}
    with open(log_path, 'r', errors='ignore') as f:
        for line in f:
            # New format: Thread N: === Scheduler Phase Breakdown: total=Xus, Y tasks ===
            m = re.search(r'Thread (\d+): === Scheduler Phase Breakdown: total=([\d.]+)us, (\d+) tasks ===', line)
            if m:
                tid = int(m.group(1))
                threads[tid] = {
                    'completed': int(m.group(3)),
                    'total_us': float(m.group(2)),
                    'format': 'two-level',
                }

            # Summary format: Thread N: Scheduler summary: total_time=Xus, loops=Y, tasks_scheduled=Z
            # Merge with existing thread dict so Phase Breakdown data (complete_us, etc.) is kept.
            m = re.search(r'Thread (\d+): Scheduler summary: total_time=([\d.]+)us, loops=(\d+), tasks_scheduled=(\d+)', line)
            if m:
                tid = int(m.group(1))
                total_us = float(m.group(2))
                loops = int(m.group(3))
                completed = int(m.group(4))
                tasks_per_loop = completed / loops if loops > 0 else 0.0
                entry = {
                    'completed': completed,
                    'total_us': total_us,
                    'loops': loops,
                    'tasks_per_loop': tasks_per_loop,
                    'format': 'summary',
                }
                if tid in threads:
                    threads[tid].update(entry)
                else:
                    threads[tid] = entry

            # New format phase lines: Thread N:   complete       : Xus (Y%)
            m = re.search(r'Thread (\d+):\s+(complete|dispatch|scan|idle)\s+:\s+([\d.]+)us \(\s*([\d.]+)%\)', line)
            if m:
                tid = int(m.group(1))

            # New format: complete with fanout/fanin stats
            m = re.search(
                r'Thread (\d+):\s+complete\s+:\s+([\d.]+)us \(\s*([\d.]+)%\)'
                r'\s+\[fanout: edges=(\d+), max_degree=(\d+), avg=([\d.]+)\]'
                r'\s+\[fanin: edges=(\d+), max_degree=(\d+), avg=([\d.]+)\]',
                line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['complete_us'] = float(m.group(2))
                    threads[tid]['complete_pct'] = float(m.group(3))
                    threads[tid]['fanout_edges'] = int(m.group(4))
                    threads[tid]['fanout_max_degree'] = int(m.group(5))
                    threads[tid]['fanout_avg'] = float(m.group(6))
                    threads[tid]['fanin_edges'] = int(m.group(7))
                    threads[tid]['fanin_max_degree'] = int(m.group(8))
                    threads[tid]['fanin_avg'] = float(m.group(9))
                continue

            # New format: dispatch with pop stats
            m = re.search(
                r'Thread (\d+):\s+dispatch\s+:\s+([\d.]+)us \(\s*([\d.]+)%\)'
                r'\s+\[pop: hit=(\d+), miss=(\d+), hit_rate=([\d.]+)%\]',
                line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    threads[tid]['dispatch_us'] = float(m.group(2))
                    threads[tid]['dispatch_pct'] = float(m.group(3))
                    threads[tid]['pop_hit'] = int(m.group(4))
                    threads[tid]['pop_miss'] = int(m.group(5))
                    threads[tid]['pop_hit_rate'] = float(m.group(6))
                continue

            # New format: scan and idle (no extra stats)
            m = re.search(r'Thread (\d+):\s+(scan|idle)\s+:\s+([\d.]+)us \(\s*([\d.]+)%\)', line)
            if m:
                tid = int(m.group(1))
                if tid in threads:
                    phase = m.group(2)
                    threads[tid][f'{phase}_us'] = float(m.group(3))
                    threads[tid][f'{phase}_pct'] = float(m.group(4))
                continue

    return threads


def validate_perf_tasks_for_overhead_analysis(tasks):
    """Validate required per-task fields for overhead deep-dive analysis.

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    required_fields = [
        "duration_us",
        "start_time_us",
        "end_time_us",
        "dispatch_time_us",
        "finish_time_us",
    ]

    missing = []
    for idx, task in enumerate(tasks):
        missing_fields = [field for field in required_fields if field not in task]
        if missing_fields:
            task_label = task.get("task_id", idx)
            missing.append(f"task={task_label} missing={','.join(missing_fields)}")
            if len(missing) >= 5:
                break

    if missing:
        detail = "; ".join(missing)
        # These fields are produced by runtime-side JSON export in:
        # src/platform/src/host/performance_collector.cpp (dispatch_time_us, finish_time_us)
        msg = "\n".join([
            "Perf JSON is incompatible with scheduler overhead deep-dive analysis.",
            f"Missing required fields (showing up to 5 tasks): {detail}",
            "",
            "Why this happens:",
            "  - The input is not a runtime-generated perf_swimlane_*.json, OR",
            "  - The runtime binary does not include / emit dispatch+finish timestamps.",
            "",
            "How to fix:",
            "  1) Re-run workload with profiling enabled (e.g. run_example.py --enable-profiling).",
            "  2) Use the newly generated outputs/perf_swimlane_*.json as --perf-json input.",
            "  3) Verify each task includes dispatch_time_us and finish_time_us.",
            "",
            "Note:",
            "  - swimlane_converter conversion can still succeed; only deep-dive analysis requires these fields.",
        ])
        return False, msg

    return True, ""


def _section_header_80(title, pad_char='-'):
    """Print an 80-char line with title centered, padded with pad_char."""
    n = len(title)
    left = (80 - n) // 2
    right = 80 - n - left
    print((pad_char * left) + title + (pad_char * right))


def _section_header_96_indent2(title, pad_char='-'):
    """Print a 96-char line with 2-space indent and title centered in the remaining 94 chars."""
    indent = '  '
    content_len = 96 - len(indent)  # 94
    n = len(title)
    left = (content_len - n) // 2
    right = content_len - n - left
    print(indent + (pad_char * left) + title + (pad_char * right))


def _run_part2_only(log_path, print_sources=True, log_label='sim log'):
    """Run only Part 2 (scheduler loop + Phase breakdown) from a log file.
    Used when --sim-log is provided and no perf JSON (e.g. after run_tests.sh aicpu_ut).
    """
    log_path = Path(log_path)
    if not log_path.exists():
        print(f"Error: Log not found: {log_path}", file=sys.stderr)
        return 1
    if print_sources:
        print(f"{log_label}: {log_path}")
    threads = parse_scheduler_threads(log_path)
    n_threads = len(threads)
    if n_threads == 0:
        print("No scheduler thread lines found in log.", file=sys.stderr)
        return 1
    print()
    _section_header_96_indent2("--- AICPU scheduler loop breakdown ---")
    print()
    indent = "  "
    fmt2 = indent + "  {:<10} {:>7} {:>10} {:>12} {:>11}"
    print(fmt2.format('Thread', 'Loops', 'Completed', 'Tasks/loop', 'Total (us)'))
    print(indent + "  " + '-' * 54)
    for tid in sorted(threads.keys()):
        t = threads[tid]
        loops = t.get('loops', 0)
        tpl = t.get('tasks_per_loop', (t['completed'] / loops if loops > 0 else 0))
        print(fmt2.format('T' + str(tid), loops, t['completed'], f"{tpl:.3f}", f"{t['total_us']:.1f}"))
    total_us = sum(t['total_us'] for t in threads.values())
    total_completed = sum(t['completed'] for t in threads.values())
    total_loops = sum(t.get('loops', 0) for t in threads.values())
    avg_tpl = total_completed / total_loops if total_loops > 0 else 0
    print(fmt2.format('SUM', total_loops, total_completed, f'{avg_tpl:.3f}', f'{total_us:.1f}'))
    print()
    phases = ['complete', 'scan', 'dispatch', 'idle']
    phase_labels = {
        'complete': 'Complete (poll handshake, resolve deps)',
        'scan': 'Scan (update perf header)',
        'dispatch': 'Dispatch (pop queue, build payload, flush)',
        'idle': 'Idle (spinning, no progress)',
    }
    fmt3 = indent + "  {:<50} {:>11} {:>10} {:>14}"
    print(fmt3.format('Phase', 'Total (us)', '% of total', 'Avg/task (us)'))
    print(indent + "  " + '-' * 89)
    for p in phases:
        key = p + '_us'
        tot = sum(t.get(key, 0) for t in threads.values())
        pct = tot / total_us * 100 if total_us > 0 else 0
        avg = tot / total_completed if total_completed > 0 else 0
        print(fmt3.format(phase_labels[p], f'{tot:.1f}', f'{pct:.1f}%', f'{avg:.2f}'))
    fanout_edges = sum(t.get('fanout_edges', 0) for t in threads.values())
    fanout_max = max((t.get('fanout_max_degree', 0) for t in threads.values()), default=0)
    fanout_avg = fanout_edges / total_completed if total_completed > 0 else 0
    print(indent + f'  Fanout (notify consumers): total edges={fanout_edges}, max_degree={fanout_max}, avg_degree={fanout_avg:.1f}')
    fanin_edges = sum(t.get('fanin_edges', 0) for t in threads.values())
    fanin_max = max((t.get('fanin_max_degree', 0) for t in threads.values()), default=0)
    fanin_avg = fanin_edges / total_completed if total_completed > 0 else 0
    print(indent + f'  Fanin  (release producers): total edges={fanin_edges}, max_degree={fanin_max}, avg_degree={fanin_avg:.1f}')
    print()
    pop_hit = sum(t.get('pop_hit', 0) for t in threads.values())
    pop_miss = sum(t.get('pop_miss', 0) for t in threads.values())
    pop_total = pop_hit + pop_miss
    pop_hit_rate = pop_hit / pop_total * 100 if pop_total > 0 else 0
    print(indent + f'  Pop: hit={pop_hit}, miss={pop_miss}, hit_rate={pop_hit_rate:.1f}%')
    print()
    return 0


def run_analysis(perf_path, log_path, print_sources=True, selection_strategy=None):
    """Run scheduler overhead analysis report.

    Args:
        perf_path: Path to perf_swimlane_*.json.
        log_path: Path to device log or sim log file.
        print_sources: Whether to print selected input files.
        selection_strategy: Optional human-readable device-log selection strategy.

    Returns:
        int: 0 on success, non-zero on failure.
    """
    perf_path = Path(perf_path)
    log_path = Path(log_path)

    if not perf_path.exists():
        print(f"Error: Perf JSON not found: {perf_path}", file=sys.stderr)
        return 1
    if not log_path.exists():
        print(f"Error: Log not found: {log_path}", file=sys.stderr)
        return 1

    if print_sources:
        print(f"Perf data:  {perf_path}")
        print(f"Log:        {log_path}")
        if selection_strategy:
            print(f"Selection:  {selection_strategy}")
        inferred_device_id = infer_device_id_from_log_path(log_path)
        if inferred_device_id is not None:
            print(f"Device ID:  {inferred_device_id}")

    # === Part 1: Per-task time breakdown from perf data ===
    with open(perf_path) as f:
        data = json.load(f)
    tasks = data['tasks']
    n_total = len(tasks)

    if n_total == 0:
        print("Error: No tasks found in perf data", file=sys.stderr)
        return 1

    valid, err = validate_perf_tasks_for_overhead_analysis(tasks)
    if not valid:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    all_exec = sum(t['duration_us'] for t in tasks)
    all_head = sum(t['start_time_us'] - t['dispatch_time_us'] for t in tasks)
    all_tail = sum(t['finish_time_us'] - t['end_time_us'] for t in tasks)
    min_disp = min(t['dispatch_time_us'] for t in tasks)
    max_fin = max(t['finish_time_us'] for t in tasks)
    wall = max_fin - min_disp

    all_latency = all_exec + all_head + all_tail

    print()
    print('=' * 90)
    print('Part 1: Per-task time breakdown (from perf profiling data)')
    print('=' * 90)
    print(f'Total tasks: {n_total}')
    print(f'Wall-clock:  {wall:.1f} us')
    print()
    fmt = "  {:<35} {:>12} {:>14} {:>13}"
    print(fmt.format('Component', 'Total (us)', 'Avg/task (us)', '% of Latency'))
    print('  ' + '-' * 78)
    print(fmt.format('Kernel Exec (end-start)', f'{all_exec:.1f}', f'{all_exec/n_total:.2f}', f'{all_exec/all_latency*100:.1f}%'))
    print(fmt.format('Head OH (start-dispatch)', f'{all_head:.1f}', f'{all_head/n_total:.2f}', f'{all_head/all_latency*100:.1f}%'))
    print(fmt.format('Tail OH (finish-end)', f'{all_tail:.1f}', f'{all_tail/n_total:.2f}', f'{all_tail/all_latency*100:.1f}%'))
    print()

    # === Part 2: AICPU scheduler loop breakdown from log ===
    threads = parse_scheduler_threads(log_path)
    n_threads = len(threads)

    print('=' * 90)
    print('Part 2: AICPU scheduler loop breakdown (from log)')
    print(f'  {n_threads} scheduler threads')
    print('=' * 90)
    print()

    fmt2 = "  {:<10} {:>7} {:>10} {:>12} {:>11}"
    print(fmt2.format('Thread', 'Loops', 'Completed', 'Tasks/loop', 'Total (us)'))
    print('  ' + '-' * 54)
    for tid in sorted(threads.keys()):
        t = threads[tid]
        loops = t.get('loops', 0)
        tpl = t.get('tasks_per_loop', (t['completed'] / loops if loops > 0 else 0))
        print(fmt2.format('T'+str(tid), loops, t['completed'], f"{tpl:.3f}", f"{t['total_us']:.1f}"))
    total_us = sum(t['total_us'] for t in threads.values())
    total_completed = sum(t['completed'] for t in threads.values())
    total_loops = sum(t.get('loops', 0) for t in threads.values())
    avg_tpl = total_completed / total_loops if total_loops > 0 else 0
    print(fmt2.format('SUM', total_loops, total_completed, f'{avg_tpl:.3f}', f'{total_us:.1f}'))
    print()

    # Phase breakdown
    phases = ['complete', 'scan', 'dispatch', 'idle']
    phase_labels = {
        'complete':    'Complete (poll handshake, resolve deps)',
        'scan':        'Scan (update perf header)',
        'dispatch':    'Dispatch (pop queue, build payload, flush)',
        'idle':        'Idle (spinning, no progress)',
    }

    fmt3 = "  {:<50} {:>11} {:>10} {:>14}"
    print(fmt3.format('Phase', 'Total (us)', '% of total', 'Avg/task (us)'))
    print('  ' + '-' * 89)
    phase_totals = {}
    for p in phases:
        key = p + '_us'
        tot = sum(t.get(key, 0) for t in threads.values())
        phase_totals[p] = tot
        pct = tot / total_us * 100 if total_us > 0 else 0
        avg = tot / total_completed if total_completed > 0 else 0
        print(fmt3.format(phase_labels[p], f'{tot:.1f}', f'{pct:.1f}%', f'{avg:.2f}'))
    sum_phase_us = sum(phase_totals.values())
    if total_us > 0 and sum_phase_us == 0:
        print('  (Phase/Fanout/Fanin/Pop below are 0: log has no per-phase breakdown.)')
        print('  To get Phase data: use --sim-log with aicpu_ut output, or run on device with PTO2_SCHED_PROFILING=ON.')
    print()

    # Fanout stats (from complete phase)
    fanout_edges = sum(t.get('fanout_edges', 0) for t in threads.values())
    fanout_max = max((t.get('fanout_max_degree', 0) for t in threads.values()), default=0)
    fanout_avg = fanout_edges / total_completed if total_completed > 0 else 0
    print(f'  Fanout (notify consumers): total edges={fanout_edges}, max_degree={fanout_max}, avg_degree={fanout_avg:.1f}')

    # Fanin stats (from complete phase)
    fanin_edges = sum(t.get('fanin_edges', 0) for t in threads.values())
    fanin_max = max((t.get('fanin_max_degree', 0) for t in threads.values()), default=0)
    fanin_avg = fanin_edges / total_completed if total_completed > 0 else 0
    print(f'  Fanin  (release producers): total edges={fanin_edges}, max_degree={fanin_max}, avg_degree={fanin_avg:.1f}')
    print()

    # Pop stats (from dispatch phase)
    pop_hit = sum(t.get('pop_hit', 0) for t in threads.values())
    pop_miss = sum(t.get('pop_miss', 0) for t in threads.values())
    pop_total = pop_hit + pop_miss
    pop_hit_rate = pop_hit / pop_total * 100 if pop_total > 0 else 0
    print(f'  Pop: hit={pop_hit}, miss={pop_miss}, hit_rate={pop_hit_rate:.1f}%')

    print()
    print('=' * 90)
    print('Part 3: Tail OH distribution & cause analysis')
    print('=' * 90)
    print()

    tails = [t['finish_time_us'] - t['end_time_us'] for t in tasks]
    tails.sort()
    n = len(tails)
    if n == 0:
        print('Error: Empty tail-overhead set', file=sys.stderr)
        return 1

    print(f'  Tail OH distribution (N={n}):')
    for pct_val in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * pct_val / 100), n - 1)
        print(f'    P{pct_val:<4}  {tails[idx]:>7.1f} us')
    print(f'    Max:   {tails[-1]:>7.1f} us')
    print(f'    Mean:  {sum(tails)/n:>7.1f} us')
    print()

    # Scheduler loop time
    avg_loop_us = total_us / total_loops if total_loops > 0 else 0
    avg_tail_oh = sum(tails) / n
    loop_ratio = avg_tail_oh / avg_loop_us if avg_loop_us > 0 else 0
    print(f'  Avg scheduler loop iteration: {avg_loop_us:.1f} us (approx avg polling interval per loop)')
    print()
    print(f'  Avg Tail OH = {avg_tail_oh:.1f} us ~= {loop_ratio:.1f} x avg loop iteration ({avg_loop_us:.1f} us)')
    print(f'  -> On average, a completed task waits ~{loop_ratio:.1f} loop iterations before being detected')
    print()

    # Data-driven insight: find the dominant phase (excluding idle which is not useful work)
    work_phases = {p: phase_totals.get(p, 0) for p in ['scan', 'complete', 'dispatch']}
    dominant_phase = max(work_phases, key=work_phases.get)
    dominant_pct = work_phases[dominant_phase] / total_us * 100 if total_us > 0 else 0
    print(f'  Key insight: {phase_labels[dominant_phase].split(" (")[0]} phase consumes ~{dominant_pct:.0f}% of scheduler CPU.')
    if dominant_phase == 'dispatch':
        print(f'  Pop hit_rate={pop_hit_rate:.1f}%: {"low hit rate suggests ready queue often empty" if pop_hit_rate < 50 else "good hit rate"}.')
        print('  Cache flush (dc cvac + dsb sy) is the dominant non-pop cost.')
    elif dominant_phase == 'complete':
        total_edges = fanout_edges + fanin_edges
        print(f'  Fanout: avg_degree={fanout_avg:.1f}, max_degree={fanout_max}.')
        print(f'  Fanin:  avg_degree={fanin_avg:.1f}, max_degree={fanin_max}.')
        if fanin_edges > fanout_edges:
            print('  Fanin traversal (release_producer + check_consumed) dominates the complete phase.')
        else:
            print('  Fanout traversal and atomic ops dominate the complete phase.')
    elif dominant_phase == 'scan':
        print('  Scan phase overhead indicates frequent perf header updates.')
    print('=' * 90)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Scheduler overhead analysis for PTO2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # auto-select latest files
  %(prog)s --perf-json outputs/perf_swimlane_*.json
  %(prog)s --device-log ~/ascend/log/debug/device-0/device-*.log
  %(prog)s --perf-json outputs/perf_swimlane_*.json -d 0
        """
    )
    parser.add_argument('--perf-json', help='Path to perf_swimlane_*.json file. If not specified, uses the latest in outputs/')
    parser.add_argument('--device-log', help='Path to device log file/path/glob. Overrides auto-resolution when provided')
    parser.add_argument('--sim-log', help='Path to sim run log (e.g. outputs/aicpu_ut_sim_run.log). Part 2 uses this; if no perf JSON, only Part 2 is run.')
    parser.add_argument('--no-sources', action='store_true', help='Do not print source/log file paths to stdout')
    parser.add_argument('-d', '--device-id', help='Device id for auto-selection from device-<id>')
    args = parser.parse_args()

    sim_log_path = Path(args.sim_log) if args.sim_log else None
    if sim_log_path is not None:
        if not sim_log_path.exists():
            print(f"Error: Sim log not found: {sim_log_path}", file=sys.stderr)
            return 1
        try:
            perf_path = Path(args.perf_json) if args.perf_json else auto_select_perf_json()
        except FileNotFoundError:
            return _run_part2_only(sim_log_path, print_sources=not args.no_sources, log_label='Sim log (aicpu_ut)')
        log_path = sim_log_path
        strategy = 'Sim log (aicpu_ut)'
    else:
        try:
            perf_path = Path(args.perf_json) if args.perf_json else auto_select_perf_json()
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        if not perf_path.exists():
            print(f"Error: Perf JSON not found: {perf_path}", file=sys.stderr)
            return 1
        log_path, strategy = resolve_device_log_path(
            device_id=args.device_id,
            device_log=args.device_log,
            perf_path=perf_path,
        )
        if log_path is None:
            print(f"Error: Failed to resolve device log ({strategy})", file=sys.stderr)
            return 1
        if not log_path.exists():
            print(f"Error: Device log not found: {log_path}", file=sys.stderr)
            return 1

    return run_analysis(perf_path, log_path, print_sources=True, selection_strategy=strategy)


if __name__ == '__main__':
    sys.exit(main())
