# Rebased PR benchmark and swimlanes

- Device: Ascend950PR_9579 card 1.
- CANN: 9.2.0.
- Base: `80dd3cd96e568e6f3ded9c11b68e7c267a31343d` (`main` at test time).
- Benchmark: 100 rounds per non-Qwen case; Qwen runs five rounds and reports the middle three by Device latency.
- Swimlane: one level-4 capture per case with `--skip-golden --manual include --enable-chip-swimlane 4`.
- Result: all eight benchmark cases and all eight swimlane captures completed successfully.

Each case directory contains a raw `merged_swimlane.json` that can be opened directly in Perfetto. Kernel name maps are stored beside the corresponding traces.

See `comparison.md` for the benchmark comparison and `summary.csv` for machine-readable values.
