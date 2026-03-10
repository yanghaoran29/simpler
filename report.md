# Batch-Dispatch Analysis: bench_data_20260310_160404

| Field | Value |
|---|---|
| Baseline (D0E0) | DISPATCH=0 (base) — 10 runs |
| Variant  (D1E0) | DISPATCH=1 — 10 runs |

Distribution format: **min/p50/p90/max** across all runs.  
Δp50: % change of median vs baseline (negative = faster).

## Overall Timing

| Metric (min/p50/p90/max) | DISPATCH=0 (base) | DISPATCH=1 | Δp50 |
|---|---|---|---|
| Wall-clock test time (us) | 2267.48/3395.20/3704.10/3704.10 | 2281.70/3010.64/3145.12/3145.12 | -11.3% |
| Avg kernel Exec (us/task) | 15.43/15.50/15.69/15.69 | 15.55/15.64/15.75/15.75 | +0.9% |
| Avg Latency dispatch→finish (us) | 21.71/26.10/27.24/27.24 | 23.49/28.33/29.17/29.17 | +8.5% |
| Exec / Latency (%) | 56.63/59.75/72.28/72.28 | 53.57/56.53/66.95/66.95 | -5.4% |

## Scheduler CPU — SUM across 3 threads

| Metric (min/p50/p90/max) | DISPATCH=0 (base) | DISPATCH=1 | Δp50 |
|---|---|---|---|
| Sched CPU (us/task) | 6.72/9.84/10.64/10.64 | 6.83/8.85/9.20/9.20 | -10.1% |
| Sched total CPU (us, SUM) | 7012.70/10268.20/11108.90/11108.90 | 7128.10/9243.20/9606.70/9606.70 | -10.0% |
| Sched loops (SUM) | 947/1157/1525/1525 | 798/906/1071/1071 | -21.7% |
| Tasks per loop | 0.70/0.90/1.10/1.10 | 1.00/1.20/1.30/1.30 | +33.3% |
| Avg loop iteration (us) | 5.90/8.80/9.20/9.20 | 7.90/10.20/10.90/10.90 | +15.9% |

## Scheduler Total (us) per Thread — distribution across runs

| Metric (min/p50/p90/max) | DISPATCH=0 (base) | DISPATCH=1 | Δp50 |
|---|---|---|---|
| T0 total (us) | 2345.90/3416.80/3687.00/3687.00 | 2406.90/3091.70/3202.60/3202.60 | -9.5% |
| T1 total (us) | 2334.30/3425.90/3700.60/3700.60 | 2361.20/3081.20/3202.70/3202.70 | -10.1% |
| T2 total (us) | 2332.50/3425.40/3721.30/3721.30 | 2360.10/3079.00/3201.40/3201.40 | -10.1% |
| max-thread total (bottleneck, us) | 2345.90/3425.90/3721.30/3721.30 | 2406.90/3091.70/3202.70/3202.70 | -9.8% |
| imbalance (max−min)/max (%) | 0.19/0.31/0.92/0.92 | 0.04/0.19/1.94/1.94 | -37.7% |

## Tail OH (finish → detection by AICPU scheduler)

| Metric (min/p50/p90/max) | DISPATCH=0 (base) | DISPATCH=1 | Δp50 |
|---|---|---|---|
| Avg Head OH (us/task) | 0.56/0.66/0.70/0.70 | 0.59/0.69/0.73/0.73 | +4.5% |
| Tail OH mean (us/task) | 5.40/9.90/11.10/11.10 | 7.20/12.00/12.80/12.80 | +21.2% |
| Tail OH P50 (us) | 4.90/8.70/9.50/9.50 | 6.50/9.60/10.80/10.80 | +10.3% |
| Tail OH P90 (us) | 9.30/18.10/21.50/21.50 | 12.30/23.20/24.60/24.60 | +28.2% |

## Per-Function Avg Exec (us)

| Metric (min/p50/p90/max) | DISPATCH=0 (base) | DISPATCH=1 | Δp50 |
|---|---|---|---|
| QK exec (us) | 17.58/17.72/17.90/17.90 | 17.61/17.82/18.01/18.01 | +0.6% |
| SF exec (us) | 12.62/12.73/13.06/13.06 | 12.80/12.94/13.08/13.08 | +1.6% |
| PV exec (us) | 16.76/16.86/16.97/16.97 | 16.81/16.96/16.99/16.99 | +0.6% |
| UP exec (us) | 14.84/15.03/15.21/15.21 | 15.08/15.17/15.20/15.20 | +0.9% |
| HUB exec (us) | 0.45/0.52/0.62/0.62 | 0.40/0.55/0.69/0.69 | +5.8% |

## Per-Function Avg Latency dispatch→finish (us)

| Metric (min/p50/p90/max) | DISPATCH=0 (base) | DISPATCH=1 | Δp50 |
|---|---|---|---|
| QK latency (us) | 23.81/27.92/29.28/29.28 | 25.93/30.45/32.37/32.37 | +9.1% |
| SF latency (us) | 19.43/24.31/26.25/26.25 | 21.51/27.52/28.97/28.97 | +13.2% |
| PV latency (us) | 22.92/27.60/28.81/28.81 | 24.54/29.56/31.10/31.10 | +7.1% |
| UP latency (us) | 20.78/24.66/25.78/25.78 | 21.74/25.04/26.24/26.24 | +1.5% |
| HUB latency (us) | 5.71/6.41/8.55/8.55 | 6.91/10.35/19.80/19.80 | +61.5% |

