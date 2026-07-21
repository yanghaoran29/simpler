# Acc ValidShape strip C2V (`srcStride` repro)

Minimal Ascend950 **simpler** sample for
`ISSUE_A5_ACC_VALIDSHAPE_C2V_SRCSTRIDE`.

## Primary test

`test_acc_c2v_strip_vs_full.py` — same Acc drained two ways:

| Output | Path |
|--------|------|
| `C_full` | one Acc TPUSH, `Valid≡Rows` |
| `C_strip` | Acc+`ValidShape(H=16)` windows @ `addr=row*64` |

Host: `C_strip ≈ C_full`. Shapes (UB-safe): `M=32, N=128, H=16, K=32`
(still `validRow < Rows`). Seed `torch.manual_seed(0)`.

### Board results (2026-07-21, device dump)

#### 修复前 (`srcStride=align(validRow)`)

典型现象：strip0（行 0…15）与 full 一致；strip1（行 16…）系统性偏离。
`col=0` 在本 seed 下仍可能碰巧相等，看同行列最大偏差处：

```text
# strip0 — 接近/相等
row=0   C_full[0,0]=10.4928   C_strip[0,0]=10.4928   diff=0
row=15  C_full[15,0]=-2.02309 C_strip[15,0]=-2.02309 diff=0

# strip1 — 明显不等（本 seed 最大差在下列）
row=16  C_full[16,71]=-12.8381  C_strip[16,71]=13.6886  |diff|=26.5268
row=17  C_full[17,44]=-11.6004  C_strip[17,44]=11.7646  |diff|=23.365
# 汇总: strip0_max=0.0  strip1_max=31.9782  → 主测 FAILED
```

#### 修复后 (`srcStride=align(Rows)`, pin `0ebbd03d`)

```text
row=0   C_full[0,0]=10.4928   C_strip[0,0]=10.4928   diff=0
row=15  C_full[15,0]=-2.02309 C_strip[15,0]=-2.02309 diff=0
row=16  C_full[16,0]=2.29007  C_strip[16,0]=2.29007  diff=0
row=17  C_full[17,0]=5.26025  C_strip[17,0]=5.26025  diff=0
# 汇总: strip0_max=0  strip1_max=0  → PASSED
```

```bash
# 在已配置 A5 的 simpler 仓库根目录：
python examples/a5/tensormap_and_ringbuffer/acc_c2v_strip_validshape/test_acc_c2v_strip_vs_full.py \
  -p a5 -d <device>

# 或用包根的 board 包装（改 SIMPLER 后）：
task-submit --device auto --max-time 900 --run \
  "bash /path/to/run_acc_c2v_strip_vs_full_board.sh"
```

## Files

| Path | Role |
|------|------|
| `kernels/mix/kernel_acc_c2v_compare.cpp` | matmul + full + strip TPUSH |
| `kernels/orchestration/acc_c2v_compare_orch.cpp` | `[A,B,C_full,C_strip]` |
| `test_acc_c2v_strip_vs_full.py` | self-compare |
