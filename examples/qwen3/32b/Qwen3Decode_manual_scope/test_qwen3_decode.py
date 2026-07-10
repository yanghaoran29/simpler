#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Qwen3-32B single-layer decode — manual scope for func4~func6.

Same architecture as ``Qwen3Decode/``, but the inner GQA attention loop
(func4=qk_matmul, func5=softmax, func6=sv_matmul) uses
``PTO2_SCOPE(PTO2ScopeMode::MANUAL)`` with explicit ``add_dep()`` edges:
``t3→t4`` (rope_kv_cache → qk_matmul) bridges the auto→manual entry,
and ``t6→t7`` bridges the manual→auto exit so the online_softmax write
to ``attn_out`` is picked up by the downstream out_proj_residual via
the runtime's auto TensorMap.

When ``params.dtype`` is ``bfloat16`` or ``float32``, each golden compare emits
output heatmaps (pass or fail): ULP/bitwise maps for bf16, tiered rtol/atol maps
for fp32; see shared ``examples/qwen3/output_compare_heatmap.py``.

Parameters aligned with ``pypto-lib/models/qwen3/32b/qwen3_32b_decode.py``:

  BATCH=16  HIDDEN=8192  KV_HIDDEN=1024  HEAD_DIM=128
  NUM_HEADS=64  NUM_KV_HEADS=8  INTERMEDIATE=25600  MAX_SEQ=4096
  CACHE_ROWS = B * NUM_KV_HEADS * MAX_SEQ = 524288
"""

import math
import importlib
import importlib.util
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import Scalar, SceneTestCase, TaskArgsBuilder, Tensor, scene_test

_SCENE_TEST_MOD = importlib.import_module(SceneTestCase.__module__)

_ROOT = Path(__file__).resolve().parent


def _load_qwen3_hooks():
    p = Path(__file__).resolve().parents[2] / "qwen3_hooks.py"
    spec = importlib.util.spec_from_file_location("qwen3_hooks_mod", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing qwen3 hooks: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_QWEN3_HOOKS = _load_qwen3_hooks()
_QWEN3_OUTPUT_COMPARE_HEATMAP = Path(__file__).resolve().parents[2] / "output_compare_heatmap.py"
_PLOT_HEATMAP_MAX_NUMEL = 1_048_576


def _safe_dir_label(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "case"
    return s[:120]


def _load_output_compare_heatmap_module():
    path = _QWEN3_OUTPUT_COMPARE_HEATMAP
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_output_compare_heatmap", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GOLDEN_RELAXED_SOFT_FRAC = 1e-2
GOLDEN_RELAXED_SOFT_ABS_CAP = 0.008
GOLDEN_RELAXED_MARGINAL_FRAC = 1e-3
GOLDEN_RELAXED_MARGINAL_ABS_CAP = 0.016


def _golden_allclose_relaxed(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float,
    atol: float,
    *,
    soft_frac: float = GOLDEN_RELAXED_SOFT_FRAC,
    soft_abs_cap: float = GOLDEN_RELAXED_SOFT_ABS_CAP,
    marginal_frac: float = GOLDEN_RELAXED_MARGINAL_FRAC,
    marginal_abs_cap: float = GOLDEN_RELAXED_MARGINAL_ABS_CAP,
) -> tuple[bool, str | None]:
    a = actual.detach().float().reshape(-1)
    e = expected.detach().float().reshape(-1)
    diff = (a - e).abs()
    n = diff.numel()
    if n == 0:
        return True, None

    thr = atol + rtol * e.abs()
    baseline_miss = diff > thr
    if not baseline_miss.any():
        return True, None

    severe = baseline_miss & (diff > marginal_abs_cap)
    if severe.any():
        return False, (
            f"max_abs_diff={diff.max().item():.6g} > marginal_abs_cap={marginal_abs_cap} "
            f"(rtol={rtol}, atol={atol})"
        )

    soft = baseline_miss & (diff <= soft_abs_cap)
    marginal = baseline_miss & (diff > soft_abs_cap) & (diff <= marginal_abs_cap)

    n_soft = int(soft.sum().item())
    n_marginal = int(marginal.sum().item())
    max_soft = int(math.floor(n * soft_frac))
    max_marginal = int(math.floor(n * marginal_frac))

    if n_soft > max_soft:
        return False, (
            f"{n_soft} / {n} elems in (thr,{soft_abs_cap}] exceed soft budget {max_soft}=floor(n*{soft_frac}); "
            f"rtol={rtol}, atol={atol}"
        )
    if n_marginal > max_marginal:
        return False, (
            f"{n_marginal} / {n} elems in ({soft_abs_cap},{marginal_abs_cap}] exceed marginal budget "
            f"{max_marginal}=floor(n*{marginal_frac}); rtol={rtol}, atol={atol}"
        )
    return True, None

# ---------------------------------------------------------------------------
# Constants (must match the compiled orchestration)
# ---------------------------------------------------------------------------

BATCH = 16
MAX_SEQ = 4096
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM          # 8192
INTERMEDIATE = 25600
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM    # 1024
CACHE_ROWS = BATCH * NUM_KV_HEADS * MAX_SEQ  # 524288
HALF_DIM = HEAD_DIM // 2
Q_PER_KV = NUM_HEADS // NUM_KV_HEADS
Q_HEAD_BATCH = 8
Q_GROUPS = Q_PER_KV // Q_HEAD_BATCH
SEQ_TILE = 256
EPS = 1e-6
BATCH_TILE = 16

REUSE_INPUT_GOLDEN_CONSTANTS = {
    "BATCH": BATCH,
    "MAX_SEQ": MAX_SEQ,
    "NUM_HEADS": NUM_HEADS,
    "NUM_KV_HEADS": NUM_KV_HEADS,
    "HEAD_DIM": HEAD_DIM,
    "HIDDEN": HIDDEN,
    "INTERMEDIATE": INTERMEDIATE,
    "KV_HIDDEN": KV_HIDDEN,
    "CACHE_ROWS": CACHE_ROWS,
    "HALF_DIM": HALF_DIM,
    "Q_PER_KV": Q_PER_KV,
    "Q_HEAD_BATCH": Q_HEAD_BATCH,
    "Q_GROUPS": Q_GROUPS,
    "SEQ_TILE": SEQ_TILE,
    "EPS": EPS,
    "BATCH_TILE": BATCH_TILE,
}


def _merge_pto2_extra_defs_qwen3_32b_manual() -> None:
    """Orchestration reads QWEN3_32B_* from ../../qwen3_32b_decode_macros.h."""
    import os as _os

    tokens = [t for t in _os.environ.get("PTO2_EXTRA_DEFS", "").split() if t]

    def upsert(key: str, val: int) -> None:
        pref = key + "="
        sval = str(val)
        for i, tok in enumerate(tokens):
            if tok.startswith(pref):
                tokens[i] = pref + sval
                return
        tokens.append(pref + sval)

    upsert("QWEN3_32B_USER_BATCH", BATCH)
    batch_padded = (BATCH + BATCH_TILE - 1) // BATCH_TILE * BATCH_TILE
    upsert("QWEN3_32B_BATCH_PADDED", batch_padded)
    if "PTO2_LARGE_EXPLICIT_DEPS" not in tokens:
        tokens.append("PTO2_LARGE_EXPLICIT_DEPS")
    _os.environ["PTO2_EXTRA_DEFS"] = " ".join(tokens)


# ---------------------------------------------------------------------------
# Reference computation (golden) — ported from qwen3_32b_decode.py
# ---------------------------------------------------------------------------

def _compute_golden(tensors: dict, params: dict | None = None) -> None:
    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    seq_lens = tensors["seq_lens"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"].clone()
    v_cache = tensors["v_cache"].clone()
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    half = HEAD_DIM // 2
    scale = 1.0 / math.sqrt(HEAD_DIM)

    # ── Scope 1 golden: RMSNorm + Q/K/V projection ──
    x_tile = hidden_states.float()
    sq_sum = (x_tile ** 2).sum(dim=-1, keepdim=True)
    variance = sq_sum / HIDDEN + EPS
    rms = torch.sqrt(variance)
    normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

    q_proj = (normed.float() @ wq.float()).float()
    k_proj = (normed.float() @ wk.float()).float()
    v_proj = (normed.float() @ wv.float()).float()

    # ── Scope 2 golden: RoPE + cache update + attention ──
    attn_out = torch.zeros(BATCH, HIDDEN, dtype=torch.float32)

    for b in range(BATCH):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + SEQ_TILE - 1) // SEQ_TILE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(NUM_KV_HEADS, HEAD_DIM)
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat([k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi], dim=-1)

        for ki in range(NUM_KV_HEADS):
            cr = b * NUM_KV_HEADS * MAX_SEQ + ki * MAX_SEQ + pos
            k_cache[cr, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cr, :] = v_proj[b, ki * HEAD_DIM : (ki + 1) * HEAD_DIM].to(torch.bfloat16)

        q_heads = q_proj[b].view(NUM_HEADS, HEAD_DIM)
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat([q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi], dim=-1)

        for kvh in range(NUM_KV_HEADS):
            for qg in range(Q_GROUPS):
                q_base = kvh * Q_PER_KV + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, HEAD_DIM, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * SEQ_TILE
                    valid_len = min(SEQ_TILE, ctx_len - s0)
                    cb = b * NUM_KV_HEADS * MAX_SEQ + kvh * MAX_SEQ + s0

                    k_tile = k_cache[cb : cb + SEQ_TILE, :]
                    v_tile = v_cache[cb : cb + SEQ_TILE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < SEQ_TILE:
                        raw_scores[:, valid_len:] = torch.finfo(torch.float32).min
                    scores = raw_scores * scale

                    cur_mi = scores.max(dim=-1, keepdim=True).values
                    exp_scores = torch.exp(scores - cur_mi)
                    exp_scores_bf16 = exp_scores.to(torch.bfloat16)
                    cur_li = exp_scores_bf16.float().sum(dim=-1, keepdim=True)

                    oi_tmp = exp_scores_bf16.float() @ v_tile.float()

                    if sb == 0:
                        oi = oi_tmp
                        li = cur_li
                        mi = cur_mi
                    else:
                        mi_new = torch.maximum(mi, cur_mi)
                        alpha = torch.exp(mi - mi_new)
                        beta = torch.exp(cur_mi - mi_new)
                        li = alpha * li + beta * cur_li
                        oi = oi * alpha + oi_tmp * beta
                        mi = mi_new

                ctx = oi / li
                for qi in range(Q_HEAD_BATCH):
                    qh = q_base + qi
                    attn_out[b, qh * HEAD_DIM : (qh + 1) * HEAD_DIM] = ctx[qi]

    # ── Scope 3 golden: output projection + residual + post RMSNorm + MLP + residual ──
    o_proj = torch.matmul(attn_out.float(), wo.float())
    resid1 = o_proj + hidden_states.float()

    variance = resid1.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + EPS)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = torch.matmul(normed_bf16.float(), w_gate.float())
    up = torch.matmul(normed_bf16.float(), w_up.float())
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = torch.matmul(mlp_bf16.float(), w_down.float())

    tensors["out"][:] = (down + resid1).bfloat16()
    tensors["k_cache"][:] = k_cache
    tensors["v_cache"][:] = v_cache


# ---------------------------------------------------------------------------
# Test case
# ---------------------------------------------------------------------------

@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen3DecodeManualScope(SceneTestCase):
    RTOL = 4e-3
    ATOL = 4e-3

    CALLABLE = {
        "orchestration": {
            "source": "orchestration/qwen3_decode.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [
                D.IN,      # 0:  hidden_states
                D.IN,      # 1:  input_rms_weight
                D.IN,      # 2:  wq
                D.IN,      # 3:  wk
                D.IN,      # 4:  wv
                D.IN,      # 5:  seq_lens
                D.IN,      # 6:  rope_cos
                D.IN,      # 7:  rope_sin
                D.INOUT,   # 8:  k_cache
                D.INOUT,   # 9:  v_cache
                D.IN,      # 10: wo
                D.IN,      # 11: post_rms_weight
                D.IN,      # 12: w_gate
                D.IN,      # 13: w_up
                D.IN,      # 14: w_down
                D.OUT,     # 15: out
            ],
        },
        "incores": [
            {
                "func_id": 0,
                "name": "rmsnorm",
                "source": "kernels/aiv/rmsnorm.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 1,
                "name": "q_proj",
                "source": "kernels/aic/q_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 2,
                "name": "kv_proj",
                "source": "kernels/aic/kv_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.IN, D.OUT, D.OUT],
            },
            {
                "func_id": 3,
                "name": "rope_kv_cache",
                "source": "kernels/aiv/rope_kv_cache.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 4,
                "name": "qk_matmul",
                "source": "kernels/aic/qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 5,
                "name": "softmax",
                "source": "kernels/aiv/softmax.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.OUT, D.OUT, D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 6,
                "name": "sv_matmul",
                "source": "kernels/aic/sv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 7,
                "name": "online_softmax",
                "source": "kernels/aiv/online_softmax.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 8,
                "name": "out_proj_residual_aic",
                "source": "kernels/aic/out_proj_residual_aic.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 9,
                "name": "out_proj_residual_aiv",
                "source": "kernels/aiv/out_proj_residual_aiv.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 10,
                "name": "post_rmsnorm",
                "source": "kernels/aiv/post_rmsnorm.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.OUT, D.IN],
            },
            {
                "func_id": 11,
                "name": "gate_proj",
                "source": "kernels/aic/gate_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 12,
                "name": "up_proj",
                "source": "kernels/aic/up_proj.cpp",
                "core_type": "aic",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 13,
                "name": "silu",
                "source": "kernels/aiv/silu.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 14,
                "name": "down_proj_residual_aic",
                "source": "kernels/aic/down_proj_residual_aic.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 15,
                "name": "down_proj_residual_aiv",
                "source": "kernels/aiv/down_proj_residual_aiv.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {
                "dtype": "bfloat16",
            },
        },
    ]

    def generate_args(self, params):
        def _build(_p):
            def rand_uniform_bf16(*shape):
                fan_in = shape[0]
                return ((torch.rand(*shape, dtype=torch.float32) - 0.5) / (fan_in ** 0.5)).to(torch.bfloat16)

            hidden_states = (torch.rand(BATCH, HIDDEN, dtype=torch.float32) - 0.5).to(torch.bfloat16)
            input_rms_weight = torch.rand(1, HIDDEN, dtype=torch.float32) - 0.5
            wq = rand_uniform_bf16(HIDDEN, HIDDEN)
            wk = rand_uniform_bf16(HIDDEN, KV_HIDDEN)
            wv = rand_uniform_bf16(HIDDEN, KV_HIDDEN)

            seq_lens = torch.randint(1, MAX_SEQ + 1, (BATCH,), dtype=torch.int32)

            rope_cos = torch.rand(MAX_SEQ, HEAD_DIM, dtype=torch.float32) - 0.5
            rope_sin = torch.rand(MAX_SEQ, HEAD_DIM, dtype=torch.float32) - 0.5

            k_cache = (torch.rand(CACHE_ROWS, HEAD_DIM, dtype=torch.float32) - 0.5).to(torch.bfloat16)
            v_cache = (torch.rand(CACHE_ROWS, HEAD_DIM, dtype=torch.float32) - 0.5).to(torch.bfloat16)

            wo = rand_uniform_bf16(HIDDEN, HIDDEN)
            post_rms_weight = torch.ones(1, HIDDEN, dtype=torch.float32)
            w_gate = rand_uniform_bf16(HIDDEN, INTERMEDIATE)
            w_up = rand_uniform_bf16(HIDDEN, INTERMEDIATE)
            w_down = ((torch.rand(INTERMEDIATE, HIDDEN, dtype=torch.float32) - 0.5) / (INTERMEDIATE ** 0.5)).to(torch.bfloat16)
            out = torch.zeros(BATCH, HIDDEN, dtype=torch.bfloat16)

            specs = [
                Tensor("hidden_states", hidden_states),
                Tensor("input_rms_weight", input_rms_weight),
                Tensor("wq", wq),
                Tensor("wk", wk),
                Tensor("wv", wv),
                Tensor("seq_lens", seq_lens),
                Tensor("rope_cos", rope_cos),
                Tensor("rope_sin", rope_sin),
                Tensor("k_cache", k_cache),
                Tensor("v_cache", v_cache),
                Tensor("wo", wo),
                Tensor("post_rms_weight", post_rms_weight),
                Tensor("w_gate", w_gate),
                Tensor("w_up", w_up),
                Tensor("w_down", w_down),
                Tensor("out", out),
            ]
            return TaskArgsBuilder(*specs)

        return _QWEN3_HOOKS.resolve_task_args_with_constant_gated_reuse(
            qwen3_root=_QWEN3_HOOKS.examples_qwen3_root_from_test_file(__file__),
            test_cls_name=type(self).__name__,
            constants=REUSE_INPUT_GOLDEN_CONSTANTS,
            params=params,
            build_fn=_build,
        )

    def compute_golden(self, args, params):
        _QWEN3_HOOKS.run_golden_phase(
            qwen3_root=_QWEN3_HOOKS.examples_qwen3_root_from_test_file(__file__),
            test_cls_name=type(self).__name__,
            args=args,
            params=params,
            orch_signature=self.CALLABLE["orchestration"]["signature"],
            compute_core=_compute_golden,
        )

    def _run_and_validate_l2(
        self,
        worker,
        callable_obj,
        case,
        rounds=1,
        skip_golden=False,
        enable_l2_swimlane=False,
        enable_dump_tensor=False,
        enable_pmu=0,
        enable_dep_gen=False,
        output_prefix="",
    ):
        orig_compare = _SCENE_TEST_MOD._compare_outputs
        case_name = case.get("name", "case")

        def _compare_with_output_heatmaps(test_args, golden_args, output_names, r, a):
            _QWEN3_HOOKS.log_qwen3("golden_compare_start", case=case_name, outputs=",".join(output_names))
            try:
                mod = _load_output_compare_heatmap_module()
                if mod is not None:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    work_dir = _ROOT / "outputs" / "golden_compare_heatmap" / f"{_safe_dir_label(case_name)}_{ts}"
                    bundles_bf16: list[tuple[str, torch.Tensor, torch.Tensor]] = []
                    bundles_fp32: list[tuple[str, torch.Tensor, torch.Tensor]] = []
                    for name in output_names:
                        act = getattr(test_args, name)
                        exp = getattr(golden_args, name)
                        if exp.dtype == torch.bfloat16:
                            bundles_bf16.append((name, act.detach().cpu(), exp.detach().cpu()))
                        elif exp.dtype == torch.float32:
                            bundles_fp32.append((name, act.detach().cpu(), exp.detach().cpu()))
                    all_png: list[str] = []
                    try:
                        if bundles_bf16:
                            paths = mod.emit_bf16_output_compare(
                                work_dir=work_dir,
                                tensors=bundles_bf16,
                                max_numel=_PLOT_HEATMAP_MAX_NUMEL,
                            )
                            all_png.extend(str(p.resolve()) for p in paths)
                        if bundles_fp32:
                            paths_f = mod.emit_fp32_tiered_output_compare(
                                work_dir=work_dir,
                                tensors=bundles_fp32,
                                rtol=r,
                                atol=a,
                                max_numel=_PLOT_HEATMAP_MAX_NUMEL,
                            )
                            all_png.extend(str(p.resolve()) for p in paths_f)
                        if all_png:
                            print(
                                f"[compare heatmap] artifacts under {work_dir.resolve()}\n"
                                f"  PNG: {all_png}",
                                file=sys.stderr,
                            )
                    except Exception as ex:  # noqa: BLE001
                        print(f"[compare heatmap] generation failed: {ex}", file=sys.stderr)
                else:
                    print(
                        f"[compare heatmap] skip: missing {_QWEN3_OUTPUT_COMPARE_HEATMAP}",
                        file=sys.stderr,
                    )
                mismatches: list[tuple[str, float, str | None]] = []
                for name in output_names:
                    act = getattr(test_args, name)
                    exp = getattr(golden_args, name)
                    ok, reason = _golden_allclose_relaxed(act, exp, r, a)
                    if not ok:
                        diff = (act - exp).abs().max().item()
                        mismatches.append((name, diff, reason))
                if mismatches:
                    detail_lines: list[str] = []
                    for n, d, rs in mismatches:
                        line = f"  - '{n}': max_diff={d}, rtol={r}, atol={a}"
                        if rs:
                            line += f" — {rs}"
                        detail_lines.append(line)
                    raise AssertionError(
                        "Golden mismatch: every output tensor was checked; "
                        f"{len(mismatches)} failed:\n"
                        + "\n".join(detail_lines)
                    )
                return None
            finally:
                _QWEN3_HOOKS.log_qwen3("golden_compare_end", case=case_name, outputs=",".join(output_names))

        _SCENE_TEST_MOD._compare_outputs = _compare_with_output_heatmaps
        try:
            with _QWEN3_HOOKS.log_l2_case(case_name), _QWEN3_HOOKS.log_device_run(worker):
                super()._run_and_validate_l2(
                    worker,
                    callable_obj,
                    case,
                    rounds=rounds,
                    skip_golden=skip_golden,
                    enable_l2_swimlane=enable_l2_swimlane,
                    enable_dump_tensor=enable_dump_tensor,
                    enable_pmu=enable_pmu,
                    enable_dep_gen=enable_dep_gen,
                    output_prefix=output_prefix,
                )
        finally:
            _SCENE_TEST_MOD._compare_outputs = orig_compare


def scene_test_pre_runtime_banner(*, args, selected_by_cls, by_rt_level):
    _QWEN3_HOOKS.scene_test_pre_runtime_banner_impl(
        Path(__file__), args=args, selected_by_cls=selected_by_cls, by_rt_level=by_rt_level
    )


if __name__ == "__main__":
    import os
    import sys

    _merge_pto2_extra_defs_qwen3_32b_manual()
    _QWEN3_HOOKS.print_qwen3_decode_script_startup(__file__)
    SceneTestCase.run_module(__name__)
