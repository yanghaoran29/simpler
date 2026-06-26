#!/usr/bin/env python3
from __future__ import annotations

import importlib
import importlib.util
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
from simpler.task_interface import ArgDirection as D

from simpler_setup import SceneTestCase, TaskArgsBuilder, Tensor, scene_test

_SCENE_TEST_MOD = importlib.import_module(SceneTestCase.__module__)
_ROOT = Path(__file__).resolve().parent
_PLOT_HEATMAP_MAX_NUMEL = 1_048_576


def _load_qwen3_hooks():
    p = Path(__file__).resolve().parents[3] / "qwen3_hooks.py"
    spec = importlib.util.spec_from_file_location("qwen3_hooks_mod", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"missing qwen3 hooks: {p}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_QWEN3_HOOKS = _load_qwen3_hooks()


def _bundled_output_compare_heatmap_path() -> Path:
    return Path(__file__).resolve().parents[3] / "output_compare_heatmap.py"


def _skill_plot_golden_mismatch_path() -> Path | None:
    p = os.environ.get("GOLDEN_MISMATCH_HEATMAP_SCRIPT")
    if p:
        pp = Path(p).expanduser().resolve()
        if pp.is_file():
            return pp
    p2 = Path.home() / ".cursor/skills/golden-tiered-file-validation/plot_golden_mismatch_heatmap.py"
    return p2 if p2.is_file() else None


def _load_plot_mismatch_module():
    bundled = _bundled_output_compare_heatmap_path()
    if bundled.is_file():
        spec = importlib.util.spec_from_file_location("_output_compare_heatmap", bundled)
        if spec is not None and spec.loader is not None:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    path = _skill_plot_golden_mismatch_path()
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("_golden_mismatch_heatmap", path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _safe_dir_label(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "case"
    return s[:120]


BATCH = 45
MAX_SEQ = 4096
NUM_HEADS = 40
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN = NUM_HEADS * HEAD_DIM
INTERMEDIATE = 17408
KV_HIDDEN = NUM_KV_HEADS * HEAD_DIM
BATCH_TILE = 16
BLOCK_SIZE = 128
SEQ_TILE = 128
EPS = 1e-6
Q_HEAD_BATCH = 5
INPUT_PROJ_K_CHUNK = 128
KV_PROJ_K_CHUNK = 128
Q_OUT_CHUNK = 256
KV_OUT_CHUNK = 128
K_CHUNK = 128
OUT_PROJ_K_CHUNK = 128
OUT_PROJ_N_CHUNK = 128
MLP_OUT_CHUNK = 1024
DOWN_MLP_CHUNK = 128
DOWN_OUT_CHUNK = 128
SYNTHETIC_PROJ_SCALE = 0.5

REUSE_INPUT_GOLDEN_CONSTANTS = {
    "BATCH": BATCH,
    "MAX_SEQ": MAX_SEQ,
    "NUM_HEADS": NUM_HEADS,
    "NUM_KV_HEADS": NUM_KV_HEADS,
    "HEAD_DIM": HEAD_DIM,
    "HIDDEN": HIDDEN,
    "INTERMEDIATE": INTERMEDIATE,
    "KV_HIDDEN": KV_HIDDEN,
    "BATCH_TILE": BATCH_TILE,
    "BLOCK_SIZE": BLOCK_SIZE,
    "SEQ_TILE": SEQ_TILE,
    "EPS": EPS,
    "Q_HEAD_BATCH": Q_HEAD_BATCH,
    "INPUT_PROJ_K_CHUNK": INPUT_PROJ_K_CHUNK,
    "KV_PROJ_K_CHUNK": KV_PROJ_K_CHUNK,
    "Q_OUT_CHUNK": Q_OUT_CHUNK,
    "KV_OUT_CHUNK": KV_OUT_CHUNK,
    "K_CHUNK": K_CHUNK,
    "OUT_PROJ_K_CHUNK": OUT_PROJ_K_CHUNK,
    "OUT_PROJ_N_CHUNK": OUT_PROJ_N_CHUNK,
    "MLP_OUT_CHUNK": MLP_OUT_CHUNK,
    "DOWN_MLP_CHUNK": DOWN_MLP_CHUNK,
    "DOWN_OUT_CHUNK": DOWN_OUT_CHUNK,
    "SYNTHETIC_PROJ_SCALE": SYNTHETIC_PROJ_SCALE,
}


def _max_blocks_per_seq() -> int:
    return (MAX_SEQ + BLOCK_SIZE - 1) // BLOCK_SIZE


def _cache_rows(batch: int) -> int:
    num_blocks = batch * _max_blocks_per_seq()
    return num_blocks * NUM_KV_HEADS * BLOCK_SIZE


def _merge_pto2_extra_defs_qwen3_14b_batch45_manual() -> None:
    tokens = [t for t in os.environ.get("PTO2_EXTRA_DEFS", "").split() if t]

    def upsert(k: str, v: int) -> None:
        pref = k + "="
        sval = str(v)
        for i, tok in enumerate(tokens):
            if tok.startswith(pref):
                tokens[i] = pref + sval
                return
        tokens.append(pref + sval)

    upsert("QWEN3_USER_BATCH", BATCH)
    upsert("QWEN3_BATCH_PADDED", (BATCH + BATCH_TILE - 1) // BATCH_TILE * BATCH_TILE)
    os.environ["PTO2_EXTRA_DEFS"] = " ".join(tokens)


def _merge_pto2_extra_defs_qwen3_14b_v200_manual() -> None:
    _merge_pto2_extra_defs_qwen3_14b_batch45_manual()


def _merge_pto2_extra_defs_qwen3_14b_manual() -> None:
    _merge_pto2_extra_defs_qwen3_14b_batch45_manual()


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


def _emit_golden_mismatch_heatmaps(
    *,
    work_dir: Path,
    comparisons: list[tuple[str, torch.Tensor, torch.Tensor]],
    rtol: float,
    atol: float,
) -> list[Path]:
    """Write golden/actual ``.pt`` and ``mismatch_<name>.png`` for every compared output."""
    out_dir = work_dir / "data" / "out"
    act_dir = work_dir / "data" / "actual"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    plot_mod = _load_plot_mismatch_module()
    png_paths: list[Path] = []

    for name, actual_cpu, golden_cpu in comparisons:
        g_path = out_dir / f"{name}.pt"
        a_path = act_dir / f"{name}.pt"
        torch.save(golden_cpu, g_path)
        torch.save(actual_cpu, a_path)

        png_path = work_dir / f"mismatch_{name}.png"
        if plot_mod is not None:
            g_plot, a_plot = golden_cpu, actual_cpu
            n = g_plot.numel()
            if n > _PLOT_HEATMAP_MAX_NUMEL:
                step = max(1, (n + _PLOT_HEATMAP_MAX_NUMEL - 1) // _PLOT_HEATMAP_MAX_NUMEL)
                flat_g = g_plot.reshape(-1)[::step].contiguous()
                flat_a = a_plot.reshape(-1)[::step].contiguous()
                g_plot = flat_g
                a_plot = flat_a
            try:
                plot_mod.plot_mismatch_map_tensors(
                    g_plot,
                    a_plot,
                    rtol=rtol,
                    atol=atol,
                    out_png=png_path,
                )
                png_paths.append(png_path)
            except Exception as ex:  # noqa: BLE001
                print(f"[golden compare] heatmap for {name} failed: {ex}", file=sys.stderr)
    return png_paths


def _compute_golden(tensors: dict, params: dict | None = None) -> None:
    hidden_states = tensors["hidden_states"]
    input_rms_weight = tensors["input_rms_weight"]
    wq = tensors["wq"]
    wk = tensors["wk"]
    wv = tensors["wv"]
    q_norm_weight = tensors["q_norm_weight"]
    k_norm_weight = tensors["k_norm_weight"]
    seq_lens = tensors["seq_lens"]
    block_table = tensors["block_table"]
    slot_mapping = tensors["slot_mapping"]
    rope_cos = tensors["rope_cos"]
    rope_sin = tensors["rope_sin"]
    k_cache = tensors["k_cache"]
    v_cache = tensors["v_cache"]
    wo = tensors["wo"]
    post_rms_weight = tensors["post_rms_weight"]
    w_gate = tensors["w_gate"]
    w_up = tensors["w_up"]
    w_down = tensors["w_down"]

    batch = hidden_states.shape[0]
    hidden_size = hidden_states.shape[1]
    kv_hidden = wk.shape[1]
    head_dim = rope_cos.shape[1]
    max_seq = rope_cos.shape[0]
    num_kv_heads = kv_hidden // head_dim
    num_heads = hidden_size // head_dim
    q_per_kv = num_heads // num_kv_heads
    q_groups = q_per_kv // Q_HEAD_BATCH
    half = head_dim // 2
    scale = 1.0 / math.sqrt(head_dim)
    eps = 1e-6
    max_ctx_blocks = (max_seq + BLOCK_SIZE - 1) // BLOCK_SIZE

    def tiled_matmul(lhs, rhs, k_chunk, n_chunk):
        out = torch.zeros(lhs.shape[0], rhs.shape[1], dtype=torch.float32)
        for n0 in range(0, rhs.shape[1], n_chunk):
            acc = torch.zeros(lhs.shape[0], n_chunk, dtype=torch.float32)
            for k0 in range(0, lhs.shape[1], k_chunk):
                acc = acc + lhs[:, k0 : k0 + k_chunk].float() @ rhs[
                    k0 : k0 + k_chunk,
                    n0 : n0 + n_chunk,
                ].float()
            out[:, n0 : n0 + n_chunk] = acc
        return out

    def chunked_row_sq_sum(x, k_chunk):
        acc = torch.zeros(x.shape[0], 1, dtype=torch.float32)
        for k0 in range(0, x.shape[1], k_chunk):
            x_chunk = x[:, k0 : k0 + k_chunk]
            acc = acc + (x_chunk * x_chunk).sum(dim=-1, keepdim=True)
        return acc

    q_proj = torch.zeros(batch, hidden_size, dtype=torch.float32)
    k_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)
    v_proj = torch.zeros(batch, kv_hidden, dtype=torch.float32)

    for b0 in range(0, batch, BATCH_TILE):
        b_end = min(b0 + BATCH_TILE, batch)
        x_tile = hidden_states[b0:b_end, :].float()

        sq_sum = torch.zeros(b_end - b0, 1, dtype=torch.float32)
        for k0 in range(0, hidden_size, INPUT_PROJ_K_CHUNK):
            x_chunk = x_tile[:, k0 : k0 + INPUT_PROJ_K_CHUNK]
            sq_sum = sq_sum + (x_chunk**2).sum(dim=-1, keepdim=True)
        variance = sq_sum / hidden_size + EPS
        rms = torch.sqrt(variance)
        normed = (x_tile / rms * input_rms_weight.float()).bfloat16()

        q_proj[b0:b_end, :] = tiled_matmul(normed, wq, INPUT_PROJ_K_CHUNK, Q_OUT_CHUNK)
        k_proj[b0:b_end, :] = tiled_matmul(normed, wk, KV_PROJ_K_CHUNK, KV_OUT_CHUNK)
        v_proj[b0:b_end, :] = tiled_matmul(normed, wv, KV_PROJ_K_CHUNK, KV_OUT_CHUNK)

    attn_out = torch.zeros(batch, hidden_size, dtype=torch.bfloat16)

    for b in range(batch):
        ctx_len = seq_lens[b].item()
        pos = ctx_len - 1
        ctx_blocks = (ctx_len + BLOCK_SIZE - 1) // BLOCK_SIZE

        cos_row = rope_cos[pos : pos + 1, :]
        sin_row = rope_sin[pos : pos + 1, :]
        cos_lo, cos_hi = cos_row[:, :half], cos_row[:, half:]
        sin_lo, sin_hi = sin_row[:, :half], sin_row[:, half:]

        k_heads = k_proj[b].view(num_kv_heads, head_dim)
        k_variance = k_heads.pow(2).mean(dim=-1, keepdim=True)
        k_heads = k_heads * torch.rsqrt(k_variance + eps) * k_norm_weight.float()
        k_lo_h, k_hi_h = k_heads[:, :half], k_heads[:, half:]
        k_rot = torch.cat(
            [k_lo_h * cos_lo - k_hi_h * sin_lo, k_hi_h * cos_hi + k_lo_h * sin_hi],
            dim=-1,
        )
        slot = int(slot_mapping[b].item())
        slot_block = slot // BLOCK_SIZE
        slot_offset = slot % BLOCK_SIZE

        for ki in range(num_kv_heads):
            cache_row = (slot_block * num_kv_heads + ki) * BLOCK_SIZE + slot_offset
            k_cache[cache_row, :] = k_rot[ki].to(torch.bfloat16)
            v_cache[cache_row, :] = v_proj[b, ki * head_dim : (ki + 1) * head_dim].to(torch.bfloat16)

        q_heads = q_proj[b].view(num_heads, head_dim)
        q_variance = q_heads.pow(2).mean(dim=-1, keepdim=True)
        q_heads = q_heads * torch.rsqrt(q_variance + eps) * q_norm_weight.float()
        q_lo_h, q_hi_h = q_heads[:, :half], q_heads[:, half:]
        q_rot = torch.cat(
            [q_lo_h * cos_lo - q_hi_h * sin_lo, q_hi_h * cos_hi + q_lo_h * sin_hi],
            dim=-1,
        )

        attn_row = torch.zeros(1, hidden_size, dtype=torch.bfloat16)
        for kvh in range(num_kv_heads):
            for qg in range(q_groups):
                q_base = kvh * q_per_kv + qg * Q_HEAD_BATCH
                q_grp_bf16 = q_rot[q_base : q_base + Q_HEAD_BATCH, :].to(torch.bfloat16)

                oi = torch.zeros(Q_HEAD_BATCH, head_dim, dtype=torch.float32)
                li = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)
                mi = torch.zeros(Q_HEAD_BATCH, 1, dtype=torch.float32)

                for sb in range(ctx_blocks):
                    s0 = sb * BLOCK_SIZE
                    valid_len = min(BLOCK_SIZE, ctx_len - s0)
                    pbid = int(block_table[b * max_ctx_blocks + sb].item())
                    cache_row0 = (pbid * num_kv_heads + kvh) * BLOCK_SIZE
                    k_tile = k_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]
                    v_tile = v_cache[cache_row0 : cache_row0 + BLOCK_SIZE, :]

                    raw_scores = q_grp_bf16.float() @ k_tile.float().T
                    if valid_len < BLOCK_SIZE:
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
                ctx_flat_bf16 = ctx.reshape(1, -1).to(torch.bfloat16)
                attn_row[
                    :,
                    q_base * head_dim : (q_base + Q_HEAD_BATCH) * head_dim,
                ] = ctx_flat_bf16

        attn_out[b : b + 1, :] = attn_row

    o_proj = tiled_matmul(attn_out, wo, OUT_PROJ_K_CHUNK, OUT_PROJ_N_CHUNK)
    resid1 = o_proj + hidden_states.float()

    variance = chunked_row_sq_sum(resid1, K_CHUNK) / hidden_size
    inv_rms = torch.rsqrt(variance + eps)
    normed_bf16 = (resid1 * inv_rms * post_rms_weight).bfloat16()

    gate = tiled_matmul(normed_bf16, w_gate, K_CHUNK, MLP_OUT_CHUNK)
    up = tiled_matmul(normed_bf16, w_up, K_CHUNK, MLP_OUT_CHUNK)
    mlp_bf16 = (gate * torch.sigmoid(gate) * up).bfloat16()
    down = tiled_matmul(mlp_bf16, w_down, DOWN_MLP_CHUNK, DOWN_OUT_CHUNK)

    tensors["out"][:] = (down + resid1).bfloat16()


@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestQwen314bBatch45ManualScopeDecode(SceneTestCase):
    RTOL = 3e-3
    ATOL = 3e-3

    CALLABLE = {
        "orchestration": {
            "source": "orchestration/qwen3_decode.cpp",
            "function_name": "aicpu_orchestration_entry",
            "signature": [
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.INOUT,
                D.INOUT,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.IN,
                D.OUT,
            ],
        },
        "incores": [
            {"func_id": 0, "name": "rmsnorm", "source": "kernels/aiv/rmsnorm.cpp", "core_type": "aiv", "signature": [D.IN, D.OUT, D.IN]},
            {"func_id": 1, "name": "q_proj", "source": "kernels/aic/q_proj.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.OUT]},
            {"func_id": 2, "name": "k_proj", "source": "kernels/aic/k_proj.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.OUT]},
            {"func_id": 3, "name": "v_proj", "source": "kernels/aic/v_proj.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.OUT]},
            {"func_id": 4, "name": "k_proj_0", "source": "kernels/aic/k_proj_0.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.INOUT]},
            {"func_id": 5, "name": "v_proj_0", "source": "kernels/aic/v_proj_0.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.INOUT]},
            {
                "func_id": 6,
                "name": "qk_norm",
                "source": "kernels/aiv/qk_norm.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 7,
                "name": "rope_kv_cache",
                "source": "kernels/aiv/rope_kv_cache.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 8,
                "name": "qk_matmul",
                "source": "kernels/aic/qk_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 9,
                "name": "softmax",
                "source": "kernels/aiv/softmax.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.OUT, D.OUT, D.OUT, D.OUT, D.OUT, D.IN, D.IN],
            },
            {
                "func_id": 10,
                "name": "sv_matmul",
                "source": "kernels/aic/sv_matmul.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.OUT, D.IN, D.IN, D.IN, D.IN],
            },
            {
                "func_id": 11,
                "name": "online_softmax",
                "source": "kernels/aiv/online_softmax.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.IN, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 12,
                "name": "out_proj_residual_aic",
                "source": "kernels/aic/out_proj_residual_aic.cpp",
                "core_type": "aic",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {
                "func_id": 13,
                "name": "out_proj_residual_aiv",
                "source": "kernels/aiv/out_proj_residual_aiv.cpp",
                "core_type": "aiv",
                "signature": [D.OUT, D.IN, D.IN, D.IN, D.OUT],
            },
            {"func_id": 14, "name": "post_rmsnorm", "source": "kernels/aiv/post_rmsnorm.cpp", "core_type": "aiv", "signature": [D.IN, D.OUT, D.IN]},
            {"func_id": 15, "name": "gate_proj", "source": "kernels/aic/gate_proj.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.INOUT]},
            {"func_id": 16, "name": "up_proj", "source": "kernels/aic/up_proj.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.INOUT]},
            {"func_id": 17, "name": "silu", "source": "kernels/aiv/silu.cpp", "core_type": "aiv", "signature": [D.IN, D.IN, D.INOUT]},
            {"func_id": 18, "name": "down_proj", "source": "kernels/aic/down_proj.cpp", "core_type": "aic", "signature": [D.IN, D.IN, D.INOUT]},
            {
                "func_id": 19,
                "name": "down_proj_residual",
                "source": "kernels/aiv/down_proj_residual.cpp",
                "core_type": "aiv",
                "signature": [D.IN, D.IN, D.OUT],
            },
        ],
    }

    CASES = [
        {
            "name": "Case1",
            "platforms": ["a2a3sim", "a2a3"],
            "config": {"aicpu_thread_num": 4, "block_dim": 24},
            "params": {"dtype": "bfloat16"},
        },
    ]

    def generate_args(self, params):
        def _build(_p):
            batch = BATCH
            max_blocks = _max_blocks_per_seq()
            num_blocks = batch * max_blocks
            cache_rows = _cache_rows(batch)

            hidden_states = ((torch.rand(batch, HIDDEN, dtype=torch.float32) - 0.5)).to(torch.bfloat16)
            input_rms_weight = torch.rand(1, HIDDEN, dtype=torch.float32) - 0.5
            wq = torch.rand(HIDDEN, HIDDEN, dtype=torch.float32) / HIDDEN**0.5
            wq = wq.to(torch.bfloat16)
            wk = torch.rand(HIDDEN, KV_HIDDEN, dtype=torch.float32) / HIDDEN**0.5
            wk = wk.to(torch.bfloat16)
            wv = SYNTHETIC_PROJ_SCALE * (torch.rand(HIDDEN, KV_HIDDEN, dtype=torch.float32) / HIDDEN**0.5)
            wv = wv.to(torch.bfloat16)

            q_norm_weight = torch.ones(1, HEAD_DIM, dtype=torch.float32)
            k_norm_weight = torch.ones(1, HEAD_DIM, dtype=torch.float32)

            seq_lens = torch.randint(1, MAX_SEQ + 1, (batch,), dtype=torch.int32)
            block_table = torch.arange(num_blocks, dtype=torch.int32)

            slot_mapping = torch.empty(batch, dtype=torch.int32)
            for b in range(batch):
                pos = int(seq_lens[b].item()) - 1
                logical_block = pos // BLOCK_SIZE
                page_offset = pos % BLOCK_SIZE
                phys_block = b * max_blocks + logical_block
                slot_mapping[b] = phys_block * BLOCK_SIZE + page_offset

            rope_cos = torch.rand(MAX_SEQ, HEAD_DIM, dtype=torch.float32) - 0.5
            rope_sin = torch.rand(MAX_SEQ, HEAD_DIM, dtype=torch.float32) - 0.5

            k_cache = ((torch.rand(cache_rows, HEAD_DIM, dtype=torch.float32) - 0.5)).to(torch.bfloat16)
            v_cache = (SYNTHETIC_PROJ_SCALE * (torch.rand(cache_rows, HEAD_DIM, dtype=torch.float32) - 0.5)).to(
                torch.bfloat16
            )

            wo = (SYNTHETIC_PROJ_SCALE * (torch.rand(HIDDEN, HIDDEN, dtype=torch.float32) - 0.5) / HIDDEN**0.5).to(
                torch.bfloat16
            )
            post_rms_weight = torch.ones(1, HIDDEN, dtype=torch.float32)
            w_gate = (SYNTHETIC_PROJ_SCALE * (torch.rand(HIDDEN, INTERMEDIATE, dtype=torch.float32) - 0.5) / HIDDEN**0.5).to(
                torch.bfloat16
            )
            w_up = (SYNTHETIC_PROJ_SCALE * (torch.rand(HIDDEN, INTERMEDIATE, dtype=torch.float32) - 0.5) / HIDDEN**0.5).to(
                torch.bfloat16
            )
            w_down = (SYNTHETIC_PROJ_SCALE * (torch.rand(INTERMEDIATE, HIDDEN, dtype=torch.float32) - 0.5) / INTERMEDIATE**0.5).to(
                torch.bfloat16
            )
            out = torch.zeros(batch, HIDDEN, dtype=torch.bfloat16)

            return TaskArgsBuilder(
                Tensor("hidden_states", hidden_states),
                Tensor("input_rms_weight", input_rms_weight),
                Tensor("wq", wq),
                Tensor("wk", wk),
                Tensor("wv", wv),
                Tensor("q_norm_weight", q_norm_weight),
                Tensor("k_norm_weight", k_norm_weight),
                Tensor("seq_lens", seq_lens),
                Tensor("block_table", block_table),
                Tensor("slot_mapping", slot_mapping),
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
            )

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
        case_name = case.get("name", "case")
        orig_compare = _SCENE_TEST_MOD._compare_outputs

        def _compare_with_heatmaps(test_args, golden_args, output_names, r, a):
            _QWEN3_HOOKS.log_qwen3("golden_compare_start", case=case_name, outputs=",".join(output_names))
            try:
                mismatches: list[tuple[str, float, torch.Tensor, torch.Tensor, str | None]] = []
                for name in output_names:
                    act = getattr(test_args, name)
                    exp = getattr(golden_args, name)
                    ok, reason = _golden_allclose_relaxed(act, exp, r, a)
                    if not ok:
                        diff = (act - exp).abs().max().item()
                        mismatches.append((name, diff, act.detach().cpu(), exp.detach().cpu(), reason))
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                work_dir = _ROOT / "outputs" / "golden_mismatch" / f"{_safe_dir_label(case_name)}_{ts}"
                comparisons = [
                    (n, getattr(test_args, n).detach().cpu(), getattr(golden_args, n).detach().cpu())
                    for n in output_names
                ]
                pngs = _emit_golden_mismatch_heatmaps(
                    work_dir=work_dir,
                    comparisons=comparisons,
                    rtol=r,
                    atol=a,
                )
                print(
                    f"[golden compare] heatmaps under {work_dir.resolve()}\n"
                    f"  data/out/*.pt (golden)  data/actual/*.pt (device)\n"
                    f"  heatmaps: {[str(p.resolve()) for p in pngs]}",
                    file=sys.stderr,
                )
                if not pngs and (not _bundled_output_compare_heatmap_path().is_file()) and _skill_plot_golden_mismatch_path() is None:
                    print(
                        "[golden compare] heatmap module not found. Expected "
                        f"{_bundled_output_compare_heatmap_path()} or set GOLDEN_MISMATCH_HEATMAP_SCRIPT.\n",
                        file=sys.stderr,
                    )
                if mismatches:
                    detail_lines: list[str] = []
                    for n, d, _a, _e, rs in mismatches:
                        line = f"  - '{n}': max_diff={d}, rtol={r}, atol={a}"
                        if rs:
                            line += f" — {rs}"
                        detail_lines.append(line)
                    raise AssertionError(
                        "Golden mismatch: every output tensor was checked; "
                        f"{len(mismatches)} failed:\n"
                        + "\n".join(detail_lines)
                        + f"\nSee heatmaps and .pt files under {work_dir.resolve()}"
                    )
            finally:
                _QWEN3_HOOKS.log_qwen3("golden_compare_end", case=case_name, outputs=",".join(output_names))

        _SCENE_TEST_MOD._compare_outputs = _compare_with_heatmaps
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
    _merge_pto2_extra_defs_qwen3_14b_batch45_manual()
    _QWEN3_HOOKS.print_qwen3_decode_script_startup(__file__)
    SceneTestCase.run_module(__name__)
