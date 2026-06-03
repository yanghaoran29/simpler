#!/usr/bin/env python3
"""Golden vs actual output compare: heatmaps for float32 (tiered rtol/atol) and bfloat16 (ULP).

This module lives at ``simpler/examples/qwen3/output_compare_heatmap.py`` and is loaded by
Qwen3 decode tests under ``examples/qwen3/32b/`` and ``examples/qwen3/14b/Qwen3Decode/``.

**Float32** uses the same tiered colormap rules as
``golden-tiered-file-validation/plot_golden_mismatch_heatmap.py``:

* Green: ``torch.isclose`` at baseline rtol/atol
* Gray: abnormal zero (golden != 0, actual == 0)
* Yellow: not baseline match but within 2× rtol / 2× atol
* Light red: within 2×..5× rtol/atol band
* Red: otherwise

**BFloat16** (both tensors as bf16):

* Always emit one PNG per tensor pair at the caller (pass or fail).
* Green: bitwise identical; light green: 1 ULP; yellow: 2 ULP; pink: 3–5 ULP;
  gray: abnormal zero (golden nonzero bits, actual all-zero bits); red: other.

Plot text is English so default Matplotlib fonts render cleanly.
Optional: set ``PYPTO_MISMATCH_HEATMAP_FONT`` to register a custom font (fp32 plots only).
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import ListedColormap
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Shared layout helpers
# ---------------------------------------------------------------------------


def floor_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << int(math.floor(math.log2(n)))


def choose_cols_pow2(numel: int) -> int:
    base = int(math.sqrt(max(numel, 1)))
    cols = floor_pow2(base)
    return max(cols, 1)


def choose_tick_step_pow2(axis_len: int, target_ticks: int = 16) -> int:
    raw = max(1, axis_len // max(target_ticks, 1))
    return floor_pow2(raw)


def _maybe_downsample(
    golden: torch.Tensor,
    actual: torch.Tensor,
    max_numel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = golden.numel()
    if n <= max_numel:
        return golden, actual
    step = max(1, (n + max_numel - 1) // max_numel)
    flat_g = golden.reshape(-1)[::step].contiguous()
    flat_a = actual.reshape(-1)[::step].contiguous()
    return flat_g, flat_a


# ---------------------------------------------------------------------------
# Float32: tiered mismatch (aligned with plot_golden_mismatch_heatmap.py)
# ---------------------------------------------------------------------------

_REGISTERED_FONT_PATHS: set[str] = set()
_plot_font_configured = False
_module_font_file_override: str | None = None

_FP32_CATEGORY_COLORS = [
    "#27ae60",  # green baseline match
    "#f1c40f",  # yellow 2× band
    "#f5b7b1",  # light red 2×..5×
    "#c0392b",  # red
    "#7f8c8d",  # gray abnormal zero
    "#ecf0f1",  # padding
]


def set_mismatch_heatmap_user_font(path: str | Path | None) -> None:
    """Register a font file before fp32 plots (same as ``PYPTO_MISMATCH_HEATMAP_FONT``)."""
    global _module_font_file_override, _plot_font_configured
    _module_font_file_override = os.path.expanduser(str(path)) if path else None
    _plot_font_configured = False


def _path_known_to_matplotlib(sp: str) -> bool:
    for info in fm.fontManager.ttflist:
        try:
            if Path(info.fname).resolve() == Path(sp).resolve():
                return True
        except OSError:
            if info.fname == sp:
                return True
    return False


def _try_register_font_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        sp = str(path.resolve())
    except OSError:
        sp = str(path)
    try:
        prop = fm.FontProperties(fname=sp)
        name = prop.get_name()
    except Exception:
        return None
    if not _path_known_to_matplotlib(sp) and sp not in _REGISTERED_FONT_PATHS:
        try:
            fm.fontManager.addfont(sp)
            _REGISTERED_FONT_PATHS.add(sp)
        except Exception:
            return None
    return name


def _configure_plot_fonts_if_needed() -> None:
    global _plot_font_configured
    if _plot_font_configured:
        return

    chain: list[str] = ["DejaVu Sans"]
    for src in filter(
        None,
        (
            _module_font_file_override,
            (os.environ.get("PYPTO_MISMATCH_HEATMAP_FONT") or "").strip() or None,
        ),
    ):
        n = _try_register_font_file(Path(os.path.expanduser(src)))
        if n is not None and n not in chain:
            chain.insert(1, n)

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = chain
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Glyph \d+ .* missing from font",
    )
    _plot_font_configured = True


def classify_fp32_tiered_pixels(
    golden: torch.Tensor,
    actual: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    """Category codes 0..4 (5 = grid padding), same priority as tiered skill script."""
    g = golden.detach().cpu().float()
    a = actual.detach().cpu().float()

    base_ok = torch.isclose(a, g, rtol=rtol, atol=atol, equal_nan=False)
    ok_2x = torch.isclose(a, g, rtol=2.0 * rtol, atol=2.0 * atol, equal_nan=False)
    ok_5x = torch.isclose(a, g, rtol=5.0 * rtol, atol=5.0 * atol, equal_nan=False)
    abnormal_zero = (g != 0) & (a == 0)

    cat = torch.full_like(g, 3, dtype=torch.int64)
    cat[base_ok] = 0
    not_ok = ~base_ok
    cat[not_ok & abnormal_zero] = 4
    rest = not_ok & ~abnormal_zero
    cat[rest & ok_2x] = 1
    cat[rest & (~ok_2x) & ok_5x] = 2
    return cat


def plot_fp32_tiered_mismatch_heatmap(
    golden: torch.Tensor,
    actual: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    out_png: str | Path,
    title: str | None = None,
    verbose_save: bool = False,
) -> None:
    """Tiered fp32 / fp-like heatmap (golden vs actual), same rules as skill ``plot_mismatch_map_tensors``."""
    _configure_plot_fonts_if_needed()

    golden = golden.detach().cpu()
    actual = actual.detach().cpu()

    if golden.shape != actual.shape:
        raise ValueError(f"shape mismatch: golden={tuple(golden.shape)} actual={tuple(actual.shape)}")

    cat = classify_fp32_tiered_pixels(golden, actual, rtol=rtol, atol=atol)
    flat = cat.reshape(-1).numpy()
    n = int(flat.size)

    cols = choose_cols_pow2(n)
    rows = (n + cols - 1) // cols
    total = rows * cols

    data = np.full(total, 5, dtype=np.int32)
    data[:n] = flat.astype(np.int32, copy=False)
    grid = data.reshape(rows, cols)

    cmap = ListedColormap(_FP32_CATEGORY_COLORS)
    fig_w = min(18, max(8, cols / 64))
    fig_h = min(18, max(8, rows / 64))
    plt.figure(figsize=(fig_w, fig_h), dpi=120)
    plt.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=5, aspect="auto")

    x_step = choose_tick_step_pow2(cols)
    y_step = choose_tick_step_pow2(rows)
    xt = np.arange(0, cols, x_step)
    yt = np.arange(0, rows, y_step)
    plt.xticks(xt, xt, fontsize=7)
    plt.yticks(yt, yt, fontsize=7)
    plt.xlabel(f"Column index (tick step={x_step}, power of 2)", fontsize=11)
    plt.ylabel(f"Row index (tick step={y_step}, power of 2)", fontsize=11)

    g_f = golden.float()
    a_f = actual.float()
    base_close = torch.isclose(a_f, g_f, rtol=rtol, atol=atol, equal_nan=False)
    mismatch_count = int((~base_close).sum().item())
    mismatch_ratio = mismatch_count / max(n, 1)
    if title is None:
        title = (
            "Golden mismatch map (tiered, float32)\n"
            f"shape={tuple(actual.shape)}  numel={n}  grid={rows}x{cols} (cols=2^k)\n"
            f"baseline mismatch={mismatch_count}/{n} ({mismatch_ratio:.4%}), rtol={rtol}, atol={atol}\n"
            "Legend: green=match | gray=abnormal zero | yellow=within 2× | "
            "light red=2×..5× band | red=>5× | pale gray=padding"
        )
    plt.title(title, fontsize=11)
    plt.grid(which="both", color="white", linewidth=0.2)
    plt.tight_layout()

    legend_elems = [
        mpatches.Patch(color=_FP32_CATEGORY_COLORS[0], label="(1) match"),
        mpatches.Patch(color=_FP32_CATEGORY_COLORS[4], label="(2) abnormal zero"),
        mpatches.Patch(color=_FP32_CATEGORY_COLORS[1], label="(3) within 2× tol"),
        mpatches.Patch(color=_FP32_CATEGORY_COLORS[2], label="(4) within 2×..5× tol"),
        mpatches.Patch(color=_FP32_CATEGORY_COLORS[3], label="(5) other error"),
        mpatches.Patch(color=_FP32_CATEGORY_COLORS[5], label="padding"),
    ]
    plt.legend(handles=legend_elems, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=8)

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    if verbose_save:
        print(f"[saved] {out_path.resolve()}")
    plt.close()


# Drop-in alias (same signature as skill script) for tooling compatibility.
def plot_mismatch_map_tensors(
    golden: torch.Tensor,
    actual: torch.Tensor,
    rtol: float = 3e-3,
    atol: float = 3e-3,
    out_png: str | Path | None = None,
    *,
    verbose_save: bool = False,
) -> None:
    if out_png is None:
        raise ValueError("out_png is required")
    plot_fp32_tiered_mismatch_heatmap(
        golden,
        actual,
        rtol=rtol,
        atol=atol,
        out_png=out_png,
        verbose_save=verbose_save,
    )


# ---------------------------------------------------------------------------
# BFloat16: bitwise + ULP distance in total-order space
# ---------------------------------------------------------------------------

_BF16_CATEGORY_COLORS = [
    "#27ae60",  # strict equal
    "#abebc6",  # 1 ULP
    "#f1c40f",  # 2 ULP
    "#f5b7b1",  # 3–5 ULP (pink)
    "#7f8c8d",  # abnormal zero
    "#c0392b",  # other / large ULP / non-finite
    "#ecf0f1",  # padding
]


def _bf16_bits_u16(t: torch.Tensor) -> torch.Tensor:
    return t.detach().cpu().to(torch.bfloat16).contiguous().view(torch.uint16)


def _sortable_int_bf16_u16(u16: torch.Tensor) -> torch.Tensor:
    u = u16.to(torch.int64) & 0xFFFF
    neg = (u >> 15) & 1
    pos_key = (u & 0x7FFF) + 0x8000
    neg_key = 0x8000 - (u & 0x7FFF)
    return torch.where(neg != 0, neg_key, pos_key)


def _bf16_ulps_distance(u_g: torch.Tensor, u_a: torch.Tensor) -> torch.Tensor:
    sg = _sortable_int_bf16_u16(u_g)
    sa = _sortable_int_bf16_u16(u_a)
    return (sg - sa).abs()


def classify_bf16_ulp_pixels(
    golden: torch.Tensor,
    actual: torch.Tensor,
) -> torch.Tensor:
    g = golden.detach().cpu().to(torch.bfloat16)
    a = actual.detach().cpu().to(torch.bfloat16)
    if g.shape != a.shape:
        raise ValueError(f"shape mismatch: golden={tuple(g.shape)} actual={tuple(a.shape)}")

    ug = _bf16_bits_u16(g)
    ua = _bf16_bits_u16(a)

    strict_eq = ug == ua
    abnormal_zero = (ug != 0) & (ua == 0)

    g_f = g.float()
    a_f = a.float()
    finite = torch.isfinite(g_f) & torch.isfinite(a_f)

    ulps = _bf16_ulps_distance(ug, ua)
    bad_float = ~finite | torch.isnan(g_f) | torch.isnan(a_f)

    cat = torch.full_like(ug, 5, dtype=torch.int64)
    cat[strict_eq] = 0
    rest = ~strict_eq
    cat[rest & abnormal_zero] = 4
    rest2 = rest & ~abnormal_zero
    cat[rest2 & bad_float] = 5
    rest3 = rest2 & ~bad_float
    u = ulps.to(torch.int64)
    cat[rest3 & (u == 1)] = 1
    cat[rest3 & (u == 2)] = 2
    cat[rest3 & (u >= 3) & (u <= 5)] = 3
    cat[rest3 & (u > 5)] = 5
    return cat


def plot_bf16_ulp_heatmap(
    golden: torch.Tensor,
    actual: torch.Tensor,
    *,
    out_png: str | Path,
    title: str | None = None,
) -> None:
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"Glyph \d+ .* missing from font",
    )
    plt.rcParams["axes.unicode_minus"] = False

    golden = golden.detach().cpu()
    actual = actual.detach().cpu()

    cat = classify_bf16_ulp_pixels(golden, actual)
    flat = cat.reshape(-1).numpy()
    n = int(flat.size)

    cols = choose_cols_pow2(n)
    rows = (n + cols - 1) // cols
    total = rows * cols

    data = np.full(total, 6, dtype=np.int32)
    data[:n] = flat.astype(np.int32, copy=False)
    grid = data.reshape(rows, cols)

    cmap = ListedColormap(_BF16_CATEGORY_COLORS)
    fig_w = min(18, max(8, cols / 64))
    fig_h = min(18, max(8, rows / 64))
    plt.figure(figsize=(fig_w, fig_h), dpi=120)
    plt.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=6, aspect="auto")

    x_step = choose_tick_step_pow2(cols)
    y_step = choose_tick_step_pow2(rows)
    plt.xticks(np.arange(0, cols, x_step))
    plt.yticks(np.arange(0, rows, y_step))
    plt.xlabel("column (flattened index / cols)")
    plt.ylabel("row (flattened index // cols)")
    ttl = title or "bfloat16 golden vs actual (ULP / bitwise)"
    plt.title(ttl)

    legend_elems = [
        mpatches.Patch(color=_BF16_CATEGORY_COLORS[0], label="strict bf16 equal (bits)"),
        mpatches.Patch(color=_BF16_CATEGORY_COLORS[1], label="1 ULP"),
        mpatches.Patch(color=_BF16_CATEGORY_COLORS[2], label="2 ULP"),
        mpatches.Patch(color=_BF16_CATEGORY_COLORS[3], label="3–5 ULP"),
        mpatches.Patch(color=_BF16_CATEGORY_COLORS[4], label="abnormal zero (g≠0, a=0 bits)"),
        mpatches.Patch(color=_BF16_CATEGORY_COLORS[5], label="other / ≥6 ULP / non-finite"),
    ]
    plt.legend(handles=legend_elems, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Emit bundles (PNG + .pt snapshots)
# ---------------------------------------------------------------------------


def emit_fp32_tiered_output_compare(
    *,
    work_dir: Path,
    tensors: list[tuple[str, torch.Tensor, torch.Tensor]],
    rtol: float,
    atol: float,
    max_numel: int = 1_048_576,
) -> list[Path]:
    """Write tiered fp32 mismatch PNG + .pt for each (name, actual, golden) pair with float32 golden."""
    work_dir = Path(work_dir)
    out_dir = work_dir / "data" / "out"
    act_dir = work_dir / "data" / "actual"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    png_paths: list[Path] = []
    for name, actual_cpu, golden_cpu in tensors:
        if golden_cpu.dtype != torch.float32:
            continue
        if actual_cpu.dtype != torch.float32:
            actual_cpu = actual_cpu.to(torch.float32)

        torch.save(golden_cpu, out_dir / f"{name}.pt")
        torch.save(actual_cpu, act_dir / f"{name}.pt")

        g_plot, a_plot = _maybe_downsample(golden_cpu, actual_cpu, max_numel)
        png_path = work_dir / f"mismatch_{name}.png"
        subsampled = g_plot.numel() < golden_cpu.numel()
        fp32_title = (
            f"float32 tiered compare: {name} (subsampled for display)\nrtol={rtol}, atol={atol}"
            if subsampled
            else None
        )
        plot_fp32_tiered_mismatch_heatmap(
            g_plot,
            a_plot,
            rtol=rtol,
            atol=atol,
            out_png=png_path,
            title=fp32_title,
        )
        png_paths.append(png_path)
    return png_paths


def emit_bf16_output_compare(
    *,
    work_dir: Path,
    tensors: list[tuple[str, torch.Tensor, torch.Tensor]],
    max_numel: int = 1_048_576,
) -> list[Path]:
    """Write bf16 ULP-class PNG + .pt for each pair whose golden dtype is ``torch.bfloat16``."""
    work_dir = Path(work_dir)
    out_dir = work_dir / "data" / "out"
    act_dir = work_dir / "data" / "actual"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    png_paths: list[Path] = []
    for name, actual_cpu, golden_cpu in tensors:
        if golden_cpu.dtype != torch.bfloat16:
            continue
        if actual_cpu.dtype != torch.bfloat16:
            actual_cpu = actual_cpu.to(torch.bfloat16)

        torch.save(golden_cpu, out_dir / f"{name}.pt")
        torch.save(actual_cpu, act_dir / f"{name}.pt")

        g_plot, a_plot = _maybe_downsample(golden_cpu, actual_cpu, max_numel)
        png_path = work_dir / f"bf16_heatmap_{name}.png"
        subsampled = g_plot.numel() < golden_cpu.numel()
        plot_bf16_ulp_heatmap(
            g_plot,
            a_plot,
            out_png=png_path,
            title=(
                f"bfloat16 compare: {name} (subsampled for display)"
                if subsampled
                else f"bfloat16 compare: {name}"
            ),
        )
        png_paths.append(png_path)
    return png_paths
