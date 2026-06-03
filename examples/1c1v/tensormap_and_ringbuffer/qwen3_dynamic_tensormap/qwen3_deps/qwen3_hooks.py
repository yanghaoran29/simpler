# Copyright (c) PyPTO Contributors.
# Shared hooks for Qwen3 example tests: phase logging, golden disk cache, device-run wrappers.
# -----------------------------------------------------------------------------------------------------------
"""Utilities used by ``examples/qwen3/**/test_qwen3_decode.py``.

Environment:
  QWEN3_SKIP_GOLDEN_DISK_CACHE=1 — disable loading/saving golden from disk (always recompute).
  QWEN3_SKIP_INPUT_GOLDEN_REUSE=1 — disable constant-gated input+golden pair reuse (always random inputs + CPU golden).

Standalone ``SceneTestCase.run_module``:
  Tests may define a module-level ``scene_test_pre_runtime_banner(*, args, selected_by_cls, by_rt_level)``
  callable; ``simpler_setup.scene_test`` invokes it immediately before printing
  ``=== Runtime: … ===`` (see ``print_qwen3_decode_script_startup`` /
  ``scene_test_pre_runtime_banner_impl`` below).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def log_qwen3(phase: str, **fields: Any) -> None:
    """Single-line stderr log with wall-clock time and monotonic counter."""
    ts = time.strftime("%H:%M:%S", time.localtime())
    mono = f"{time.perf_counter():.3f}s"
    extra = ""
    if fields:
        parts = [f"{k}={v!r}" for k, v in sorted(fields.items())]
        extra = " " + " ".join(parts)
    print(f"[qwen3 {ts} mono={mono}] {phase}{extra}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Standalone CLI banners (``SceneTestCase.run_module``)
# ---------------------------------------------------------------------------


def print_qwen3_decode_script_startup(test_file: str | Path) -> None:
    """Call from ``if __name__ == "__main__"`` before ``SceneTestCase.run_module``."""
    script = Path(test_file).resolve()
    extra_defs = (os.environ.get("PTO2_EXTRA_DEFS") or "").strip()
    extra_line = f"\n  PTO2_EXTRA_DEFS: {extra_defs}\n" if extra_defs else "\n"
    argv_tail = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "(empty)"
    print(
        "\n[qwen3 decode test] Script startup\n"
        f"  path: {script}\n"
        f"  argv: {argv_tail}\n"
        f"  cwd:  {os.getcwd()}"
        f"{extra_line}",
        flush=True,
    )


def scene_test_pre_runtime_banner_impl(
    test_file: str | Path,
    *,
    args: Any,
    selected_by_cls: dict[type, list[dict]],
    by_rt_level: dict[tuple[str, int], list[type]],
) -> None:
    """Body for module-level ``scene_test_pre_runtime_banner`` in ``test_qwen3_decode.py`` scripts."""
    script = Path(test_file).resolve()
    labels: list[str] = []
    for cls, cases in selected_by_cls.items():
        for c in cases:
            labels.append(f"{cls.__name__}::{c.get('name', '?')}")
    groups = [f"{rt} L{lv}" for (rt, lv) in by_rt_level]
    extra_defs = (os.environ.get("PTO2_EXTRA_DEFS") or "").strip()
    ed = f"\n  PTO2_EXTRA_DEFS: {extra_defs}" if extra_defs else ""
    print(
        "\n[qwen3 decode test] About to run SceneTestCase runtime/level groups\n"
        f"  script: {script}{ed}\n"
        f"  platform={getattr(args, 'platform', None)!r}  "
        f"device={getattr(args, 'device', None)!r}  rounds={getattr(args, 'rounds', None)!r}\n"
        f"  groups ({len(by_rt_level)}): {', '.join(groups) if groups else '(none)'}\n"
        f"  cases ({len(labels)}): {', '.join(labels) if labels else '(none)'}\n",
        flush=True,
    )


def examples_qwen3_root_from_test_file(test_file: str) -> Path:
    """``.../qwen3/<N>b/<CaseDir>/test_qwen3_decode.py`` → ``.../qwen3``."""
    return Path(test_file).resolve().parents[2]


def _safe_cache_token(s: str) -> str:
    t = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_") or "x"
    return t[:80]


# ---------------------------------------------------------------------------
# Golden disk cache
# ---------------------------------------------------------------------------


def _skip_golden_disk_cache() -> bool:
    return (os.environ.get("QWEN3_SKIP_GOLDEN_DISK_CACHE") or "").strip() in ("1", "true", "TRUE", "yes", "YES")


def _skip_input_golden_reuse() -> bool:
    return (os.environ.get("QWEN3_SKIP_INPUT_GOLDEN_REUSE") or "").strip() in ("1", "true", "TRUE", "yes", "YES")


def _params_for_byte_fingerprint(params: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strip ``_qwen3_*`` keys so internal reuse metadata does not change the byte-level golden cache key."""
    if not params:
        return {}
    return {k: v for k, v in params.items() if not str(k).startswith("_qwen3_")}


def constants_fingerprint(constants: Mapping[str, Any]) -> str:
    """Stable SHA256 over JSON-serialized constant dict (sorted keys)."""
    payload = json.dumps(dict(constants), sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def input_golden_reuse_paths(qwen3_root: Path, test_cls_name: str, const_fp: str) -> tuple[Path, Path]:
    d = qwen3_root / "outputs" / "input_golden_reuse"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe_cache_token(test_cls_name)}_{const_fp}"
    return d / f"{stem}_inputs.pt", d / f"{stem}_golden.pt"


def _save_inputs_reuse_blob(path: Path, *, test_cls_name: str, const_fp: str, task_args: Any) -> None:
    import torch  # noqa: PLC0415

    from simpler_setup import Tensor  # noqa: PLC0415

    names: list[str] = []
    tensors: dict[str, Any] = {}
    for spec in task_args.specs:
        if isinstance(spec, Tensor):
            names.append(spec.name)
            tensors[spec.name] = spec.value.detach().cpu().clone()
    torch.save(
        {"version": 1, "test_cls_name": test_cls_name, "const_fp": const_fp, "tensor_names": names, "tensors": tensors},
        path,
    )


def _load_task_args_from_inputs_blob(path: Path) -> Any:
    import torch  # noqa: PLC0415

    from simpler_setup import TaskArgsBuilder, Tensor  # noqa: PLC0415

    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location="cpu")
    names: list[str] = list(blob["tensor_names"])
    td: dict[str, Any] = blob["tensors"]
    return TaskArgsBuilder(*[Tensor(n, td[n].clone()) for n in names])


def _reuse_blob_header_ok(blob: Any, *, test_cls_name: str, const_fp: str) -> bool:
    if not isinstance(blob, dict) or blob.get("version") != 1:
        return False
    return blob.get("test_cls_name") == test_cls_name and blob.get("const_fp") == const_fp


def resolve_task_args_with_constant_gated_reuse(
    *,
    qwen3_root: Path,
    test_cls_name: str,
    constants: Mapping[str, Any],
    params: dict[str, Any],
    build_fn: Callable[[dict[str, Any]], Any],
) -> Any:
    """Build or restore ``TaskArgsBuilder``; pair with ``run_golden_phase`` for golden reuse.

    When ``<stem>_inputs.pt`` and ``<stem>_golden.pt`` exist under
    ``qwen3_root/outputs/input_golden_reuse/`` and headers match ``test_cls_name`` +
    ``constants_fingerprint(constants)``, sets ``params["_qwen3_reuse_full"]=True`` so
    ``run_golden_phase`` loads golden from disk without ``compute_core``.

    Otherwise materializes via ``build_fn``, saves inputs blob, sets
    ``params["_qwen3_reuse_full"]=False`` (golden will be computed and saved).
    """
    const_fp = constants_fingerprint(constants)
    inp_p, gold_p = input_golden_reuse_paths(qwen3_root, test_cls_name, const_fp)

    def _publish_paths(*, reuse_full: bool) -> None:
        params["_qwen3_const_fp"] = const_fp
        params["_qwen3_reuse_inputs_path"] = str(inp_p)
        params["_qwen3_reuse_golden_path"] = str(gold_p)
        params["_qwen3_reuse_cls"] = test_cls_name
        params["_qwen3_reuse_full"] = reuse_full

    if _skip_input_golden_reuse():
        tb = build_fn(params)
        try:
            _save_inputs_reuse_blob(inp_p, test_cls_name=test_cls_name, const_fp=const_fp, task_args=tb)
        except Exception as ex:  # noqa: BLE001
            log_qwen3("input_golden_reuse_save_inputs_failed", error=str(ex), path=str(inp_p))
        _publish_paths(reuse_full=False)
        return tb

    if inp_p.is_file() and gold_p.is_file():
        try:
            import torch as _torch  # noqa: PLC0415

            try:
                ib = _torch.load(inp_p, map_location="cpu", weights_only=False)
            except TypeError:
                ib = _torch.load(inp_p, map_location="cpu")
            try:
                gb = _torch.load(gold_p, map_location="cpu", weights_only=False)
            except TypeError:
                gb = _torch.load(gold_p, map_location="cpu")
        except Exception as ex:  # noqa: BLE001
            log_qwen3("input_golden_reuse_pair_load_failed", error=str(ex))
        else:
            if _reuse_blob_header_ok(ib, test_cls_name=test_cls_name, const_fp=const_fp) and _reuse_blob_header_ok(
                gb, test_cls_name=test_cls_name, const_fp=const_fp
            ):
                try:
                    tb = _load_task_args_from_inputs_blob(inp_p)
                except Exception as ex:  # noqa: BLE001
                    log_qwen3("input_golden_reuse_restore_inputs_failed", error=str(ex))
                else:
                    outs = gb.get("outputs") or {}
                    onames = gb.get("output_names") or []
                    if isinstance(onames, list) and onames and all(n in outs for n in onames):
                        _publish_paths(reuse_full=True)
                        log_qwen3(
                            "input_golden_reuse_hit",
                            const_fp_prefix=const_fp[:16],
                            inputs=str(inp_p.name),
                            golden=str(gold_p.name),
                        )
                        return tb
            log_qwen3("input_golden_reuse_stale_pair", inputs=str(inp_p), golden=str(gold_p))

    tb = build_fn(params)
    try:
        _save_inputs_reuse_blob(inp_p, test_cls_name=test_cls_name, const_fp=const_fp, task_args=tb)
    except Exception as ex:  # noqa: BLE001
        log_qwen3("input_golden_reuse_save_inputs_failed", error=str(ex), path=str(inp_p))
    _publish_paths(reuse_full=False)
    return tb


def _fingerprint_inputs(args: Any, params: dict | None) -> str:
    """Stable SHA256 over params JSON + every Tensor spec (name, shape, dtype, raw bytes)."""
    import torch  # noqa: PLC0415

    from simpler_setup import Tensor  # noqa: PLC0415

    h = hashlib.sha256()
    p = params if params is not None else {}
    h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    tensors = [s for s in args.specs if isinstance(s, Tensor)]
    for spec in sorted(tensors, key=lambda s: s.name):
        t = spec.value
        h.update(spec.name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        tc = t.detach().contiguous()
        if tc.dtype == torch.bfloat16:
            arr = tc.view(torch.uint16).cpu().numpy()
        else:
            arr = tc.cpu().numpy()
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _golden_cache_path(qwen3_root: Path, test_cls_name: str, fingerprint: str) -> Path:
    d = qwen3_root / "outputs" / "golden_disk_cache"
    d.mkdir(parents=True, exist_ok=True)
    short = fingerprint[:48]
    return d / f"{_safe_cache_token(test_cls_name)}_{short}.pt"


def _output_names(args: Any, orch_signature: list) -> list[str]:
    # ``from simpler_setup import scene_test`` is the @scene_test decorator, not the module.
    from simpler_setup.scene_test import _build_chip_task_args  # noqa: PLC0415

    _chip, names = _build_chip_task_args(args, orch_signature)
    return names


def run_golden_phase(
    *,
    qwen3_root: Path,
    test_cls_name: str,
    args: Any,
    params: dict | None,
    orch_signature: list,
    compute_core: Callable[[dict[str, Any], dict | None], None],
) -> None:
    """Log golden start/end; skip ``compute_core`` when a valid disk cache exists."""
    import torch  # noqa: PLC0415

    from simpler_setup import Tensor  # noqa: PLC0415

    params_x: dict[str, Any] = params if params is not None else {}
    output_names = _output_names(args, orch_signature)
    log_qwen3("golden_start", cls=test_cls_name, outputs=",".join(output_names))

    # --- Constant-gated pair: restore golden outputs only (inputs already from disk). ---
    if (
        not _skip_input_golden_reuse()
        and params_x.get("_qwen3_reuse_full") is True
        and params_x.get("_qwen3_reuse_golden_path")
        and params_x.get("_qwen3_reuse_cls") == test_cls_name
    ):
        gold_p = Path(str(params_x["_qwen3_reuse_golden_path"]))
        exp_cf = params_x.get("_qwen3_const_fp")
        if gold_p.is_file() and isinstance(exp_cf, str) and exp_cf:
            try:
                try:
                    gb = torch.load(gold_p, map_location="cpu", weights_only=False)
                except TypeError:
                    gb = torch.load(gold_p, map_location="cpu")
            except Exception as ex:  # noqa: BLE001
                log_qwen3("input_golden_reuse_golden_load_failed", error=str(ex), path=str(gold_p))
            else:
                if _reuse_blob_header_ok(gb, test_cls_name=test_cls_name, const_fp=exp_cf):
                    outs = gb.get("outputs") or {}
                    if outs and all(n in outs for n in output_names):
                        for name in output_names:
                            dst = getattr(args, name)
                            src = outs[name]
                            if isinstance(src, torch.Tensor):
                                dst.copy_(src.to(device=dst.device, dtype=dst.dtype, non_blocking=False))
                        log_qwen3(
                            "golden_end",
                            reused=True,
                            path=str(gold_p),
                            source="input_golden_constant_pair",
                        )
                        return
        params_x["_qwen3_reuse_full"] = False
        log_qwen3("input_golden_reuse_golden_incomplete_recompute")

    params_fp = _params_for_byte_fingerprint(params_x)
    fp = _fingerprint_inputs(args, params_fp)
    params_json = json.dumps(params_fp, sort_keys=True, default=str)
    cache_path = _golden_cache_path(qwen3_root, test_cls_name, fp)

    if not _skip_golden_disk_cache() and cache_path.is_file():
        try:
            try:
                blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            except TypeError:
                blob = torch.load(cache_path, map_location="cpu")
        except Exception as ex:  # noqa: BLE001
            log_qwen3("golden_cache_load_failed", error=str(ex), path=str(cache_path))
        else:
            if blob.get("fingerprint") == fp and blob.get("params_json") == params_json:
                outs = blob.get("outputs") or {}
                if outs and all(n in outs for n in output_names):
                    for name in output_names:
                        dst = getattr(args, name)
                        src = outs[name]
                        if not isinstance(src, torch.Tensor):
                            continue
                        dst.copy_(src.to(device=dst.device, dtype=dst.dtype, non_blocking=False))
                    log_qwen3("golden_end", reused=True, path=str(cache_path))
                    return
            log_qwen3("golden_cache_stale_or_incomplete", path=str(cache_path))

    tensors: dict[str, Any] = {s.name: s.value for s in args.specs if isinstance(s, Tensor)}
    compute_core(tensors, params)
    for s in args.specs:
        if isinstance(s, Tensor) and s.name in tensors:
            getattr(args, s.name)[:] = tensors[s.name]

    if not _skip_golden_disk_cache():
        try:
            outs_cpu = {n: getattr(args, n).detach().cpu().clone() for n in output_names}
            torch.save(
                {
                    "fingerprint": fp,
                    "params_json": params_json,
                    "outputs": outs_cpu,
                },
                cache_path,
            )
            log_qwen3("golden_saved", path=str(cache_path))
        except Exception as ex:  # noqa: BLE001
            log_qwen3("golden_save_failed", error=str(ex), path=str(cache_path))

    if not _skip_input_golden_reuse():
        gp_s = params_x.get("_qwen3_reuse_golden_path")
        cf = params_x.get("_qwen3_const_fp")
        if gp_s and isinstance(cf, str) and cf:
            gp = Path(str(gp_s))
            try:
                torch.save(
                    {
                        "version": 1,
                        "test_cls_name": test_cls_name,
                        "const_fp": cf,
                        "output_names": list(output_names),
                        "outputs": {n: getattr(args, n).detach().cpu().clone() for n in output_names},
                    },
                    gp,
                )
                log_qwen3("input_golden_reuse_saved_golden", path=str(gp))
            except Exception as ex:  # noqa: BLE001
                log_qwen3("input_golden_reuse_save_golden_failed", error=str(ex), path=str(gp))

    log_qwen3("golden_end", reused=False)


# ---------------------------------------------------------------------------
# Device (on-board) run — wrap Worker.run (post-unification of run/run_prepared, PR #797)
# ---------------------------------------------------------------------------


@contextmanager
def log_device_run(worker: Any):
    """Emit ``device_run_start`` / ``device_run_end`` around each device launch.

    Wraps ``worker.run`` (the unified entry that replaced the older
    ``run_prepared`` in pypto/runtime PR #797). Falls back to ``run_prepared``
    when running against an older simpler that still exposes it.
    """
    method_name = "run" if hasattr(worker, "run") else "run_prepared"
    orig = getattr(worker, method_name)

    def _wrapped(callable_id: int, args=None, config=None, **kwargs):
        log_qwen3("device_run_start", callable_id=callable_id)
        try:
            return orig(callable_id, args, config=config, **kwargs)
        finally:
            log_qwen3("device_run_end", callable_id=callable_id)

    setattr(worker, method_name, _wrapped)  # type: ignore[arg-type]
    try:
        log_qwen3("device_wrap_installed")
        yield
    finally:
        setattr(worker, method_name, orig)  # type: ignore[arg-type]
        log_qwen3("device_wrap_removed")


@contextmanager
def log_l2_case(case_name: str):
    """Bracket the whole L2 validation (golden + device rounds) for one case."""
    log_qwen3("l2_case_start", case=case_name)
    try:
        yield
    finally:
        log_qwen3("l2_case_end", case=case_name)
