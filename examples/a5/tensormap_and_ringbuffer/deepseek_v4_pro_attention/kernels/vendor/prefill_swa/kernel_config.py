# Kernel and Orchestration Configuration

from pathlib import Path

from simpler.task_interface import ArgDirection as _D

_ROOT_DIR = Path(__file__).parent

# Runtime configuration for tensormap_and_ringbuffer.
# AICPU thread count 0 selects the runtime's architecture default (a2a3: 4; a5: 5).
RUNTIME_CONFIG = {
	"runtime": "tensormap_and_ringbuffer",
	"aicpu_thread_num": 0,
}

ORCHESTRATION = {
	"source": str(_ROOT_DIR / "orchestration" / "prefill_attention_swa_test.cpp"),
	"function_name": "aicpu_orchestration_entry",
	"signature": [_D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.INOUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN, _D.OUT],
}

KERNELS = [
	{"func_id": 0, "name": "hc_pre_rms", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_pre_rms.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 1, "name": "hc_pre_linear", "source": str(_ROOT_DIR / "kernels" / "aic" / "hc_pre_linear.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.OUT]},
	{"func_id": 2, "name": "split_pre_post", "source": str(_ROOT_DIR / "kernels" / "aiv" / "split_pre_post.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 3, "name": "comb_sinkhorn", "source": str(_ROOT_DIR / "kernels" / "aiv" / "comb_sinkhorn.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.IN, _D.OUT]},
	{"func_id": 4, "name": "mix_x", "source": str(_ROOT_DIR / "kernels" / "aiv" / "mix_x.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 5, "name": "rms_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rms_norm.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 6, "name": "qkv_rope_rows", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qkv_rope_rows.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 7, "name": "q_rope_prepare", "source": str(_ROOT_DIR / "kernels" / "aiv" / "q_rope_prepare.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT]},
	{"func_id": 8, "name": "qr_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qr_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 9, "name": "qr_rms_norm_quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qr_rms_norm_quant.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.IN, _D.OUT, _D.OUT, _D.OUT]},
	{"func_id": 10, "name": "qproj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "qproj_matmul.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 11, "name": "qproj_dequant_rms_nope_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qproj_dequant_rms_nope_rope.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 12, "name": "kv_proj_matmul", "source": str(_ROOT_DIR / "kernels" / "aic" / "kv_proj_matmul.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 13, "name": "kv_proj_reduce", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_proj_reduce.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT]},
	{"func_id": 14, "name": "kv_rms_norm_rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "kv_rms_norm_rope.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 15, "name": "prefill_swa_cache_write", "source": str(_ROOT_DIR / "kernels" / "aiv" / "prefill_swa_cache_write.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 16, "name": "prefill_swa_window_indices", "source": str(_ROOT_DIR / "kernels" / "aiv" / "prefill_swa_window_indices.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN]},
	{"func_id": 17, "name": "prefill_swa_dummy_cmp_table", "source": str(_ROOT_DIR / "kernels" / "aiv" / "prefill_swa_dummy_cmp_table.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 18, "name": "prefill_swa_empty_cmp_meta", "source": str(_ROOT_DIR / "kernels" / "aiv" / "prefill_swa_empty_cmp_meta.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
	{"func_id": 19, "name": "rope_cs", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope_cs.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 20, "name": "gather_kv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "gather_kv.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 21, "name": "build_bias", "source": str(_ROOT_DIR / "kernels" / "aiv" / "build_bias.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN]},
	{"func_id": 22, "name": "qk_pv_aic", "source": str(_ROOT_DIR / "kernels" / "aic" / "qk_pv_aic.cpp"), "core_type": "aic", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 23, "name": "qk_pv_aiv", "source": str(_ROOT_DIR / "kernels" / "aiv" / "qk_pv_aiv.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 24, "name": "merge_norm", "source": str(_ROOT_DIR / "kernels" / "aiv" / "merge_norm.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 25, "name": "rope", "source": str(_ROOT_DIR / "kernels" / "aiv" / "rope.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.IN, _D.IN, _D.IN]},
	{"func_id": 26, "name": "proj_a_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "proj_a_mm.cpp"), "core_type": "aic", "signature": [_D.IN, _D.IN, _D.INOUT]},
	{"func_id": 27, "name": "quant", "source": str(_ROOT_DIR / "kernels" / "aiv" / "quant.cpp"), "core_type": "aiv", "signature": [_D.INOUT, _D.INOUT, _D.IN]},
	{"func_id": 28, "name": "proj_b_mm", "source": str(_ROOT_DIR / "kernels" / "aic" / "proj_b_mm.cpp"), "core_type": "aic", "signature": [_D.INOUT, _D.IN, _D.IN]},
	{"func_id": 29, "name": "proj_b_act", "source": str(_ROOT_DIR / "kernels" / "aiv" / "proj_b_act.cpp"), "core_type": "aiv", "signature": [_D.IN, _D.OUT, _D.IN, _D.IN]},
	{"func_id": 30, "name": "hc_post_prefill", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_post_prefill.cpp"), "core_type": "aiv", "signature": [_D.OUT, _D.IN, _D.IN, _D.IN, _D.IN]},
	{"func_id": 31, "name": "hc_post_inactive_pad", "source": str(_ROOT_DIR / "kernels" / "aiv" / "hc_post_inactive_pad.cpp"), "core_type": "aiv", "signature": [_D.OUT]},
]
