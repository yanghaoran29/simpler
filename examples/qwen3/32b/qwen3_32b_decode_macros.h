#ifndef QWEN3_32B_DECODE_MACROS_H
#define QWEN3_32B_DECODE_MACROS_H

/* Build-time layout for Qwen3-32B manual orchestration (Qwen3Decode_manual_scope).
 * Defaults mirror qwen3_32b_decode.py and test_qwen3_decode.py (BATCH=16, HIDDEN=8192, …).
 * Override via PTO2_EXTRA_DEFS, e.g. QWEN3_32B_USER_BATCH=16 QWEN3_32B_BATCH_PADDED=16 */

#ifndef QWEN3_32B_USER_BATCH
#define QWEN3_32B_USER_BATCH 16 /* default */
#endif

#ifndef QWEN3_32B_BATCH_TILE
#define QWEN3_32B_BATCH_TILE 16 /* default; full batch is one tile for 32B decode test */
#endif

#ifndef QWEN3_32B_BATCH_PADDED
#define QWEN3_32B_BATCH_PADDED ((((QWEN3_32B_USER_BATCH) + (QWEN3_32B_BATCH_TILE) - 1) / (QWEN3_32B_BATCH_TILE)) * (QWEN3_32B_BATCH_TILE)) /* default 16 */
#endif

#ifndef QWEN3_32B_MAX_SEQ
#define QWEN3_32B_MAX_SEQ 4096 /* default */
#endif

#ifndef QWEN3_32B_NUM_HEADS
#define QWEN3_32B_NUM_HEADS 64 /* default */
#endif

#ifndef QWEN3_32B_NUM_KV_HEADS
#define QWEN3_32B_NUM_KV_HEADS 8 /* default */
#endif

#ifndef QWEN3_32B_HEAD_DIM
#define QWEN3_32B_HEAD_DIM 128 /* default */
#endif

#ifndef QWEN3_32B_INTERMEDIATE
#define QWEN3_32B_INTERMEDIATE 25600 /* default */
#endif

#ifndef QWEN3_32B_Q_HEAD_PAD
#define QWEN3_32B_Q_HEAD_PAD 16 /* default */
#endif

#ifndef QWEN3_32B_SEQ_TILE
#define QWEN3_32B_SEQ_TILE 256 /* default */
#endif

#ifndef QWEN3_32B_Q_OUT_CHUNK
#define QWEN3_32B_Q_OUT_CHUNK 256 /* default; q_proj outer step */
#endif

#ifndef QWEN3_32B_KV_OUT_CHUNK
#define QWEN3_32B_KV_OUT_CHUNK 256 /* default; kv_proj fused outer step */
#endif

#ifndef QWEN3_32B_MLP_OUT_CHUNK
#define QWEN3_32B_MLP_OUT_CHUNK 256 /* default */
#endif

#ifndef QWEN3_32B_SOFTMAX_ACC_DIM
#define QWEN3_32B_SOFTMAX_ACC_DIM 128 /* default; partial L/M rows per softmax kernel */
#endif

#define QWEN3_32B_HIDDEN ((QWEN3_32B_NUM_HEADS) * (QWEN3_32B_HEAD_DIM))                 /* default 8192 */
#define QWEN3_32B_KV_HIDDEN ((QWEN3_32B_NUM_KV_HEADS) * (QWEN3_32B_HEAD_DIM))           /* default 1024 */
#define QWEN3_32B_Q_PER_KV ((QWEN3_32B_NUM_HEADS) / (QWEN3_32B_NUM_KV_HEADS))           /* default 8 */
#define QWEN3_32B_HALF_DIM ((QWEN3_32B_HEAD_DIM) / 2)                                    /* default 64 */
#define QWEN3_32B_ALL_Q_PAD_STRIDE ((QWEN3_32B_NUM_KV_HEADS) * (QWEN3_32B_Q_HEAD_PAD))  /* default 128 */
#define QWEN3_32B_ALL_Q_PADDED_ROWS ((QWEN3_32B_BATCH_PADDED) * (QWEN3_32B_ALL_Q_PAD_STRIDE)) /* default 2048 */
#define QWEN3_32B_Q_PROJ_STEPS ((QWEN3_32B_HIDDEN) / (QWEN3_32B_Q_OUT_CHUNK))           /* default 32 */
#define QWEN3_32B_KV_PROJ_STEPS ((QWEN3_32B_KV_HIDDEN) / (QWEN3_32B_KV_OUT_CHUNK))      /* default 4 */
#define QWEN3_32B_OUT_PROJ_OB_MAX ((QWEN3_32B_HIDDEN) / (QWEN3_32B_Q_OUT_CHUNK))         /* default 32; loop ob+=2 */
#define QWEN3_32B_OUT_PROJ_OB_STRIDE 2 /* default */
#define QWEN3_32B_MLP_SILU_STEPS ((QWEN3_32B_INTERMEDIATE) / (QWEN3_32B_MLP_OUT_CHUNK)) /* default 100 */
#define QWEN3_32B_DOWN_MIXED_OB_MAX ((QWEN3_32B_HIDDEN) / (QWEN3_32B_Q_OUT_CHUNK))       /* default 32; loop db+=2 */
#define QWEN3_32B_DOWN_MIXED_OB_STRIDE 2 /* default */
#define QWEN3_32B_GM_PIPE_NUMEL ((QWEN3_32B_NUM_KV_HEADS) * (QWEN3_32B_Q_HEAD_PAD) * (QWEN3_32B_Q_OUT_CHUNK)) /* default 32768 */

#endif /* QWEN3_32B_DECODE_MACROS_H */
