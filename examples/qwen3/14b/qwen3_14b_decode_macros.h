#ifndef QWEN3_14B_DECODE_MACROS_H
#define QWEN3_14B_DECODE_MACROS_H

/* Build-time layout for Qwen3-14B manual orchestration (examples/.../Qwen3Decode_manual_scope).
 * QWEN3_USER_BATCH defaults to 16 (same as ``qwen3_14b_decode.py`` demo BATCH). The manual_scope test
 * uses BATCH=240 — pass QWEN3_USER_BATCH / QWEN3_BATCH_PADDED via PTO2_EXTRA_DEFS when running that test.
 * HIDDEN=5120, …; override with PTO2_EXTRA_DEFS to match your host tensors.
 * Override at compile time via PTO2_EXTRA_DEFS (space-separated -D tokens), e.g.:
 *   QWEN3_USER_BATCH=32 QWEN3_BATCH_PADDED=32
 * If QWEN3_BATCH_PADDED is omitted, it defaults to BATCH_TILE–aligned padding of QWEN3_USER_BATCH. */

#ifndef QWEN3_USER_BATCH
#define QWEN3_USER_BATCH 16 /* default; align with qwen3_14b_decode.py BATCH */
#endif

#ifndef QWEN3_BATCH_TILE
#define QWEN3_BATCH_TILE 16 /* default */
#endif

#ifndef QWEN3_BATCH_PADDED
#define QWEN3_BATCH_PADDED ((((QWEN3_USER_BATCH) + (QWEN3_BATCH_TILE) - 1) / (QWEN3_BATCH_TILE)) * (QWEN3_BATCH_TILE)) /* default */
#endif

#ifndef QWEN3_MAX_SEQ
#define QWEN3_MAX_SEQ 4096 /* default */
#endif

#ifndef QWEN3_NUM_HEADS
#define QWEN3_NUM_HEADS 40 /* default */
#endif

#ifndef QWEN3_NUM_KV_HEADS
#define QWEN3_NUM_KV_HEADS 8 /* default */
#endif

#ifndef QWEN3_HEAD_DIM
#define QWEN3_HEAD_DIM 128 /* default */
#endif

#ifndef QWEN3_INTERMEDIATE
#define QWEN3_INTERMEDIATE 17408 /* default */
#endif

#ifndef QWEN3_Q_HEAD_PAD
#define QWEN3_Q_HEAD_PAD 16 /* default */
#endif

#ifndef QWEN3_SEQ_TILE
#define QWEN3_SEQ_TILE 256 /* default */
#endif

#ifndef QWEN3_BLOCK_SIZE
#define QWEN3_BLOCK_SIZE 256 /* default */
#endif

#ifndef QWEN3_INPUT_PROJ_K_CHUNK
#define QWEN3_INPUT_PROJ_K_CHUNK 256 /* default */
#endif

#ifndef QWEN3_KV_PROJ_K_CHUNK
#define QWEN3_KV_PROJ_K_CHUNK 256 /* default */
#endif

#ifndef QWEN3_OUT_PROJ_K_CHUNK
#define QWEN3_OUT_PROJ_K_CHUNK 256 /* default */
#endif

#ifndef QWEN3_MLP_OUT_CHUNK
#define QWEN3_MLP_OUT_CHUNK 256 /* default */
#endif

#define QWEN3_HIDDEN ((QWEN3_NUM_HEADS) * (QWEN3_HEAD_DIM))                 /* default 5120 */
#define QWEN3_KV_HIDDEN ((QWEN3_NUM_KV_HEADS) * (QWEN3_HEAD_DIM))           /* default 1024 */
#define QWEN3_Q_PER_KV ((QWEN3_NUM_HEADS) / (QWEN3_NUM_KV_HEADS))           /* default 5 */
#define QWEN3_HALF_DIM ((QWEN3_HEAD_DIM) / 2)                               /* default 64 */
#define QWEN3_ALL_Q_PAD_STRIDE ((QWEN3_NUM_KV_HEADS) * (QWEN3_Q_HEAD_PAD))  /* default 128 */
#define QWEN3_NUM_TILES ((QWEN3_BATCH_PADDED) / (QWEN3_BATCH_TILE))
#define QWEN3_Q_PROJ_STEPS ((QWEN3_HIDDEN) / (QWEN3_INPUT_PROJ_K_CHUNK))
#define QWEN3_KV_PROJ_STEPS ((QWEN3_KV_HIDDEN) / (QWEN3_KV_PROJ_K_CHUNK))
#define QWEN3_MAX_BLOCKS_PER_SEQ (((QWEN3_MAX_SEQ) + (QWEN3_BLOCK_SIZE) - 1) / (QWEN3_BLOCK_SIZE)) /* default 16 */
#define QWEN3_OUT_PROJ_OB_COUNT ((QWEN3_HIDDEN) / (QWEN3_OUT_PROJ_K_CHUNK))   /* default 20 */
#define QWEN3_MLP_SILU_STEPS ((QWEN3_INTERMEDIATE) / (QWEN3_MLP_OUT_CHUNK))  /* default 68 */
#define QWEN3_DOWN_OUTPUT_CHUNKS ((QWEN3_HIDDEN) / (QWEN3_OUT_PROJ_K_CHUNK))   /* default 20 */
#define QWEN3_GM_PIPE_NUMEL ((QWEN3_NUM_KV_HEADS) * (QWEN3_Q_HEAD_PAD) * (QWEN3_OUT_PROJ_K_CHUNK)) /* default 32768 */

#endif /* QWEN3_14B_DECODE_MACROS_H */
