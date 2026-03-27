/*
 * QEMU TCG plugin: count guest instructions executed (per dynamic execution).
 *
 * Modes:
 *   filter: self-only in [filter_low, filter_high)
 *   inclusive=1: AArch64 flat self + callee stack (BL/BLR..RET)
 *   hierarchy=1: AArch64 tree under root; needs symfile= (nm -S --defined-only)
 *   markers=1: count strict dynamic instructions between marker instructions
 *   bias=<int>: guest_pc - bias == nm file address (PIE/slide); optional if symtab works
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

#define MAX_VCPU 64
#define MAX_STACK 192
#define MAX_MARKER_STACK 32

static uint64_t g_total_insns;
static uint64_t g_self_insns;
static uint64_t g_inclusive_insns;
static char* g_outfile;
static int g_filter_enabled;
static uint64_t g_filter_low;
static uint64_t g_filter_high;
static int g_inclusive_mode;
static int g_hierarchy_mode;
static int g_marker_mode;
static int g_is_aarch64;
static char* g_root_display_name;
static uint32_t g_marker_start_enc = 0xaa030063u; /* orr x3, x3, x3 */
static uint32_t g_marker_end_enc = 0xaa040084u;   /* orr x4, x4, x4 */
static int g_marker_phases_mode;
static uint64_t g_between_markers_insns;
static uint64_t g_between_markers_sessions;
typedef struct {
    uint64_t session_id;
    uint32_t phase_id;
    uint32_t cpu_id;
    uint64_t insn_count;
} MarkerSession;
static MarkerSession* g_marker_sessions;
static size_t g_marker_sessions_count;
static size_t g_marker_sessions_cap;
static int64_t g_active_marker_session_stack[MAX_VCPU][MAX_MARKER_STACK];
static uint32_t g_active_marker_phase_stack[MAX_VCPU][MAX_MARKER_STACK];
static uint32_t g_active_marker_depth[MAX_VCPU];

typedef struct {
    uint32_t phase_id;
    uint32_t start_enc;
    uint32_t end_enc;
    const char* phase_name;
} MarkerPair;

static MarkerPair g_marker_pairs_default[] = {
    {0, 0xaa030063u, 0xaa040084u, "submit_total"},
};

static const MarkerPair g_marker_pairs_phases[] = {
    {0, 0xaa030063u, 0xaa040084u, "submit_total"},
    {1, 0xaa0500a5u, 0xaa0600c6u, "alloc"},
    {2, 0xaa0700e7u, 0xaa080108u, "sync"},
    {3, 0xaa090129u, 0xaa0a014au, "lookup"},
    {4, 0xaa0b016bu, 0xaa0c018cu, "insert"},
    {5, 0xaa0d01adu, 0xaa0e01ceu, "params"},
    {6, 0xaa010021u, 0xaa020042u, "fanin"},
};

static const MarkerPair* g_marker_pairs = g_marker_pairs_default;
static size_t g_marker_pair_count = sizeof(g_marker_pairs_default) / sizeof(g_marker_pairs_default[0]);

typedef struct {
    uint64_t lo;
    uint64_t hi;
    char* name;
} SymEnt;

static SymEnt* g_syms;
static size_t g_n_syms;

typedef struct TNode {
    char* name;
    uint64_t inclusive;
    uint64_t self;
    uint64_t calls;
    struct TNode* children;
    struct TNode* next;
} TNode;

static TNode* g_root_accum;

static int marker_append_session(uint32_t cpu_id, uint32_t phase_id)
{
    if (g_marker_sessions_count == g_marker_sessions_cap) {
        size_t new_cap = (g_marker_sessions_cap == 0) ? 1024 : (g_marker_sessions_cap * 2);
        MarkerSession* ns = realloc(g_marker_sessions, new_cap * sizeof(MarkerSession));
        if (!ns) {
            return -1;
        }
        g_marker_sessions = ns;
        g_marker_sessions_cap = new_cap;
    }
    size_t idx = g_marker_sessions_count++;
    g_marker_sessions[idx].session_id = g_between_markers_sessions + 1;
    g_marker_sessions[idx].phase_id = phase_id;
    g_marker_sessions[idx].cpu_id = cpu_id;
    g_marker_sessions[idx].insn_count = 0;
    g_between_markers_sessions++;
    return (int)idx;
}

/* Guest link-time offset: guest_pc - bias == nm symbol file address (learned at runtime). */
static int g_have_pc_bias;
static int64_t g_pc_bias;

/* When outside call stack (g_hsp<0): previous insn was "in root" symbol (substring match). */
static int g_prev_sym_in_root[MAX_VCPU];

static void tnode_free(TNode* n)
{
    if (!n) {
        return;
    }
    for (TNode* c = n->children; c;) {
        TNode* nx = c->next;
        tnode_free(c);
        c = nx;
    }
    free(n->name);
    free(n);
}

static TNode* tnode_find_child(TNode* parent, const char* name)
{
    for (TNode* c = parent->children; c; c = c->next) {
        if (strcmp(c->name, name) == 0) {
            return c;
        }
    }
    return NULL;
}

static void tnode_append_child(TNode* parent, TNode* ch)
{
    ch->next = parent->children;
    parent->children = ch;
}

static TNode* tnode_get_or_create_child(TNode* parent, const char* name)
{
    TNode* f = tnode_find_child(parent, name);
    if (f) {
        return f;
    }
    TNode* n = calloc(1, sizeof(TNode));
    if (!n) {
        return NULL;
    }
    n->name = strdup(name);
    tnode_append_child(parent, n);
    return n;
}

/* Merge src into dst; frees src after merging (recursive) */
static void merge_tree_into(TNode* dst, TNode* src)
{
    if (!src) {
        return;
    }
    dst->inclusive += src->inclusive;
    dst->self += src->self;
    dst->calls += src->calls;
    while (src->children) {
        TNode* sc = src->children;
        src->children = sc->next;
        sc->next = NULL;
        TNode* dc = tnode_find_child(dst, sc->name);
        if (!dc) {
            tnode_append_child(dst, sc);
        } else {
            merge_tree_into(dc, sc);
        }
    }
    free(src->name);
    free(src);
}

static int sym_cmp(const void* a, const void* b)
{
    const SymEnt* x = a;
    const SymEnt* y = b;
    if (x->lo < y->lo) {
        return -1;
    }
    if (x->lo > y->lo) {
        return 1;
    }
    return 0;
}

static int load_symfile(const char* path)
{
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    char line[1024];
    size_t cap = 256;
    g_syms = calloc(cap, sizeof(SymEnt));
    if (!g_syms) {
        fclose(f);
        return -1;
    }
    while (fgets(line, sizeof line, f)) {
        unsigned long lo = 0, sz = 0;
        char type = 0;
        char name[512];
        if (sscanf(line, "%lx %lx %c %511s", &lo, &sz, &type, name) < 4) {
            continue;
        }
        if (type != 'T' && type != 't') {
            continue;
        }
        if (sz == 0) {
            continue;
        }
        if (g_n_syms >= cap) {
            cap *= 2;
            SymEnt* ns = realloc(g_syms, cap * sizeof(SymEnt));
            if (!ns) {
                fclose(f);
                return -1;
            }
            g_syms = ns;
        }
        g_syms[g_n_syms].lo = (uint64_t)lo;
        g_syms[g_n_syms].hi = (uint64_t)lo + (uint64_t)sz;
        g_syms[g_n_syms].name = strdup(name);
        if (!g_syms[g_n_syms].name) {
            fclose(f);
            return -1;
        }
        g_n_syms++;
    }
    fclose(f);
    if (g_n_syms == 0) {
        return -1;
    }
    qsort(g_syms, g_n_syms, sizeof(SymEnt), sym_cmp);
    return 0;
}

static uint64_t file_pc_for_lookup(uint64_t guest_pc)
{
    if (!g_have_pc_bias) {
        return guest_pc;
    }
    return (uint64_t)((int64_t)guest_pc - g_pc_bias);
}

static int sym_matches_root(const char* sym)
{
    if (!sym || !g_root_display_name) {
        return 0;
    }
    return strstr(sym, g_root_display_name) != NULL;
}

/* First time we see a guest PC + symbol that exactly matches an nm row, learn bias. */
static void maybe_learn_pc_bias(uint64_t guest_pc, const char* sym)
{
    if (g_have_pc_bias || !sym || sym[0] == '?' || !g_syms) {
        return;
    }
    for (size_t i = 0; i < g_n_syms; i++) {
        if (strcmp(g_syms[i].name, sym) == 0) {
            g_pc_bias = (int64_t)guest_pc - (int64_t)g_syms[i].lo;
            g_have_pc_bias = 1;
            return;
        }
    }
}

/*
 * Stripped guests: qemu_plugin_insn_symbol is often "?". If guest VA matches nm file
 * addresses (bias 0), learn that so addr_to_sym / root PC check work.
 */
static void maybe_learn_pc_bias_from_guest_filter_range(uint64_t guest_pc, const char* sym)
{
    (void)sym;
    if (g_have_pc_bias || !g_filter_enabled) {
        return;
    }
    if (guest_pc >= g_filter_low && guest_pc < g_filter_high) {
        g_pc_bias = 0;
        g_have_pc_bias = 1;
    }
}

static int guest_pc_in_root_nm_range(uint64_t guest_pc)
{
    if (!g_filter_enabled) {
        return 0;
    }
    uint64_t fp = file_pc_for_lookup(guest_pc);
    return (fp >= g_filter_low && fp < g_filter_high);
}

/* Root "body" for hierarchy: symbol substring match, or fallback to PC in nm range (after bias). */
static int in_root_for_hier(uint64_t guest_pc, const char* sym)
{
    if (sym_matches_root(sym)) {
        return 1;
    }
    if (guest_pc_in_root_nm_range(guest_pc)) {
        return 1;
    }
    return 0;
}

static const char* addr_to_sym(uint64_t guest_pc)
{
    if (!g_syms || g_n_syms == 0) {
        return NULL;
    }
    uint64_t pc = file_pc_for_lookup(guest_pc);
    const char* best = NULL;
    uint64_t best_span = UINT64_MAX;
    for (size_t i = 0; i < g_n_syms; i++) {
        if (pc >= g_syms[i].lo && pc < g_syms[i].hi) {
            uint64_t sp = g_syms[i].hi - g_syms[i].lo;
            if (sp < best_span) {
                best_span = sp;
                best = g_syms[i].name;
            }
        }
    }
    return best;
}

struct insn_ud {
    uint64_t pc;
    uint32_t enc;
    char* sym;
};

static uint32_t g_depth[MAX_VCPU];
static uint64_t g_last_pc[MAX_VCPU];
static int g_have_last[MAX_VCPU];

typedef struct {
    char* name;
    int name_pending;
    TNode* node;
    uint64_t ret_pc;
    int has_ret_pc;
} HFrame;

static HFrame g_hstk[MAX_VCPU][MAX_STACK];
static int g_hsp[MAX_VCPU];
static TNode* g_inv_root[MAX_VCPU];

static void hier_flush_all_active_roots(void)
{
    for (int c = 0; c < MAX_VCPU; c++) {
        if (g_inv_root[c]) {
            if (g_root_accum) {
                merge_tree_into(g_root_accum, g_inv_root[c]);
            } else {
                g_root_accum = g_inv_root[c];
            }
            g_inv_root[c] = NULL;
            g_hsp[c] = -1;
        }
    }
}

static bool aarch64_is_bl(uint32_t w)
{
    return ((w >> 26) & 0x3fu) == 0x25u;
}

static bool aarch64_is_blr(uint32_t w)
{
    return ((w >> 21) & 0x7ffu) == 0x6b4u;
}

static bool aarch64_is_return_like(uint32_t w)
{
    uint32_t m = w & 0xfffffc1fu;
    return m == 0xd61f03c0u || m == 0xd65f03c0u;
}

static uint64_t aarch64_bl_target(uint64_t pc, uint32_t w)
{
    uint32_t imm26 = w & 0x03ffffffu;
    int64_t off = (int64_t)imm26;
    if (off & (1LL << 25)) {
        off |= ~((1LL << 26) - 1);
    }
    off <<= 2;
    return (uint64_t)((int64_t)pc + off);
}

static void hier_push(unsigned int cpu, const char* name, int pending, uint64_t ret_pc, int has_ret_pc)
{
    if (cpu >= MAX_VCPU) {
        cpu = 0;
    }
    if (g_hsp[cpu] + 1 >= MAX_STACK) {
        return;
    }
    g_hsp[cpu]++;
    int sp = g_hsp[cpu];
    HFrame* fr = &g_hstk[cpu][sp];
    memset(fr, 0, sizeof(*fr));
    fr->name_pending = pending;
    fr->ret_pc = ret_pc;
    fr->has_ret_pc = has_ret_pc;
    if (name && !pending) {
        fr->name = strdup(name);
    }

    if (sp == 0) {
        fr->node = calloc(1, sizeof(TNode));
        if (fr->node) {
            const char* rn = g_root_display_name ? g_root_display_name : "?";
            fr->node->name = strdup(rn);
            fr->node->calls = 1;
            g_inv_root[cpu] = fr->node;
        }
    } else {
        TNode* parent = g_hstk[cpu][sp - 1].node;
        const char* ch = (name && !pending) ? name : "?";
        fr->node = tnode_get_or_create_child(parent, ch);
        if (fr->node) {
            fr->node->calls++;
        }
    }
}

static void hier_pop(unsigned int cpu)
{
    if (cpu >= MAX_VCPU) {
        cpu = 0;
    }
    if (g_hsp[cpu] < 0) {
        return;
    }
    HFrame* fr = &g_hstk[cpu][g_hsp[cpu]];
    free(fr->name);
    fr->name = NULL;

    int was_root = (g_hsp[cpu] == 0);
    TNode* inv = g_inv_root[cpu];
    g_hsp[cpu]--;

    if (was_root && inv) {
        if (g_root_accum) {
            merge_tree_into(g_root_accum, inv);
        } else {
            g_root_accum = inv;
        }
        g_inv_root[cpu] = NULL;
    }
}

/* Pop frames when PC reaches saved return address (pc+4 at BL/BLR). */
static void hier_unwind_by_return_pc(unsigned int cpu, uint64_t pc)
{
    if (cpu >= MAX_VCPU) {
        cpu = 0;
    }
    while (g_hsp[cpu] >= 0) {
        HFrame* fr = &g_hstk[cpu][g_hsp[cpu]];
        if (!fr->has_ret_pc || fr->ret_pc != pc) {
            break;
        }
        hier_pop(cpu);
    }
}

static void vcpu_insn_exec_self(unsigned int cpu_index, void* udata)
{
    (void)udata;
    if (cpu_index >= MAX_VCPU) {
        cpu_index = 0;
    }
    __atomic_add_fetch(&g_total_insns, 1, __ATOMIC_RELAXED);
}

static void vcpu_insn_exec_inclusive(unsigned int cpu_index, void* udata)
{
    struct insn_ud* u = (struct insn_ud*)udata;
    uint64_t pc = u->pc;
    uint32_t enc = u->enc;

    if (cpu_index >= MAX_VCPU) {
        cpu_index = 0;
    }

    if (g_depth[cpu_index] == 0) {
        bool in_f = (pc >= g_filter_low && pc < g_filter_high);
        bool last_in_f = false;
        if (g_have_last[cpu_index]) {
            uint64_t lp = g_last_pc[cpu_index];
            last_in_f = (lp >= g_filter_low && lp < g_filter_high);
        }
        if (in_f && !last_in_f) {
            g_depth[cpu_index] = 1u;
        }
    }

    if (g_depth[cpu_index] > 0) {
        __atomic_add_fetch(&g_inclusive_insns, 1, __ATOMIC_RELAXED);
    }
    if (pc >= g_filter_low && pc < g_filter_high) {
        __atomic_add_fetch(&g_self_insns, 1, __ATOMIC_RELAXED);
    }

    if (g_depth[cpu_index] > 0 && g_is_aarch64) {
        if (aarch64_is_bl(enc) || aarch64_is_blr(enc)) {
            if (g_depth[cpu_index] < 0xffffffffu) {
                g_depth[cpu_index]++;
            }
        } else if (aarch64_is_return_like(enc)) {
            if (g_depth[cpu_index] > 0) {
                g_depth[cpu_index]--;
            }
        }
    }

    g_last_pc[cpu_index] = pc;
    g_have_last[cpu_index] = 1;
}

static void vcpu_insn_exec_markers(unsigned int cpu_index, void* udata)
{
    struct insn_ud* u = (struct insn_ud*)udata;
    uint32_t enc = u->enc;
    if (cpu_index >= MAX_VCPU) {
        cpu_index = 0;
    }
    for (size_t i = 0; i < g_marker_pair_count; i++) {
        if (enc == g_marker_pairs[i].start_enc) {
            int idx = marker_append_session((uint32_t)cpu_index, g_marker_pairs[i].phase_id);
            if (idx >= 0) {
                uint32_t d = g_active_marker_depth[cpu_index];
                if (d < MAX_MARKER_STACK) {
                    g_active_marker_session_stack[cpu_index][d] = idx;
                    g_active_marker_phase_stack[cpu_index][d] = g_marker_pairs[i].phase_id;
                    g_active_marker_depth[cpu_index] = d + 1;
                }
            }
            return;
        }
    }
    for (size_t i = 0; i < g_marker_pair_count; i++) {
        if (enc == g_marker_pairs[i].end_enc) {
            uint32_t d = g_active_marker_depth[cpu_index];
            while (d > 0) {
                uint32_t top = d - 1;
                if (g_active_marker_phase_stack[cpu_index][top] == g_marker_pairs[i].phase_id) {
                    g_active_marker_depth[cpu_index] = top;
                    break;
                }
                d--;
            }
            return;
        }
    }
    uint32_t d = g_active_marker_depth[cpu_index];
    if (d > 0) {
        for (uint32_t k = 0; k < d; k++) {
            int64_t active_idx = g_active_marker_session_stack[cpu_index][k];
            if (active_idx >= 0 && (size_t)active_idx < g_marker_sessions_count) {
                g_marker_sessions[active_idx].insn_count++;
            }
        }
        g_between_markers_insns++;
    }
}

static void vcpu_insn_exec_hier(unsigned int cpu_index, void* udata)
{
    struct insn_ud* u = (struct insn_ud*)udata;
    uint64_t pc = u->pc;
    uint32_t enc = u->enc;
    const char* sym = u->sym ? u->sym : "?";

    if (cpu_index >= MAX_VCPU) {
        cpu_index = 0;
    }

    maybe_learn_pc_bias_from_guest_filter_range(pc, sym);
    maybe_learn_pc_bias(pc, sym);
    hier_unwind_by_return_pc(cpu_index, pc);

    int in_root_sym = in_root_for_hier(pc, sym);

    /* Root session entry: symbol match or PC in nm filter range (stripped ELF + bias). */
    if (g_hsp[cpu_index] < 0 && in_root_sym && !g_prev_sym_in_root[cpu_index]) {
        hier_push(cpu_index, g_root_display_name ? g_root_display_name : sym, 0, 0, 0);
    }

    if (g_hsp[cpu_index] >= 0) {
        HFrame* top = &g_hstk[cpu_index][g_hsp[cpu_index]];
        if (top->name_pending && top->node) {
            free(top->name);
            top->name = strdup(sym);
            top->name_pending = 0;
            free(top->node->name);
            top->node->name = strdup(sym);
        }
    }

    if (g_hsp[cpu_index] >= 0) {
        HFrame* top = &g_hstk[cpu_index][g_hsp[cpu_index]];
        if (top->node) {
            top->node->inclusive++;
            int self_hit = 0;
            if (g_hsp[cpu_index] == 0) {
                self_hit = in_root_sym;
            } else if (top->name && strcmp(sym, top->name) == 0) {
                self_hit = 1;
            }
            if (self_hit) {
                top->node->self++;
            }
        }
    }

    if (g_hsp[cpu_index] == 0 && !in_root_sym) {
        hier_pop(cpu_index);
    }

    if (g_hsp[cpu_index] >= 0 && g_is_aarch64) {
        if (aarch64_is_bl(enc)) {
            uint64_t tgt = aarch64_bl_target(pc, enc);
            const char* cn = addr_to_sym(tgt);
            if (!cn) {
                cn = "?";
            }
            hier_push(cpu_index, cn, 0, pc + 4u, 1);
        } else if (aarch64_is_blr(enc)) {
            hier_push(cpu_index, NULL, 1, pc + 4u, 1);
        }
    }

    if (g_hsp[cpu_index] < 0) {
        g_prev_sym_in_root[cpu_index] = in_root_sym ? 1 : 0;
    }

    g_last_pc[cpu_index] = pc;
    g_have_last[cpu_index] = 1;
}

static int tnode_child_count(TNode* n)
{
    int c = 0;
    for (TNode* x = n->children; x; x = x->next) {
        c++;
    }
    return c;
}

static int tnode_cmp_name(const void* a, const void* b)
{
    TNode* const* x = a;
    TNode* const* y = b;
    return strcmp((*x)->name, (*y)->name);
}

static void print_subtree(FILE* f, TNode* n, int depth)
{
    int i;
    if (!n) {
        return;
    }
    for (i = 0; i < depth; i++) {
        fputc('\t', f);
    }
    fprintf(f, "%s: 指令=%" PRIu64 ", 调用=%" PRIu64 "\n", n->name, n->inclusive, n->calls);
    for (i = 0; i < depth; i++) {
        fputc('\t', f);
    }
    fprintf(f, "本级指令: %" PRIu64 "\n", n->self);

    int nc = tnode_child_count(n);
    if (nc <= 0) {
        return;
    }
    TNode** arr = calloc((size_t)nc, sizeof(TNode*));
    if (!arr) {
        return;
    }
    int k = 0;
    for (TNode* c = n->children; c; c = c->next) {
        arr[k++] = c;
    }
    qsort(arr, (size_t)nc, sizeof(TNode*), tnode_cmp_name);
    for (k = 0; k < nc; k++) {
        print_subtree(f, arr[k], depth + 1);
    }
    free(arr);
}

static void vcpu_tb_trans(qemu_plugin_id_t id, struct qemu_plugin_tb* tb)
{
    (void)id;
    size_t n = qemu_plugin_tb_n_insns(tb);

    if (g_hierarchy_mode) {
        struct insn_ud* block = calloc(n, sizeof(struct insn_ud));
        if (!block) {
            return;
        }
        for (size_t i = 0; i < n; i++) {
            struct qemu_plugin_insn* insn = qemu_plugin_tb_get_insn(tb, i);
            block[i].pc = qemu_plugin_insn_vaddr(insn);
            const void* d = qemu_plugin_insn_data(insn);
            memcpy(&block[i].enc, d, sizeof(uint32_t));
            const char* s = qemu_plugin_insn_symbol(insn);
            block[i].sym = s ? strdup(s) : strdup("?");
            qemu_plugin_register_vcpu_insn_exec_cb(
                insn, vcpu_insn_exec_hier, QEMU_PLUGIN_CB_NO_REGS, &block[i]);
        }
        return;
    }

    if (g_marker_mode) {
        struct insn_ud* block = calloc(n, sizeof(struct insn_ud));
        if (!block) {
            return;
        }
        for (size_t i = 0; i < n; i++) {
            struct qemu_plugin_insn* insn = qemu_plugin_tb_get_insn(tb, i);
            block[i].pc = qemu_plugin_insn_vaddr(insn);
            const void* d = qemu_plugin_insn_data(insn);
            memcpy(&block[i].enc, d, sizeof(uint32_t));
            block[i].sym = NULL;
            qemu_plugin_register_vcpu_insn_exec_cb(
                insn, vcpu_insn_exec_markers, QEMU_PLUGIN_CB_NO_REGS, &block[i]);
        }
        return;
    }

    if (g_inclusive_mode) {
        struct insn_ud* block = calloc(n, sizeof(struct insn_ud));
        if (!block) {
            return;
        }
        for (size_t i = 0; i < n; i++) {
            struct qemu_plugin_insn* insn = qemu_plugin_tb_get_insn(tb, i);
            block[i].pc = qemu_plugin_insn_vaddr(insn);
            const void* d = qemu_plugin_insn_data(insn);
            memcpy(&block[i].enc, d, sizeof(uint32_t));
            block[i].sym = NULL;
            qemu_plugin_register_vcpu_insn_exec_cb(
                insn, vcpu_insn_exec_inclusive, QEMU_PLUGIN_CB_NO_REGS, &block[i]);
        }
        return;
    }

    for (size_t i = 0; i < n; i++) {
        struct qemu_plugin_insn* insn = qemu_plugin_tb_get_insn(tb, i);
        uint64_t va = qemu_plugin_insn_vaddr(insn);
        if (g_filter_enabled && (va < g_filter_low || va >= g_filter_high)) {
            continue;
        }
        qemu_plugin_register_vcpu_insn_exec_cb(
            insn, vcpu_insn_exec_self, QEMU_PLUGIN_CB_NO_REGS, NULL);
    }
}

static void plugin_exit(qemu_plugin_id_t id, void* userdata)
{
    (void)id;
    (void)userdata;
    char line[384];

    if (g_hierarchy_mode) {
        /* Return tracking can miss some unwind edges; preserve unfinished roots. */
        hier_flush_all_active_roots();
    }

    if (g_hierarchy_mode && g_root_accum) {
        snprintf(line, sizeof line,
            "QEMU_TCG hierarchical instruction counts (root symbol ~'%s', nm range [0x%" PRIx64 ", 0x%" PRIx64 ")):\n",
            g_root_display_name ? g_root_display_name : "?", g_filter_low, g_filter_high);
        qemu_plugin_outs(line);
        if (g_outfile && g_outfile[0]) {
            FILE* f = fopen(g_outfile, "w");
            if (f) {
                fputs(line, f);
                print_subtree(f, g_root_accum, 0);
                fclose(f);
            }
        }
        tnode_free(g_root_accum);
        g_root_accum = NULL;
        goto cleanup;
    }

    if (g_hierarchy_mode && !g_root_accum) {
        snprintf(line, sizeof line,
            "QEMU_TCG hierarchy: no root calls recorded (root ~'%s'; nm [0x%" PRIx64 ", 0x%" PRIx64
            ")). If insn symbol is '?' (stripped ELF), set plugin bias= guest_pc_minus_nm_low, or keep symtab.\n",
            g_root_display_name ? g_root_display_name : "?", g_filter_low, g_filter_high);
        qemu_plugin_outs(line);
        if (g_outfile && g_outfile[0]) {
            FILE* f = fopen(g_outfile, "w");
            if (f) {
                fputs(line, f);
                fclose(f);
            }
        }
        goto cleanup;
    }

    if (g_marker_mode) {
        uint64_t phase_sum[16] = {0};
        uint64_t phase_cnt[16] = {0};
        uint64_t phase_max[16] = {0};
        uint64_t phase_min[16];
        for (size_t i = 0; i < 16; i++) {
            phase_min[i] = UINT64_MAX;
        }
        uint64_t max_session_insns = 0;
        uint64_t min_session_insns = 0;
        uint64_t avg_session_insns = 0;
        if (g_marker_sessions_count > 0) {
            min_session_insns = UINT64_MAX;
            for (size_t i = 0; i < g_marker_sessions_count; i++) {
                uint64_t v = g_marker_sessions[i].insn_count;
                if (v > max_session_insns) {
                    max_session_insns = v;
                }
                if (v < min_session_insns) {
                    min_session_insns = v;
                }
                uint32_t pid = g_marker_sessions[i].phase_id;
                if (pid < 16) {
                    phase_sum[pid] += v;
                    phase_cnt[pid] += 1;
                    if (v > phase_max[pid]) {
                        phase_max[pid] = v;
                    }
                    if (v < phase_min[pid]) {
                        phase_min[pid] = v;
                    }
                }
            }
            avg_session_insns = g_between_markers_insns / g_marker_sessions_count;
        }
        snprintf(line, sizeof line,
            "QEMU_TCG between_markers_insns: %" PRIu64
            " (sessions=%" PRIu64 ", avg_per_session=%" PRIu64
            ", max_session_insns=%" PRIu64 ", min_session_insns=%" PRIu64
            ", start=0x%08x, end=0x%08x, phase_markers=%d)\n",
            g_between_markers_insns,
            g_between_markers_sessions,
            avg_session_insns,
            max_session_insns,
            min_session_insns,
            g_marker_start_enc,
            g_marker_end_enc,
            g_marker_phases_mode);
        qemu_plugin_outs(line);
        char summary_line[384];
        snprintf(summary_line, sizeof summary_line,
            "QEMU_TCG between_markers_insns: %" PRIu64
            " (sessions=%" PRIu64 ", avg_per_session=%" PRIu64
            ", max_session_insns=%" PRIu64 ", min_session_insns=%" PRIu64
            ", start=0x%08x, end=0x%08x, phase_markers=%d)\n",
            g_between_markers_insns,
            g_between_markers_sessions,
            avg_session_insns,
            max_session_insns,
            min_session_insns,
            g_marker_start_enc,
            g_marker_end_enc,
            g_marker_phases_mode);
        for (size_t i = 0; i < g_marker_pair_count; i++) {
            uint32_t pid = g_marker_pairs[i].phase_id;
            uint64_t cnt = (pid < 16) ? phase_cnt[pid] : 0;
            uint64_t sum = (pid < 16) ? phase_sum[pid] : 0;
            uint64_t avg = (cnt > 0) ? (sum / cnt) : 0;
            uint64_t pmax = (pid < 16) ? phase_max[pid] : 0;
            uint64_t pmin = (pid < 16 && phase_min[pid] != UINT64_MAX) ? phase_min[pid] : 0;
            snprintf(line, sizeof line,
                "QEMU_TCG phase_insns: phase_id=%u name=%s sessions=%" PRIu64
                " avg=%" PRIu64 " max=%" PRIu64 " min=%" PRIu64 "\n",
                pid, g_marker_pairs[i].phase_name, cnt, avg, pmax, pmin);
            qemu_plugin_outs(line);
        }
        if (g_outfile && g_outfile[0]) {
            FILE* f = fopen(g_outfile, "w");
            if (f) {
                fputs(summary_line, f);
                for (size_t i = 0; i < g_marker_pair_count; i++) {
                    uint32_t pid = g_marker_pairs[i].phase_id;
                    uint64_t cnt = (pid < 16) ? phase_cnt[pid] : 0;
                    uint64_t sum = (pid < 16) ? phase_sum[pid] : 0;
                    uint64_t avg = (cnt > 0) ? (sum / cnt) : 0;
                    uint64_t pmax = (pid < 16) ? phase_max[pid] : 0;
                    uint64_t pmin = (pid < 16 && phase_min[pid] != UINT64_MAX) ? phase_min[pid] : 0;
                    fprintf(f,
                        "QEMU_TCG phase_insns: phase_id=%u name=%s sessions=%" PRIu64
                        " avg=%" PRIu64 " max=%" PRIu64 " min=%" PRIu64 "\n",
                        pid, g_marker_pairs[i].phase_name, cnt, avg, pmax, pmin);
                }
                fputs("session_id,phase_id,cpu_id,insn_count\n", f);
                for (size_t i = 0; i < g_marker_sessions_count; i++) {
                    fprintf(f, "%" PRIu64 ",%u,%u,%" PRIu64 "\n",
                        g_marker_sessions[i].session_id,
                        g_marker_sessions[i].phase_id,
                        g_marker_sessions[i].cpu_id,
                        g_marker_sessions[i].insn_count);
                }
                fclose(f);
            }
        }
        goto cleanup;
    }

    if (g_inclusive_mode && g_filter_enabled) {
        snprintf(line, sizeof line,
            "QEMU_TCG guest instructions: self-only (PC in [0x%" PRIx64 ", 0x%" PRIx64 ")) %" PRIu64
            "; inclusive (root + callees via BL/BLR..RET) %" PRIu64 "\n",
            g_filter_low, g_filter_high, g_self_insns, g_inclusive_insns);
    } else if (g_filter_enabled) {
        snprintf(line, sizeof line,
            "QEMU_TCG guest instructions (PC in [0x%" PRIx64 ", 0x%" PRIx64 ")): %" PRIu64 "\n",
            g_filter_low, g_filter_high, g_total_insns);
    } else {
        snprintf(line, sizeof line, "QEMU_TCG guest instructions executed: %" PRIu64 "\n", g_total_insns);
    }
    qemu_plugin_outs(line);
    if (g_outfile && g_outfile[0]) {
        FILE* f = fopen(g_outfile, "w");
        if (f) {
            fputs(line, f);
            fclose(f);
        }
    }

cleanup:
    free(g_marker_sessions);
    g_marker_sessions = NULL;
    g_marker_sessions_count = 0;
    g_marker_sessions_cap = 0;
    if (g_syms) {
        for (size_t i = 0; i < g_n_syms; i++) {
            free(g_syms[i].name);
        }
        free(g_syms);
        g_syms = NULL;
        g_n_syms = 0;
    }
    free(g_root_display_name);
    g_root_display_name = NULL;
}

QEMU_PLUGIN_EXPORT int qemu_plugin_install(
    qemu_plugin_id_t id, const qemu_info_t* info, int argc, char** argv)
{
    const char* target = info && info->target_name ? info->target_name : "";
    g_is_aarch64 = (strstr(target, "aarch64") != NULL || strstr(target, "arm64") != NULL);

    int have_low = 0, have_high = 0;
    char* symfile = NULL;

    for (int i = 0; i < argc; i++) {
        if (strncmp(argv[i], "outfile=", 8) == 0) {
            free(g_outfile);
            g_outfile = strdup(argv[i] + 8);
        } else if (strncmp(argv[i], "filter_low=", 11) == 0) {
            g_filter_low = strtoull(argv[i] + 11, NULL, 0);
            have_low = 1;
        } else if (strncmp(argv[i], "filter_high=", 12) == 0) {
            g_filter_high = strtoull(argv[i] + 12, NULL, 0);
            have_high = 1;
        } else if (strcmp(argv[i], "inclusive=1") == 0) {
            g_inclusive_mode = 1;
        } else if (strcmp(argv[i], "hierarchy=1") == 0) {
            g_hierarchy_mode = 1;
        } else if (strcmp(argv[i], "markers=1") == 0) {
            g_marker_mode = 1;
        } else if (strncmp(argv[i], "symfile=", 8) == 0) {
            free(symfile);
            symfile = strdup(argv[i] + 8);
        } else if (strncmp(argv[i], "rootname=", 9) == 0) {
            free(g_root_display_name);
            g_root_display_name = strdup(argv[i] + 9);
        } else if (strncmp(argv[i], "bias=", 5) == 0) {
            errno = 0;
            char* end = NULL;
            long long v = strtoll(argv[i] + 5, &end, 0);
            if (end != argv[i] + 5 && errno == 0) {
                g_pc_bias = (int64_t)v;
                g_have_pc_bias = 1;
            }
        } else if (strncmp(argv[i], "marker_start=", 13) == 0) {
            g_marker_start_enc = (uint32_t)strtoul(argv[i] + 13, NULL, 0);
        } else if (strncmp(argv[i], "marker_end=", 11) == 0) {
            g_marker_end_enc = (uint32_t)strtoul(argv[i] + 11, NULL, 0);
        } else if (strcmp(argv[i], "marker_phases=1") == 0) {
            g_marker_phases_mode = 1;
        }
    }
    if (have_low && have_high && g_filter_low < g_filter_high) {
        g_filter_enabled = 1;
    }

    if (g_hierarchy_mode) {
        if (!g_filter_enabled || !symfile || load_symfile(symfile) != 0) {
            qemu_plugin_outs("insn_count: hierarchy=1 needs filter_low/high and valid symfile=\n");
            free(symfile);
            return -1;
        }
        if (!g_is_aarch64) {
            qemu_plugin_outs("insn_count: hierarchy=1 is only for aarch64\n");
            free(symfile);
            return -1;
        }
        g_inclusive_mode = 0;
        if (!g_root_display_name) {
            g_root_display_name = strdup("pto2_submit_mixed_task");
        }
        for (int c = 0; c < MAX_VCPU; c++) {
            g_hsp[c] = -1;
            g_prev_sym_in_root[c] = 0;
        }
    }
    free(symfile);

    if (g_inclusive_mode) {
        if (!g_filter_enabled) {
            qemu_plugin_outs("insn_count: inclusive=1 requires filter_low and filter_high\n");
            return -1;
        }
        if (!g_is_aarch64) {
            qemu_plugin_outs("insn_count: inclusive=1 is only supported for aarch64 user-mode guests\n");
            return -1;
        }
    }

    if (g_marker_mode) {
        if (!g_is_aarch64) {
            qemu_plugin_outs("insn_count: markers=1 is only supported for aarch64 user-mode guests\n");
            return -1;
        }
        if (g_marker_phases_mode) {
            g_marker_pairs = g_marker_pairs_phases;
            g_marker_pair_count = sizeof(g_marker_pairs_phases) / sizeof(g_marker_pairs_phases[0]);
        } else {
            g_marker_pairs_default[0].start_enc = g_marker_start_enc;
            g_marker_pairs_default[0].end_enc = g_marker_end_enc;
        }
        for (int c = 0; c < MAX_VCPU; c++) {
            g_active_marker_depth[c] = 0;
            for (int k = 0; k < MAX_MARKER_STACK; k++) {
                g_active_marker_session_stack[c][k] = -1;
                g_active_marker_phase_stack[c][k] = 0;
            }
        }
    }
    qemu_plugin_register_vcpu_tb_trans_cb(id, vcpu_tb_trans);
    qemu_plugin_register_atexit_cb(id, plugin_exit, NULL);
    return 0;
}
