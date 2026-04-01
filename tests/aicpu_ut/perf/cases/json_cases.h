/**
 * json_cases.h — Minimal JSON loader for PerfTestCase arrays
 *
 * Parses a JSON file containing an array of test case objects.
 * Supported field types: string, int, float, int[].
 * Unknown fields are skipped.
 */

#pragma once
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>

// ─── Data structure ───────────────────────────────────────────────────────────

struct PerfTestCase {
    char     name[256];
    int      batch;
    int      num_heads;
    int      kv_head_num;
    int      head_dim;
    int      block_size;
    int      block_num;
    float    scale_value;
    int      context_lens[64];
    int      context_lens_count;
};

// ─── Internal parser helpers ──────────────────────────────────────────────────

static inline void jc_skip_ws(const char** p) {
    while (**p && (unsigned char)**p <= ' ') (*p)++;
}

static inline bool jc_expect(const char** p, char c) {
    jc_skip_ws(p);
    if (**p == c) { (*p)++; return true; }
    return false;
}

static inline bool jc_parse_string(const char** p, char* out, int max_len) {
    jc_skip_ws(p);
    if (**p != '"') return false;
    (*p)++;
    int i = 0;
    while (**p && **p != '"') {
        if (**p == '\\') (*p)++;  // skip escape prefix
        if (i < max_len - 1) out[i++] = **p;
        if (**p) (*p)++;
    }
    if (**p == '"') (*p)++;
    out[i] = '\0';
    return true;
}

static inline bool jc_parse_int(const char** p, int* out) {
    jc_skip_ws(p);
    char* end;
    long v = strtol(*p, &end, 10);
    if (end == *p) return false;
    *out = (int)v;
    *p = end;
    return true;
}

static inline bool jc_parse_float(const char** p, float* out) {
    jc_skip_ws(p);
    char* end;
    double v = strtod(*p, &end);
    if (end == *p) return false;
    *out = (float)v;
    *p = end;
    return true;
}

static inline bool jc_parse_int_array(const char** p, int* arr, int* count, int max) {
    jc_skip_ws(p);
    if (!jc_expect(p, '[')) return false;
    *count = 0;
    jc_skip_ws(p);
    if (**p == ']') { (*p)++; return true; }
    while (**p) {
        if (*count >= max) return false;
        if (!jc_parse_int(p, &arr[(*count)++])) return false;
        jc_skip_ws(p);
        if (**p == ',') { (*p)++; continue; }
        if (**p == ']') { (*p)++; break; }
        return false;
    }
    return true;
}

static inline void jc_skip_value(const char** p) {
    jc_skip_ws(p);
    if (**p == '"') {
        char tmp[512];
        jc_parse_string(p, tmp, sizeof(tmp));
        return;
    }
    if (**p == '[') {
        int tmp[256]; int cnt;
        jc_parse_int_array(p, tmp, &cnt, 256);
        return;
    }
    // Skip number or bare word (true/false/null)
    while (**p && **p != ',' && **p != '}' && **p != ']') (*p)++;
}

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Load test cases from a JSON file.
 *
 * @param filename  Path to JSON file containing an array of case objects.
 * @param out       Output array (caller allocated).
 * @param max_cases Maximum number of cases to read.
 * @return Number of cases loaded, or -1 on error.
 */
static inline int load_test_cases(const char* filename, PerfTestCase* out, int max_cases) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "json_cases: cannot open '%s'\n", filename);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return -1; }
    if (fread(buf, 1, sz, f) != (size_t)sz) { free(buf); fclose(f); return -1; }
    buf[sz] = '\0';
    fclose(f);

    const char* p = buf;
    int count = 0;

    jc_skip_ws(&p);
    if (!jc_expect(&p, '[')) { free(buf); return -1; }
    jc_skip_ws(&p);
    if (*p == ']') { free(buf); return 0; }

    while (*p && count < max_cases) {
        jc_skip_ws(&p);
        if (*p == ']') break;
        if (*p == ',') { p++; continue; }
        if (!jc_expect(&p, '{')) break;

        PerfTestCase* tc = &out[count];
        memset(tc, 0, sizeof(*tc));
        tc->scale_value = 1.0f;  // default

        while (*p) {
            jc_skip_ws(&p);
            if (*p == '}') { p++; break; }
            if (*p == ',') { p++; continue; }

            char key[64];
            if (!jc_parse_string(&p, key, sizeof(key))) break;
            jc_skip_ws(&p);
            if (!jc_expect(&p, ':')) break;

            if      (strcmp(key, "name")          == 0) jc_parse_string(&p, tc->name, sizeof(tc->name));
            else if (strcmp(key, "batch")         == 0) jc_parse_int(&p, &tc->batch);
            else if (strcmp(key, "num_heads")     == 0) jc_parse_int(&p, &tc->num_heads);
            else if (strcmp(key, "kv_head_num")   == 0) jc_parse_int(&p, &tc->kv_head_num);
            else if (strcmp(key, "head_dim")      == 0) jc_parse_int(&p, &tc->head_dim);
            else if (strcmp(key, "block_size")    == 0) jc_parse_int(&p, &tc->block_size);
            else if (strcmp(key, "block_num")     == 0) jc_parse_int(&p, &tc->block_num);
            else if (strcmp(key, "scale_value")   == 0) jc_parse_float(&p, &tc->scale_value);
            else if (strcmp(key, "context_lens")  == 0)
                jc_parse_int_array(&p, tc->context_lens, &tc->context_lens_count, 64);
            else jc_skip_value(&p);
        }

        count++;
    }

    free(buf);
    return count;
}
