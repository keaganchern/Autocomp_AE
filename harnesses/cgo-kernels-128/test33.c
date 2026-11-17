// x32-packw-gemm-goi (nr=2, kr=1, sr=1): GEMM weight-packing routine — reads
// GOI-ordered u32 weights (g groups × nc output channels × kc input
// channels, row-major) plus an optional bias array, and emits the nr=2-tiled
// layout GEMM kernels consume: per 2-column group, [bias[2], then per k:
// w[2] interleaved], advancing by extra_bytes after each group. Pure u32
// moves — no arithmetic — so the compare is bit-exact.
//
// Sol asserts nr==2, kr==1, sr==1. nc is kept EVEN here: the odd-remainder
// path writes only lane 0 of each 2-wide slot, leaving lane 1 uninitialized,
// which a full-buffer bit compare would flag spuriously. extra_bytes=0.
// The bias pointer advances continuously ACROSS groups (never rewinds), so
// bias holds g*nc entries.
//
// Output u32 count = g * (nc/2) * (2 + 2*kc).
//
// Two configs:
//   {g=1, nc=64, kc=128} — single group, deep K
//   {g=2, nc=32, kc=64}  — two groups, shallow K
//
// Pack-aware: candidate may define pack_weights (4-byte-word units) to feed
// the kernel a friendlier INPUT layout than GOI (e.g. pre-transposed for
// unit-stride loads — the canonical inner loop uses strided vlse32). Score
// denominator = output u32 count.
#include "harness_common.h"

static const size_t configs[][3] = {   // {g, nc, kc}
    {1, 64, 128},
    {2, 32,  64},
};
#define NUM_CONFIGS    (sizeof(configs) / sizeof(configs[0]))
// Max input words g*nc*kc: 8192 / 4096. Max bias g*nc: 64.
#define MAX_W_IN       8192
#define MAX_BIAS       64
// Max output words g*(nc/2)*(2+2*kc): 1*32*258=8256 / 2*16*130=4160.
#define MAX_OUT        8256
// Scratch for the candidate's repacked input layout.
#define CAND_CAP       (4 * MAX_W_IN)

static uint32_t ref_weights_buf [MAX_W_IN];   // canonical GOI input (reference reads this)
static uint32_t cand_weights_buf[CAND_CAP];   // scratch for candidate's repacked input
static uint32_t bias_buf        [MAX_BIAS];
static uint32_t output_buf      [MAX_OUT];
static uint32_t gold_buf        [MAX_OUT];

// Macros the sol body uses — preamble only reaches reference.c.
#ifndef XNN_LIKELY
#define XNN_LIKELY(cond) (cond)
#endif
#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(cond) (cond)
#endif

typedef void (*candidate_fn_t)(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const uint32_t* weights,
    const uint32_t* bias,
    const void* scale,
    uint32_t* packed_weights,
    size_t extra_bytes,
    const void* params);

extern void reference_kernel(
    size_t g,
    size_t nc,
    size_t kc,
    size_t nr,
    size_t kr,
    size_t sr,
    const uint32_t* weights,
    const uint32_t* bias,
    const void* scale,
    uint32_t* packed_weights,
    size_t extra_bytes,
    const void* params);

static size_t output_elems(size_t si) {
    return configs[si][0] * (configs[si][1] / 2) * (2 + 2 * configs[si][2]);
}
static size_t ref_weights_n(size_t si) {   // 4-byte words
    return configs[si][0] * configs[si][1] * configs[si][2];
}

static inline uint32_t rand_u32(void) {
    return ((uint32_t)rand() << 16) ^ (uint32_t)rand();
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t n_w = ref_weights_n(si);
    for (size_t i = 0; i < n_w; i++) ref_weights_buf[i] = rand_u32();
    size_t n_b = configs[si][0] * configs[si][1];
    for (size_t i = 0; i < n_b; i++) bias_buf[i] = rand_u32();
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes ref_weights_buf
// on the reference path and cand_weights_buf (candidate-packed) on the
// candidate path. The buffers are u32; the macro's float* casts are
// type-puns over the same 4-byte-word storage.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, uint32_t* out) {
    fn(/*g=*/configs[si][0], /*nc=*/configs[si][1], /*kc=*/configs[si][2],
       /*nr=*/2, /*kr=*/1, /*sr=*/1,
       (const uint32_t*)weights, bias_buf, /*scale=*/NULL,
       out, /*extra_bytes=*/0, /*params=*/NULL);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep (see HARNESS_MAIN_MULTI_PACK doc in
// harness_common.h). Bit-exact compare — the kernel only moves u32s.
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, u32_bit_equal)
