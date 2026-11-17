// f32-igemm-minmax 1x8: indirect GEMM (GEMV-shaped) — conv expressed as GEMM
// over an indirection buffer of input-row pointers. Sol asserts mr <= 1 and
// consumes ONE row pointer per spatial group (`a += 1`), so:
//   ks = KS_COUNT * sizeof(void*)        (9 spatial positions, a 3x3 conv)
//   indirection has KS_COUNT entries     (one row per position)
// `kc` is in BYTES. A row pointer equal to `zero` denotes padding (kernel
// must NOT add a_offset to it). After each nc-tile the kernel rewinds `a`
// by ks, so the same indirection serves all tiles. cm_stride is unused
// (single output row).
//
// Packed-weight layout per nr=8 column tile:
//   [bias[8], then for each spatial position s (KS_COUNT of them):
//      for each k step (kc_elems of them): 8 weights]
// = 8*(1 + KS_COUNT*kc_elems) floats per tile; nc/8 tiles.
//
// Two configs (3x3-conv-as-IGEMM, MobileNet-class):
//   {nc=128, kc=64  elems} — wide N, shallow K
//   {nc=64,  kc=128 elems} — narrow N, deep K
//
// Pack-aware: candidate may define pack_weights to retile the canonical nr=8
// layout. Score denominator = nc output elements (mr=1). Tolerant compare —
// the ks×kc accumulation is associative.
#include "harness_common.h"

#define MR        1
#define KS_COUNT  9   // 3x3 kernel positions
static const size_t configs[][2] = {   // {nc, kc_elems}
    {128,  64},
    { 64, 128},
};
#define NUM_CONFIGS    (sizeof(configs) / sizeof(configs[0]))
#define MAX_NC         128
#define MAX_KC_ELEMS   128
#define MAX_C          (MR * MAX_NC)
// One distinct input row per spatial position.
#define NUM_ROWS       KS_COUNT
#define MAX_ROW_BUF    (NUM_ROWS * MAX_KC_ELEMS)
// Packed weights: (nc/8 tiles) × 8 × (1 + KS_COUNT*kc_elems) floats.
// Max over configs: 128*(1+9*64)=73856, 64*(1+9*128)=73792.
#define MAX_PACKED_W   73856
// Scratch for the candidate's repacked layout. 2× canonical (≈590 KB) —
// enough for any same-size retile plus padding without blowing up bss.
#define CAND_CAP       (2 * MAX_PACKED_W)

static float row_buf         [MAX_ROW_BUF];
static float zero_buf        [MAX_KC_ELEMS];   // stays zero — padding rows
static const float* indirection_buf[NUM_ROWS];
static float ref_weights_buf [MAX_PACKED_W];  // canonical nr=8 layout (reference reads this)
static float cand_weights_buf[CAND_CAP];      // scratch for candidate's repacked layout
static float output_buf      [MAX_C];
static float gold_buf        [MAX_C];

// Local mirror — the preamble in saturn_eval.py reaches reference.c only.
struct xnn_f32_minmax_params { struct { float min, max; } scalar; };
// Wide clamp keeps the path side-effect-free.
static const struct xnn_f32_minmax_params mparams = {
    .scalar = { .min = -INFINITY, .max = INFINITY }
};

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif
#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(cond) (cond)
#endif

typedef void (*candidate_fn_t)(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* params);

extern void reference_kernel(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** a,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* params);

static size_t output_elems(size_t si) { return MR * configs[si][0]; }
static size_t ref_weights_n(size_t si) {
    return configs[si][0] * (1 + KS_COUNT * configs[si][1]);
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t kc_elems = configs[si][1];
    for (size_t i = 0; i < NUM_ROWS * kc_elems; i++) row_buf[i] = rand_f1();
    // Weights in [-0.5, 0.5] so the ks*kc-deep accumulators stay in range.
    size_t n_w = ref_weights_n(si);
    for (size_t i = 0; i < n_w; i++) ref_weights_buf[i] = rand_f1() * 0.5f;
    // Indirection: distinct row per spatial position; position 0 points at
    // `zero_buf` to exercise the padding path deterministically.
    for (size_t s = 0; s < KS_COUNT; s++) {
        indirection_buf[s] = &row_buf[s * kc_elems];
    }
    indirection_buf[0] = zero_buf;
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes ref_weights_buf
// on the reference path and cand_weights_buf (candidate-packed) on the
// candidate path.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, float* out) {
    size_t nc       = configs[si][0];
    size_t kc_bytes = configs[si][1] * sizeof(float);
    fn(/*mr=*/MR, nc, kc_bytes,
       /*ks=*/KS_COUNT * MR * sizeof(void*),
       indirection_buf, weights, out,
       /*cm_stride=*/nc * sizeof(float),
       /*cn_stride=*/8 * sizeof(float),
       /*a_offset=*/0, zero_buf, &mparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep (see HARNESS_MAIN_MULTI_PACK doc in
// harness_common.h). Tolerant compare: the ks×kc accumulation order is
// associative; candidate is free to reorder.
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, f32_approx_equal_default)
