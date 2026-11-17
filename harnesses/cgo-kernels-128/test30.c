// f32-gemm-minmax 4x8: C[mr×nc] = A[mr×kc] · W(packed) + bias, with min/max
// clamp. `kc` is in BYTES (kernel asserts kc % sizeof(float) == 0 and steps
// k -= sizeof(float) per element). Packed-weight layout per nr=8 column tile:
// [bias[8], then for each k: w[8]] — the kernel advances w by 8 floats per
// step regardless of vl, so nc must be a multiple of 8.
//
// Two configs mirror 1x1-conv-as-GEMM operating points (MobileNet-class):
//   {nc=64,  kc=128 elems} — narrow N, deep K
//   {nc=256, kc=64  elems} — wide N, shallow K
//
// Pack-aware: candidate may define pack_weights to retile the canonical
// nr=8 layout (e.g. to nr=16/32 for VLEN=512). Score denominator = mr*nc
// output elements. Tolerant compare — K-loop accumulation is associative.
#include "harness_common.h"

#define MR 4
static const size_t configs[][2] = {   // {nc, kc_elems}
    { 64, 128},
    {256,  64},
};
#define NUM_CONFIGS    (sizeof(configs) / sizeof(configs[0]))
#define MAX_NC         256
#define MAX_KC_ELEMS   128
#define MAX_A          (MR * MAX_KC_ELEMS)
#define MAX_C          (MR * MAX_NC)
// Packed weights: (nc/8 tiles) × (8 bias + kc*8 weights) = nc*(kc+1) floats.
// Max over configs: 64*129=8256, 256*65=16640.
#define MAX_PACKED_W   16640
// Scratch for the candidate's repacked layout (4× canonical covers retiling
// to wider nr plus padded variants).
#define CAND_CAP       (4 * MAX_PACKED_W)

static float a_buf           [MAX_A];
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

// Branch-hint macros the -128 sol body uses (main.c has no xnnpack headers;
// reference.c gets them via the saturn_eval preamble).
#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(cond) (cond)
#endif
#ifndef XNN_UNLIKELY
#define XNN_UNLIKELY(cond) (cond)
#endif

typedef void (*candidate_fn_t)(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* params);

extern void reference_kernel(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* a,
    size_t a_stride,
    const float* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params* params);

static size_t output_elems(size_t si) { return MR * configs[si][0]; }
static size_t ref_weights_n(size_t si) {
    return configs[si][0] * (configs[si][1] + 1);   // nc*(kc_elems+1)
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t kc_elems = configs[si][1];
    for (size_t i = 0; i < MR * kc_elems; i++) a_buf[i] = rand_f1();
    // Weights in [-0.5, 0.5] so K-deep accumulators stay well within float range.
    size_t n_w = ref_weights_n(si);
    for (size_t i = 0; i < n_w; i++) ref_weights_buf[i] = rand_f1() * 0.5f;
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes ref_weights_buf
// on the reference path and cand_weights_buf (candidate-packed) on the
// candidate path.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, float* out) {
    size_t nc       = configs[si][0];
    size_t kc_bytes = configs[si][1] * sizeof(float);
    fn(/*mr=*/MR, nc, kc_bytes,
       a_buf, /*a_stride=*/kc_bytes,
       weights, out,
       /*cm_stride=*/nc * sizeof(float),
       /*cn_stride=*/8 * sizeof(float),
       &mparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep (see HARNESS_MAIN_MULTI_PACK doc in
// harness_common.h). Tolerant compare: K-loop accumulation order is
// associative; candidate is free to reorder (multiple accumulators,
// tree reduce, wider tiles).
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, f32_approx_equal_default)
