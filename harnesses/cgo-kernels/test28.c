// f32-dwconv2d-chw 3x3s1p1: single-plane depthwise 3x3 conv with stride 1 and
// padding=1 on top/sides. `input_width` is in BYTES (kernel asserts
// `input_width % sizeof(float) == 0` and reads `w = input_width >> log2(sizeof(float))`).
//
// Two configs mirror typical MobileNet 3x3 dwconv2d-chw shapes:
//   {input_height=112, input_width_bytes=112*sizeof(float)}
//   {input_height=56,  input_width_bytes=56 *sizeof(float)}
//
// Single channel-plane per call. Score denominator = input_height * (input_width_bytes/4)
// (output elements written — output has the same H×W as the input).
#include "harness_common.h"

static const size_t configs[][2] = {
    {112, 112 * sizeof(float)},
    { 56,  56 * sizeof(float)},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_HEIGHT  112
#define MAX_WIDTH   112
#define MAX_PLANE   (MAX_HEIGHT * MAX_WIDTH)
#define NUM_WEIGHTS 10  // 1 bias + 9 kernel taps
// Scratch capacity for the candidate's repacked layout (4× canonical covers
// padded/duplicated layouts, e.g. broadcast-friendly per-tap splats).
#define CAND_CAP    (4 * NUM_WEIGHTS)

static float input_buf       [MAX_PLANE];
static float ref_weights_buf [NUM_WEIGHTS];  // canonical layout (reference reads this)
static float cand_weights_buf[CAND_CAP];     // scratch for candidate's repacked layout
static float zero_row        [MAX_WIDTH];    // implicitly zero — kernel reads when output_height < 2..4
static float output_buf      [MAX_PLANE];
static float gold_buf        [MAX_PLANE];

// Local mirror — the preamble in saturn_eval.py reaches reference.c only.
struct xnn_f32_minmax_params { struct { float min, max; } scalar; };
// Wide clamp keeps the path side-effect-free.
static const struct xnn_f32_minmax_params mparams = {
    .scalar = { .min = -INFINITY, .max = INFINITY }
};

// Macros the sol body uses — preamble only reaches reference.c, so the candidate
// copy injected into main.c needs these visible. All are semantic no-ops here.
#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif
#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(cond) (cond)
#endif
#ifndef XNN_LOG2_SIZEOF_FLOAT
#define XNN_LOG2_SIZEOF_FLOAT 2
#endif
static inline size_t doz(size_t a, size_t b) { return b < a ? a - b : 0; }

typedef void (*candidate_fn_t)(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* params);

extern void reference_kernel(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const struct xnn_f32_minmax_params* params);

static size_t output_elems(size_t si) {
    return configs[si][0] * (configs[si][1] / sizeof(float));
}
static size_t ref_weights_n(size_t si) { (void)si; return NUM_WEIGHTS; }

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t n = output_elems(si);
    for (size_t i = 0; i < n; i++) input_buf[i] = rand_f1();
    // Weights in [-0.5, 0.5] so 9-tap accumulators stay well within float range.
    for (size_t i = 0; i < NUM_WEIGHTS; i++) ref_weights_buf[i] = rand_f1() * 0.5f;
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes ref_weights_buf
// on the reference path and cand_weights_buf (candidate-packed) on the
// candidate path.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, float* out) {
    fn(configs[si][0], configs[si][1],
       input_buf, weights, zero_row, out,
       /*padding_top=*/1, &mparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep (see HARNESS_MAIN_MULTI_PACK doc in
// harness_common.h). Tolerant compare: 9-tap dwconv accumulator order is
// associative; candidate is free to reorder.
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, f32_approx_equal_default)
