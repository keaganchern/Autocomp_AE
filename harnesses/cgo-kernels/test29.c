// f32-conv-hwc2chw 3x3s2p1c3: first-layer HWC→CHW convolution. 3 input channels,
// 16 output channels, kernel 3x3 stride 2 padding 1. Input is 224x224x3 floats
// in HWC order; output is 16 planes of 112x112 floats in CHW order. Weights
// pack as oc-tile=4: per 4-output-channel block there are 4 bias floats then
// 3 input-channels * 3*3 kernel weights * 4 lanes = 108 weight floats, totalling
// 112 floats per tile; 16/4 = 4 tiles ⇒ 448 weight floats.
//
// Single config (matches the MobileNet-v3 first-layer shape) — output is large
// enough that the per-element measurement is meaningful on its own.
//
// Score denominator = output_channels * (output_y_end - output_y_start) * output_width.
#include "harness_common.h"

#define INPUT_H        224
#define INPUT_W        224
#define INPUT_C        3
#define OUTPUT_C       16
#define OUTPUT_H       (INPUT_H / 2)   // 112
#define OUTPUT_W       (INPUT_W / 2)   // 112 (= (W+1)/2 since W is even)
#define INPUT_PAD_TOP  1
#define WEIGHTS_PER_TILE   112         // 4 bias + 3*3*3*4 weights
#define NUM_WEIGHTS    ((OUTPUT_C / 4) * WEIGHTS_PER_TILE)
// Scratch capacity for the candidate's repacked layout. 4× canonical covers
// retiling to oc_tile=16/32 (same float count) plus padded variants.
#define CAND_CAP       (4 * NUM_WEIGHTS)

static const size_t configs[][1] = { {0} };  // single config
#define NUM_CONFIGS 1

static float input_buf       [INPUT_H * INPUT_W * INPUT_C];
static float ref_weights_buf [NUM_WEIGHTS];  // canonical oc-tile=4 layout (reference reads this)
static float cand_weights_buf[CAND_CAP];     // scratch for candidate's repacked layout
static float zero_row        [INPUT_W * INPUT_C];  // one HWC row of zeros (top-padding fallback)
static float output_buf      [OUTPUT_C * OUTPUT_H * OUTPUT_W];
static float gold_buf        [OUTPUT_C * OUTPUT_H * OUTPUT_W];

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
static inline size_t doz(size_t a, size_t b) { return b < a ? a - b : 0; }

typedef void (*candidate_fn_t)(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const struct xnn_f32_minmax_params* params);

extern void reference_kernel(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const struct xnn_f32_minmax_params* params);

static size_t output_elems(size_t si) {
    (void)si;
    return (size_t)OUTPUT_C * OUTPUT_H * OUTPUT_W;
}
static size_t ref_weights_n(size_t si) { (void)si; return NUM_WEIGHTS; }

static void init_inputs(size_t si) {
    (void)si;
    srand(0x1234);
    size_t n_in = (size_t)INPUT_H * INPUT_W * INPUT_C;
    for (size_t i = 0; i < n_in; i++) input_buf[i] = rand_f1();
    // Weights in [-0.5, 0.5] so 27-tap accumulators stay well within float range.
    for (size_t i = 0; i < NUM_WEIGHTS; i++) ref_weights_buf[i] = rand_f1() * 0.5f;
    // zero_row stays zero-initialized (static).
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes ref_weights_buf
// on the reference path and cand_weights_buf (candidate-packed) on the
// candidate path.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, float* out) {
    (void)si;
    fn(/*input_height=*/INPUT_H,
       /*input_width=*/INPUT_W,
       /*output_y_start=*/0,
       /*output_y_end=*/OUTPUT_H,
       input_buf, zero_row, weights, out,
       /*input_padding_top=*/INPUT_PAD_TOP,
       /*output_channels=*/OUTPUT_C,
       /*output_height_stride=*/OUTPUT_W * sizeof(float),
       /*output_channel_stride=*/(size_t)OUTPUT_H * OUTPUT_W * sizeof(float),
       &mparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep (see HARNESS_MAIN_MULTI_PACK doc in
// harness_common.h). Tolerant compare: 27-tap (3 channels × 3×3) conv
// accumulator order is associative; candidate is free to reorder taps and
// use lane-parallel per-channel accumulation. Default tol covers K≤512 over
// U[-1,1] inputs × [-0.5,0.5] weights.
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, f32_approx_equal_default)
