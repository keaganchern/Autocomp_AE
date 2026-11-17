// f32-dwconv-minmax: 3x3 depthwise conv along an indirection buffer of input
// row pointers. Per output column the kernel reads 3 row pointers from the
// indirection buffer, then advances the buffer by `input_stride` BYTES.
// Weights are reused across columns: per channel-tile vl-block, the layout is
// {bias[vl], k0[vl], k1[vl], k2[vl]}.
//
// Two configs mirror XNNPACK DepthwiseConvParameters (MobileNet-class):
//   (channels=32,  output_width=112) — wide spatial, narrow channels
//   (channels=128, output_width=56)  — narrow spatial, wide channels
//
// Score denominator = channels * output_width (the number of f32 output
// elements produced).
#include "harness_common.h"

#define KERNEL_ELEMENTS 3
static const size_t configs[][2] = {
    { 32, 112},
    {128,  56},
};
#define NUM_CONFIGS    (sizeof(configs) / sizeof(configs[0]))
#define MAX_CHANNELS   128
#define MAX_OUT_WIDTH  112
#define MAX_OUT_ELEMS  (MAX_CHANNELS * MAX_OUT_WIDTH)
// Each of the `output_width` columns consumes 3 input rows of `channels` floats.
#define MAX_INPUT_ROWS (MAX_OUT_WIDTH * KERNEL_ELEMENTS)
#define MAX_INPUT      (MAX_INPUT_ROWS * MAX_CHANNELS)
#define MAX_WEIGHTS    (MAX_CHANNELS * (1 + KERNEL_ELEMENTS))
// Scratch capacity for the candidate's repacked layout. 4× the canonical max
// covers any reasonable repacking (oc-tile-of-N for VLEN=512 e32m1/m2/m4 etc.).
#define CAND_CAP       (4 * MAX_WEIGHTS)

static float    input_buf       [MAX_INPUT];
static float    ref_weights_buf [MAX_WEIGHTS];   // canonical XNNPACK layout (reference reads this)
static float    cand_weights_buf[CAND_CAP];      // scratch for candidate's repacked layout
static float    zero_row        [MAX_CHANNELS];  // unused (no padding rows) but kernel keeps a stable pointer
static const float* indirection_buf[MAX_OUT_WIDTH * KERNEL_ELEMENTS];

static float output_buf[MAX_OUT_ELEMS];
static float gold_buf  [MAX_OUT_ELEMS];

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
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params* params);

extern void reference_kernel(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params* params);

static size_t output_elems(size_t si) { return configs[si][0] * configs[si][1]; }
static size_t ref_weights_n(size_t si) { return configs[si][0] * (1 + KERNEL_ELEMENTS); }

static void rebuild_indirection(size_t channels, size_t output_width) {
    // Distinct row per (column, k) — no padding, no aliasing.
    for (size_t c = 0; c < output_width; c++) {
        for (size_t k = 0; k < KERNEL_ELEMENTS; k++) {
            size_t row = c * KERNEL_ELEMENTS + k;
            indirection_buf[row] = &input_buf[row * channels];
        }
    }
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t channels     = configs[si][0];
    size_t output_width = configs[si][1];
    size_t n_in = output_width * KERNEL_ELEMENTS * channels;
    for (size_t i = 0; i < n_in; i++) input_buf[i] = rand_f1();
    // Keep weights in [-0.5, 0.5] so 3-tap accumulators stay well within float range.
    size_t n_w = channels * (1 + KERNEL_ELEMENTS);
    for (size_t i = 0; i < n_w; i++) ref_weights_buf[i] = rand_f1() * 0.5f;
    rebuild_indirection(channels, output_width);
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes either
// ref_weights_buf (reference path) or cand_weights_buf (candidate path) into
// `weights`. A candidate's `pack_weights` (if defined) populated the latter
// from the former, OUTSIDE the cycle window.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, float* out) {
    size_t channels     = configs[si][0];
    size_t output_width = configs[si][1];
    intptr_t input_stride = (intptr_t)(KERNEL_ELEMENTS * sizeof(const float*));
    size_t output_increment = channels * sizeof(float);
    fn(channels, output_width,
       indirection_buf, weights, out,
       input_stride, output_increment,
       /*input_offset=*/0, /*input_pixel_stride=*/0,
       zero_row, &mparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep. Candidate may define `pack_weights` in its
// sol body to repack ref_weights_buf into a preferred layout (see
// HARNESS_MAIN_MULTI_PACK doc in harness_common.h). Tolerant compare since
// the 9-tap accumulator is associative.
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, f32_approx_equal_default)
