// f32-maxpool: per-output-pixel, per-channel max over `kernel_elements` input
// rows, with [vmin, vmax] clamp. Two-config sweep matching XNNPACK
// PoolingParameters (3x3 → kernel_elements=9). Score denominator =
// output_pixels * channels (number of f32 output elements produced per call).
#include "harness_common.h"

#define KERNEL_ELEMENTS 9
static const size_t configs[][2] = {
    {64,  32},
    {256, 64},
};
#define NUM_CONFIGS    (sizeof(configs) / sizeof(configs[0]))
#define MAX_OUT_PIXELS 256
#define MAX_CHANNELS   64
#define MAX_OUT_ELEMS  (MAX_OUT_PIXELS * MAX_CHANNELS)
#define MAX_INPUT_ROWS (MAX_OUT_PIXELS * KERNEL_ELEMENTS)
#define MAX_INPUT      (MAX_INPUT_ROWS * MAX_CHANNELS)

static float    input_buf[MAX_INPUT];
static const float* indirection_buf[MAX_OUT_PIXELS * KERNEL_ELEMENTS];

static float output_buf[MAX_OUT_ELEMS];
static float gold_buf  [MAX_OUT_ELEMS];

// Local mirror — the preamble in saturn_eval.py reaches reference.c only.
struct xnn_f32_minmax_params { struct { float min, max; } scalar; };
// Wide clamp: keeps the path side-effect-free so the comparison sees the
// actual max-pool result rather than the clamp bound.
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
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f32_minmax_params* params);

extern void reference_kernel(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f32_minmax_params* params);

static size_t output_elems(size_t si) { return configs[si][0] * configs[si][1]; }

static void rebuild_indirection(size_t output_pixels, size_t channels) {
    for (size_t p = 0; p < output_pixels; p++) {
        for (size_t k = 0; k < KERNEL_ELEMENTS; k++) {
            size_t row = p * KERNEL_ELEMENTS + k;
            indirection_buf[row] = &input_buf[row * channels];
        }
    }
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t output_pixels = configs[si][0];
    size_t channels      = configs[si][1];
    size_t n = output_pixels * KERNEL_ELEMENTS * channels;
    for (size_t i = 0; i < n; i++) input_buf[i] = rand_f1();
    rebuild_indirection(output_pixels, channels);
}

static void run_kernel(size_t si, candidate_fn_t fn, float* out) {
    size_t output_pixels = configs[si][0];
    size_t channels      = configs[si][1];
    size_t input_increment  = KERNEL_ELEMENTS * sizeof(const float*);
    size_t output_increment = channels * sizeof(float);
    fn(output_pixels, KERNEL_ELEMENTS, channels,
       indirection_buf,
       /*input_offset=*/0, /*input_pixel_stride=*/0,
       out,
       input_increment, output_increment, &mparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_bit_equal)
