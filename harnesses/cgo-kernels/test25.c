// f32-argmaxpool: per-output-pixel, per-channel argmax pool over `pooling_elements`
// input rows. Writes BOTH a float `output` (pool max) and a uint32 `index` (argmax).
//
// XNNPACK PoolingParameters sweep (utils.h, pool benchmarks) is parameterised by
// output_pixels × channels with kernel_elements=9 (3x3 pool). Two configs:
//   (output_pixels=64,  channels=32,  kernel_elements=9) — small in-cache pool
//   (output_pixels=256, channels=64,  kernel_elements=9) — exercises vl-tail loop
//
// Score denominator = output_pixels * channels (number of f32 output elements
// produced per call; the index buffer carries the same element count so the
// per-element cost is identical either way).
//
// Comparison: byte-exact across BOTH the f32 output buffer AND the uint32 index
// buffer. HARNESS_MAIN_MULTI invokes a single comparator on (OUT, GOLD, _n) —
// our custom comparator additionally walks the index buffers (file-scope) so a
// mismatch in either path fails the test.
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
// Distinct input row per (pixel, kernel_element).
#define MAX_INPUT_ROWS (MAX_OUT_PIXELS * KERNEL_ELEMENTS)
#define MAX_INPUT      (MAX_INPUT_ROWS * MAX_CHANNELS)

static float    input_buf [MAX_INPUT];
static const float* indirection_buf[MAX_OUT_PIXELS * KERNEL_ELEMENTS];

static float    output_buf [MAX_OUT_ELEMS];
static float    gold_buf   [MAX_OUT_ELEMS];
static uint32_t index_buf  [MAX_OUT_ELEMS];
static uint32_t index_gold [MAX_OUT_ELEMS];

// Local mirror — the preamble in saturn_eval.py reaches reference.c only.
struct xnn_f32_minmax_params { struct { float min, max; } scalar; };

#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif
#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(cond) (cond)
#endif

typedef void (*candidate_fn_t)(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    size_t index_increment);

extern void reference_kernel(
    size_t output_pixels,
    size_t pooling_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    size_t input_pixel_stride,
    float* output,
    uint32_t* index,
    size_t input_increment,
    size_t output_increment,
    size_t index_increment);

static size_t output_elems(size_t si) { return configs[si][0] * configs[si][1]; }

static void rebuild_indirection(size_t output_pixels, size_t channels) {
    // Each (pixel, k) gets its own distinct row pointer into input_buf, so
    // there are no aliased reads and the argmax is unambiguous.
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
    // Index buffer pairs with the output buffer; pick the matching gold/out
    // companion via the buffer address. HARNESS_MAIN_MULTI memsets the float
    // buffer between runs but not the index buffer — zero it ourselves so a
    // stale prior run cannot mask a bug.
    uint32_t* idx_out = (out == output_buf) ? index_buf : index_gold;
    memset(idx_out, 0, output_pixels * channels * sizeof(uint32_t));

    size_t input_increment  = KERNEL_ELEMENTS * sizeof(const float*);
    size_t output_increment = channels * sizeof(float);
    size_t index_increment  = channels * sizeof(uint32_t);
    fn(output_pixels, KERNEL_ELEMENTS, channels,
       indirection_buf,
       /*input_offset=*/0, /*input_pixel_stride=*/0,
       out, idx_out,
       input_increment, output_increment, index_increment);
}

// Comparator: compare f32 output bit-exact AND uint32 index bit-exact.
static int f32_plus_idx_bit_equal(const float* got, const float* ref, size_t n) {
    if (!f32_bit_equal(got, ref, n)) return 0;
    return u32_bit_equal(index_buf, index_gold, n);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, f32_plus_idx_bit_equal)
