// qu8-rdsum: output[c] += sum_{r=0..rows-1} input[r*input_stride + c] for each
// channel c. Row-direction reduction over u8 → u32-per-channel output.
//
// Sizes from XNNPACK's ReduceDiscontiguousParameters (utils.h:109-115) — the
// canonical rdsum bench list is {(8,1024),(16,1024),(1024,1024),(32768,5)}.
// We pick two trimmed configs that exercise both wide-channels (vl-tail loop)
// and tall-rows (the 252-row inner accumulator boundary) within Spike memory.
#include "harness_common.h"

static const size_t configs[][2] = {
    {16,  1024},  // wide channels — vl-tail + small rows prologue
    {256, 128},   // tall rows — exercises the 252-row inner accumulator
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_ROWS     256
#define MAX_CHANNELS 1024
#define MAX_INPUT    (MAX_ROWS * MAX_CHANNELS)

static uint8_t  input_buf [MAX_INPUT];
static uint8_t  zero_buf  [MAX_CHANNELS];  // implicitly zero-initialized
static uint32_t output_buf[MAX_CHANNELS];
static uint32_t gold_buf  [MAX_CHANNELS];

// Layout matches XNNPACK's struct xnn_qs8_rsum_params (microparams.h:419-421).
// Reused unchanged by qs8/qu8 rsum *and* rdsum kernels (an opaque stub).
struct xnn_qs8_rsum_params { char _; };
static const struct xnn_qs8_rsum_params rparams = { 0 };

// The sol uses XNNPACK's branch-hint macro `XNN_UNPREDICTABLE`; xnnpack/common.h
// is not on main.c's include path, so we provide a no-op shim. Semantically
// neutral — only affects the compiler's branch-prediction hint.
#ifndef XNN_UNPREDICTABLE
#define XNN_UNPREDICTABLE(cond) (cond)
#endif

typedef void (*candidate_fn_t)(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* params);

extern void reference_kernel(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params* params);

// Comparison element count: channels (one u32 reduction per output channel).
static size_t output_elems(size_t si) { return configs[si][1]; }

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t n = configs[si][0] * configs[si][1];
    for (size_t i = 0; i < n; i++) input_buf[i] = rand_u8();
}

static void run_kernel(size_t si, candidate_fn_t fn, uint32_t* out) {
    // rdsum accumulates (output[c] += sum_r input[r,c]); the macro's 3-iter
    // warmup loop would otherwise leave GOLD = 3*sum vs the candidate's
    // per-iter-zeroed 1*sum. Zero here so both paths see a fresh accumulator.
    size_t channels = configs[si][1];
    for (size_t i = 0; i < channels; i++) out[i] = 0;
    size_t input_stride = channels * sizeof(uint8_t);
    fn(configs[si][0], channels, input_buf, input_stride, zero_buf,
       out, &rparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, u32_bit_equal)
