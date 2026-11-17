// x32-transposec: out[c][r] = in[r][c] for an H×W block of u32 words. Strides
// are in BYTES (input_stride between input rows, output_stride between output
// rows). The kernel walks W in vl-column strips, gathering a strip down the
// rows and scatter-storing it as a row of the output (vsse32 at output_stride).
//
// Reference == the injected sol, so gold is the sol's own transpose — the
// harness only needs valid inputs/strides. Square configs keep in/out
// contiguous (input_stride = W words, output_stride = H words). Bit-exact:
// the kernel only moves u32 words.
//
// Configs exercise VLMAX (e32m8 → 32 lanes at VLEN=128) plus a non-multiple
// tail (48) and a two-strip size (64).
#include "harness_common.h"

static const size_t configs[][1] = {   // {dim} (square: block_width = block_height)
    {32},
    {48},
    {64},
};
#define NUM_CONFIGS (sizeof(configs) / sizeof(configs[0]))
#define MAX_DIM     64
// Small tail pad absorbs the sol's XNN_OOB_READS (may load a full vl past the
// last strip). MAX_DIM*MAX_DIM elements + one VLMAX strip of headroom.
#define BUF_ELEMS   (MAX_DIM * MAX_DIM + 64)

static uint32_t input_buf [BUF_ELEMS];
static uint32_t output_buf[BUF_ELEMS];
static uint32_t gold_buf  [BUF_ELEMS];

typedef void (*candidate_fn_t)(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

extern void reference_kernel(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height);

static size_t output_elems(size_t si) { return configs[si][0] * configs[si][0]; }

static inline uint32_t rand_u32(void) {
    return ((uint32_t)rand() << 16) ^ (uint32_t)rand();
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t n = configs[si][0] * configs[si][0];
    for (size_t i = 0; i < n; i++) input_buf[i] = rand_u32();
}

static void run_kernel(size_t si, candidate_fn_t fn, uint32_t* out) {
    size_t dim = configs[si][0];
    fn(input_buf, out,
       /*input_stride =*/ dim * sizeof(uint32_t),
       /*output_stride=*/ dim * sizeof(uint32_t),
       /*block_width  =*/ dim,
       /*block_height =*/ dim);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

HARNESS_MAIN_MULTI(reference_kernel, output_buf, gold_buf, NUM_CONFIGS, u32_bit_equal)
