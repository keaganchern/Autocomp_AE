// qd8-f32-qc4w-gemm-minmax 1x16: quantized GEMV — int8 dynamically-quantized
// activations × 4-bit packed weights → f32 output. Sol asserts mr <= 1.
// `kc` is in int8 elements (kernel rounds it up to a multiple of 2 and
// consumes activations in pairs). nr=16 column tiles (w advances 16 lanes
// per chunk, vl = min(16, nc) via e32m8 at VLEN=512).
//
// Packed-weight layout per nr=16 tile (mixed-type, walked via void*):
//   [int32 ksum[16],
//    then per k-pair (kc/2): 16 bytes — low nibble = even k, high = odd k,
//    float filter_scale[16],
//    float bias[16]]
// Tile size = 16*4 + (kc/2)*16 + 16*4 + 16*4 bytes = (48 + 2*kc) words.
// The semantic coupling between ksum/nibbles doesn't matter for the harness:
// reference and candidate read the SAME buffer, so any deterministic fill
// validates equivalence.
//
// Two configs (per-token LLM GEMV operating points):
//   {nc=128, kc=256} — narrow N, deep K
//   {nc=256, kc=128} — wide N, shallow K
//
// Pack-aware: candidate may define pack_weights (4-byte-word units) to
// retile, e.g. widen the column tile or pre-expand nibbles to int8. Score
// denominator = nc output floats (mr=1). Tolerant compare — the f32
// epilogue (scale/bias) may be reordered; integer accumulation is exact.
#include "harness_common.h"

#define MR 1
static const size_t configs[][2] = {   // {nc, kc_elems}
    {128, 256},
    {256, 128},
};
#define NUM_CONFIGS    (sizeof(configs) / sizeof(configs[0]))
#define MAX_NC         256
#define MAX_KC         256
// Packed weights in 4-byte words: (nc/16) * (48 + 2*kc).
// Max over configs: 8*(48+512)=4480, 16*(48+256)=4864.
#define MAX_PACKED_W   4864
// Scratch for the candidate's repacked layout (4× covers nibble→int8
// expansion, which doubles the weight bytes, plus wider tiles).
#define CAND_CAP       (4 * MAX_PACKED_W)

static int8_t a_buf           [MAX_KC];
static float  ref_weights_buf [MAX_PACKED_W];  // canonical mixed-type layout (word-aligned)
static float  cand_weights_buf[CAND_CAP];      // scratch for candidate's repacked layout
static float  output_buf      [MAX_NC];
static float  gold_buf        [MAX_NC];

// Local mirrors — the preamble in saturn_eval.py reaches reference.c only.
// Layouts match xnnpack/microparams.h exactly.
struct xnn_f32_qc4w_minmax_params { struct { float min; float max; int32_t kernel_zero_point; } scalar; };
struct xnn_qd8_quantization_params { int32_t zero_point; float inv_scale; };

// Wide clamp keeps the path side-effect-free.
static const struct xnn_f32_qc4w_minmax_params mparams = {
    .scalar = { .min = -INFINITY, .max = INFINITY, .kernel_zero_point = 8 }
};
static const struct xnn_qd8_quantization_params qparams[1] = {
    { .zero_point = 3, .inv_scale = 0.5f }
};

// Helpers the sol body uses — preamble only reaches reference.c.
#ifndef XNN_OOB_READS
#define XNN_OOB_READS
#endif
static inline size_t round_up_po2(size_t n, size_t q) {
    return (n + q - 1) & ~(q - 1);
}

typedef void (*candidate_fn_t)(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* params,
    const struct xnn_qd8_quantization_params* quantization_params);

extern void reference_kernel(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* a,
    size_t a_stride,
    const void* w,
    float* c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_qc4w_minmax_params* params,
    const struct xnn_qd8_quantization_params* quantization_params);

static size_t output_elems(size_t si) { return configs[si][0]; }
static size_t ref_weights_n(size_t si) {   // 4-byte words
    return (configs[si][0] / 16) * (48 + 2 * configs[si][1]);
}

static void init_inputs(size_t si) {
    srand(0x1234);
    size_t nc = configs[si][0];
    size_t kc = configs[si][1];
    for (size_t i = 0; i < kc; i++) a_buf[i] = rand_s8();
    // Walk the mixed-type packed layout tile by tile.
    uint8_t* bytes = (uint8_t*)ref_weights_buf;
    size_t off = 0;
    for (size_t t = 0; t < nc / 16; t++) {
        int32_t* ksum = (int32_t*)(bytes + off);
        for (size_t j = 0; j < 16; j++) ksum[j] = (int32_t)rand_s8();
        off += 16 * sizeof(int32_t);
        for (size_t kp = 0; kp < kc / 2; kp++) {
            for (size_t j = 0; j < 16; j++) bytes[off + j] = rand_u8();
            off += 16;
        }
        float* scale = (float*)(bytes + off);
        // Small positive scales keep outputs O(10) so rel-tol dominates.
        for (size_t j = 0; j < 16; j++)
            scale[j] = 0.005f + 0.005f * ((float)rand() / (float)RAND_MAX);
        off += 16 * sizeof(float);
        float* bias = (float*)(bytes + off);
        for (size_t j = 0; j < 16; j++) bias[j] = rand_f1();
        off += 16 * sizeof(float);
    }
}

// 4-arg contract for HARNESS_MAIN_MULTI_PACK: the macro passes ref_weights_buf
// on the reference path and cand_weights_buf (candidate-packed) on the
// candidate path.
static void run_kernel(size_t si, candidate_fn_t fn,
                       const float* weights, float* out) {
    size_t nc = configs[si][0];
    size_t kc = configs[si][1];
    fn(/*mr=*/MR, nc, /*kc=*/kc,
       a_buf, /*a_stride=*/kc,
       (const void*)weights, out,
       /*cm_stride=*/nc * sizeof(float),
       /*cn_stride=*/16 * sizeof(float),
       &mparams, qparams);
}

// SUBSTITUTE HERE
// SUBSTITUTE END

// Pack-aware multi-config sweep (see HARNESS_MAIN_MULTI_PACK doc in
// harness_common.h). Tolerant compare: integer accumulation is exact but the
// f32 epilogue (inv_scale, filter_scale, bias) may be reordered/fused.
HARNESS_MAIN_MULTI_PACK(reference_kernel,
                        output_buf, gold_buf,
                        ref_weights_buf, cand_weights_buf, CAND_CAP,
                        NUM_CONFIGS, f32_approx_equal_default)
