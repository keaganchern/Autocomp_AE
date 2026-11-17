// f32-vsqrt, raw-NEON version for the Neon2RVV baseline lane.
//
// Hand-inlined from XNNPACK's f32-vsqrt-aarch64-neon-sqrt.c (u4 variant):
// that source is written against the xnn_simd abstraction layer
// (src/xnnpack/simd/f32-neon.h), which neon2rvv cannot translate. Every
// abstraction call is replaced by its NEON definition from that header:
//   xnn_simd_f32_t       -> float32x4_t
//   xnn_loadu_f32        -> vld1q_f32
//   xnn_sqrt_f32         -> vsqrtq_f32
//   xnn_storeu_f32       -> vst1q_f32
//   xnn_simd_size_f32    -> 4
//   xnn_load_tail_f32    -> vld1q_f32 (XNN_OOB_READS semantics)
//   xnn_store_tail_f32   -> vget_low/high + vst1_f32/vst1_lane_f32
// Loop structure and operation order are unchanged.

void xnn_f32_vsqrt_ukernel__neon_sqrt_u4_inlined(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* unused_params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;
    float32x4_t vy = vsqrtq_f32(vx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    size_t num_elements = batch >> XNN_LOG2_SIZEOF_FLOAT;
    float32x4_t vx = vld1q_f32(input);   /* XNN_OOB_READS tail load */
    float32x4_t vy = vsqrtq_f32(vx);
    float32x2_t vy_low = vget_low_f32(vy);
    if (num_elements & 2) {
      vst1_f32(output, vy_low); output += 2;
      vy_low = vget_high_f32(vy);
    }
    if (num_elements & 1) {
      vst1_lane_f32(output, vy_low, 0);
    }
  }
}
