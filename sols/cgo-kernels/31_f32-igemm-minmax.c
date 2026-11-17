// Canonical layout in `src` (src_n = nc*(1 + ks_groups*kc_elems) floats,
// nr=8 column tiles, ks_groups = ks/sizeof(void*) spatial positions): per
// tile of 8 output columns, [bias[8], then for each spatial position s:
// for each of the kc_elems k-steps: 8 weights — one per column in the
// tile]. Tiles follow each other in increasing column order. This identity
// copy keeps that order; replace the loop to emit a different layout (e.g.
// a wider nr tile for VLEN=512) and update the kernel's weight reads to
// match.
static void pack_weights(const void* src_v, size_t src_n,
                         void* dst_v, size_t dst_cap)
{
    const float* src = (const float*)src_v;
    float*       dst = (float*)dst_v;
    (void)dst_cap;  // stay within dst_cap floats when emitting a larger layout
    for (size_t i = 0; i < src_n; i++) {
        dst[i] = src[i];
    }
}

void test_rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  (void) cm_stride;

  float* c0 = c;

  do {
    size_t vl = nc > 8 ? 8 : nc;

    vfloat32m4_t vacc = __riscv_vle32_v_f32m4(w, vl);
    w += 8;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k > 0) {
        float va = *a0++;
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(w, vl);
        w += 8;
        vacc = __riscv_vfmacc_vf_f32m4(vacc, va, vb, vl);
        k -= sizeof(float);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc = __riscv_vfmin_vf_f32m4(vacc, params->scalar.max, vl);
    vacc = __riscv_vfmax_vf_f32m4(vacc, params->scalar.min, vl);

    __riscv_vse32_v_f32m4(c0, vacc, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
    nc -= vl;
  } while (nc != 0);
}