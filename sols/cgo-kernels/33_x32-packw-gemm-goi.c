// Canonical layout in `src` (src_n = g*nc*kc 4-byte words): GOI order —
// g groups back to back, each holding nc output-channel rows of kc
// consecutive u32 input-channel values (row-major nc×kc per group). This is
// the kernel's INPUT array; the kernel's job is to emit the nr=2-tiled GEMM
// layout into packed_weights. This identity copy keeps GOI order; replace
// the loop to feed the kernel a friendlier input layout (e.g. pre-transposed
// so the inner loop uses unit-stride loads instead of strided ones) and
// update the kernel's reads to match.
static void pack_weights(const void* src_v, size_t src_n,
                         void* dst_v, size_t dst_cap)
{
    const uint32_t* src = (const uint32_t*)src_v;
    uint32_t*       dst = (uint32_t*)dst_v;
    (void)dst_cap;  // stay within dst_cap words when emitting a larger layout
    for (size_t i = 0; i < src_n; i++) {
        dst[i] = src[i];
    }
}

void test_rvv(
  size_t g,
  size_t nc,
  size_t kc,
  size_t nr,
  size_t kr,
  size_t sr,
  const uint32_t* weights,
  const uint32_t* bias,
  const void* scale,
  uint32_t* packed_weights,
  size_t extra_bytes,
  const void* params)
{
  assert(g != 0);
  assert(nc != 0);
  assert(kc != 0);
  assert(nr == 2);
  assert(kr == 1);
  assert(sr == 1);
  assert(weights != NULL);
  assert(packed_weights != NULL);

  // NR=2 stores: clamp vl>=1 so vlen=0/1 cannot zero out the store.
  size_t vl2 = __riscv_vsetvl_e32m1(2);
  if (vl2 < 2) vl2 = 2;
  const size_t kc_bstride = kc * sizeof(uint32_t);

  do {
    const uint32_t* w0 = weights;
    size_t n = nc;

    // NC main loop multiple of 2
    for (; n >= 2; n -= 2) {
      if XNN_LIKELY(bias != NULL) {
        vuint32m1_t vb0 = __riscv_vle32_v_u32m1(bias, 2); bias += 2;
        __riscv_vse32_v_u32m1(packed_weights, vb0, 2); packed_weights += 2;
      } else {
        vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 2);
        __riscv_vse32_v_u32m1(packed_weights, vzero, 2); packed_weights += 2;
      }

      const uint32_t* w1 = w0 + kc;

      size_t k = kc;
      for (; k != 0; --k) {
        vuint32m1_t vw = __riscv_vlse32_v_u32m1(w0, kc_bstride, 2);
        __riscv_vse32_v_u32m1(packed_weights, vw, 2);
        packed_weights += 2;
        w0 += 1;
        w1 += 1;
      }
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
      w0 = w1;
    }

    if XNN_UNLIKELY(n != 0) {
      // NC remainder of 1
      if XNN_LIKELY(bias != NULL) {
        *packed_weights = *bias++;
      } else {
        vuint32m1_t vzero = __riscv_vmv_v_x_u32m1(0, 2);
        __riscv_vse32_v_u32m1(packed_weights + 0, vzero, 2);
      }
      packed_weights += 2;
      size_t k = kc;
      do {
        *packed_weights = *w0++;
        packed_weights += 2;
      } while (--k);
      packed_weights = (uint32_t*) ((uintptr_t) packed_weights + extra_bytes);
    }
    weights += nc * kc;
  } while (--g != 0);
}
