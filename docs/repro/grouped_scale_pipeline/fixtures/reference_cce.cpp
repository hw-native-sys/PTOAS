// Direct A5 peer for one 256-lane group-of-eight body.  This is deliberately
// generic AscendC and has no dependency on a higher-level kernel framework.
#include "vector_helpers.h"

__simd_vf__ inline void grouped_scale_reference(__ubuf__ bfloat16_t *src,
                                                __ubuf__ bfloat16_t *scale_out,
                                                __ubuf__ float8_e4m3_t *dst) {
  using namespace direct_vec;
  vector_bool m16 = pset_b16(PAT_ALL);
  vector_bool m8 = pset_b8(PAT_ALL);
  vector_u16 abs_bits; ::vdup(abs_bits, (uint16_t)32767, m16, MODE_ZEROING);
  bf16_pair pair = load_deinterleaved(src);
  vector_u16 a0 = bit_and(*(vector_u16 *)&pair.lo, abs_bits, m16);
  vector_u16 a1 = bit_and(*(vector_u16 *)&pair.hi, abs_bits, m16);
  vector_u16 maxima = grouped_max(max(a0, a1, m16), m16);
  vector_bf16 scale = *(vector_bf16 *)&maxima;
  store(scale, scale_out, 0, m16);
  // E2B expands the eight packed group values over the 256 logical lanes.
  vector_bf16 expanded; ::vlds(expanded, scale_out, 0, E2B_B16);
  vector_bf16 y0 = mul(pair.lo, expanded, m16);
  vector_bf16 y1 = mul(pair.hi, expanded, m16);
  vector_f32 f0e = convert<vector_f32>(y0, m16, PART_EVEN);
  vector_f32 f0o = convert<vector_f32>(y0, m16, PART_ODD);
  vector_f32 f1e = convert<vector_f32>(y1, m16, PART_EVEN);
  vector_f32 f1o = convert<vector_f32>(y1, m16, PART_ODD);
  vector_f8e4m3 q0 = convert_fp8<vector_f8e4m3>(f0e, m8, PART_P0);
  vector_f8e4m3 q1 = convert_fp8<vector_f8e4m3>(f1e, m8, PART_P1);
  vector_f8e4m3 q2 = convert_fp8<vector_f8e4m3>(f0o, m8, PART_P2);
  vector_f8e4m3 q3 = convert_fp8<vector_f8e4m3>(f1o, m8, PART_P3);
  vector_u8 t0, t1, packed;
  ::vor(t0, *(vector_u8 *)&q0, *(vector_u8 *)&q2, m8, MODE_ZEROING);
  ::vor(t1, *(vector_u8 *)&q1, *(vector_u8 *)&q3, m8, MODE_ZEROING);
  ::vor(packed, t0, t1, m8, MODE_ZEROING);
  store_fp8(*(vector_f8e4m3 *)&packed, dst, 0, m8);
}

extern "C" __global__ __aicore__ void grouped_scale_reference_kernel(
    __gm__ bfloat16_t *src_gm, __gm__ float8_e4m3_t *dst_gm,
    __gm__ bfloat16_t *scale_gm) {
  AscendC::InitSocState();
  __ubuf__ bfloat16_t *src = (__ubuf__ bfloat16_t *)0;
  __ubuf__ float8_e4m3_t *dst = (__ubuf__ float8_e4m3_t *)(32768);
  __ubuf__ bfloat16_t *scale = (__ubuf__ bfloat16_t *)(33024);
  copy_gm_to_ubuf_align_v2(src, src_gm, 0, 1, 512, 0, 0, false, 0, 512, 512);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  grouped_scale_reference(src, scale, dst);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  copy_ubuf_to_gm_align_v2(dst_gm, dst, 0, 1, 256, 0, 256, 256);
  copy_ubuf_to_gm_align_v2(scale_gm, scale, 0, 1, 16, 0, 16, 16);
}

extern "C" void launch_grouped_scale_reference(void *x, void *y, void *s,
                                                void *stream) {
  grouped_scale_reference_kernel<<<1, nullptr, stream>>>(
      (__gm__ bfloat16_t *)x, (__gm__ float8_e4m3_t *)y,
      (__gm__ bfloat16_t *)s);
}
