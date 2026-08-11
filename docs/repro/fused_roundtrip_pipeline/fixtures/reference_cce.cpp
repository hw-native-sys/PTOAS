#include "kernel_operator.h"

// Direct reference keeps group maxima, packed FP8 values, and dequant scales
// in registers across the round trip.
__simd_vf__ inline void fused_roundtrip_reference(__ubuf__ bfloat16_t *src,
                                                  __ubuf__ bfloat16_t *dst,
                                                  __ubuf__ float *scale_out) {
  vector_bool m16 = pset_b16(PAT_ALL);
  vector_bool m8 = pset_b8(PAT_ALL);
  vector_bf16 lo, hi; ::vld(lo, hi, src, DINTLV_B16);
  vector_u16 mask; ::vdup(mask, (uint16_t)32767, m16, MODE_ZEROING);
  vector_u16 a0, a1, a;
  ::vand(a0, *(vector_u16 *)&lo, mask, m16, MODE_ZEROING);
  ::vand(a1, *(vector_u16 *)&hi, mask, m16, MODE_ZEROING);
  ::vmax(a, a0, a1, m16, MODE_ZEROING);
  vector_u16 maxima; ::vcgmax(maxima, a, m16, MODE_ZEROING);
  vector_bf16 group = *(vector_bf16 *)&maxima;
  vector_f32 group_f; ::vcvt(group_f, group, m16, PART_EVEN, MODE_ZEROING);
  ::vsts(group_f, scale_out, 0, NORM_B32, pset_b32(PAT_ALL));
  ::vsts(group, dst, 0, PK_B32, m16);
  vector_bf16 expanded; ::vlds(expanded, dst, 0, E2B_B16);
  vector_bf16 n0, n1; ::vmul(n0, lo, expanded, m16, MODE_ZEROING);
  ::vmul(n1, hi, expanded, m16, MODE_ZEROING);
  vector_f32 f0, f1; ::vcvt(f0, n0, m16, PART_EVEN, MODE_ZEROING);
  ::vcvt(f1, n1, m16, PART_EVEN, MODE_ZEROING);
  vector_f8e4m3 q0, q1;
  ::vcvt(q0, f0, m8, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
  ::vcvt(q1, f1, m8, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);
  vector_f32 dq0, dq1; ::vcvt(dq0, q0, m8, PART_P0, MODE_ZEROING);
  ::vcvt(dq1, q1, m8, PART_P1, MODE_ZEROING);
  vector_f32 s; ::vlds(s, scale_out, 0, E2B_B32);
  ::vmul(dq0, dq0, s, pset_b32(PAT_ALL), MODE_ZEROING);
  ::vmul(dq1, dq1, s, pset_b32(PAT_ALL), MODE_ZEROING);
  vector_bf16 o0, o1;
  ::vcvt(o0, dq0, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
  ::vcvt(o1, dq1, m16, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING);
  ::vsts(o0, dst, 0, PK_B32, m16);
  ::vsts(o1, dst, 1, PK_B32, m16);
}

extern "C" __global__ __aicore__ void fused_roundtrip_reference_kernel(
    __gm__ bfloat16_t *src_gm, __gm__ bfloat16_t *dst_gm,
    __gm__ float *scale_gm) {
  AscendC::InitSocState();
  auto src = (__ubuf__ bfloat16_t *)0;
  auto dst = (__ubuf__ bfloat16_t *)32768;
  auto scale = (__ubuf__ float *)33792;
  copy_gm_to_ubuf_align_v2(src, src_gm, 0, 1, 512, 0, 0, false, 0, 512, 512);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  fused_roundtrip_reference(src, dst, scale);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  copy_ubuf_to_gm_align_v2(dst_gm, dst, 0, 1, 512, 0, 512, 512);
  copy_ubuf_to_gm_align_v2(scale_gm, scale, 0, 1, 32, 0, 32, 32);
}
