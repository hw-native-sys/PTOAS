#include "kernel_operator.h"

// Complete direct CCE reference.  The four PART conversions keep the packed
// FP8 input/output in registers; the grouped input and output scales use the
// A5 BRC/E2B representation rather than a UB-expanded temporary.
__simd_vf__ inline void requant_reference(__ubuf__ float8_e4m3_t *src,
                                          __ubuf__ float *input_scale,
                                          __ubuf__ float8_e4m3_t *dst,
                                          __ubuf__ float *output_scale) {
  vector_bool m8 = pset_b8(PAT_ALL);
  vector_bool m32 = pset_b32(PAT_ALL);
  vector_f8e4m3 q; ::vlds(q, src, 0, NORM);
  vector_f32 qf; ::vcvt(qf, q, m8, PART_P0, MODE_ZEROING);
  vector_f32 scale; ::vlds(scale, input_scale, 0, E2B_B32);
  vector_f32 x; ::vmul(x, qf, scale, m32, MODE_ZEROING);
  vector_f32 ax; ::vabs(ax, x, m32, MODE_ZEROING);
  vector_f32 maxima; ::vcgmax(maxima, ax, m32, MODE_ZEROING);
  ::vsts(maxima, output_scale, 0, NORM_B32, m32);
  vector_f32 inverse; ::vlds(inverse, output_scale, 0, E2B_B32);
  vector_f32 normalized; ::vmul(normalized, x, inverse, m32, MODE_ZEROING);
  vector_f8e4m3 out;
  ::vcvt(out, normalized, m8, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
  ::vsts(out, dst, 0, NORM_B8, m8);
}

extern "C" __global__ __aicore__ void requant_reference_kernel(
    __gm__ float8_e4m3_t *src_gm, __gm__ float *input_scale_gm,
    __gm__ float8_e4m3_t *dst_gm, __gm__ float *output_scale_gm) {
  AscendC::InitSocState();
  auto src = (__ubuf__ float8_e4m3_t *)0;
  auto in_s = (__ubuf__ float *)32768;
  auto dst = (__ubuf__ float8_e4m3_t *)33792;
  auto out_s = (__ubuf__ float *)34816;
  copy_gm_to_ubuf_align_v2(src, src_gm, 0, 1, 256, 0, 0, false, 0, 256, 256);
  copy_gm_to_ubuf_align_v2(in_s, input_scale_gm, 0, 1, 32, 0, 0, false, 0, 32, 32);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  requant_reference(src, in_s, dst, out_s);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
  copy_ubuf_to_gm_align_v2(dst_gm, dst, 0, 1, 256, 0, 256, 256);
  copy_ubuf_to_gm_align_v2(output_scale_gm, out_s, 0, 1, 32, 0, 32, 32);
}
