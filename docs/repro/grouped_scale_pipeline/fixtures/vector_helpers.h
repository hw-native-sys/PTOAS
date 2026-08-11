#pragma once
#include "kernel_operator.h"

namespace direct_vec {
struct bf16_pair { vector_bf16 lo; vector_bf16 hi; };

__simd_callee__ inline bf16_pair load_deinterleaved(__ubuf__ bfloat16_t *p) {
  bf16_pair r;
  ::vld(r.lo, r.hi, p, DINTLV_B16);
  return r;
}
template <class V> __simd_callee__ inline V bit_and(V a, V b, vector_bool m) {
  V r; ::vand(r, a, b, m, MODE_ZEROING); return r;
}
template <class V> __simd_callee__ inline V max(V a, V b, vector_bool m) {
  V r; ::vmax(r, a, b, m, MODE_ZEROING); return r;
}
template <class V> __simd_callee__ inline V grouped_max(V a, vector_bool m) {
  V r; ::vcgmax(r, a, m, MODE_ZEROING); return r;
}
template <class V> __simd_callee__ inline V mul(V a, V b, vector_bool m) {
  V r; ::vmul(r, a, b, m, MODE_ZEROING); return r;
}
template <class T, class V, class P> __simd_callee__ inline T convert(V a, vector_bool m,
                                                                      P p) {
  T r; ::vcvt(r, a, m, p, MODE_ZEROING); return r;
}
template <class T, class V, class P> __simd_callee__ inline T convert_fp8(V a, vector_bool m,
                                                                         P p) {
  T r; ::vcvt(r, a, m, ROUND_R, RS_ENABLE, p, MODE_ZEROING); return r;
}
template <class V> __simd_callee__ inline void store(V a, __ubuf__ bfloat16_t *p,
                                                     int off, vector_bool m) {
  ::vsts(a, p, off, PK_B32, m);
}
template <class V> __simd_callee__ inline void store_fp8(V a, __ubuf__ float8_e4m3_t *p,
                                                         int off, vector_bool m) {
  ::vsts(a, p, off, NORM_B8, m);
}
} // namespace direct_vec
