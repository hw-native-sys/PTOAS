#pragma once

// Only the typed vector forms exercised by the direct CCE control.
#include "a5_runtime.hpp"
namespace a5 {
template <typename T> struct vec;
template <> struct vec<float> { using type = vector_f32; };
#if (__NPU_ARCH__ == 3510)
template <> struct vec<bfloat16_t> { using type = vector_bf16; };
template <> struct vec<fp8_e4_t> { using type = vector_f8e4m3; };
#endif
template <> struct vec<uint16_t> { using type = vector_u16; };
template <typename T> using vec_t = typename vec<T>::type;
template <typename T> struct pair { T v0; T v1; };
template <typename T, typename O, typename D> __simd_callee__ inline vec_t<T>
vlds(__ubuf__ T *p, O o, D d) { vec_t<T> x; ::vlds(x, p, o, d); return x; }
template <typename U, typename S, typename P, typename M> __simd_callee__ inline vec_t<U>
vcvt(S x, vector_bool m, P p, M z) { vec_t<U> y; ::vcvt(y, x, m, p, z); return y; }
template <typename U, typename S, typename R, typename RS, typename P, typename M> __simd_callee__ inline vec_t<U>
vcvt(S x, vector_bool m, R r, RS rs, P p, M z) { vec_t<U> y; ::vcvt(y, x, m, r, rs, p, z); return y; }
template <typename V> __simd_callee__ inline pair<V> vintlv(V x, V y) { pair<V> z; ::vintlv(z.v0, z.v1, x, y); return z; }
template <typename V, typename M> __simd_callee__ inline V vmul(V x, V y, vector_bool m, M z) { V r; ::vmul(r,x,y,m,z); return r; }
template <typename V, typename M> __simd_callee__ inline V vmax(V x, V y, vector_bool m, M z) { V r; ::vmax(r,x,y,m,z); return r; }
template <typename V, typename M> __simd_callee__ inline V vand(V x, V y, vector_bool m, M z) { V r; ::vand(r,x,y,m,z); return r; }
template <typename V, typename M> __simd_callee__ inline V vdiv(V x, V y, vector_bool m, M z) { V r; ::vdiv(r,x,y,m,z); return r; }
// Keep the generated reference's correctly-rounded f32 division sequence.
// It is deliberately local to this small standalone fixture.
template <typename M> __simd_callee__ inline vector_f32
vdiv_precise(vector_f32 x, vector_f32 y, vector_bool mask, M mode) {
  constexpr uint32_t kInfNan = 0xff800000u, kSign = 0x80000000u;
  vector_f32 sign, q, original, neg_y, residual, q_prev, q_next;
  vector_u32 bits;
  ::vdup((vector_u32 &)sign, kSign, mask, mode);
  ::vdiv(q, x, y, mask, mode); original = q;
  ::vor(bits, (vector_u32 &)q, (vector_u32 &)sign, mask, mode);
  vector_bool zero, special, keep_original, choose_prev;
  ::vcmps_eq(zero, q, 0.0f, mask);
  ::vcmps_ge(special, (vector_u32 &)bits, kInfNan, mask);
  ::por(keep_original, special, zero, mask);
  ::vmuls(neg_y, y, -1.0f, mask, mode);
  residual = x; ::vmula(residual, q, neg_y, mask, mode);
  ::vadds((vector_s32 &)q_prev, (vector_s32 &)q, -1, mask, mode);
  ::vadds((vector_s32 &)q_next, (vector_s32 &)q, 1, mask, mode);
  vector_f32 r_prev = x, r_next = x;
  ::vmula(r_prev, q_prev, neg_y, mask, mode);
  ::vmula(r_next, q_next, neg_y, mask, mode);
  ::vabs(residual, residual, mask, mode);
  ::vabs(r_prev, r_prev, mask, mode);
  ::vabs(r_next, r_next, mask, mode);
  ::vcmp_lt(choose_prev, residual, r_prev, mask);
  ::vsel(residual, residual, r_prev, choose_prev);
  ::vsel(q, q, q_prev, choose_prev);
  ::vcmp_lt(choose_prev, r_next, residual, mask);
  ::vsel(q, q_next, q, choose_prev);
  vector_f32 result; ::vsel(result, original, q, keep_original);
  return result;
}
template <typename T, typename M> __simd_callee__ inline vec_t<T> vdup(T x, vector_bool m, M z) { vec_t<T> r; ::vdup(r,x,m,z); return r; }
template <typename T, typename D> __simd_callee__ inline void vsts(vec_t<T> x, __ubuf__ T *p, int o, D d, vector_bool m) { ::vsts(x,p,o,d,m); }
template <typename T> __simd_callee__ inline void mem_bar(T x) { ::mem_bar(x); }
}  // namespace a5
