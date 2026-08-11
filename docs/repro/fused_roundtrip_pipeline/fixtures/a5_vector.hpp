#pragma once

// Small typed layer over the CANN vector register intrinsics used below.
#include "kernel_operator.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_fp8.h"
#include "simt_api/asc_simt.h"

using fp8_e4_t = float8_e4m3_t;

namespace a5 {
template <typename T> struct vec;
template <> struct vec<float> { using type = vector_f32; };
#if (__NPU_ARCH__ == 3510)
template <> struct vec<bfloat16_t> { using type = vector_bf16; };
template <> struct vec<fp8_e4_t> { using type = vector_f8e4m3; };
#endif
template <> struct vec<uint16_t> { using type = vector_u16; };
template <> struct vec<uint32_t> { using type = vector_u32; };
template <typename T> using vec_t = typename vec<T>::type;
template <typename T> struct pair { T v0; T v1; };

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_t<T> vlds(__ubuf__ T *src, Offset offset, Dist dist) {
  vec_t<T> dst; ::vlds(dst, src, offset, dist); return dst;
}
template <typename T, typename Dist>
__simd_callee__ inline pair<vec_t<T>> vld_x2(__ubuf__ T *src, Dist dist) {
  pair<vec_t<T>> dst; ::vld(dst.v0, dst.v1, src, dist); return dst;
}
template <typename U, typename Src, typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(Src src, vector_bool mask, Part part, Mode mode) {
  vec_t<U> dst; ::vcvt(dst, src, mask, part, mode); return dst;
}
template <typename U, typename Src, typename Round, typename Rs, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(Src src, vector_bool mask, Round round, Rs rs, Mode mode) {
  vec_t<U> dst; ::vcvt(dst, src, mask, round, rs, mode); return dst;
}
template <typename U, typename Src, typename Round, typename Rs, typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(Src src, vector_bool mask, Round round, Rs rs, Part part, Mode mode) {
  vec_t<U> dst; ::vcvt(dst, src, mask, round, rs, part, mode); return dst;
}
template <typename V> __simd_callee__ inline pair<V> vdintlv(V x, V y) {
  pair<V> dst; ::vdintlv(dst.v0, dst.v1, x, y); return dst;
}
template <typename V> __simd_callee__ inline pair<V> vintlv(V x, V y) {
  pair<V> dst; ::vintlv(dst.v0, dst.v1, x, y); return dst;
}
template <typename V, typename Mode> __simd_callee__ inline V vsub(V x, V y, vector_bool m, Mode mode) {
  V dst; ::vsub(dst, x, y, m, mode); return dst;
}
template <typename V, typename Mode> __simd_callee__ inline V vmul(V x, V y, vector_bool m, Mode mode) {
  V dst; ::vmul(dst, x, y, m, mode); return dst;
}
template <typename V, typename Mode> __simd_callee__ inline V vmax(V x, V y, vector_bool m, Mode mode) {
  V dst; ::vmax(dst, x, y, m, mode); return dst;
}
template <typename V, typename Mode> __simd_callee__ inline V vand(V x, V y, vector_bool m, Mode mode) {
  V dst; ::vand(dst, x, y, m, mode); return dst;
}
template <typename V, typename Mode> __simd_callee__ inline V vcgmax(V x, vector_bool m, Mode mode) {
  V dst; ::vcgmax(dst, x, m, mode); return dst;
}
template <typename T, typename Mode> __simd_callee__ inline vec_t<T> vdup(T x, vector_bool m, Mode mode) {
  vec_t<T> dst; ::vdup(dst, x, m, mode); return dst;
}
template <typename V, typename S, typename Mode> __simd_callee__ inline V vadds(V x, S y, vector_bool m, Mode mode) {
  V dst; ::vadds(dst, x, y, m, mode); return dst;
}
template <typename V, typename S, typename Mode> __simd_callee__ inline V vmaxs(V x, S y, vector_bool m, Mode mode) {
  V dst; ::vmaxs(dst, x, y, m, mode); return dst;
}
template <typename T, typename Dist>
__simd_callee__ inline void vsts(vec_t<T> x, __ubuf__ T *p, int offset, Dist dist, vector_bool m) {
  ::vsts(x, p, offset, dist, m);
}
}  // namespace a5
