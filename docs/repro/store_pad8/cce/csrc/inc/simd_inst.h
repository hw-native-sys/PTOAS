#pragma once

// Minimal CCE SIMD helpers for store_pad8_vf_sim_kernel.cpp.
// Only float vlds / vcadd / vsts — not a full TileLang simd_inst dump.

#include "kernel_operator.h"

namespace simd_inst {

template <typename T> struct vec {};

template <> struct vec<float> {
  using type = vector_f32;
};

template <typename T> using vec_t = typename vec<T>::type;

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_t<T> vlds(__ubuf__ T *src, Offset offset,
                                     Dist dist) {
  vec_t<T> dst;
  ::vlds(dst, src, offset, dist);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcadd(dst, src, mask, mode);
  return dst;
}

template <typename T, typename Dist>
__simd_callee__ inline void vsts(vec_t<T> data, __ubuf__ T *base,
                                 int32_t offset, Dist dist, vector_bool mask) {
  ::vsts(data, base, offset, dist, mask);
}

} // namespace simd_inst
