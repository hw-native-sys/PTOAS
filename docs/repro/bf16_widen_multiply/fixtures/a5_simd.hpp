#pragma once

#include "kernel_operator.h"
#include "simt_api/asc_fp8.h"

namespace simd_inst {
template <typename T> struct vec {};

template <> struct vec<float> {
  using type = vector_f32;
};
template <> struct vec<half> {
  using type = vector_f16;
};

#if (__NPU_ARCH__ == 3510)
template <> struct vec<bfloat16_t> {
  using type = vector_bf16;
};
template <> struct vec<fp8_e4_t> {
  using type = vector_f8e4m3;
};
// template <> struct vec<float8_e4m3_t> {
//   using type = vector_f8e4m3;
// };
template <> struct vec<float8_e5m2_t> {
  using type = vector_f8e5m2;
};
template <> struct vec<float8_e8m0_t> {
  using type = vector_f8e8m0;
};
template <> struct vec<float4_e2m1x2_t> {
  using type = vector_f4e2m1x2;
};
template <> struct vec<float4_e1m2x2_t> {
  using type = vector_f4e1m2x2;
};
#endif

template <> struct vec<uint32_t> {
  using type = vector_u32;
};

template <> struct vec<uint16_t> {
  using type = vector_u16;
};

template <> struct vec<uint8_t> {
  using type = vector_u8;
};

template <> struct vec<int32_t> {
  using type = vector_s32;
};

template <> struct vec<int16_t> {
  using type = vector_s16;
};

template <> struct vec<int8_t> {
  using type = vector_s8;
};

template <> struct vec<int64_t> {
  using type = vector_s64;
};

template <> struct vec<uint64_t> {
  using type = vector_u64;
};

template <typename SrcVec> struct vec_pair {
  SrcVec v0;
  SrcVec v1;
};

template <typename T> using vec_t = typename vec<T>::type;

template <typename SrcVec> struct widen_vec {
  using type = SrcVec;
};

template <> struct widen_vec<vector_s8> {
  using type = vector_s16;
};

template <> struct widen_vec<vector_u8> {
  using type = vector_u16;
};

template <> struct widen_vec<vector_s16> {
  using type = vector_s32;
};

template <> struct widen_vec<vector_u16> {
  using type = vector_u32;
};

template <typename SrcVec> using widen_vec_t = typename widen_vec<SrcVec>::type;

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_t<T> vlds(__ubuf__ T *src, Offset offset,
                                     Dist dist) {
  vec_t<T> dst;
  ::vlds(dst, src, offset, dist);
  return dst;
}

// Dual-dest memory load (ASC DIST_DINTLV_B16). Prefer ::vld(dst0,dst1,...)
// which wraps ::vlds on dav-3510; matches asc_loadalign_v2_impl.h.
template <typename T, typename Dist>
__simd_callee__ inline vec_pair<vec_t<T>> vld_x2(__ubuf__ T *src, Dist dist) {
  vec_pair<vec_t<T>> dst;
  ::vld(dst.v0, dst.v1, src, dist);
  return dst;
}

template <typename T, typename Offset, typename Dist>
__simd_callee__ inline vec_pair<vec_t<T>> vld_x2(__ubuf__ T *src, Offset offset,
                                                 Dist dist) {
  vec_pair<vec_t<T>> dst;
  ::vld(dst.v0, dst.v1, src, offset, dist);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx) {
  vec_t<T> dst;
  ::vgatherb(dst, base, idx);
  return dst;
}

template <typename T>
__simd_callee__ inline vec_t<T> vgatherb(__ubuf__ T *base, vector_u32 idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  ::vgatherb(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline vec_t<T> vgather2(__ubuf__ T *base, IdxVec idx,
                                         vector_bool mask) {
  vec_t<T> dst;
  ::vgather2(dst, base, idx, mask);
  return dst;
}

template <typename IdxVec>
__simd_callee__ inline widen_vec_t<vec_t<int8_t>>
vgather2(__ubuf__ int8_t *base, IdxVec idx, vector_bool mask) {
  widen_vec_t<vec_t<int8_t>> dst;
  ::vgather2(dst, base, idx, mask);
  return dst;
}

template <typename IdxVec>
__simd_callee__ inline widen_vec_t<vec_t<uint8_t>>
vgather2(__ubuf__ uint8_t *base, IdxVec idx, vector_bool mask) {
  widen_vec_t<vec_t<uint8_t>> dst;
  ::vgather2(dst, base, idx, mask);
  return dst;
}

template <typename T, typename IdxVec>
__simd_callee__ inline void vscatter(vec_t<T> data, __ubuf__ T *base,
                                     IdxVec idx, vector_bool mask) {
  ::vscatter(data, base, idx, mask);
}

__simd_callee__ inline vector_bool pand(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  ::pand(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool por(vector_bool src_0, vector_bool src_1,
                                       vector_bool mask) {
  vector_bool dst;
  ::por(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool pxor(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  ::pxor(dst, src_0, src_1, mask);
  return dst;
}

__simd_callee__ inline vector_bool pnot(vector_bool src, vector_bool mask) {
  vector_bool dst;
  ::pnot(dst, src, mask);
  return dst;
}

__simd_callee__ inline vector_bool psel(vector_bool src_0, vector_bool src_1,
                                        vector_bool mask) {
  vector_bool dst;
  ::psel(dst, src_0, src_1, mask);
  return dst;
}

// f16→f32, f8→f32, i32→f32: ::vcvt(dst, src, mask, part/round, mode)
template <typename U, typename SrcVec, typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask, Part part,
                                     Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, part, mode);
  return dst;
}

// f32→i32: ::vcvt(dst, src, mask, round, rs, mode)
template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, mode);
  return dst;
}

// f32→f16, f32→f8, f16→f8: ::vcvt(dst, src, mask, round, rs, part, mode)
template <typename U, typename SrcVec, typename RoundMode, typename Rs,
          typename Part, typename Mode>
__simd_callee__ inline vec_t<U> vcvt(SrcVec src, vector_bool srcMask,
                                     RoundMode roundingMode, Rs rsMode,
                                     Part part, Mode mode) {
  vec_t<U> dst;
  ::vcvt(dst, src, srcMask, roundingMode, rsMode, part, mode);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vdintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  ::vdintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

template <typename SrcVec>
__simd_callee__ inline vec_pair<SrcVec> vintlv(SrcVec src_0, SrcVec src_1) {
  vec_pair<SrcVec> dst;
  ::vintlv(dst.v0, dst.v1, src_0, src_1);
  return dst;
}

// -- Binary arithmetic
// ----------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vadd(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vadd(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsub(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vsub(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmul(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmul(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline void vmula(SrcVec *dst, SrcVec src_0, SrcVec src_1,
                                  vector_bool mask, Mode mode) {
  ::vmula(*dst, src_0, src_1, mask, mode);
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline void vmadd(SrcVec *dst, SrcVec src_0, SrcVec src_1,
                                  vector_bool mask, Mode mode) {
  ::vmadd(*dst, src_0, src_1, mask, mode);
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline void vaxpy(SrcVec *dst, SrcVec src, ScalarT scalar,
                                  vector_bool mask, Mode mode) {
  ::vaxpy(*dst, src, scalar, mask, mode);
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vdiv(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vdiv(dst, src_0, src_1, mask, mode);
  return dst;
}

// ============================================================================
// Precision division matching Ascend DivPrecisionImpl / torch.npu behavior.
// This intentionally keeps hardware FTZ/special-value behavior and only applies
// the 0ULP error-correction core.
// ============================================================================

// Precision f32 division exposed under the historical vdiv_precise name.
template <typename Mode>
__simd_callee__ inline vector_f32 vdiv_precise(vector_f32 src0, vector_f32 src1,
                                               vector_bool mask, Mode mode) {
  constexpr uint32_t infNanBound = 0xff800000u;
  constexpr uint32_t signBitNum = 0x80000000u;

  vector_f32 regNegZero;
  ::vdup((vector_u32 &)regNegZero, signBitNum, mask, mode);

  vector_f32 z;
  ::vdiv(z, src0, src1, mask, mode);

  vector_u32 infNan;
  ::vor(infNan, (vector_u32 &)z, (vector_u32 &)regNegZero, mask, mode);

  vector_f32 tmpDst = z;

  vector_bool zeroCmp;
  ::vcmps_eq(zeroCmp, z, 0.0f, mask);
  vector_bool infNanCmp;
  ::vcmps_ge(infNanCmp, infNan, infNanBound, mask);
  ::por(infNanCmp, infNanCmp, zeroCmp, mask);

  vector_f32 y;
  ::vmuls(y, src1, -1.0f, mask, mode);
  vector_f32 r = src0;
  ::vmula(r, z, y, mask, mode);

  vector_f32 rPre, rNext, zPre, zNext;
  ::vadds((vector_s32 &)zPre, (vector_s32 &)z, -1, mask, mode);
  ::vadds((vector_s32 &)zNext, (vector_s32 &)z, 1, mask, mode);

  rPre = src0;
  rNext = src0;
  ::vmula(rPre, zPre, y, mask, mode);
  ::vmula(rNext, zNext, y, mask, mode);

  ::vabs(r, r, mask, mode);
  ::vabs(rPre, rPre, mask, mode);
  ::vabs(rNext, rNext, mask, mode);

  vector_bool cmpMaskReg;
  ::vcmp_lt(cmpMaskReg, r, rPre, mask);
  ::vsel(r, r, rPre, cmpMaskReg);
  ::vsel(z, z, zPre, cmpMaskReg);

  ::vcmp_lt(cmpMaskReg, rNext, r, mask);
  ::vsel(z, zNext, z, cmpMaskReg);

  vector_f32 dst;
  ::vsel(dst, tmpDst, z, infNanCmp);
  return dst;
}

template <typename Mode>
__simd_callee__ inline void vdiv_precise(vector_f32 &dst, vector_f32 src0,
                                         vector_f32 src1, vector_bool mask,
                                         Mode mode) {
  (void)mode;
  // The in-place overload implements MODE_MERGING: update active lanes while
  // preserving the old destination value for inactive lanes.
  vector_f32 result = vdiv_precise(src0, src1, mask, MODE_ZEROING);
  ::vsel(dst, result, dst, mask);
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmax(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmax(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vmin(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vmin(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vand(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vand(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vor(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                  Mode mode) {
  SrcVec dst;
  ::vor(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vxor(SrcVec src_0, SrcVec src_1, vector_bool mask,
                                   Mode mode) {
  SrcVec dst;
  ::vxor(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename ShiftVec, typename Mode>
__simd_callee__ inline SrcVec vshl(SrcVec src_0, ShiftVec src_1,
                                   vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshl(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename SrcVec, typename ShiftVec, typename Mode>
__simd_callee__ inline SrcVec vshr(SrcVec src_0, ShiftVec src_1,
                                   vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshr(dst, src_0, src_1, mask, mode);
  return dst;
}

// -- Unary
// ----------------------------------------------------------------------

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vln(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vln(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqrt(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vsqrt(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vabs(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vabs(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vneg(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vneg(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vrelu(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vrelu(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vnot(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vnot(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vexp(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vexp(dst, src, mask, mode);
  return dst;
}

// -- Cross-lane reductions
// ------------------------------------------------------
template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcpadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcpadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline widen_vec_t<SrcVec> vcadd(SrcVec src, vector_bool mask,
                                                 Mode mode) {
  widen_vec_t<SrcVec> dst;
  ::vcadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcmax(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcmin(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcmin(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgadd(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcgadd(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgmax(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcgmax(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vcgmin(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vcgmin(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vsqz(SrcVec src, vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vsqz(dst, src, mask, mode);
  return dst;
}

// -- Index ramp / compare
// ------------------------------------------------------------

// index ramp: T = int32_t / float / ...; returns dst[lane] = index (+/-) lane
template <typename T, typename Order>
__simd_callee__ inline vec_t<T> vci(T index, Order order) {
  vec_t<T> dst;
  ::vci(dst, index, order);
  return dst;
}

#define __SIMD_INST_VCMP(OP)                                                   \
  template <typename SrcVec>                                                   \
  __simd_callee__ inline vector_bool vcmp_##OP(SrcVec src_0, SrcVec src_1,     \
                                               vector_bool mask) {             \
    vector_bool dst;                                                           \
    ::vcmp_##OP(dst, src_0, src_1, mask);                                      \
    return dst;                                                                \
  }                                                                            \
  template <typename SrcVec, typename ScalarT>                                 \
  __simd_callee__ inline vector_bool vcmps_##OP(SrcVec src, ScalarT scalar,    \
                                                vector_bool mask) {            \
    vector_bool dst;                                                           \
    ::vcmps_##OP(dst, src, scalar, mask);                                      \
    return dst;                                                                \
  }
__SIMD_INST_VCMP(eq)
__SIMD_INST_VCMP(ne)
__SIMD_INST_VCMP(gt)
__SIMD_INST_VCMP(ge)
__SIMD_INST_VCMP(lt)
__SIMD_INST_VCMP(le)
#undef __SIMD_INST_VCMP

// -- Broadcast
// ------------------------------------------------------------------

// scalar broadcast: T = float / half / int32_t / ...
template <typename T, typename Mode>
__simd_callee__ inline vec_t<T> vdup(T src, vector_bool mask, Mode mode) {
  vec_t<T> dst;
  ::vdup(dst, src, mask, mode);
  return dst;
}

template <typename SrcVec, typename Pos, typename Mode>
__simd_callee__ inline SrcVec vdupv(SrcVec src, vector_bool mask, Pos pos,
                                    Mode mode) {
  SrcVec dst;
  ::vdup(dst, src, mask, pos, mode);
  return dst;
}

// -- Select
// ---------------------------------------------------------------------

template <typename SrcVec>
__simd_callee__ inline SrcVec vsel(SrcVec src_0, SrcVec src_1,
                                   vector_bool mask) {
  SrcVec dst;
  ::vsel(dst, src_0, src_1, mask);
  return dst;
}

template <typename SrcVec, typename IdxVec>
__simd_callee__ inline SrcVec vselr(SrcVec src, IdxVec idx) {
  SrcVec dst;
  ::vselr(dst, src, idx);
  return dst;
}

// -- Scalar-vector ops
// ----------------------------------------------------------

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vadds(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vadds(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmaxs(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmaxs(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmins(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmins(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vmuls(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vmuls(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vshls(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshls(dst, src, scalar, mask, mode);
  return dst;
}

template <typename SrcVec, typename ScalarT, typename Mode>
__simd_callee__ inline SrcVec vshrs(SrcVec src, ScalarT scalar,
                                    vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vshrs(dst, src, scalar, mask, mode);
  return dst;
}

// -- Exponential difference
// -----------------------------------------------------

template <typename SrcVec, typename Part>
__simd_callee__ inline SrcVec vexpdif(SrcVec src_0, SrcVec src_1,
                                      vector_bool mask, Part part) {
  SrcVec dst;
  ::vexpdif(dst, src_0, src_1, mask, part);
  return dst;
}

template <typename SrcVec, typename Mode>
__simd_callee__ inline SrcVec vabsdif(SrcVec src_0, SrcVec src_1,
                                      vector_bool mask, Mode mode) {
  SrcVec dst;
  ::vabsdif(dst, src_0, src_1, mask, mode);
  return dst;
}

template <typename U, typename SrcVec, typename Part>
__simd_callee__ inline vec_t<U> vpack(SrcVec src, Part part) {
  vec_t<U> dst;
  ::vpack(dst, src, part);
  return dst;
}

template <typename T>
__simd_callee__ inline void vsstb(vec_t<T> src, __ubuf__ T *base,
                                  int32_t stride, vector_bool mask) {
  ::vsstb(src, base, stride, mask);
}

template <typename T, typename Post>
__simd_callee__ inline __ubuf__ T *vsstb(vec_t<T> src, __ubuf__ T *base,
                                         int32_t stride, vector_bool mask,
                                         Post post) {
  ::vsstb(src, base, stride, mask, post);
  return base;
}

template <typename T, typename Dist>
__simd_callee__ inline void vsts(vec_t<T> data, __ubuf__ T *base,
                                 int32_t offset, Dist dist, vector_bool mask) {
  ::vsts(data, base, offset, dist, mask);
}

template <typename T> __simd_callee__ inline void mem_bar(T mem_type) {
  ::mem_bar(mem_type);
}

} // namespace simd_inst
