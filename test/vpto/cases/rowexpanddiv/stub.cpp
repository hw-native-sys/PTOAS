// ---------------------------------------------------------------------------
// PTOAS compatibility layer
//
// The upstream pto-isa headers reference some FP8/FP4 types and the
// __VEC_SCOPE__ marker that are not available on every AICore arch/toolchain
// combination (e.g. __NPU_ARCH__==2201).
//
// For our PTOAS-generated kernels we don't rely on these types today, but the
// headers still mention them in templates/static_asserts. Provide minimal
// fallbacks to keep compilation working on dav-c220.
// ---------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

// AICore printf support is gated behind `--cce-enable-print` on some
// toolchains. When enabled, include the CCE print header so `cce::printf`
// resolves in device compilation.
#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

// Some PTO-ISA types are only available in the __CCE_AICORE__ compilation
// path, but `bisheng -xcce` still performs a host-side parse pass.
// Provide minimal fallbacks only when the corresponding header wasn't
// pulled in by the selected arch implementation.
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

#include "pto/pto-inst.hpp"
using namespace pto;
extern "C" __global__ AICORE void vec_trowexpanddiv_kernel_2d(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  unsigned v4 = 1024;
  unsigned v5 = 32;
  unsigned v6 = 1;
  unsigned v7 = 0;
  int32_t v8 = 32;
  int32_t v9 = 1;
  int64_t v10 = 0;
  int64_t v11 = 4096;
  int64_t v12 = 4224;
  using T = float;
  pto::Shape<1, 1, 1, 32, 32> v13 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v14 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v15 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v13, v14);
  pto::Shape<1, 1, 1, 32, 1> v16 = pto::Shape<1, 1, 1, 32, 1>();
  pto::Stride<32, 32, 32, 1, 1> v17 = pto::Stride<32, 32, 32, 1, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<32, 32, 32, 1, 1>, pto::Layout::DN> v18 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<32, 32, 32, 1, 1>, pto::Layout::DN>(v2 + (v7 + v7 * (unsigned) v9 + v7 * (unsigned) v9), v16, v17);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v19;
  TASSIGN(v19, v10);
  Tile<TileType::Vec, float, 32, 1, BLayout::ColMajor, 32, 1, SLayout::NoneBox, 512, PadValue::Null> v20;
  TASSIGN(v20, v11);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v21;
  TASSIGN(v21, v12);
  TLOAD(v19, v15);
  TLOAD(v20, v18);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWEXPANDDIV(v21, v19, v20);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 32> v22 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v23 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v7 + v7 * (unsigned) v8 + v7 * (unsigned) v9), v22, v23);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v24, v21);
  pipe_barrier(PIPE_ALL);
  return;
}

