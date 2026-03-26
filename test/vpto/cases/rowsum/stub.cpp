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
extern "C" __global__ AICORE void rowsum_kernel_2d(__gm__ float* v1, __gm__ float* v2) {
  unsigned v3 = 1024;
  unsigned v4 = 32;
  unsigned v5 = 1;
  unsigned v6 = 0;
  int32_t v7 = 32;
  int32_t v8 = 1;
  int64_t v9 = 0;
  int64_t v10 = 4096;
  using T = float;
  pto::Shape<1, 1, 1, 32, 32> v11 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v12 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v13 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v1 + (v6 + v6 * (unsigned) v7 + v6 * (unsigned) v8), v11, v12);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v14;
  TASSIGN(v14, v9);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v15;
  TASSIGN(v15, v10);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 1, SLayout::NoneBox, 512, PadValue::Null> v16;
  TASSIGN(v16, v10);
  TLOAD(v14, v13);
  set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
  TROWSUM(v16, v14, v15);
  set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  pto::Shape<1, 1, 1, 32, 1> v17 = pto::Shape<1, 1, 1, 32, 1>();
  pto::Stride<32, 32, 32, 1, 1> v18 = pto::Stride<32, 32, 32, 1, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<32, 32, 32, 1, 1>, pto::Layout::ND> v19 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 1>, pto::Stride<32, 32, 32, 1, 1>, pto::Layout::ND>(v2 + (v6 + v6 * (unsigned) v8 + v6 * (unsigned) v8), v17, v18);
  wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
  TSTORE(v19, v16);
  pipe_barrier(PIPE_ALL);
  return;
}

