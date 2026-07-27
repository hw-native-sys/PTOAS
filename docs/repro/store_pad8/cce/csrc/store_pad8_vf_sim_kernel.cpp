/*===========================================================================
 * store_pad8 microbench — CCE ONEPT path
 *
 * Per accumulator: vcadd then a single scalar vsts ... ONEPT_B32 store.
 * No padding, no broadcast — the CCE baseline the VMI pad-8 port
 * (../vmi/store_pad8_vmi.py) is compared against.
 *===========================================================================*/

#include "inc/common.h"

namespace {

template <int32_t N_ACC>
__simd_vf__ inline void simd_vf_store_pad8_cce(__ubuf__ uint8_t *ub) {
  auto *acc = (__ubuf__ float *)(ub + kAccUb);
  auto *reduced = (__ubuf__ float *)(ub + kReducedUb);
#pragma unroll
  for (int32_t i = 0; i < N_ACC; ++i) {
    vector_bool v = pset_b32(PAT_ALL);
    auto acc_v = simd_inst::vlds<float>(acc + i * kVL, 0, NORM);
    auto red = simd_inst::vcadd(acc_v, v, MODE_ZEROING);
    simd_inst::vsts(red, reduced + i, 0, ONEPT_B32, v);
  }
}

}  // namespace

template <int32_t N_ACC>
__global__ __vector__ void store_pad8_cce_kernel(__gm__ float *acc_g,
                                                  __gm__ float *reduced_g) {
  AscendC::InitSocState();
  __ubuf__ uint8_t *ub = (__ubuf__ uint8_t *)0x0;

  copy_gm_to_ubuf_align_v2(ub + kAccUb, (__gm__ uint8_t *)acc_g, 0, 1,
                           N_ACC * kVL * 4, 0, 0, 0, 0, N_ACC * kVL * 4,
                           N_ACC * kVL * 4);
  AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);

  simd_vf_store_pad8_cce<N_ACC>(ub);

  AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
  AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
  copy_ubuf_to_gm_align_v2((__gm__ void *)reduced_g,
                           (__ubuf__ void *)(ub + kReducedUb), 0, 1, N_ACC * 4,
                           4, N_ACC * 4, N_ACC * 4);
}

extern "C" {

void call_store_pad8_cce_large(void *stream, void *acc, void *reduced) {
  store_pad8_cce_kernel<20><<<1, kUbBytes, stream>>>(
      (__gm__ float *)acc, (__gm__ float *)reduced);
}

void call_store_pad8_cce_small(void *stream, void *acc, void *reduced) {
  store_pad8_cce_kernel<4><<<1, kUbBytes, stream>>>(
      (__gm__ float *)acc, (__gm__ float *)reduced);
}

}  // extern "C"
