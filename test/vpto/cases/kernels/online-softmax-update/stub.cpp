// -----------------------------------------------------------------------------
// case: kernels/online-softmax-update
// family: kernels
// target_ops: pto.copy_gm_to_ubuf, pto.copy_ubuf_to_gm, pto.vlds, pto.vcmax, pto.vdup, pto.vmax, pto.vexpdiff, pto.vcadd, pto.vadd, pto.vmul, pto.vdiv, pto.vsts
// scenarios: online-softmax-update, dynamic-rows-and-seq, max-seq-128, block-rows-8, oldmax-oldsum-qk-to-newmax-newsum-expmax-out
// -----------------------------------------------------------------------------
#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void online_softmax_update_kernel_2d(
    __gm__ float *v1, __gm__ float *v2, __gm__ float *v3,
    __gm__ float *v4, __gm__ float *v5, __gm__ float *v6,
    __gm__ float *v7, int32_t v8, int32_t v9) {
  (void)v1; (void)v2; (void)v3; (void)v4;
  (void)v5; (void)v6; (void)v7; (void)v8; (void)v9;
}
