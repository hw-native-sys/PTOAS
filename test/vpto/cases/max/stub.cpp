#include <pto/common/type.hpp>

#ifndef __global__
#define __global__
#endif

#ifndef __gm__
#define __gm__
#endif

extern "C" __global__ AICORE void max_kernel_2d(__gm__ float *v1,
                                                 __gm__ float *v2,
                                                 __gm__ float *v3) {
  (void)v1;
  (void)v2;
  (void)v3;
}
