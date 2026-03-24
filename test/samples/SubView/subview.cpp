#include "common/pto_instr.hpp"
using namespace pto;
template <typename T, int M, int N, int validM, int validN>
__global__ AICORE void Run_test_double_buffer_step(__gm__ T* v1, __gm__ T* v2, __ub__ T* v3) {
  unsigned v4 = 32;
  unsigned v5 = 0;
  __ub__ T* v6 = v3 + v5;
  __ub__ T* v7 = v3 + v4;
  TADD(v6, v6, v6);
  TLOAD(v7, v1);
  TSTORE(v2, v6);
  return;
}

extern "C" [aicore] void test_double_buffer_step(__gm__ float* v1, __gm__ float* v2, __ub__ float* v3) {
  Run_test_double_buffer_step<float, 32, 32, 32, 32>(v1, v2, v3);
  return;
}