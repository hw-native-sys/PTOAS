#include <stdint.h>
extern "C" __global__ [aicore] void indirect_reference(
    __gm__ int64_t *, __gm__ uint16_t *, __gm__ int64_t *, __gm__ int64_t *,
    __gm__ int64_t *, __gm__ int64_t *, __gm__ int64_t *, int32_t);
extern "C" void launch_indirect_reference(void *stream, void *comb, void *initial,
    void *layer_in, void *layer_out, void *post, void *pre, void *residual) {
  indirect_reference<<<72, 196864, stream>>>(
      (__gm__ int64_t *)comb, (__gm__ uint16_t *)initial,
      (__gm__ int64_t *)layer_in, (__gm__ int64_t *)layer_out,
      (__gm__ int64_t *)post, (__gm__ int64_t *)pre,
      (__gm__ int64_t *)residual, 8192);
}
