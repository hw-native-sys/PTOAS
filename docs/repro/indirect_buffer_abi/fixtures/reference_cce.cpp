#include "kernel_operator.h"

// Direct CCE control: load one address from a GM table and DMA the pointed-to
// payload without copying all buffers into a contiguous staging tensor.
extern "C" __global__ __aicore__ void indirect_buffer_reference(
    __gm__ uint64_t *source_table, __gm__ uint64_t *destination_table,
    uint32_t count, uint32_t bytes_per_buffer) {
  AscendC::InitSocState();
  __ubuf__ uint8_t *ub = (__ubuf__ uint8_t *)0;
  for (uint32_t layer = 0; layer < count; ++layer) {
    __gm__ uint8_t *src = (__gm__ uint8_t *)source_table[layer];
    __gm__ uint8_t *dst = (__gm__ uint8_t *)destination_table[layer];
    copy_gm_to_ubuf_align_v2(ub, src, 0, 1, bytes_per_buffer,
                             0, 0, false, 0, bytes_per_buffer, bytes_per_buffer);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm_align_v2(dst, ub, 0, 1, bytes_per_buffer,
                             0, bytes_per_buffer, bytes_per_buffer);
  }
}
