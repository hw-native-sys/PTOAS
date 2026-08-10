// Direct CCE supports the fundamental operation with an address table:
// GM_ADDR table[]; auto src = (__gm__ float *)table[layer];
// copy_gm_to_ubuf(ub, src + offset, ...);
__global__ __aicore__ void indirect_buffer_reference(uint64_t *table);
