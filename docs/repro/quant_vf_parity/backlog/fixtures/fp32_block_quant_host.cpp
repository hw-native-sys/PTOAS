// Host entry for AscendC fp32_block_quant (ctypes). Appended to
// reference_asc_fp32_block_quant.asc by scripts/build_fp32_block_quant_asc.sh.

namespace {
constexpr int32_t kBlocks = 72;
constexpr uint32_t kUbBytes = 163968;
}  // namespace

extern "C" {

void call_fp32_block_quant(void* stream, void* out, void* output_sf, void* x,
                           int32_t num_tokens, int32_t sf_stride) {
  fp32_block_quant_kernel<<<kBlocks, kUbBytes, stream>>>(
      (__gm__ fp8_e4_t*)out, (__gm__ float*)output_sf, (__gm__ float*)x,
      num_tokens, sf_stride);
}

}  // extern "C"
