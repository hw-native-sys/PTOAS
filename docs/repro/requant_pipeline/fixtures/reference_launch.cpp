// Stream-first host ABI for the standalone direct vector reference.
extern "C" __global__ [aicore] void requant_reference(
    __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, __gm__ uint8_t *, int32_t, int32_t);
extern "C" void launch_requant_reference(void *stream, void *src, void *input_scale,
                                           void *dst, void *output_scale) {
  requant_reference<<<72, 156096, stream>>>(
      (__gm__ uint8_t *)dst, (__gm__ uint8_t *)output_scale,
      (__gm__ uint8_t *)src, (__gm__ uint8_t *)input_scale, 8064, 224);
}
