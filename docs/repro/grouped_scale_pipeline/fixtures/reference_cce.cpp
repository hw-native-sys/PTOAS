// Minimal stream-first launch ABI for the standalone vector reference.
extern "C" __global__ [aicore] void grouped_reference(
    __gm__ uint8_t *out, __gm__ float *scales, __gm__ uint16_t *input,
    int rows, int stride);
extern "C" void launch_grouped_reference(void *stream, void *out, void *scales,
                                          void *input, int rows, int stride) {
  grouped_reference<<<1, 204800, stream>>>((__gm__ uint8_t *)out,
      (__gm__ float *)scales, (__gm__ uint16_t *)input, rows, stride);
}
