// Stream-first C ABI: this is intentionally usable from ctypes without a
// framework runtime or a tensor-handle ABI.
extern "C" __global__ [aicore] void roundtrip_reference(
    __gm__ uint16_t *data, int rows);

extern "C" void launch_roundtrip_reference(void *stream, void *data,
                                            int rows) {
  roundtrip_reference<<<1, 102912, stream>>>(
      (__gm__ uint16_t *)data, rows);
}
