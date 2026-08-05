# AscendC runtime for backlog FP32 block-quant (former FR3)

**Issue solved. Not a blocker now.** This artifact is kept for backlog /
regression device benches only.

Built by [`scripts/build_fp32_block_quant_asc.sh`](../../scripts/build_fp32_block_quant_asc.sh)
from [`reference_asc_fp32_block_quant.asc`](../reference_asc_fp32_block_quant.asc)
plus [`fp32_block_quant_host.cpp`](../fp32_block_quant_host.cpp).

| File | Role |
|------|------|
| `libfp32_block_quant.so` | bisheng `--shared` library; exports `call_fp32_block_quant` for ctypes |

Host launch:

```text
call_fp32_block_quant(stream, out, sf, x, num_tokens, sf_stride)
  → fp32_block_quant_kernel<<<72, 163968, stream>>>(...)
```
