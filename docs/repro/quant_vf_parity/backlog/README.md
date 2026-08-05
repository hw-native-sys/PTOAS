# Backlog — solved / not a blocker

**Issue solved. Not a blocker now. Kept only as backlog.**

Former feature request 3 (FP32 double-buffered block-quant wall-clock) closed
after serial abs-max (~0.98× AscendC at 8192×2048). Open blockers live under
`../fixtures/` and `../FEATURE_REQUEST.md` (Ask 1 + Ask 2 only).

| Path | Contents |
|------|----------|
| `fixtures/` | AscendC + VMI FP32 kernels, strip fragments, `PERF_FINDINGS.md` |
| `scripts/` | ctypes build, device bench, msopprof wrappers |

Optional re-run from package root:

```bash
PTOAS_ROOT=/path/to/PTOAS NPU_DEVICE=1 ./backlog/scripts/run_fp32_block_quant_device.sh
./backlog/scripts/run_msopprof_fp32_block_quant.sh both
```

`../scripts/check_vmi_asc_residual.sh` does **not** exercise this tree.
