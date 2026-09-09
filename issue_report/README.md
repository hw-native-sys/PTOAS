# Quant leftover PTOAS issues (0909 pins)

PTOAS `b465f26b`, CANN 9.2.0, TileLang `5038c468`, A5.

Kernel-side work on `quant_missing_impl_0909` closed fp32 TMA, per_channel TMA-in, fused e2m1 rescale, and large-H e4m3 SF. These reports are the remaining **cast_back** leftovers after one legal-VMI schedule attempt (PTOAS-legal mask pad 32→64).

Each directory has legal desired VMI (`desired_vmi.ptodsl.py` = production dump), TileLang PTODSL dump, Nightly ASC C++ (`asc_reference.cpp`), `run_repro.sh`, and recorded logs.

| Dir | Variant | Recorded |
|---|---|---|
| `cast_back_e4m3_fp32_tma_npt32_h128` | e4m3→fp32 TMA npt=32 H=128 | compile-OK, ACL 507035 |
| `cast_back_e4m3_fp32_tma_npt1` | e4m3→fp32 TMA npt=1 H=128 | compile-OK, ACL 507035 |
| `cast_back_row_npt1` | row npt=1 e2m1/e4m3→bf16/fp32 | compile-OK, numerical mismatch |
