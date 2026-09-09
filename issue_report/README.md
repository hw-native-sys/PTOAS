# Quant leftover PTOAS issues (0909 pins)

PTOAS `b465f26b`, CANN 9.2.0, TileLang `5038c468`, A5.

Revalidated on TileKernels-vmi `perf/per-channel-bk128-fused` @ `933c329` (fused bk128 + legal-width UE8M0 decode). That PR does **not** close these three cases — they still fail on the same pins. The small-H packed decode / compose bug that PR fixed was a VMI bug, not a PTOAS leftover; it is not reported here.

These reports remain the **cast_back** blockers after one legal-VMI schedule attempt (PTOAS-legal mask pad 32→64, then PR73 chunked legal-width decode).

Each directory has legal desired VMI (`desired_vmi.ptodsl.py` = production dump), TileLang PTODSL dump, Nightly ASC C++ (`asc_reference.cpp`), `run_repro.sh`, and recorded logs.

| Dir | Variant | 0909 recorded | Revalidated on PR73 @ 933c329 |
|---|---|---|---|
| `cast_back_e4m3_fp32_tma_npt32_h128` | e4m3→fp32 TMA npt=32 H=128 | compile-OK, ACL 507035 | still ACL 507035 |
| `cast_back_e4m3_fp32_tma_npt1` | e4m3→fp32 TMA npt=1 H=128 | compile-OK, ACL 507035 | still ACL 507035 |
| `cast_back_row_npt1` | row npt=1 e2m1/e4m3→bf16/fp32 | compile-OK, numerical mismatch | still mismatch (e4m3→bf16 count=127437 max_abs=5.25; e2m1→bf16 112474 / 4.5; e4m3→fp32 141102 / 5.25; e2m1→fp32 119210 / 4.5) |

Maps to [cann/pto-as issue #7](https://gitcode.com/cann/pto-as/issues/7). Keep all three; none became a non-blocker.
