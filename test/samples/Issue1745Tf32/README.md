# Issue #1745 A5 EmitC runtime validation

`issue1745_tf32.pto` preserves the issue's 80x64 and 64x32 FP32 matmul
shapes and round-away TF32 mode, and adds GM/L1/L0 transfers and output storage.
It computes `C = 2 * A @ B`. `issue1745_tf32_loop.pto` executes 80 products
inside a structured loop, computing `C = 80 * A @ B` with one CTRL region.
This is a runtime reproduction of the issue, not the unavailable full hc_linear.

The signed inputs are exactly representable in TF32, so the independent float64
CPU oracle isolates accumulation and synchronization from rounding tolerance.
All 2560 output elements are checked; an output initialized to NaN detects
missing stores. These cases do not exhaustively test TF32 rounding semantics.

Source a complete A5 CANN environment, then set `PTO_ISA_ROOT`, `PTOAS_BIN`
and optionally `WORK_SPACE`, `PYTHON_BIN`, `BUILD_JOBS`, `ACL_DEVICE_ID`:

```bash
bash test/samples/Issue1745Tf32/run_validation.sh
```

On shared hardware, run through the local device scheduler/lock. The compiler
may run on another host: transfer the generated testcase directory, compile
with the A5 CANN toolchain, generate inputs with `golden.py --products 2` (or
`80` for the loop), execute the host, and run `compare.py`. Preserve input PTO,
generated C++, build logs and device task logs with the validation result.
