# FP32 accumulator barrier experiment

This sample makes the cost of per-MMAD `PIPE_M` barriers reproducible. It
contains an automatic-sync code-generation reproducer and a separate native
A5 benchmark. It does not change automatic synchronization or a model's
default implementation.

## Automatic-sync reproducer

`acc_chain.pto` loads FP32 operands into L0, initializes an accumulator on the
first iteration, and accumulates on subsequent iterations. The output has
16 rows, 24 valid columns, and a 32-element row stride. The runtime `repeats`
argument must be positive.

```bash
"$PTOAS_BIN" test/samples/MatmulAccBarrier/acc_chain.pto \
  --enable-insert-sync --pto-level=level3 -o /tmp/acc_chain.cpp
rg -n 'pipe_barrier\(PIPE_M\)|TMATMUL' /tmp/acc_chain.cpp
```

PTOAS 0.72 emits a `PIPE_M` barrier in each of the initialization and
accumulation branches. Automatic synchronization stays enabled in this
reproduction. This is a diagnostic input, not a regression test asserting
that the current conservative behavior is incorrect.

## Native comparison

`native/barrier_kernel.cpp` compares two compile-time versions of the
same FP32 projection. `Serialize=true` executes `pipe_barrier(PIPE_M)` before
every MMAD; `Serialize=false` omits just that barrier. Both retain operand
readiness/reuse events, the final M-to-FIX event, and the final `PIPE_ALL`.

The full problem has `M=rows`, `N=24`, and `K=20480`. Each task handles up to
16 rows and one of four K5120 splits, using twenty K256 MMADs and double
buffering. Partial results have shape `[4 * padded_rows, 32]`, where
`padded_rows=ceil(rows/16)*16`. Only the valid rows and first 24 columns are
checked; output padding is not part of the numerical contract.

Select the CANN and PTO ISA versions explicitly. The shared library build
does not need a device or a PTOAS/LLVM source build:

```bash
source /path/to/cann/set_env.sh
export PTO_ISA_ROOT=/path/to/selected/pto-isa
bash test/samples/MatmulAccBarrier/native/build.sh
```

The Python runner requires NumPy. Run it inside a device allocation on shared
servers; it accepts `TASK_DEVICE` or an explicit `--device` on other systems.
For a queue that supports `task-submit`:

```bash
task-submit --device auto --env ASCEND_HOME_PATH --env LD_LIBRARY_PATH \
  'OPENBLAS_NUM_THREADS=1 .venv/bin/python test/samples/MatmulAccBarrier/native/run.py'
```

Useful runner options are `--library`, `--rows`, `--rounds`, `--warmup`,
`--batches`, `--seeds`, and `--output`. Defaults are 72 rows, 100 rounds,
five warmups, five batches, and three fixed random seeds.

The runner checks each partial against an FP64 reference and requires
bitwise equality between the two FP32 variants. It then randomizes their
order within each timed pair. Every timed call has its output overwritten
with NaNs before the device-event window and checked after it. Raw paired
samples, validation errors, the library hash, and per-batch medians are
written to JSON.

## Recorded native comparison

One run on 2026-11-06 used Ascend950DT_9581, CANN 9.3.0, PTO ISA
`dcebd22bec597256e264b1bff93226823d782962`, and 72 rows. Each batch had
100 timed pairs after five warmups. All three random seeds passed the FP64
reference check, and all 1,000 timed-call outputs matched the validated
FP32 baseline bitwise. Separate correctness runs covered 1, 16, 17, and
80 rows, including partial tiles.

| Batch | Serialized median, us | Barrier-free median, us | Median paired change |
| --- | ---: | ---: | ---: |
| 1 | 20.406 | 19.509 | -4.87% |
| 2 | 20.265 | 19.184 | -5.50% |
| 3 | 19.726 | 18.532 | -5.61% |
| 4 | 19.721 | 18.726 | -4.85% |
| 5 | 20.254 | 18.969 | -5.85% |

The median of the five within-batch paired percentage changes was **-5.50%**.
That paired metric differs from taking the ratio of two independently
aggregated timing medians; the JSON retains both. This result characterizes
the two native sample implementations under the protocol above.

## Interpretation

Device-event timings cover the complete native kernel, including its input
preparation and final store. Input allocation, host-to-device poisoning,
output downloads, and correctness checks are outside those windows. There
is no PyPTO graph or AICPU task scheduler in this benchmark.

Compare only matching protocols, shapes, devices, toolchains, and cache
conditions. The inputs are reused without flushing L2; the result is not a
cold-HBM measurement. Do not compare these numbers to a PyPTO invocation or
infer a whole-model speedup.

Successful numerical tests on this shape do not prove that all `PIPE_M`
barriers can be removed. A production auto-sync optimization needs its own
proof and regressions for aliases, loop-carried dependencies, initialization
paths, and cross-pipeline consumers. The native experiment intentionally
keeps those decisions separate from the compiler reproducer.
