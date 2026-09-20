# C/V cost model stage-one fixture

This directory contains a serial A5 two-kernel, four-stage communication example and independently specified graph
expectations. It is not FlashAttention: the vector stages are negation and addition, with real Cube matmuls between
the three pipes. The two AIVs use different output slices. No hardware execution is claimed by code-generation tests.

Use the matching PTOAS build/package:

```bash
ptoas --pto-arch=a5 four_stage_serial.pto -o serial.cpp
ptoas costmodel export four_stage_serial.pto --profile a5_profile.json --output package
```

The cost model reads the package and returns a plan in the format described in
[the exchange ADR](../../../docs/designs/ptoas-cv-costmodel-exchange-v1.md).
For a PTOAS-only smoke test, the following deliberately hand-authored plan sets each local allocation to one slot:

```python
import json
from pathlib import Path

manifest = json.loads(Path("package/manifest.json").read_text())
keys = ("schema_version", "input_fingerprint", "target_profile_fingerprint",
        "bindings_fingerprint", "pipeline_id", "schedule_kind",
        "preload_semantics", "local_schedule")
plan = {key: manifest[key] for key in keys}
plan["preload_count"] = 2
plan["buffers"] = [{"buffer_id": buf["id"], "count": 1}
                   for buf in manifest["buffers"] if buf["multi_buffer_eligible"]]
Path("manual_plan.json").write_text(json.dumps(plan, indent=2))
```

```bash
ptoas costmodel import package --plan manual_plan.json --output result
```

Read `result/apply_report.json`: `status=annotation_only`, `optimization_applied=false`,
`physical_feasibility=not_checked`, `schedule_legality=not_proven`.
This validates transport of configuration, not the optimized program's correctness or performance.

Run the native/MLIR-backed contract suite with the configured build tree:

```bash
llvm-lit -v build/test/lit --filter 'cv_costmodel_exchange.pto$'
```

This test is automatically included in the existing `check-pto` CI target. It needs the build's matching
Python interpreter and staged PTOAS package, as configured by CMake. It does not require a cost model checkout.

## Exchange 2.0

The opt-in [2.0 ADR](../../../docs/designs/ptoas-costmodel-exchange-v2.md) adds typed `program.json`,
independent adapter processes, bound candidate schedules, and static candidate compilation.
The [implementation plan](../../../docs/designs/ptoas-costmodel-exchange-implementation-plan.md)
tracks the memory-reuse contract and post-compile feedback. Explicit compilation now produces
`memory_plan.json` and `validation_report.json` from the same native invocation as the C++.
G2 completion checking is restricted to the declared static A5 profile; unknown conditions do not pass.
PTOAS consumes the optional `tilesim.selection.v1` ranking, freezes the selected Plan, and rejects
selection/configuration/schedule mismatches. Paired exact-workload certification remains opt-in.

```bash
ptoas costmodel v2 export four_stage_serial.pto --profile a5_profile.json --output package-v2
ptoas costmodel v2 propose package-v2 --adapter adapter.json --preloads 0 1 2 3 --output model-run
ptoas costmodel v2 validate package-v2 --plan model-run/plan-1.json
ptoas costmodel v2 apply package-v2 --plan model-run/plan-1.json --output annotated-v2
```

`adapter.json` is a user-owned JSON object with `argv`, `cwd` and optional `environment`.
For the protocol reference use `argv=["python", "-m", "pto_costmodel.reference"]` and the directory
containing the installed SDK as `cwd`. For TileSim use its own Python and
`-m core.frontend.adaptor.ptoas_exchange`, with its checkout as `cwd`. Install the standalone SDK
in that interpreter (`pip install ./ptodsl/pto_costmodel`) or add `ptodsl` to its `PYTHONPATH`.
The reference adapter deliberately returns unknown latency and cannot certify performance.

For a compile experiment, copy the inferred bindings into a new bindings file and set
`alias_contract="disjoint"` only if the launch really uses disjoint Q/K/V/output allocations.
Re-export with `--bindings`, rerun the model, and apply the new plan with `--mode compile`.
The output includes materialized PTO, native C++, lowering IR, a trace, and an apply report.
Compilation explicitly runs level2 memory planning and `--enable-insert-sync`.
Existing plans from the package with unknown aliasing must not be reused after changing bindings.

The result contains one complete `plan-N.json` per evaluated candidate. When an adapter returns
`tilesim.selection.v1`, PTOAS also writes `selected-plan.json` and a frozen selection report. Unsupported
timing remains `null` and cannot be selected. Selection still does not automatically apply a candidate.
`--cache` stores predictions by
request and model fingerprints; it does not cache hardware measurements.

```bash
llvm-lit -v build/test/lit --filter 'cv_costmodel_exchange(_v2)?.pto$'
ptoas costmodel v2 certify --policy policy.json --evidence measured.json --output certification
```

`certify` checks supplied measurement evidence; it neither runs a device nor authenticates a report.
No built-in numeric performance thresholds or performance certificate are supplied for this micro.

Run the complete fixed-input integration with the matching PTOAS Python environment:

```bash
python test/samples/CVCostModel/run_joint.py \
  --model-python /path/to/tilesim/venv/bin/python \
  --model-root /path/to/tilesim \
  --output /path/to/new-run-directory
```

The runner exports disjoint bindings for this fixture, invokes the real adapter, imports its
bound plan, and explicitly compiles P=2. Selecting P=2 here exercises the interface; it is not
a model-certified performance choice. Keep `summary.json` and the complete run directory together.


## Fixed-candidate G2/G3 device validation

The interface micro above is deliberately retained. It is **not a device golden fixture**:
its 8-row vector NZ tile is undersized for the installed A5 ND-to-NZ 16-row stride.
Native G2 reports `NATIVE_LAYOUT_OVERFLOW` for that configuration. `runtime_fixture.py`
uses 32x32 FP32 matrices (16 rows per AIV), distinct Q/K/V per iteration, and independent
integer NumPy golden. Basic computes `2*(-Q@K)@V`; crossing computes `(-Q@K)@V-Q@K`.

```bash
# Run in the matching PTOAS Python environment; output must be fresh.
python test/samples/CVCostModel/prepare_runtime.py --output /path/to/frozen-matrix
# Compiler-backed feedback and failure tests, including N=0 (no hardware).
llvm-lit -v build/test/lit --filter 'cv_costmodel_g2.pto$'
```

Commit the task locally and build a clean compiler before preparing an accepted matrix.
The manifest records source, compiler, Python and runtime-tool provenance. Do not edit the
prepared files; regenerate after any change. The matrix has N=1/2/4/8, unique P=0/1/2/N/N+1,
minimum legal slots, explicit vector slots 1/2/3, original serial and P=0-unrolled controls.
Rejected slot configurations cannot enter the device runner. Capacity is max(4,N) at each N.

Upload the matrix and the three runtime files (`build_runtime.py`, `run_runtime.py`,
`runtime_main.cpp`) into a new private A5 experiment. Activate the verified toolkit and build:

```bash
python run_runtime.py --mode build --matrix /experiment/inputs/matrix \
  --artifacts /experiment/artifacts --experiment /experiment --soc ACTUAL_SOC
```

Run the following through the site's exclusive device queue and private experiment lock:

```bash
python run_runtime.py --mode correctness --matrix /experiment/inputs/matrix \
  --artifacts /experiment/artifacts --experiment /experiment --device 0 --soc ACTUAL_SOC
```

The private lab's `bin/new_run_dir.sh` must be installed. Every seed/variant/repeat gets a
fresh directory and restored input. A run has a 60-second limit; any error, mismatch or
unstable output stops the batch and preserves evidence. Each variant runs twice with seeds
0/1/2; the second order starts with a candidate. Four disjoint GM allocations have guards,
and output begins as a sentinel. Guard checking complements static bounds, not UB/L1 tracing.

Only after complete G3 success, create a **different** private performance experiment, reuse
the immutable objects by path, and run `--mode performance --g3-report /old/results/correctness.json`.
N=4/8 variants each get one warmup and five serialized msprof runs; every output is checked.
Keep original profiler files. Report mixed-kernel duration consistently, median/range and
ratios to serial and P=0. Missing timing or unresolved mixed-task boundaries mean unknown,
not zero. This command collects diagnostic data; it never issues G4 certification.

The G2 profile verifies native engine/set/wait ordering, version-to-operand mapping, local
physical overlap and borrowed-release obligations. It currently requires N<=pipe capacity,
no local/backing address overlap, supported FP32 tile operations, and compatible installed PTO
headers. Unknown batched FIFO reuse or unmodeled operations remain `unknown`. Its layout
coverage ends at final PTO; target compiler transformations must be checked in device evidence.


After copying the complete performance evidence locally, validate every saved input/output
and the profiler task boundary, then summarize into a fresh directory:

```bash
python test/samples/CVCostModel/summarize_runtime.py \
  --experiment /local/performance-experiment \
  --matrix /local/correctness-experiment/inputs/matrix \
  --output /local/new-timing-summary
llvm-lit -v build/test/lit --filter 'cv_costmodel_(g2|runtime).pto$'
```

The summary requires exactly one matching `MIX_AIC` task per profile, one AIC and two AIVs,
one warmup and five finite positive durations. It rejects missing/extra tasks and stale evidence.
Overlapping observed timing ranges are reported as `cannot_reliably_distinguish`; non-overlap
is only a diagnostic difference, not a certified ranking. See the
[2026-09-15 fixed-candidate acceptance record](../../../docs/designs/ptoas-costmodel-fixed-g2-g3-20260915.md).
The source freeze for that device campaign is `20ae930a4`; later summary/tests/docs commits
must not be presented as a rebuilt device campaign. Archived artifacts are immutable;
replay in a new experiment with a matching source/toolchain and freshly built executables.

## Frozen TileSim A/B acceptance

`prepare_ab.py` is the separate acceptance path for `basic/crossing × N=4/8`. It invokes TileSim
before any device measurement, freezes `tilesim.selection.v1`, and emits A (serial), P0 (unrolled),
and B only when the model predicts at least 2% improvement. A baseline decision never fabricates B.

```bash
python test/samples/CVCostModel/prepare_ab.py \
  --tilesim-root /path/to/tilesim --tilesim-python /path/to/python \
  --output /path/to/fresh-matrix
python test/samples/CVCostModel/run_ab.py --mode build \
  --matrix /experiment/inputs/matrix --artifacts /experiment/artifacts \
  --experiment /experiment --soc ACTUAL_SOC
python test/samples/CVCostModel/run_ab.py --mode correctness \
  --matrix /experiment/inputs/matrix --artifacts /experiment/artifacts \
  --experiment /experiment --device 0 --soc ACTUAL_SOC
```

After matching G3 evidence, `run_ab.py --mode performance --g3-report ...` collects 20 AB/BA
paired blocks per optimized workload and five P0 diagnostic samples. `summarize_ab.py` requires
one matching `MIX_AIC` task per run, reproduces the 10,000-resample bootstrap with seed 20260916,
and reports benefit separately from prediction-error certification. Baseline-retained workloads
produce no A/B performance samples and are reported as `BASELINE_RETAINED`.

## preload + multi-buffer mechanism acceptance

`prepare_mechanism.py` builds the crossing/N=8 stress fixture and asks TileSim to evaluate 129,
257, then 513 odd `tneg` repetitions. It freezes the first recommendation whose predicted gain
reaches 10%, then emits A (serial), P0 (unrolled), M (P0 plus the recommended P slots), B (the
complete recommendation), and an `INSUFFICIENT_SLOTS` P-invalid rejection. No A5 result is
available while this selection is made.

```bash
python test/samples/CVCostModel/prepare_mechanism.py \
  --tilesim-root /path/to/tilesim --tilesim-python /path/to/python \
  --output /path/to/fresh-mechanism-matrix
python test/samples/CVCostModel/run_mechanism.py --mode build \
  --matrix /experiment/inputs/matrix --artifacts /experiment/artifacts \
  --experiment /experiment --soc ACTUAL_SOC
python test/samples/CVCostModel/run_mechanism.py --mode correctness \
  --matrix /experiment/inputs/matrix --artifacts /experiment/artifacts \
  --experiment /experiment --device 0 --soc ACTUAL_SOC
python test/samples/CVCostModel/run_mechanism.py --mode performance \
  --matrix /experiment/inputs/matrix --artifacts /experiment/artifacts \
  --experiment /experiment --device 0 --soc ACTUAL_SOC \
  --g3-report /experiment/results/correctness.json
python test/samples/CVCostModel/summarize_mechanism.py \
  --experiment /experiment --matrix /experiment/inputs/matrix \
  --output /experiment/acceptance
```

The performance run has four warmups and 20 four-way paired blocks (80 profiler samples). The
summary reports `MECHANISM_BENEFIT_PASS/FAIL`; it deliberately does not issue a general G4 claim.
