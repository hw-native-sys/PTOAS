# PTOAS × TileSim preload + multi-buffer mechanism acceptance (2026-09-18)

## Verdicts kept separate

- Annotation-guided optimization: **`MECHANISM_BENEFIT_PASS`** for the exact frozen
  `multibuffer-stress-n8-r129` workload on `ptoas-a5-39`, device 0.
- Cost Model accuracy: candidate selection direction passed, absolute latency accuracy failed the
  10% policy. This is not a general G4 certificate.

TileSim selected the smallest permitted stress setting, 129 odd alternating `tneg` operations,
before any device measurement. It recommended `preload_count=1` and two slots for P Buffer
`vector.op61`, predicting 19.584% improvement over P=0. PTOAS generated and G2-validated:

- A: serial, P=0, all local Buffers single-slot;
- P0: static materialization, P=0, all local Buffers single-slot;
- M: P0 schedule with only P Buffer changed to two slots;
- B: TileSim's P=1 recommendation with the complete minimum slot configuration;
- P-invalid: B's preload with P forced to one slot, rejected with `INSUFFICIENT_SLOTS`.

## Part 1: annotation-guided optimization benefit

All 24 G3 runs passed independent exact-integer golden comparison, repeated output, input hash,
guard, timeout, and runtime checks. Performance collection completed four warmups and 20 paired
four-way blocks, for 80 valid profiles. Every retained profile contains exactly one matching
`MIX_AIC` task with one AIC and two AIV blocks.

| Comparison | Mean gain | Bootstrap 95% interval | Worst paired regression |
|---|---:|---:|---:|
| B / A | 14.64% | [14.22%, 15.07%] | -12.93% |
| B / P0 | 15.39% | [14.95%, 15.81%] | -13.61% |
| M / P0 | -0.72% | [-1.29%, -0.14%] | 3.29% |

Mean task durations were A 32.4302 us, P0 32.7171 us, M 32.9497 us, and B 27.6794 us.
Both B comparisons exceed the 2% benefit threshold with positive confidence lower bounds and no
paired regression. M/P0 remains wholly inside the specified ±2% equivalence interval. The result
therefore attributes the observed gain to the legal cross-iteration overlap enabled by preload
and its required multi-buffer storage, rather than to allocating an extra P slot alone.

## Part 2: model prediction gap and feedback

| Quantity | TileSim | A5 measured | Difference |
|---|---:|---:|---:|
| P0 baseline latency | 26.1513 us | 32.7171 us | 20.07% absolute error |
| B candidate latency | 21.0298 us | 27.6794 us | 24.02% absolute error |
| B/P0 gain | 19.58% | 15.39% | model overestimates by 4.19 percentage points |

TileSim correctly selected a beneficial configuration, so its selection result must not be hidden
by the latency error. It nevertheless underestimates both the common workload cost and B's final
latency. The cost-model team should act on the following feedback:

1. Calibrate common launch, ND/NZ layout conversion, A5 L2L, and synchronization costs. The fact
   that both P0 and B are low indicates missing or optimistic common costs.
2. Calibrate `tneg` throughput and fixed cost with independent primitive microbenchmarks across
   several chain lengths and both AIV lanes. Do not fit this acceptance workload and then reuse it
   as independent G4 evidence.
3. Consume the compiler's lowered operation counts, inserted waits/synchronization, memory plan,
   and final schedule fingerprint before retaining a prediction for G4.
4. Report three fields separately: selection quality, absolute-latency accuracy, and speedup
   accuracy. For this run they are respectively pass, fail, and an overestimate of 4.19 points.

The optimization mechanism passes. Model calibration and G4 remain open.

## Frozen identities and evidence

- PTOAS experiment source: `210f9f03f43687619bf948087ef26ec321d3e195`
- TileSim source: `0ea5ba8f548595771c0c92f51f797e0dac7e0ee9`
- Target: `Ascend950DT_9592`, A5, 1 AIC + 2 AIV
- Remote experiment: `20260918-cv-mechanism-acceptance-02`
- Immutable provenance correction: `20260918-cv-mechanism-acceptance-02-attestation`
- Local evidence: `_private_ptoas_lab/experiments/20260918-cv-mechanism-acceptance-02`
- Machine-readable result: `results/acceptance/report.json`

`npu-smi` was unavailable. ACL context creation and device synchronization passed for devices 0
and 1; the accepted matrix used device 0 under `task-submit`/`npu-lock`. This proves the ACL runtime
and cooperative serialization, but does not provide temperature, ECC, firmware alarm, or
non-cooperating-process telemetry.

The initial CANN profiler export lacked an unversioned `libsqlite3.so` dependency. All 80 raw
profiles had already been captured and correctness-checked. They were parsed and exported offline
from the same immutable raw directories using the host's `libsqlite3.so.0`; no kernel was rerun and
no candidate was changed.

The remote experiment wrapper's manually supplied `SOURCE_COMMIT` contains a transcription error.
The matrix manifest and acceptance report contain the correct full commit shown above. The separate
read-only attestation binds that correction to SHA-256 values for the matrix, G3, performance, and
acceptance reports; it changes no measured evidence.
