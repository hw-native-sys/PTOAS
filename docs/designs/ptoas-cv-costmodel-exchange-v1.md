# ADR: CV cost model exchange, annotation-only v1

Status: implemented PTOAS-side protocol; external model sign-off pending.

This implements the G0/G1 boundary of the [C/V pipelining design, PR #1292](https://github.com/hw-native-sys/PTOAS/pull/1292).
A model receives a normalized serial PTO program and returns one pipeline preload and per-allocation slot recommendations.
PTOAS validates identity, ownership and the supported protocol before attaching annotations. It does not reorder
operations, allocate multiple slots, prove the proposed schedule legal, or claim a speedup in this version.

## Checkpoint and interface

The native `--emit-cv-costmodel-ir` mode stops after section normalization, serial frontend pipe lowering,
pipe-init inference/validation and explicit `tfree` verification. It returns PTO IR through the normal `-o` output.
It uses MLIR generic operation syntax to round-trip pre-sync pipes whose `flag_base` has not been assigned yet.
It runs before layout/fusion, memory planning, synchronization insertion and buffer-select resolution.
The native mode is A5-only. Existing compilation paths remain available without this flag.

The Python package exposes both `ptoas costmodel export/import` and
`python -m ptoas.costmodel export/import`. It parses native-checkpoint output with PTOAS's actual MLIR bindings.
It does not parse PTO syntax with regular expressions. Native subprocesses use the active package/interpreter,
not an unrelated `ptoas` executable from PATH.

```
ptoas costmodel export four_stage_serial.pto --profile a5_profile.json --output package
ptoas costmodel import package --plan plan.json --output result
```

`--bindings bindings.json` is optional; v1 supports the empty object because kernels have static loops and pointer-only
arguments. Import accepts `--input current.pto`, `--profile current_profile.json`, and `--bindings current_bindings.json`
to verify that a recommendation is still valid for the current request. Without `--input`, the package's canonical
PTO is the input being annotated; this is not validation against an unrelated source checkout.

Export writes `canonical.pto`, `manifest.json`, `runtime_bindings.json`, and `target_profile.json`.
Import writes `annotated.pto` and `apply_report.json`. Output directories must be fresh. Invalid input cannot leave a
partially accepted result. CLI rejection is a JSON diagnostic on stderr with nonzero exit status.

## Supported semantic subset

V1 accepts exactly one Cube function and one Vector function, each containing one direct, constant-bound `scf.for`.
It supports A5 unidirectional tile-entry L2L pipes backed by explicit reserved buffers/imports, with no-split or
row-split transactions. The serial pipe topology is `C_QK -> V_P -> C_PV -> V_O`. Stages are identified from validated
transaction order, not from function names. Each pipe has exactly one push, one pop and one free per iteration.
There are no nested branches, nested loops or SSA iter_args. The op allowlist is explicit in `_cv_ir.py`.
Unknown operations, unresolved alias roots, compact storage and unsupported types are rejected, never assigned
zero latency or zero storage. This initial micro-kernel subset is intentionally smaller than full FlashAttention.

The sample computes QK=Q*K, P=-QK, PV=P*V and O=PV+PV. It is an interface/communication fixture, not a softmax kernel
or an independently validated hardware numerical golden. It uses 1 AIC / 2 AIV row-split traffic with disjoint output
slices. A separate FA fixture and simulator/board reference are later acceptance work.

Local allocs, pipe backing and borrowed entries are different owners. A local accumulator remains local even if it
is pushed into a pipe. Pop entries and their aliases are not eligible for local multi-buffer annotations. Alias
readers must stay inside the corresponding pop/free lifetime. Memory access records preserve in-place reads/writes;
GM-pointer disjointness is not proven, and the manifest says `gm_aliasing=not_proven`.

Tile storage records use the explicit normalized shape, valid shape, dtype, layouts, element bytes and profile
alignment. Compact storage is rejected. `logical_bytes`, `allocation_bytes` and `slot_stride_bytes` are separate;
final physical placement/fragmentation remains PlanMemory's authority. The sample A5 local capacities and alignments
match `PTOPlanMemoryModern.cpp::getMemSpec` at the implementation baseline. Its L2 model is explicitly off; no hardware
L2 capacity is guessed. Profile changes are fingerprinted. Private Vector buffers are per-AIV allocations, not
one allocation shared by both AIVs.

## Stable identity and plan

The protocol is `pto.cv_costmodel.v1`. IDs are deterministic preorder structural IDs scoped by function symbols.
The fingerprint serializes op names, parent IDs, types, semantic attributes, operands and block arguments; it excludes
locations, printed SSA spellings and the known output annotations. Changing bounds, shape, topology or semantic
attributes invalidates the plan. There is no promise to preserve identity through CSE/fusion/splitting: re-export
when structure changes. Supplied IDs never override IDs recomputed from the canonical input.

The plan has these required fields:

```
schema_version
input_fingerprint
target_profile_fingerprint
bindings_fingerprint
pipeline_id = "cv0"
schedule_kind = "prefix_suffix_v1"
preload_semantics = "iteration_distance"
preload_count = nonnegative integer
local_schedule = "off"
buffers = [{"buffer_id": "<manifest local allocation id>", "count": <positive integer>}, ...]
```

Every eligible local allocation must appear exactly once. Unknown/duplicate IDs, missing allocations, nonpositive
counts, booleans masquerading as counts, values above signed 32-bit range, unsupported schedules, unknown fields and
physical-address injection are rejected. Optional `metrics` and `model_version` are diagnostic metadata and do not
change annotations. Reordering the buffer list also does not change the selected plan identity.

`Pe=min(preload_count,trip_count)`. A nonempty loop's P pipe must have effective capacity at least `max(Pe,1)`.
Each recommended allocation must individually fit the selected memory budget. These are necessary prechecks, not
whole-program physical feasibility or schedule-liveness proof. The report explicitly marks both checks as pending.
PPT task-preissue depth is not accepted as iteration distance without an external verified mapping. With trip_count=2,
P=2 and P=3 have the same Pe=2. Requested P is retained in the annotation; effective P is reported separately.

Import attaches:

- `pto.costmodel.buffer_id` to local allocations/declarations/reserved allocations;
- `pto.costmodel.loop_id` and `pto.cv_preload_count` to both loops;
- `pto.pipeline.multi_buffer_count` to each local alloc;
- `pto.costmodel.plan_id` and `pto.costmodel.status="annotation_only"` to the module.

`count=1` stays an ordinary alloc. In this stage, count=2/3 also remains an ordinary alloc with pending metadata;
materialization into `alloc_multi_tile` belongs to G2. The existing multi-buffer type requires count>=2.
No physical address attributes are generated by this interface. Reimporting the same plan into the same unchanged
structure is idempotent. Normal compilation of annotated IR remains serial until a transformation consumer is added.

## Acceptance and rollout

G0 reuses the twelve selected mainline multi-buffer/pipe lit tests. G1's
`test/lit/pto/cv_costmodel_exchange.pto` runs actual native-checkpoint and MLIR-backed tests, including the frozen
sample graph/storage counts, 1/2/3-slot annotations, SSA renaming, idempotence, stale plans, ownership, unsupported ops,
JSON errors, capacity prechecks and serial code generation. Fixtures/expected counts are checked in independently
of generated export output. This is not the earlier reference-only oracle self-test.

The CLI is opt-in. Default codegen does not activate any scheduling optimization. No migration of existing PTO inputs
is required. Tests use hand-authored plans and must be reported as PTOAS-side contract acceptance, not as a real
Tilesim round trip. Real model sign-off requires the cost model team to return a plan and configuration/trace from
an actual model run. PPT raw IR, runtime bindings and model revision remain required for PPT golden replay.
