# ADR: PTOAS and cost models through a versioned semantic exchange

Status: proposed architecture; opt-in static implementation under validation.
Revision: 2026-09-15, memory reuse and compiler feedback requirements.

This extends [the v1 checkpoint contract](ptoas-cv-costmodel-exchange-v1.md) and the
[CV pipeline design, PR #1292](https://github.com/hw-native-sys/PTOAS/pull/1292).
The v1 CLI and wire format remain supported. No performance certification is implied by this ADR.
The [implementation and acceptance plan](ptoas-costmodel-exchange-implementation-plan.md) separates
existing evidence from pending work. The memory/feedback contracts below are proposed additions;
they are not implemented by the current 2.0 SDK, schema, CLI or TileSim adapter.

## Decision and ownership

PTOAS owns source semantics, stable identities, candidate schedules, legal transformations,
memory planning and synchronization. A model owns predictions and their coverage. An adapter
translates those facts without inventing missing dependencies, resource budgets or latencies.
Models run in independent processes. The `pto-costmodel` Python package has no MLIR, native
PTOAS or TileSim imports; it can also be installed separately from its own `pyproject.toml`.

The implemented protocol is `2.0`, operation semantics `pto.static_cv.1`, native checkpoint `1`,
and schedule `prefix_suffix_v1`. Public wire shapes are in the SDK's `exchange.schema.json`.
Schema validation does not replace semantic validation or reconstruction from canonical PTO.
New major versions may change meaning. Minor additions preserve existing field semantics and
add optional information. Semantic extensions needed for a request must also be negotiated through
`required_features`; they cannot be hidden in ignorable diagnostic extensions. Unknown required
capabilities fail closed. Exact feature names and the extension version must be frozen with schema
and compatibility tests before implementation. Existing v1 and 2.0 requests retain their behavior;
new memory-feasibility decisions require the negotiated extension, not a silent reinterpretation.

## Package and identity

Export writes `canonical.pto`, `program.json`, `runtime_bindings.json`, `target_profile.json`
and `manifest.json`. The manifest binds all four file digests and the semantic program, target
and bindings fingerprints. IDs are structural, independent of printed SSA names; transformations
invalidate old plans and require re-export. They are not cross-pass lineage IDs.

`program.json` contains typed values, typed attributes, operand references, parent relationships,
function arguments, loop domains, tasks, transactions, local memory accesses and ownership.
Arithmetic and view operands express addresses as a value graph. Models do not parse MLIR text.
Only explicitly mapped enum/type/attribute forms are exported; opaque forms are rejected.
Compiler import reconstructs this representation from the actual MLIR module and compares it.

Buffers carry allocation/valid shape, dtype, layout, bytes, alignment, root IDs, aliases,
ownership and physical instances. AIV0 and AIV1 are distinct private memory instances. Pipe
backing is charged once per declared owner; borrowed entries never acquire local slots.
Model scratch storage must be reported separately, not assigned a source allocation ID.

Bindings include rank-2 argument shape/strides/dtype, scenario and an alias contract. Static
view dimensions can be inferred, but disjointness is never inferred. Compile mode requires
the caller's explicit `disjoint` contract, verifies view bounds and rejects cross-core store
overlap and GM read/write recurrences outside the supported analysis. The launch caller must
honor that contract. Current scalar kernel arguments, multi-axis loops and branches are unsupported.

The target separates a compiler budget from a hardware profile. A missing hardware profile is
explicitly `null`. The initial budget is the existing PTOAS A5 profile; it is not silently
identified with TileSim's `davidV100` configuration. Model hardware, compiler usable capacities,
clock/performance tables, and cache policy must be reconciled before whole-kernel predictions
can be certified.

## Candidate semantics and executable subset

The compiler constructs `Pe=min(P,N)` schedules with Prefix before Suffix at the same step.
Requests carry the expanded per-core operation order, RAW dependencies, pipe transactions,
FIFO reuse edges, write generations and selected local slots. Models return the same schedule
fingerprint. A TileSim task depth is not part of this public contract. Equal core order for
two saturated P values does not collapse their configuration or resource identities.

The static implementation accepts at most 256 iterations, 65,536 operation instances and 32
search candidates including the baseline. It resolves write versions in original serial order,
then assigns slots to their live intervals under the candidate order. Reads before definitions
and insufficient hard slot counts fail. The P pipe must satisfy the ADR minimum capacity;
an independent DAG check rejects cycles from FIFO and per-core execution order.

Before physical planning the implementation conservatively budgets all local slots and pipe
backing with alignment, without assuming memory reuse. This may reject some candidates that
a future read-only PlanMemory query could accept. It never shrinks slots to make a plan fit.
This is an implementation limitation: the proposed extension treats an over-budget no-reuse sum
as inconclusive, not proof that reuse cannot fit. The current position-based slot analysis does
not itself prove asynchronous access completion.

`apply --mode compile` materializes static schedules by unrolling the two loops and creating
native `pto.alloc_multi_tile` / `pto.multi_tile_get` operations for multiple physical slots,
with version-aware slot rotation. This is an explicit bounded implementation,
not the future symbolic prologue/steady-state/epilogue pass. It uses the same schedule object
as evaluation, maps every tile operand to its write generation, and runs the native compiler
through memory planning, synchronization and C++ generation. Subview/reshape materialization,
out-of-loop tile accesses and unmodeled scalar expressions are rejected. Each candidate starts
from a freshly parsed canonical module; a failed compile publishes no partial result.

The compiler preserves the existing preload/count/ID annotations for annotation-only mode.
Predictions, model selection, search settings and ranking stay in sidecar files. Materialized
code is an explicit candidate; default compilation of annotated v1/v2 IR remains unchanged.
The compile report records the compiler binary and output artifact fingerprints and requires
post-lowering reevaluation. Native compile success is not device-level numerical or sync proof.

## Proposed memory reuse contract

### Provenance across compilation

PTOAS maintains explicit relations between source `buffer_id`, candidate-bound write versions
and slots, and compilation-bound physical allocations. These relations are not necessarily
one-to-one: versions rotate through slots, different buffers reuse an allocation range, and
lowering may split storage or eliminate an unused object. Preserve each source identity even
when addresses coincide. Record materialized, eliminated and compiler-generated objects explicitly;
do not invent source buffers for compiler temporaries or allocations for eliminated slots.

An allocation identifies its physical core instance, memory space, allocation ID and half-open
byte range. Views additionally identify their storage root and covered region. Equal numeric
addresses on AIV0 and AIV1 denote different private storage. Shared spaces use their actual shared
storage domain. The adapter must consume compiler provenance rather than recover it from SSA
names or addresses. Current multi-slot materialization filters source metadata without exporting
a replacement lineage map; repairing this is required work, not an existing guarantee.
Physical layout feedback is compiler-produced and read-only to the model. Its presence does not
authorize address injection in a candidate Plan or replacement of compiler allocation decisions.

No new user-visible annotation is required. Internal provenance metadata or compiler-owned side
tables must survive relevant transformations and be exported before metadata is discarded.

### Estimate basis, coverage and feasibility

Resource reports separate the following dimensions; these are proposed concepts, not current fields:

| Dimension | Required meaning |
|---|---|
| Basis | Known-storage sum without reuse; model-estimated lifetime reuse; compiler-realized layout |
| Coverage | Included local slots, backing, compiler temporaries, alignment, reservations and unknown components |
| Feasibility | Unknown/pending planning; realized layout verified; current planning attempt failed |

A no-reuse sum only bounds the storage it covers. Unknown lowering temporaries prevent a claim
that this bounds the final program. A model's reuse estimate does not prove the compiler can
realize that layout. Even a verified physical layout does not prove asynchronous reuse safety
or runtime correctness. Report payload, allocated/reserved bytes and arena extent separately
where relevant; account for padding, holes and reserved regions rather than merely summing
logical objects or deduplicating address numbers.

In the negotiated extension, over-budget conservative estimates may enter a bounded compiler
planning trial. Search budgets separately limit planning attempts/time. Unknown, unsupported
and timeout outcomes are not capacity proofs. A failed planner means that this compiler and
strategy did not realize the candidate; it does not prove that all possible layouts fail.
Automatic search may discard such an attempt with its reason. An explicitly requested candidate
must return its unresolved or failed outcome, never silently substitute a baseline. Corrupt
packages, invalid identities and protocol violations still terminate the request. Never reduce
slots inside a schedule that already depends on them.

### Completion conditions and safe reuse

A write version's storage may be overwritten only after every relevant access has completed
and the communication protocol's release conditions are satisfied. Issuing a reader, its position
in linear IR, or a model-predicted timestamp is not a completion proof. In-place access requires
operation-specific semantics; equality of interval endpoints alone does not authorize reuse.

PTOAS defines access regions, required completion/release events and happens-before constraints.
The model estimates event times for performance. PTOAS verifies that final hardware ordering,
synchronization or protocol rules enforce those constraints, with traceable evidence. Unknown
dependencies or release conditions block G2 for the affected transformation. Conservative
serialization is acceptable only when implemented, verified and included in performance feedback.
Recheck dependency/FIFO progress after adding reuse waits; a previously acyclic candidate can
deadlock after new constraints are inserted.

## Proposed executable compiler feedback

The current `re_evaluation_required` flag is diagnostic only. The required next workflow is:

```text
logical package + candidate schedule -> model estimate
  -> fresh candidate compile, memory planning and synchronization
  -> realized memory map + final execution constraints
  -> verify assumptions / reevaluate affected costs
  -> G3 runtime checks and G4 performance certification
```

Proposed `memory_plan.json` contains provenance relations, physical domains/ranges, reuse groups,
completion conditions and proof references, actual budget consumption, compiler temporaries and
coverage limitations. Execution feedback additionally describes inserted/removed operations,
transfers, synchronization and ordering constraints in structured form. A memory map or operation
counter alone is insufficient to reevaluate added waits. The two parts may be separate files in
one feedback package. The bounded implementation below freezes its own compiler-feedback
versions; general adapter feedback capabilities still require schema review.

All feedback must originate from the same final compilation checkpoint that produces the candidate
artifact. Bind the input identity, candidate/schedule, compiler binary and pipeline configuration,
target, final execution graph, memory plan and emitted artifact fingerprints. Separate reruns of
planning and code emission are not evidence of a common compilation unless consistency is proven.
Further cost-relevant lowering invalidates that assessment and requires another bound revision.

Adapters declare whether they consume realized layouts, reuse-induced waits and address-dependent
costs such as bank conflicts. Physical-address simulation is optional; respecting required execution
constraints is not. Unsupported costs remain unknown. Prediction validity is separate from G2:

- Retained only when a checked comparison establishes equivalence within the negotiated model abstraction.
- Stale when compilation changes relevant resource or execution assumptions; reevaluation is required.
- Unsupported when the model cannot assess the change; keep diagnostics but no certified full latency.

Post-compile prediction caches bind the feedback revision, execution/layout fingerprints, compiler
configuration, model/adapter configuration and target in addition to existing input/candidate keys.
Measurement caches additionally bind device and measurement environment. Never reuse a logical
estimate as a realized-program estimate merely because configuration or addresses look unchanged.
Each retry starts from a fresh canonical module and publishes a complete bound artifact set atomically.

## Process API and failure behavior

`capabilities`, `evaluate` and `propose` are adapter subprocess actions. PTOAS `validate` and
`apply` accept fingerprint-bound plans. `evaluate` requires a specific plan; `propose` prepares
a bounded prechecked candidate set and asks the model to evaluate/rank it. The current
TileSim adapter does not yet search additional slot combinations using full timed liveness.

Adapter configuration is user-owned `{argv, cwd, environment?}`. It is never taken from a model
result. Calls use argv without a shell, a timeout and bounded readback. Requests and results are
retained with capability negotiation, rejected candidates, selected plans and provenance.
Prediction caches are keyed by the complete request and model revision/configuration. Cache
hits are validated again. Hardware measurement evidence is a separate artifact family and
is never reused as a prediction cache entry.

Malformed packages, stale identities, unsupported required capabilities and protocol violations
terminate a request. Search can discard capacity-infeasible candidates, but a specifically
selected invalid plan fails. No optimization is automatically enabled without certification.
Without a recognized selection extension the existing behavior remains baseline with an explicit
reason. The optional `tilesim.selection.v1` extension binds a complete ranking and frozen candidate;
PTOAS validates and records it but does not automatically apply it. Partial model coverage returns
`latency.value=null` and cannot participate in selection.

## Verification levels and current limitations

| Level | Evidence required | Current behavior |
|---|---|---|
| G1 | Package/plan identity, ownership, mappings, schedule binding | Executable contract and subprocess tests |
| G2 | Provenance, actual layout/budget, completion-safe reuse, final dependency/FIFO/sync constraints and artifact consistency all verified | Bounded A5 completion/layout checker emits pass/fail/unknown; acceptance is candidate-specific |
| G3 | Independent golden, runtime bounds and deadlock validation | Frozen 32x32 fixed matrix passes 282 runs; frozen-selection A/P0 matrix passes another 48 runs on A5; broader workloads unvalidated |
| G4 | Same-workload predictions, paired measurements and explicit policy | Paired A/B harness and bootstrap evidence validator supplied; the first frozen selection retained baseline, so no certificate was applicable |

For the supported transformation, every necessary G2 check must pass; unknown is not pass.
Compiler success, source-order liveness and set/wait counts are insufficient individually or
collectively without the required constraint evidence. Preserve the existing historical result
as restricted candidate compile success, not complete memory-reuse verification. G2 remains a
compiler proof gate; it neither substitutes for G3 execution nor requires a successful cost
prediction. Prediction validity and coverage are reported independently and gate performance use.

`certify` validates supplied paired measurement evidence against an explicit policy for minimum
sample count, mean speedup, worst paired regression, prediction error and numerical tolerances.
It binds the exact input, candidate artifact, model, device and measurement environment. It
does not run hardware, authenticate a measurement producer, or certify unmeasured workloads.
Synthetic contract tests for this validator are not performance evidence. Automatic consumption
of certificates remains disabled pending G4; the fixed 32x32 G3 result does not certify broader workloads.

The first TileSim adapter runs actual TileOp cost functions on operations from this package.
It preserves stable buffer provenance into the legacy DSL/MIR/event/liveness path and excludes
borrowed, backing and model-temporary storage from local-slot recommendations. For the fixed static
four-stage micro, it lowers stage instances and dependencies to TileSim MIR, expands two AIV lanes,
and uses `EventEvaluator` for candidate makespan. L2L transfer cost uses the A5 `davidV100`
L0C→UB and UB→L1 bandwidth curves; pop/free are synchronization/lifecycle events. Negation and
layout mappings remain visible approximations. This complete coverage claim does not extend beyond
the fixed micro, and slot counts remain compiler-proven minima. No hand-written FA graph substitutes
for the input.

The original FA source referenced by the model repository is still a second fixture to obtain.
Generalized multi-axis/recursive/dynamic inputs and local scheduling require new capabilities,
semantic analysis and independent acceptance before use.


## Bounded compiler feedback implementation (2026-09-15)

Explicit `--mode compile` and the P=0/all-one `--mode compile_serial` baseline path emit:

- `memory_plan.json`, `schema_version=pto.compiler_memory.v1`: source buffer/slot allocations,
  materialized/reserved-unused/eliminated state, compiler-generated storage, borrowed versions
  mapped to backing slots, per-AIC/AIV address ranges and union/arena budget accounting.
- `validation_report.json`, `schema_version=pto.compiler_validation.v1`: physical/static-GM
  checks, required RAW/physical-overlap/release obligations, actual native engine and set/wait
  edges, unresolved conditions, and independent G2/G3/G4 status.
- C++, final PTO and materialization trace. A single native call writes both C++ and the final
  PTO snapshot via `--cv-costmodel-final-ir-file`; no second independent lowering supplies proof.

Both reports share input identity, candidate, schedule, native compiler binary, flags, final IR
and C++ fingerprints. The validation report binds the memory report digest; the runtime matrix
manifest binds all files, wrapper, clean source and launcher tools. Reports are read-only
compiler outputs, not fields accepted in a model Plan. Neither exchange v1 nor 2.0 meanings change.

Completion profile `a5_l2l_tile_entry_v1` is bounded to mapped FP32 operations, constant domains,
N<=each pipe capacity (no within-loop backing wrap), no local/backing overlap, and matching
installed A5 TPipe engine semantics. It verifies actual native operands against source versions,
checks reader completion before free/overwrite, and rejects cycles in the augmented completion
graph. Unknown engines, missing completion edges or unimplemented reuse return `unknown`.
Wrong versions, live clobbers, cycles or known layout overflow return `fail`.

Target library storage geometry is part of the gate. The retained 16x16 interface micro has an
8-row vector NZ tile, while installed A5 ND-to-NZ rounds row stride to 16. That device-invalid
shape is explicitly rejected. Device fixtures use 32x32 matrices and 16 rows per AIV; all Q/K/V
and output iterations are distinct. No tolerance relaxation is used to accommodate the failure.
The feedback boundary is final PTO. Bisheng/library lowering, stack spills and device execution
remain separately bound target evidence; this profile does not certify arbitrary native layouts.

The harness stops at the first runtime failure, preserves evidence and collects diagnostic
performance only after complete G3. G4, general memory-reuse search, full M01–M12 coverage and
model consumption/reevaluation remain pending; existing historical evidence is not relabeled.

Frozen runtime commit `20ae930a4` passed all 282 G3 executions (47 legal variants, three seeds,
two orders) on `ptoas-a5-52`, device 0, actual SOC `Ascend950PR_9589`. Three insufficient-slot
configurations were rejected before execution. Distinct allocation ranges do not overlap in this
board matrix: cross-buffer address reuse has compiler tests only. No G4 certificate is issued.

Diagnostic timing collected 130 measured samples plus 26 warmups on the same G3 objects.
All candidate ranges overlap their serial and P0 controls: no reliable performance ranking
is established. See the [fixed-candidate record](ptoas-costmodel-fixed-g2-g3-20260915.md) for
per-candidate results, provenance and retained raw evidence. This is calibration data, not G4.

## Frozen selection acceptance (2026-09-16)

PTOAS `170e78a2b` and TileSim `0ea5ba8f5` froze recommendations for
`basic/crossing × N=4/8` before device execution. The best P=1 candidates predicted only
0.730% (N=4) and 0.739% (N=8) improvement over P=0. Both values are below the protocol's
2% recommendation threshold, so all four workloads returned `BASELINE_RETAINED`.

PTOAS therefore emitted only A and P0, as required by the protocol. All eight artifacts passed
candidate-specific G2 and all 48 three-seed/two-order G3 executions passed on `ptoas-a5-39`,
device 0, actual SOC `Ascend950DT_9592`. The matching performance experiment contains no A/B
samples because there is no B. It does not issue a benefit result or G4 certificate. See the
[frozen selection acceptance record](ptoas-costmodel-ab-acceptance-20260916.md) for identities,
predictions, evidence boundaries and archived experiment locations.
