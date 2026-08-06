# PTOAS Unified Intra-Core Sync Allocator Design

## Intent

`--enable-unified-sync` allocates intra-core synchronization for one function
across all three hardware mechanisms from a single model, instead of committing to
a mechanism before the cost of that choice is known.

The existing passes each own one mechanism. `--enable-insert-sync` allocates event
ids and falls back to a `PIPE_ALL` barrier when a pool is exhausted.
`--enable-bufid_sync` allocates buffer-id tokens (see `docs/bufid_sync_a5_design.md`).
Neither can trade one resource against the other: when a pipe direction runs out of
event ids, insert-sync must serialize with a barrier even where buffer-id tokens are
free.

This pass makes the mechanism a decision rather than a premise. Hazards whose
direction would overflow its event pool are routed to buffer-id tokens where the
buffer is eligible; what remains is coloured into the event pools by live interval;
anything that still does not fit spills to a barrier.

The dependence analysis is **shared** with `pto-insert-sync` rather than
reimplemented. Only the assignment of mechanisms and ids differs. That is a
deliberate constraint, not a convenience: it means both passes see the same program
and the same hazards, which is what makes a differential comparison between them
meaningful rather than a comparison of two front ends.

Non-goals:

- Cross-core synchronization. `pto.sync.set` / `pto.sync.wait` carry FFTS or
  intra-block flag ids from a different id space, and are untouched.
- Replacing any existing pass. This is one more automatic-synchronization mode,
  mutually exclusive with the other four and off by default. Passing more than one is
  rejected with a diagnostic rather than resolved by precedence.
- Scheduling. The pass does not move compute ops; it only decides how the orderings
  the front end reports are enforced.

## Mechanisms

| mechanism | ops | pool | cost |
|---|---|---|---|
| event id | `pto.set_flag` / `pto.wait_flag` | 8 per `(src, dst)` pipe direction, disjoint hardware per direction | directed: orders one direction |
| buffer id | `pto.get_buf` / `pto.rls_buf` | one pool of K, shared across directions; K=0 on A3, K=32 on A5 | a ticket protocol on one buffer; co-consumers of one token serialize |
| barrier | `pto.barrier <PIPE_ALL>` | unlimited | orders every anchor pair spanning it |

Event pools are per direction and physically disjoint: id 0 in `MTE2->V` is a
different flag from id 0 in `V->MTE3`. The buffer-id pool is not -- it is one global
K, so a buffer-id token consumed anywhere reduces what is available everywhere.

A hazard always ends on some mechanism. There is no resting state in which a
hazard has been decided but carries no resource; `SyncCodegen` fails the pass if a
set/wait pair reaches emission with no event id.

## Enabling the pass

```bash
ptoas input.pto --pto-arch=a5 --enable-unified-sync -o output.cpp
```

`--enable-unified-sync` is mutually exclusive with the other four automatic modes --
`--enable-insert-sync`, `--enable-bufid_sync`, `--enable-inject-barrier-all-sync` and
`--enable-graph-sync-solver` -- and passing two is an error rather than a precedence
question. There is no compiler default: a run that passes none of them inserts no
synchronization at all, which is why every measurement in this document names its
mode explicitly.

K is derived from `--pto-arch`: 0 for a3, 32 for a5. On a3 the routing step is
therefore inert and the pass allocates events and barriers only.

Functions that already contain explicit synchronization (`pto.set_flag`,
`pto.wait_flag`, `pto.record_event`, `pto.wait_event`) are left untouched.

### Flags

Ten user-facing flags ship with the pass. The first two drive it; the rest are the
oracle described under **Oracle gates**.

| flag | purpose |
|---|---|
| `--enable-unified-sync` | run the allocator |
| `--enable-unified-sync-debug` | print the model, routing, colouring and emission reports to stderr. Off by default: the reports are verbose enough on a mid-sized kernel to bury anything a caller needs. Gate violations print either way, since they precede a failure and would otherwise leave it unexplained |
| `--dump-sync-extract` | the oracle's structured view of the emitted sync ops |
| `--dump-sync-coverage` | the happens-before edges the emitted sync induces; produces a G1 reference profile |
| `--dump-sync-counts` | kernel-body sync-op counts; produces a G4 reference profile |
| `--check-sync-ids` | gate G3, device-id legality |
| `--check-sync-interference` | gate G2, non-interference |
| `--check-sync-coverage=<file>` | gate G1, differential coverage against a reference profile |
| `--check-sync-self-coverage` | gate G1-self, absolute coverage against the dependence analysis |
| `--check-sync-count-floor=<file>` | gate G4, sync-count floor against a reference profile |

Three further flags exist for tests and triage. All three are `cl::Hidden`, so they
appear under `--help-hidden` and not under `--help`:

- `--check-sync-ids-inject-fault` feeds G3 a synthetic illegal id, which is how the
  gate is shown to fail on a fault rather than passing everything.
- `--check-sync-interference-inject-fault` appends a synthetic record pair to G2's own
  list, in two shapes: a compensation record colliding with an unrelated hazard, and a
  pair colliding only on a non-first id. Neither shape can be written as input IR,
  because both exist only before codegen.
- `--unified-sync-force-mechanism` stamps hazard group 0 off its type-derived
  mechanism onto buffer-id, so the value and its plumbing are exercised on kernels the
  router declines. It accepts `bufid` and `bufid-spill`; any other value is an error.
  Note that stamping is not routing -- routing is a per-buffer decision this flag does
  not make -- so `bufid` deliberately ends in a codegen refusal.

## Pass placement

The pass runs as a nested function pass in the same slot as `pto-insert-sync`,
selected by an `else if` chain over the five automatic modes. It therefore sees the
same IR the incumbent sees, which is a precondition for the differential gates.

The slot itself is constrained: synchronization runs BEFORE
`PTOResolveBufferSelect`, so it still sees per-use `pto.multi_tile_get` operations and
can keep their slot identity for alias and event-id analysis. Moving it later would
erase the distinction rotation depends on.

## Flow

The order of the four steps is the design, not an implementation detail.

1. **Settle every operation before any id exists.** Loop-carried compensation
   synthesis runs here: a hazard whose consumer precedes its producer in program
   order needs a head-set before the loop and a tail-wait after it, or the first
   iteration waits on a flag nobody set. An id assigned to an op that later moves or
   disappears is worse than no id, so nothing is allocated until the op set is final.
2. **Route buffers whose direction would overflow.** For each buffer, if its
   direction's peak concurrent demand exceeds the event pool and the buffer is
   eligible, every hazard on that buffer moves to buffer-id. Routing is
   all-or-nothing per hazard: a hazard that took a token on one side and an event id
   on the other would be split across two mechanisms, and a split pair hangs.
3. **Colour the rest into the event pools.** Per direction, hazards are coloured by
   live interval, so two hazards whose intervals are disjoint share an id. Where the
   pool cannot hold the peak, the hazard spills to a `PIPE_ALL` barrier.
4. **Emit, then check.** Emission reuses `SyncCodegen` for events and barriers and
   the `bufid_sync` codegen for tokens. The oracle gates then run over the result,
   and a gate violation fails the pass.

## Data model

One `Hazard` per sync group -- the granularity at which the both-halves invariant
must hold. Each carries:

- the `(src, dst)` pipe direction;
- a live interval in SyncIR index space, from the producer's index to the
  consumer's;
- whether a reverse WAR exists on the same buffer, which is what makes buffer-id
  more expensive than an event for that buffer: the token protocol orders the
  reverse direction too, so co-consumers serialize;
- the mechanism, once decided.

Buffer clustering is reused from `BufidSyncAnalysis`: one virtual buffer id per clique
of aliasing tiles, computed on the freshly translated SyncIR before any sync op is
inserted.

Hazards are joined to those clusters with `MemAlias`, the same relation
`BufidSyncAnalysis` itself uses to decide two tiles are the same memory. Three
plausible keys are all wrong. The root allocation is far coarser than a buffer, so it
conflates tiles that do not alias. A `(scope, base address, size)` tuple is only a
BUCKET inside `BufidSyncAnalysis`, confirmed with `MemAlias` afterwards, and it
degenerates outright at this point in the pipeline: base addresses are not yet
assigned, so the address collapses to zero and distinct clusters share one key.
Pointer identity is exact but incomplete, because the tile dedup drops the losing
`BaseMemInfo` objects. The implementation therefore tries pointer identity first and
falls back to `MemAlias`.

## Oracle gates

The allocator is written from scratch, so it cannot be validated by byte-comparing its
output against the incumbent -- and in fact does not reproduce it op for op, though its
counts match on every kernel in the sample corpus (see **Equivalence to the
incumbent**). Correctness is
defined instead by gates over the extracted sync. Two are differential -- they need a
reference profile from a second compiler run -- and four are absolute.

| gate | kind | asserts |
|---|---|---|
| G1 | differential | every ordering the reference establishes is also established here |
| G1-self | absolute | every dependency `DepBetween` reports is ordered by some emitted sync |
| G2 | absolute | no two overlapping intervals share a sync id; no leaked or unclosed id |
| G3 | absolute | every id is legal for its direction, op and arch |
| G4 | differential | sync-op counts do not regress, per class: total, barriers, `PIPE_ALL` barriers |
| macro reservation | absolute | no compiler hazard takes an id a macro's library implementation uses internally |

The last has no flag: it runs on every `--enable-unified-sync` compilation and fails
the pass, because no other gate can see the fault. G3 checks that an id is in range,
not that something reserved it, and G2 cannot see an id consumed inside a library
call.

G1-self is the load-bearing one, because it is the only gate whose green result
means something in its own right. G1 green means "matches the incumbent" and
its strength is bounded by that reference. G1-self builds its expected set by
running the translator and calling `MemoryDependentAnalyzer::DepBetween` directly
over the resulting def/use vectors. Nothing in it reads the incumbent's decisions.
It deliberately does not call `IsMemInfoHasDependency`, which layers sync policy on
top of `DepBetween`: policy is what the gate audits, so policy cannot be part of
the oracle.

Running a differential is two runs and a file:

```bash
ptoas k.pto --pto-arch=a5 --pto-level=level3 --enable-insert-sync  --dump-sync-coverage 2> ref.cov > /dev/null
ptoas k.pto --pto-arch=a5 --pto-level=level3 --enable-unified-sync --check-sync-coverage=ref.cov
```

Both reference dumps go to **stderr**, which matters: redirecting stderr away
discards the profile and leaves a gate that silently checks nothing.

## Known limitations

Everything currently known to be incomplete, in one place. Each entry states
whether it is a gap in the allocator or in the gate that checks it, and whether the
incumbent shares it.

### 1. A larger macro kernel could be pushed into the spill path

Macro ops are now supported: the (direction, id) pairs a macro's library
implementation consumes internally are reserved over the call's span, padded one
SyncIR index each side, and the colourer treats them as occupancy.

What remains is a second-order effect. A reservation shrinks effective pool capacity,
so a kernel that fit in eight ids can be pushed into the spill path -- and on A3,
where K is 0, no buffer-id routing can absorb that. No macro kernel in the sample
corpus overflows, spills or routes, because those kernels are small. One with more
concurrently live hazards per direction could be pushed over, which is why the spill
path remains and remains tested.

### 2. G2 does not validate rotating ids from the command line

Two gaps, both about rotating (multi-id) hazards. Neither is a live defect, because
the colourer hands out disjoint ids per direction -- but that is a property of the
current colourer, not an invariant.

- The standalone `--check-sync-interference` gate does not validate rotating event
  ids; only the in-pass check does. The CLI gate reads the emitted-IR view, where
  `set_flag_dyn` / `wait_flag_dyn` carry a runtime id recorded as `-1`, so those ops
  never enter its id check. The ids are statically recoverable -- the emitted selector
  is an N-way `arith.select` chain over constants -- so closing this is mechanical.
- Order-dependent conflicts between two hazards within one SyncIR element are
  unresolved for rotating ops in either view. The syncops view orders by `irIndex`,
  which is element-granular; the emitted-IR view has true program order but cannot
  see the rotating ops. For rotating hazards neither view is the authority on order.

### 3. Same-pipe dependencies are reported uncovered, identically in both modes

Summed over `test/samples`, across 124 functions producing a gate line, G1-self reports
433 uncovered dependencies: 429 same-pipe, 4 cross-pipe. **The totals are identical
under `--enable-insert-sync`** -- 433, 429 and 4 again -- so this is inherited, not
introduced by this pass. The equality is the load-bearing part; the magnitudes track
the size of the sample corpus and move when it grows.

They are dominated by A3 `PIPE_V -> PIPE_V` pairs between vector ops: on
`gqa_attention_block` all 61 uncovered dependencies are of that shape, with no
cross-pipe pair among them. Neither pass emits sync for them, and whether they need any
is unresolved. The gate does not distinguish a same-buffer pair from a different-buffer
one, so resolving them means classifying them by allocation identity first.

### 4. G1-self never enumerates self-pairs

The pair loop is `for j = i + 1`, so a compound is never paired with itself and a
**same-op loop-carried hazard** -- op x in iteration k against the same op in k+1 -- is
never asked about. Such a pair is absent from every bucket alike: not in
`dependencies`, so not in `covered`, `uncovered`, `unmappable`, `armExcluded` or
`archGuaranteed`. It can neither fail nor pass; it is outside the question.

Most such pairs are writes to one tile buffer, which program order already orders. The
case that matters is a `tstore` whose iterations write one loop-invariant GM region --
an `MTE3 -> MTE3` carried WAW this gate does not excuse. On `overflow_writeback` it is
ordered, but transitively and by accident of that kernel's shape, through the carried
`MTE3 -> V` pair composed with the in-iteration `V -> MTE3` pair. A kernel that stores
without that round trip would have the same hazard, no chain, and this gate would still
say nothing.

The count is not reported by the gate, because a pair outside the question is in no
bucket to be counted; establishing it needs instrumentation the gate does not carry.

Closing it is a widening, not a bug fix, and it is not free: same-op carried WAW on a
tile buffer is the common case, so enumerating self-pairs needs a same-tile-buffer
exemption the check does not have. Without one, every `tload` writing its tile each
iteration would report uncovered.

### 5. G1 is one-sided when the reference emits more barriers

A `PIPE_ALL` barrier at slot `s` contributes `s * (n - s)` forward orderings by
itself, so a reference that discharges hazards with barriers has an edge set denser
by construction than one built from events and tokens. A candidate needing fewer
barriers therefore reports missing orderings for a *reduction* in
over-synchronization, and the superset requirement cannot distinguish that from a
real regression.

This is not hypothetical, and it is pinned rather than described. On
`overflow_writeback` (a5, level3) G1 reports 665 missing orderings while this run's
edge set is a strict subset of the reference's -- the reverse comparison reports zero
violations, so nothing is established here that the reference lacks. The reference
emits 8 in-body `PIPE_ALL` barriers where this pass emits none, because routing keeps
the event pool from overflowing, and those account for the entire gap. G4 reads the
same fact as an improvement: 55 sync ops against 63. Both directions of the comparison
are pinned in `sync_forced_overflow.pto`.

The verdict is deliberately left failing, and the gate draws no conclusion from the
barrier counts it prints. The predicate "the reference emits more barriers" also
holds when the candidate is simply broken -- an empty allocation shows it too, since
the reference carries the auto-sync tail barrier and an empty one does not -- so
keying the verdict on it, or even explaining it away at the failure site, would
excuse the exact fault class the gate exists to catch. When the barrier counts
differ, G1-self is the gate to read.

The violation count is also not a count of orderings lost: `checkCoverageSuperset`
compares raw edge sets with no transitive closure, unlike G1-self, and compares
forward and carried sets separately. Both effects inflate it. Treat it as a
tripwire, not a measurement.

### 6. The A5 PIPE_V exemption is keyed on the pipe alone

G1-self treats an A5 `PIPE_V -> PIPE_V` dependency that emitted sync did not cover
as satisfied by the target rather than as a gap. What it rests on is that neither pass
emits sync for such a pair on any A5 kernel measured, so demanding one would make this
gate report every such pair on every A5 kernel it sees. That is an observation about the
kernels available, not a target guarantee this document can cite.

The test asks only whether both endpoints sit on `PIPE_V`; it does not distinguish a
same-allocation pair from a cross-allocation one. A substantial share of the pairs it
excuses span two distinct allocations that overlap in memory, which is how aliasing
views are expressed once addresses are assigned, so the exemption is broader than a
per-allocation test would be.

It is deliberately not narrowed: the incumbent treats those cross-allocation pairs
identically, excusing all of them and covering none, so the exemption models the
behaviour of both allocators rather than a divergence between them. Narrowing it would
move a large number of corpus dependencies from excused to uncovered in both modes at
once, changing what the gate reports without changing what either pass emits.

### 7. Rotation depth greater than one is barely exercised

A rotating hazard uses `pto.alloc_multi_tile` plus `multi_tile_get`, emitting
`set_flag_dyn` / `wait_flag_dyn` with an N-way `arith.select` chain over constant
ids indexed by `slot % N`. G1-self models these at distance N, recovered from the
selector's own modulus.

In-tree coverage is thin. Most files using `multi_tile_get` or `alloc_multi_tile`
declare a3, and most do not compile at `--pto-level=level3`, so the rotating path is
predominantly an A3, pre-level3 shape.

The interaction with buffer-id routing -- which exists only where K > 0, that is on A5
-- is reached but not exercised. `sync_event_rotation.pto` is an A5 rotating kernel, so
the router does examine its buffer: the rotating hazard reports `revwar=yes`, which
prices a token at parity with an event rather than above it, and routing declines it
only for want of overflow. A rotating kernel that also overflowed would be the first
input to combine the two, and none exists in the tree.

### 8. Buffer-id rotation is not expressible in this ISA

Rotation requires selecting a resource id at runtime. Events support it: the dialect
has `SetFlagDynOp` and `WaitFlagDynOp` alongside the immediate forms. Buffer-id does
not, and this is a property of the ISA rather than a missing feature of the pass:
`pto.get_buf` declares `buf_id` as an `I32Attr` -- a compile-time attribute with no
operand form -- there is no `GetBufDynOp`, and the only intrinsics the emitters
produce are `llvm.hivm.GET.BUFI.mode` and `llvm.hivm.RLS.BUFI.mode`, both immediate,
with no register counterpart.

Buffer-id is therefore a distance-one mechanism here: no per-slot rotation of a token
is expressible.

What this does NOT establish is that a rotating hazard must never be routed. A token is
per aliasing CLIQUE, not per slot -- a multi-slot buffer's slots fall in one cluster, so
one token would cover the whole rotation rather than needing one per slot. Whether that
is sound is unresolved, and it is not currently decided by anything: the router has no
test on rotation depth, and the case is unreached only because no rotating kernel
overflows. If one ever does, this is the question to settle before trusting the
result.

### 9. The A3 same-pipe barrier population is not reducible in-house

A same-pipe hazard cannot use an event, because an event names two distinct pipes.
It is therefore resolved by a barrier or by a target guarantee. On A5 the guarantee
applies to `PIPE_V` (limitation 6). On A3 nothing is dropped, so the barriers are
emitted, and both allocators emit the same ones -- `CanPrunePipeVBarrier` sits in
`InsertSyncAnalysis`, which both pipelines run.

That prune accepts a `PIPE_V -> PIPE_V` pair only when it did not arrive from the
loop back-edge phase, both endpoints are `PIPE_V`, every dependency pair describes
the exact same access, that access is provably contiguous, and the producer's repeat
is at least `kPipeVPruneMinRepeat` (16). `repeat` is derived from the access shape as
`ceil(validElems / (256 / elementBytes))` -- the number of 256-byte vector-register
passes the producer takes -- not from any loop trip count.

The population is substantial and dominated by `PIPE_V`. Across the a3 sample kernels
that emit any same-pipe barrier -- 49 of them -- there are 267 in total: 208 `PIPE_V`,
42 `PIPE_M`, 16 `PIPE_MTE3` and one `PIPE_MTE1`. Most `PIPE_V -> PIPE_V` candidates the
predicate examines are refused rather than pruned, and the refusals divide between not
being the exact same access, arriving from the back-edge phase, and a producer repeat
below the threshold. These counts scale with the sample corpus and are cited for
magnitude, not as fixed values.

What the population costs, as a bound: deleting the same-pipe barriers from emitted
C++ without replacing them takes rmsnorm a3 from 16.10 us to 14.93 us and
rope_kv_cache a3 from 5.57 us to 4.38 us. These figures bound what the barriers cost.
They are not a statement that any particular barrier is unnecessary.

Three things would let the question be settled. None is obtainable from this
repository:

- **The content of issue 646.** `test/lit/pto/issue646_pipev_repeat_prune.pto` pins
  the threshold's behaviour -- a producer repeat of 15 keeps the barrier, 16 prunes it
  -- and also pins keep-cases for a non-contiguous access and for a shared temporary.
  The commit that introduced the prune and that test carries a subject line and no
  body. The issue is in an external tracker and is referenced nowhere in this tree.
- **A definition of "interval".** The vector op references give per-op A2/A3 figures
  of the form "throughput: 2 cycles/repeat (f32), 4 cycles/repeat (f16); interval: 18
  cycles". The term is not defined in those documents nor in the cost-model
  reference, so the figure cannot be converted into a required repeat.
- **Hardware, or a simulator that models the hazard.** See limitation 10.

### 10. The simulator detects cross-pipe sync removal, not same-pipe barrier removal

An output comparison is only evidence if the comparison is capable of failing, so both
directions were established with control experiments rather than assumed.

**It detects cross-pipe removal, down to a single pair.** On `post_rmsnorm` a3 the
`PIPE_MTE2 -> PIPE_V` pair that separates a `TLOAD` from the `TMUL` reading that
tile is the only op ordering them. Removing just that pair changes the stored result
(output digest `5b079fd5` to `e657a1cd`); removing all six pairs on that direction
changes it again (`327898c6`).

**It does not detect same-pipe barrier removal.** A purpose-built kernel with a
same-buffer `PIPE_V -> PIPE_V` RAW chain at producer repeat 15 -- `TADD` writing a
tile, `TROWEXPANDDIV` reading it, with the compiler-inserted `pipe_barrier(PIPE_V)`
as the only separator between them -- produces an identical result with and without
that barrier (`74d57487` either way), on output that is non-degenerate and
input-dependent. `issue646_pipev_repeat_prune` asserts the barrier is required at
repeat 15, so this is a removal the tree treats as unsafe, and it is invisible here.

This is not merely a serialised pipe model: in a collected trace, five of six
consecutive vector-pipe operations overlap in time, so intra-pipe concurrency is
represented in the timing model.

**Consequence.** An output comparison can support a claim about cross-pipe
synchronisation. It cannot support one about same-pipe barriers in either direction:
agreement is not evidence of safety, and it cannot be used to argue that a barrier is
removable. A same-pipe claim needs an ordering argument, not a simulator run.

Two further properties of the harness constrain any such comparison:

- The generated `golden.py` fills most input tensors with zeros. Comparing outputs
  across variants on all-zero input is vacuous, and poisoning an output buffer with
  zeros is a no-op when the true output is also zeros. Randomised inputs and a
  non-zero poison pattern are both needed for the comparison to mean anything.
- Some kernels cannot serve as an oracle at all. `Qwen3DecodeA3/rmsnorm.pto` at
  `--pto-level=level3` does write its output buffer -- a `0xFF` poison is cleared --
  but writes all zeros whatever the input.

## Testing

Eight lit files drive `--enable-unified-sync`, across 34 RUN lines that invoke it.

| file | pins |
|---|---|
| `sync_unified_model.pto` | the data model, the chosen ids, and that hazards on different buffers report DIFFERENT clusters -- the check that separates the `MemAlias` join from a degenerate key |
| `sync_mechanism_field.pto` | that mechanism is derived from type rather than defaulted flat, that it is stamped on both halves of a hazard, and that the spill drops type and mechanism together |
| `sync_extract_oracle_views.pto` | both extractor renderings, their field names and their program order; the only test that would notice a renamed field |
| `sync_forced_overflow.pto` | forced event-pool overflow as a dial, the routing predicate discriminating written-back from forward-only, and both directions of G1's barrier-delta comparison |
| `sync_overflow_id_gap.pto` | buffer-id routing ABSORBING an overflow, so nothing spills, and that no hazard reaches codegen without a realised mechanism |
| `sync_unified_macro_pinning.pto` | that a macro's library-internal event ids are reserved before colouring, that the emitted id matches the incumbent's, and separately that the reservation was actually seeded |
| `sync_unified_compensation_g2.pto` | that the G2 compensation exemption scopes to a hazard's own compensation pair and no further |
| `sync_event_rotation.pto` | rotating pairs at depth 2, that both ids are primed ahead of the loop, and that G1-self models them at distance N |

Each of the eight has been shown to fail when the behaviour it pins is deliberately
broken, so none of them passes vacuously.

The gate fixtures are separate and do not drive the allocator:
`sync_oracle_self_validation.pto` and the `sync_gate_g*.pto` family pin that each gate
fails on an injected fault.

No kernel in the sample corpus overflows an event pool -- per-direction peak demand
stays inside the pool of 8 on every one measured -- so the overflow, routing and spill paths would be unreachable
without synthetic inputs that overflow on purpose. That is what `sync_forced_overflow.pto`
exists for, and why it pins a non-overflowing control alongside the overflowing case: a
synthetic that always overflows could be broken rather than saturated.

## Equivalence to the incumbent

Three claims of different strength, kept apart deliberately. Op-count identity does not
entail cycle identity, and the cycle measurements below show a case where it does not.

### Proven: sync-op count parity across the corpus

The allocator replaces only the assignment of mechanisms and ids; the dependence
analysis, hoisting and redundancy pruning ahead of it are shared. Comparing emitted PTO
IR between `--enable-insert-sync` and `--enable-unified-sync`, at the highest level each
kernel compiles at, over the sample corpus: of 130 kernels, 115 emit under both modes --
56 byte-identical, 59 differing, 15 emitting under neither. No kernel fails under the
unified allocator that succeeds under the incumbent.

Every one of the 59 differences is at an IDENTICAL per-kernel sync-op count: 2242 ops on
each side across those 59, and 2603 across all 115 that emit -- a delta of zero either
way. The differences are ordering, not quantity. This holds for A5 as well as A3: all 21
A5 kernels compile at level3 under both modes, 20 differing and 1 identical, every one at
an equal count.

### Measured: two A5 kernels, in the cycle domain

Ordering is not free, so op-count parity is not a cycle result. Two A5 kernels were
measured on the cycle-accurate model, per-core latency, with a determinism control:

| kernel | insert-sync | unified-sync | delta |
|---|---|---|---|
| `rmsnorm` (vector) | 18.35 us | 18.38 us | +0.16% |
| `qwen3_decode_incore_10` (cube) | 82.15 us | 82.15 us | 0.00% |

The control matters: repeating the `rmsnorm` insert-sync run reproduced 18.35 us with a
bit-identical instruction-cycle sum, so +0.16% is a signal rather than run variance. Only
the SoC tick counter moves between repeats, and it is not used here.

Note that the two available metrics disagree in direction on `rmsnorm`: unified's summed
instruction cycles are LOWER by 3.2% while its per-core latency is marginally higher. The
trace span is the metric to trust -- a schedule can retire fewer cycles in total and still
finish later -- and quoting the cycle sum instead would turn a small regression into an
apparent win.

### Not claimed: cycle neutrality as a general property

Two kernels are not the corpus. What the measurements support is that reordering at equal
op count costs little on the kernels tested -- nothing worse than a fraction of a percent,
in one case a regression and in the other nothing. They do not establish that the
allocator is cycle-neutral in general, and no claim here depends on it being so: the
allocator's purpose is to make the mechanism a decision rather than a premise, and on the
production corpus it declines to route because no kernel exhausts a pool.
