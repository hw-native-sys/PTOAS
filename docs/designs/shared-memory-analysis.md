# Shared memory analysis

This change extracts MR379's value-range, access-region and dependence proofs
into `PTOAnalysis`. Scheduler ordering and alias policies are unchanged.
Compared with master, membar also gains MR379's bounded-loop, indexed-access
and dependence-redundancy proofs, which can remove previously conservative
barriers. This is not a behavior-preserving refactor of master. Neither
consumer owns a second copy of these proofs. The functional changes originate in
`0841b66ac` (bounded loops and symbols), `83e53e75c` (indexed accesses),
`cf12d0fad` (narrow loops; launch contract subsequently removed) and
`823a51cd4` (MoE predicates and recurrences).

## Layers and clients

| Layer | Responsibility | Clients |
| --- | --- | --- |
| `PTOValueEvolutionAnalysis` | Typed expressions; exact evolution; contextual signed ranges and cast preservation | Address analysis, access analysis, loop optimizations |
| `PTOIntegerRange` | Conservative loop domains, certified arithmetic, guarded bounds and remainder ranges | Value evolution, memory dependence |
| `PTOAddressAnalysis` | Pointer provenance, offset units and certified byte differences | Memory alias analysis, post-update transforms |
| `PTOMemoryAccessAnalysis` | Instruction footprints, masks, paired/packed accesses, indexed lanes and immutable tables | Membar and independent queries |
| `PTOMemoryAliasAnalysis` | Resolve instruction scheduling contracts and compare point accesses | VPTO scheduler |
| `PTOMemoryDependenceAnalysis` | Same-execution and loop-carried conservative overlap relations | Membar and independent queries |

The access layer reuses the IR's address and memory-distribution contracts.
The scheduler keeps its `VPTOMemoryAccess` adapter: extending its supported
footprint domain during this refactor would change scheduling. Its byte
interval comparison and typed address difference are shared queries.

Membar retains instruction order, hazard categories, SSA ordering witnesses,
existing-barrier coverage, redundancy and placement. The scheduler retains
its frontier and edge policy. The analysis library does not include either
transform and does not return barrier kinds or placement anchors.

## Proof contracts

* `getRangeAt(value, context)` returns signed bounds at that program point.
  It checks function membership and value availability. Dominating guards
  are not cached as unconditional facts about the value.
* `preservesValueAt` forwards a cast only when its entire signed input range
  is representable in the output type. Flattening a cast input additionally
  requires transitive non-wrapping arithmetic certification. Zero extension requires
  nonnegative input. Unknown is not a successful preservation proof.
* `PTOLoopDomain::maxTripCount()` bounds executions of a conservative envelope.
  `exactTripCount()` is available only for constant bounds; nonpositive steps
  are unsupported. The final IV
  increment must remain representable for an envelope to be usable.
* Negative and zero loop steps are not modeled; they conservatively retain
  dependencies, including cases that older constant-domain code could resolve.
* Domain-free point inference deliberately retains the scheduler's original
  leaf policy. Contextual inference may use guards and loop envelopes; these
  queries are not interchangeable. `getDifferenceBytes` can retain cast
  boundaries for clients whose existing proof domain requires that policy.
* `getMemoryFootprints` retains separate block-strided segments. Empty means
  unsupported, not that an operation accesses no bytes; membar aborts the
  analysis rather than dropping that operation. The descriptor overload avoids
  rebuilding access facts already checked by a consumer.
* Dependence scope must be a strict common non-loop ancestor. Invalid context
  returns Unknown, including a null-loop Unknown entry in carried queries.
  Membar retains this entry at the access and every shared loop latch.
* A nonempty conservative relation is **MayDependence**, not proof of actual
  overlap. Only an empty relation permits removing a dependency. Unknown
  provenance, wrap, unsupported expressions and unbounded loops stay unknown.
* Different SSA roots in physical UB do not establish disjoint allocations.
  Explicit memory effects and pointer provenance remain separate facts.
* Shared symbols cancel across iterations only when invariant in every
  carrying loop. The UB accumulation chain from issue33 remains dependent.

## Reuse example

```cpp
PTOValueEvolutionAnalysis values(function);
auto range = values.getRangeAt(bound16, loop);
auto faithful = values.preservesValueAt(bound16.getDefiningOp(), loop);
auto footprints = getMemoryFootprints(load, enclosingIVs);
PTOMemoryAccessOccurrence a{store, storeFootprint};
PTOMemoryAccessOccurrence b{load, loadFootprint};
auto relation = querySequentialMemoryOverlap(a, b, function, true);
auto carried = queryCarriedMemoryOverlap(a, b, function);
```

These queries do not insert or remove IR. Clients must recompute after IR
mutation; proofs tied to a program point cannot be transferred after motion.
`pto-print-address-analysis="memory-queries=true"` exposes direct queries
without requiring a vecscope or running barrier insertion.

Existing consumer tests remain in place. The short `memory_analysis.pto`
test separately covers guarded and unguarded truncation, actual truncation,
guard scope, exact/envelope distinction, IV backedge overflow, adjacent
accesses and real UB accumulation. Validation also compares frozen original
inputs through both the standalone pass and the complete VPTO pipeline.


## Review safety boundaries

An unbounded pointer loop recurrence cannot use its initializer as an address
lower bound merely because its update is usually positive. The initializer
fallback now requires the yielded pointer to be unchanged; backward, swapped
and unproved advancing pointers stay unknown. Constant bounded recurrences
retain their separate exact-coordinate proof.

PK_B16 stores the lower eight bits of each of 128 physical 16-bit lanes:
a 256-byte register has a 128-byte packed footprint. The dedicated
`vecscope_membar_pk_b16.pto` test checks adjacent stores at offsets 0/128
and overlapping stores at 0/64. This contract predates this MR; using it to
remove barriers changes membar output and is not described as global behavior
equivalence. The moving-root and INTLV defects fixed at intermediate head
`7adb49226` were introduced in this MR, not established master defects.


The scheduler retains its frontier/edge policy, context-free leaf policy,
cast boundaries and no-wrap checks. This is not a claim that every proof
result is identical: reassociating the byte-difference expression changes
which int64 intermediate overflow checks can succeed. Public queries remain
conservative when a proof fails. Reusing the caller's analysis instances is
a separate compile-time optimization; the current per-query constructors
store the function and do not traverse the whole IR.
