# ADR: preserve A5 vector fusion boundaries during automatic synchronization

- Status: Proposed
- Issue: https://github.com/hw-native-sys/PTOAS/issues/983

## Context

FusionRegionGen runs before InsertSync. Descending through its container emits
MTE2-to-V waits between tile operations and prevents their vector loops from
fusing. The original 32x128 TADD/TSUB reproducer also loses the V-to-MTE3 handoff:
InsertSync did not associate a fusion result with the buffer yielded by its body.
With the interrupted fusion, the final VPTO loses the intermediate addition.

## Decision

For automatic event synchronization, model a flat A5 fusion group consisting of
PIPE_V destination-style tile operations and pure SSA setup as one compound.
Read/write ranges from every member, including internal temporaries, remain in
that compound. Its placement anchor is the container, so incoming waits precede
it and outgoing sets follow it. Keep the existing allocator, loop compensation,
redundancy analysis and tail barrier.

Retain the logical SyncIR node count before aggregation for both emitters'
simple-function checks. A group containing several operations still requires
the existing function-completion barrier even if it becomes one compound;
ordinary single-operation functions retain their previous policy.

Column-broadcast input ranges remain identifiable through a group's member
operations. A later writer reusing those ranges requires the existing A5
PIPE_ALL drain at the group boundary. If a later member of the same group
overwrites an earlier column-broadcast input, retain per-operation anchors
for that group so the required barrier can be placed inside it. Groups without
that internal storage-reuse hazard retain atomic aggregation.

Translate the body before associating every result with its corresponding
`pto.yield` operand. This preserves allocations and views defined inside the
region and makes downstream users participate in alias analysis. Alias mapping
also applies when a region cannot be aggregated.

Retain per-operation synchronization for A3, mixed pipes, macros, calls, nested
control flow, unknown effects and locally computed multi-buffer selectors.
A local selector might not dominate a boundary flag. An internal MTE2 producer
must never have its V wait moved ahead of the whole container.

Dependency audits retain body-level compounds by default. Only the two automatic
sync emitters request aggregation. G1-self therefore checks the individual
accesses against emitted flags; it does not accept the emitter's aggregated
node as a missing/unmappable oracle anchor.

## Alternatives

Moving FusionRegionGen after InsertSync leaves flags between group members and
prevents contiguous region construction. Blindly moving flags out of any region
can put a wait before its producer. Adding global barriers sacrifices overlap
and fusion. The restricted single-pipe model avoids those changes.

## Validation

Keep the issue's original pipeline and verify both the synchronized tile IR and
the final VPTO/LLVM arithmetic chain. Standalone regressions cover all yielded
results, consecutive groups, physical reuse of an internal temporary, dynamic
loop bounds, mixed pipes, control flow, local selectors, manual flags, unified
sync and A3 same-pipe barriers. Check G2/G3 and per-operation G1-self, including
an intentionally removed output wait that must fail coverage.

A5 numerical correctness and the current-head VF A/B/C/D performance matrix
remain a hardware acceptance gate; compilation alone cannot satisfy it.
