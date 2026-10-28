# ADR: safe A5 scale-address hoisting

Status: proposed

Issue: https://github.com/hw-native-sys/PTOAS/issues/1463

## Context

A5 TMOV normalization moves `pto.tget_scale_addr` before an earlier MAT to
SCALING `pto.tmov` using the same destination. Moving the binding alone can
place its source use before its definition. Simply skipping that move is not
a complete repair: EmitC lowers the binding to `GetScaleAddr` and `TASSIGN`,
which must establish the destination address before the matched copy.

## Decision

Use MLIR dominance to recognize operands already available before the copy,
including block arguments and definitions outside the current block. For
unavailable operands, collect a same-block dependency slice before mutating IR.
Allow logical `pto.alloc_tile` handles and region-free, effect-free,
speculatable scalar computations only. Recursively check allocation addresses
and dynamic valid extents. Move the collected definitions in dependency order,
then the binding, without cloning allocations or moving the copy.

Reject an unsupported dependency with a diagnostic at the binding and notes at
the blocking definition and matched copy. A failed collection performs no
moves for that binding. Do not silently leave its address binding after the
copy. This is a bounded repair, not a general scheduling or alias analysis.

Check every operation between the matched copy and binding, including nested
regions, for uses of the destination. Reject a binding that would cross such a
use, with notes at the captured/direct use and matched copy, before collecting
or moving dependencies. Leaving the binding after the copy would not prove that
the copy used the intended address. This conservative diagnostic also applies
to direct destination uses that previously prevented matching silently. Inputs
must establish the intended address before the copy or use a separate
destination when rebinding is required.

Repeated consumers may reuse the same scale fill (issue #1691). If an identical
source/destination binding immediately precedes the matched copy, that copy
already used the required address. Keep the later binding at its original
consumer instead of hoisting it across earlier uses. Retaining that binding
also preserves rebinding after physical scaling-buffer reuse. This exception
requires both SSA handles to match; a different established source does not
justify skipping the safety checks. It performs no motion or alias analysis.

Unmatched bindings are unchanged. No dialect syntax, public API, or command-line
option changes. Existing safe normalization and non-scaling TMOV normalization
retain their behavior.

## Consequences and validation

The reported late-allocation case compiles with the definition and binding
before TMOV. Inputs requiring movement of calls, loads, or region operations
receive a specific normalization error instead of invalid SSA. SSA dominance
alone does not authorize moving a side-effecting dependency.

Regression tests cover late/early allocations, both operand sides, captured
values and block arguments, scalar dependency chains, unsupported dependencies,
unmatched bindings, repeated execution, and final EmitC binding/copy order.
Destination-use regressions cover direct calls, `scf.if`, `scf.for`, and a loop
inside an else region, including unchanged IR after rejection and the final
EmitC pipeline's diagnostic. Regions without destination captures still allow
normalization. These compiler checks do not establish hardware numerical
correctness.

Repeated-consumer regressions verify the initial binding/copy order, retention
of the later binding, repeat-pass stability, and rejection of a conflicting
established binding.
