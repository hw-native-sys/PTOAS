# Tile provenance and descriptor capture

`TileProvenance.h` provides the shared, read-only source graph for tile
verification, view lowering, memory planning, semantic alias checks and
synchronization. It separates two properties:

- Payload storage aliases: a selected tile may access either branch's storage.
- Descriptor identity: valid extents belong to a particular descriptor object.

An allocation, declaration or frontend pipe pop is a local descriptor root.
Pipe lowering materializes the FIFO-backed pop as a declaration plus TPOP;
it does not allocate another payload. Reshape, tile bitcast, subview
and descriptor copy introduce payload source edges. An `scf.if` result has two
edges, matched by result index, carrying the condition and branch polarity.
`arith.select` has the same source relation. Function arguments, calls and
unrecognized operations are explicitly unknown.

Origin queries use a visited worklist. `scf.for` iter arguments/results include
both the initial value and yielded value, including the zero-trip path.
`scf.while` before arguments include initial/backedge values; after arguments
and results follow condition operands. This computes finite storage roots;
it does not claim a finite offset for a recurrent subview. Unknown origins
remain incomplete, and must never imply an empty memory access set or NoAlias.
The mutation verifier deliberately does not enable loop-carried descriptors.

## Descriptor ownership and capture time

Native Tile results capture descriptor values at their branch yield.
Normalization inserts `pto.copy_tile_descriptor` immediately before each
`scf.yield`. This includes static valid shapes: address binding and opaque
library metadata can still change. Capture does not depend on a function
containing an explicit `set_validshape` or on an opcode mutation whitelist:

```mlir
%selected = scf.if %condition -> (!T) {
  pto.set_validshape %a, %rows, %cols : !T
  %captured = pto.copy_tile_descriptor %a : !T
  scf.yield %captured : !T
} else {
  %captured = pto.copy_tile_descriptor %b : !T
  scf.yield %captured : !T
}
```

The copy reads current metadata and allocates a fresh descriptor identity on
`pto.tile_descriptor`, without allocating UB storage. Tile reshape, bitcast
and subview results also allocate descriptor identity. This prevents
CSE from merging separately mutable descriptors. Numeric bitcasts retain
empty memory effects. Captures cannot be speculated before branch updates.
The copy also implements `ViewLikeOpInterface` so generic MLIR alias consumers
retain its payload alias, independently of the descriptor resource allocation.

During final EmitC conversion a sole yield/assignment use fuses the capture
into the SCF result variable's assignment:

```cpp
Tile<...> selected;
if (condition) {
  // Producer operations and metadata updates precede the capture.
  selected = a;
} else {
  selected = b;
}
selected.SetValidShape(rows, cols);
```

This uses ordinary manual-address Tile value assignment. It does not alias
`selected` by reference to `a` or `b`, nor reconstruct another descriptor after
the if. If constant folding removes the if, a remaining observable capture
lowers to a value copy. General standalone captures likewise retain value-copy
semantics. The normalization is idempotent, including same-source yields and
multiple result positions. Tile `arith.select` uses the same capture semantics.
The final C++ compiler can optimize copies whose fields are unobservable;
generic SCF optimization must first preserve the descriptor value boundary.

## Views, ranges and synchronization

Subview address materialization runs after planning and sync insertion. A
selected local descriptor provides its runtime data address, and offsets use
the parent's physical row/column strides and storage element width. The
existing compact, boxed and row-plus-one layout rules remain authoritative.
Analysis does not choose one of a phi's roots as its runtime address.

The memory planner propagates payload roots through the shared view edges;
region branch entry/yield propagation and loop liveness remain in its existing
fixed-point model. Descriptor copies are aliases, never allocation candidates.
Escaping branch storage stays live until the selected value's final use.

Semantic no-alias diagnostics compare guarded alternatives, composing view
offsets and strides along each path. Opposite polarities of the same condition
are excluded. Unknown and recurrent ranges are not used to assert a proven
overlap. Bounded path enumeration limits diagnostic work; missing alternatives
do not constitute a proof of disjointness.

Both normal and unified sync use the existing regional sync graph. If results
collect both source alternatives from the shared graph, and subviews retain
the existing range refinement. An unmapped tile alias becomes an unknown range
in its address space instead of disappearing from the access map. Physical
address overlap checks remain necessary even for distinct allocation roots,
because the memory planner can reuse storage.

May-alias storage alternatives are joined by union. An already-synchronized
fact can escape a branch only when both paths establish it. Descriptor copies
move metadata only; they do not replace producer/consumer dependencies.
For the load/abs/store examples, each executable path must preserve MTE2-to-V
ordering before `TABS`, and V-to-MTE3 ordering before `TSTORE`. Allocation
inside versus outside the branches changes storage lifetime, not that contract.
Backend scalar metadata synchronization rules continue to apply.

## Validation and boundaries

The regressions cover branch capture after valid updates, independent reshape,
bitcast and subview descriptors under CSE, same-source/nested/multiple results,
constant conditions, valid reads around consumers and updates, storage
liveness, selected subviews with nonzero offsets, reshape/bitcast combinations,
and allocation inside/outside branches under normal/unified sync.

A2/A3/A5 EmitC level2/level3 code generation is covered. The host descriptor
model executes generated C++ to check selected addresses, metadata independence
and writes to the selected payload. It is not an NPU instruction or hardware
event simulator. This does not establish device numerical correctness.

Native selected-descriptor mutation remains EmitC-only; VPTO produces a
specific diagnostic. No `__PTO_AUTO__` descriptor copy support is added: that
Tile ABI disables assignment and needs a separate field-projection lowering.
