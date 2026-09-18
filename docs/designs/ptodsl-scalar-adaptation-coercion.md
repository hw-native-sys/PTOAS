# PTODSL Scalar Adaptation and Coercion Design

## Background

PTODSL has several frontend paths that accept authored scalar values and lower
them to MLIR scalar operands:

- runtime scalar arithmetic and comparisons;
- runtime scalar bitwise operators;
- loop bounds and loop steps;
- sync event IDs;
- pointer and tile offsets;
- scalar load/store values.

These paths all need the same basic decisions: identify whether an authored
value is an `index`, integer, or floating-point scalar; materialize Python
literals; convert integer values to `index` where the target semantic is an
index; and adapt fixed-width integer values without losing signedness metadata.

Today those rules are duplicated across multiple modules:

- `ptodsl/_runtime_scalar_ops.py`
- `ptodsl/_scalar_coercion.py`
- `ptodsl/_runtime_index_ops.py`
- `ptodsl/_ops.py`

Issue #794 tracks the need to make the rules shared rather than fixing each
new `index` / integer mixed case at the individual operation emitter.

## Goals

- Introduce one common scalar adaptation module for PTODSL frontend lowering.
- Preserve current user-visible behavior while moving common code.
- Make mixed `index` / integer behavior explicit by semantic target:
  - "prefer index" for loop bounds, event IDs, pointer offsets, and authored
    operations whose result stays in the index domain;
  - "prefer wider integer" for fixed-width integer arithmetic;
  - "explicit target type" for scalar stores and API parameters with a known
    result type.
- Keep operation-specific emitters focused on operation lowering, not type
  reconciliation details.
- Keep `index`/integer conversion in the existing `pto.cast` frontend surface;
  represent it as `pto.index_cast` until generic PTO operations are legalized.

## Non-Goals

- Do not change the semantics of ordinary numeric conversions.
- Do not redesign signed/unsigned semantics.
- Do not introduce a second public index-cast API.
- Do not change AST rewrite behavior.
- Do not merge tile-template tracing's private `_Value(type_text)` prototype
  model into this helper. That path does not operate on the authored
  surface/MLIR scalar values addressed by issue #794, and sharing this module
  there would require a separate tile-template tracing design change.

## Proposed Module

Add:

```text
ptodsl/ptodsl/_scalar_adaptation.py
```

The module operates on Python literals or already-unwrapped MLIR SSA values. It
must not import `_surface_values`, because `_surface_values` imports runtime
scalar operators and would otherwise create a cycle.

Initial helpers:

```python
classify_runtime_scalar_type(type_obj) -> "index" | "integer" | "float"
is_mlir_value(value) -> bool
materialize_scalar_literal(value, target_type, *, context)
coerce_scalar_value_to_type(value, target_type, *, context)
coerce_runtime_index_value(value, *, context)
coerce_runtime_integer_value(value, target_type, *, context)
coerce_runtime_i1_value(value, *, context)
normalize_runtime_binary_operands(lhs, rhs)
coerce_integer_like(value, target_type)
```

The names intentionally match existing internal vocabulary so migration stays
mechanical and reviewable.

## Semantics

### Explicit Target Type

APIs such as scalar store know their destination element type. They should call
`coerce_scalar_value_to_type(value, target_type, context=...)`.

Rules:

- Python literals materialize directly as the target type.
- `integer -> index` uses `pto.index_cast` with signedness derived from the
  integer source.
- `index -> integer` uses `pto.index_cast` with signedness derived from the
  integer destination, then restores authored integer signedness where needed.
- `integer -> integer` extends, truncates, or strips/restores signedness using
  the existing width rules.
- `float -> float` extends or truncates.
- `float -> index/integer` and `index/integer -> float` remain invalid unless a
  future API explicitly asks for such a conversion.

The public `pto.cast(value, dtype)` interface also accepts the
`index <-> integer` pair. It emits `pto.index_cast` with signedness derived
from the integer side; later legalization selects `arith.index_cast` or
`arith.index_castui`. Numeric conversion attributes do not apply to this type
bridge. APIs whose semantic
target is already `index` (such as pointer offsets, loop bounds, and memory
access offsets) adapt integer runtime values automatically.

### Prefer Index

Loop bounds, pointer offsets, tile offsets, and dynamic sync event IDs are
semantically index values. They should call `coerce_runtime_index_value(...)`.

Rules:

- Python `int` materializes as `index`.
- runtime `index` is kept as-is.
- runtime integer is cast to `index`.
- Python `bool`, floating-point literals, and runtime floating-point scalars
  are rejected.
- Surface index expression helpers normalize intermediate operands before
  emitting `arith.addi` or `arith.muli`, so expressions such as tile slices with
  integer runtime offsets are converted to the index domain before arithmetic.

### Fixed-Width Integer Targets

Micro-op parameters such as mask scalar operands, DMA burst sizes, MAD
dimensions, and other hardware integer operands have a fixed integer target
type. Their local wrappers keep operation-specific diagnostics and names, but
delegate integer/index/literal adaptation to
`coerce_runtime_integer_value(value, target_type, context=...)`.

Rules:

- Python `int` materializes directly as the target integer type.
- Python `bool` is rejected for normal integer operands.
- runtime `index` casts to the target integer type with `pto.index_cast`.
- runtime integer values extend, truncate, or strip/restore signedness using
  the shared integer adaptation helper.
- runtime floating-point values are rejected.
- `i1` operands use `coerce_runtime_i1_value(...)`, preserving the historical
  special case that Python `bool` and Python `0`/`1` are accepted.

### Runtime Binary Operators

Runtime scalar binary operators should call
`normalize_runtime_binary_operands(lhs, rhs)`.

Rules:

- At least one operand must be a traced runtime scalar.
- Python literals materialize against the other operand's type.
- matching operand types stay unchanged.
- `index` mixed with integer casts the integer operand to `index`.
- integer mixed with integer adapts both sides to the wider integer type.
- floating-point mixed with floating-point requires matching type for now.
- unsupported mixed categories raise a context-rich `TypeError`.

Bitwise operators benefit from the same normalization. After issue #484,
`index & 1` lowers directly to `arith.andi : index`, not through a fixed-width
integer cast.

## Migration Plan

1. Add `_scalar_adaptation.py` and move common logic from
   `_runtime_scalar_ops.py` into it.
2. Update `_runtime_scalar_ops.py` to import:
   - `classify_runtime_scalar_type`
   - `normalize_runtime_binary_operands`
   - `coerce_integer_like`
3. Update `_scalar_coercion.py` to become a thin surface-aware wrapper:
   - unwrap authored values;
   - delegate to `coerce_scalar_value_to_type`;
   - re-export `materialize_scalar_literal`.
4. Update `_runtime_index_ops.py` to delegate to
   `coerce_runtime_index_value`.
5. Update `_ops.py`:
   - import `classify_runtime_scalar_type` from `_scalar_adaptation`;
   - make `_coerce_index(...)` delegate to `coerce_runtime_index_value`;
   - make `_coerce_i16`, `_coerce_i32`, `_coerce_i64`, and `_coerce_i1`
     delegate to shared fixed-width integer coercion helpers.
6. Update `_surface_values.py`:
   - normalize surface index expression inputs through
     `coerce_runtime_index_value`;
   - keep static Python integer expression folding unchanged.

This keeps operation-specific wrapper names stable while moving duplicated
runtime scalar rules into one place.

## Testing Strategy

Use behavior tests rather than only helper tests:

- `test_jit_compile.py`
  - runtime scalar arithmetic remains unchanged;
  - `index` / integer mixed bitwise from issue #484 still lowers to dynamic
    sync event IDs;
  - loop bounds and pointer offsets still accept integer runtime scalars.
  - fixed-width integer parameters accept runtime index values through shared
    adaptation.
  - surface tile slice indices accept runtime integer values through shared
    index adaptation.
- `test_jit_diagnostics.py`
  - bool values remain rejected for runtime scalar operators and index-like
    contexts;
  - floating-point values remain rejected for index-like contexts;
  - floating-point bitwise remains rejected.
- `test_docs_as_test.py`
  - user guide examples continue to compile.
- `test_ptoas_frontend_verify.py`
  - generated MLIR continues to pass PTOAS frontend verification.

## Rollout

This change should land after issue #484 is merged, because #484 adds the
concrete user-facing `index` bitwise behavior and tests. The scalar adaptation
refactor should be reviewed as a separate PR so that behavior changes and
mechanical code movement are easy to distinguish.
