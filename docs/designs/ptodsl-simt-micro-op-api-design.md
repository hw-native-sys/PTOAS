# PTO-DSL SIMT Micro-op API Design

## 1. Scope

This document records the PTO-DSL frontend design plan for the SIMT micro-op
surface that is already supported by VPTO on `main`.

The design is intentionally frontend-first:

- expose Python PTO-DSL wrappers for existing VPTO SIMT operations;
- keep wrapper names and parameters close to VPTO IR;
- avoid backend changes unless the frontend generates valid IR that the
  backend incorrectly rejects.

The implementation was staged in batches so the API direction stays consistent
across launch helpers, query ops, lane collectives, scalar memory, atomics,
math, conversion, sync, and state preservation.

## 2. References

- SIMT ISA documentation: `docs/isa/micro-isa/17-simt.md`
- VPTO operation definitions: `include/PTO/IR/VPTOOps.td`
- VPTO verifier behavior: `lib/PTO/IR/VPTO.cpp`
- Existing PTO-DSL operation wrappers: `ptodsl/ptodsl/_ops.py`
- Existing PTO-DSL subkernel lowering: `ptodsl/ptodsl/_subkernels.py`
- Existing PTO-DSL tracing session: `ptodsl/ptodsl/_tracing/session.py`
- Existing PTO-DSL SIMT docs: `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
- Existing scalar docs: `ptodsl/docs/user_guide/06-scalar-and-pointer-ops.md`
- Existing SIMT VPTO lit tests: `test/lit/vpto/simt_*`
- Existing SIMT runtime samples: `test/vpto/cases/micro-op/simt/*`

## 3. Current PTO-DSL State

Current PTO-DSL already has a narrow SIMT surface:

- `@pto.simt` decorator and `with pto.simt():` inline scope.
- `@pto.simt(max_threads=...)` optional entry resource
  attributes.
- `pto.store_vfsimt_info(dim_z, dim_y, dim_x)`.
- `pto.get_tid_x()`, `pto.get_tid_y()`, `pto.get_tid_z()`.
- `pto.load(...)` and `pto.store(...)` for plain scalar element access.

Current `@pto.simt` helper calls lower to:

```mlir
%dim_z = arith.constant 1 : i32
%dim_y = arith.constant 1 : i32
%dim_x = arith.constant 1 : i32
pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
func.call @simt_body(...)
```

That path emits a reusable helper function marked with `pto.simt_entry`, but it
does not yet expose user-controlled launch dimensions and does not use
`pto.simt_launch`.

## 4. Full Migration Plan

The full SIMT micro-op PTO-DSL surface is migrated in staged batches.

### Batch 1: Launch and Query Ops

Status: implemented in `ptodsl/ptodsl/_ops.py`, exported from
`ptodsl/ptodsl/pto.py`, and covered by `ptodsl/tests/test_jit_compile.py`.

Expose launch configuration and nullary thread/lane query wrappers:

- `pto.simt_launch(...)`
- `pto.store_vfsimt_info(...)`
- `pto.get_tid_x/y/z()`
- `pto.get_block_dim_x/y/z()`
- `pto.get_grid_dim_x/y/z()`
- `pto.get_block_idx_x/y/z()`
- `pto.get_veccoreid()`
- `pto.get_clock32()`
- `pto.get_clock64()`
- `pto.get_laneid()`
- `pto.get_lanemask_eq/le/lt/ge/gt()`

### Batch 2: Lane Collective Ops

Status: implemented as direct VPTO wrappers and covered by the full SIMT
surface compile test.

Expose direct wrappers for:

- `pto.vote_all/any/uni/ballot(pred)`
- `pto.shuffle_idx/up/down/bfly(value, control, *, width=32)`
- `pto.redux_add(value)`
- `pto.redux_max/min(value, *, signedness=None)`; integer values require an
  explicit signedness

### Batch 3: SIMT Scalar Memory and Atomics

Status: implemented as direct VPTO wrappers. `pto.ldg`/`pto.stg` reuse the
same address-access normalization as `pto.load`/`pto.store`; atomics
operate on explicit pointer operands.

Expose direct wrappers for:

- `pto.ldg(ptr, offset=0, *, l1cache="cache", l2cache="nmfv")`
- `pto.stg(value, ptr, offset=0, *, l1cache="cache", l2cache="nmfv")`
- `pto.atomic_exch/add/sub/and/or/xor(ptr, value, *, l2cache="nmfv")`
- `pto.atomic_min/max(ptr, value, *, l2cache="nmfv", signedness=None)`;
  integer values require an explicit signedness
- `pto.atomic_cas(ptr, compare, value, *, l2cache="nmfv")`

Plain scalar memory is provided through `pto.load(...)` and `pto.store(...)`.

### Batch 4: SIMT Scalar Math, Convert, Sync, and State

Status: implemented as direct VPTO wrappers. `pto.keep`/`pto.resume` expose
explicit slot attributes and leave placement validation to VPTO.

Expose direct wrappers for:

- `pto.prmt(...)`
- `pto.mulhi(...)`
- `pto.mul_i32toi64(...)`
- `pto.absf(...)`, `pto.sqrt(...)`, `pto.exp(...)`, `pto.log(...)`,
  `pto.pow(...)`, `pto.ceil(...)`, `pto.floor(...)`, `pto.rint(...)`,
  `pto.round(...)`, `pto.fma(...)`
- `pto.cast(...)`, which emits `pto.ftof`, `pto.ftoi`, `pto.itof`, `pto.exti`,
  or `pto.trunci` according to the source and destination categories
- `pto.syncthreads()`, `pto.threadfence()`, `pto.threadfence_block()`
- `pto.keep(...)`, `pto.resume(...)`

`pto.sqrt/exp/log` are shared value operations. Their DSL surface is the same
inside and outside SIMT; lowering selects the generic `math.*` form for ordinary
values and retains the packed or hardware form when the type requires it.

## 5. Implemented Launch and Helper Design

### 5.1 Goals

The implemented SIMT launch layer makes launch dimensions, SIMT helper
materialization, and SIMT runtime queries authorable from PTO-DSL. Later
micro-op batches build on the same helper-lowering path.

The implementation:

- keep micro-op names aligned with VPTO op names;
- preserve the low-level `store_vfsimt_info(dim_z, dim_y, dim_x)` order;
- add an ergonomic launch wrapper that uses the launch-site `x, y, z` order;
- preserve direct `@pto.simt` calls with default launch dimensions;
- specialize reusable SIMT helper functions by argument types and static
  keyword arguments;
- expose optional SIMT entry resource attributes on generated helper functions;
- avoid backend changes.

### 5.2 Non-goals

The launch/helper layer should not implement operation-specific SIMT semantics
itself. Lane collectives, atomics, GM scalar cache policy, scalar math,
conversion, keep/resume, and validation rules are exposed as direct VPTO
wrappers in Batches 2-4.

The launch/helper layer should not change the semantics of `pto.load/store`.

### 5.3 Operation Mapping

| PTO-DSL API | VPTO IR op | Return |
|---|---|---|
| `pto.store_vfsimt_info(dim_z, dim_y, dim_x)` | `pto.store_vfsimt_info` | `None` |
| `pto.simt_launch(body, *args, dims=(dim_x, dim_y, dim_z))` | `pto.simt_launch` or equivalent `store_vfsimt_info + func.call` | `None` |
| `pto.get_tid_x()` | `pto.get_tid_x` | `i32` |
| `pto.get_tid_y()` | `pto.get_tid_y` | `i32` |
| `pto.get_tid_z()` | `pto.get_tid_z` | `i32` |
| `pto.get_block_dim_x()` | `pto.get_block_dim_x` | `i32` |
| `pto.get_block_dim_y()` | `pto.get_block_dim_y` | `i32` |
| `pto.get_block_dim_z()` | `pto.get_block_dim_z` | `i32` |
| `pto.get_grid_dim_x()` | `pto.get_grid_dim_x` | `i32` |
| `pto.get_grid_dim_y()` | `pto.get_grid_dim_y` | `i32` |
| `pto.get_grid_dim_z()` | `pto.get_grid_dim_z` | `i32` |
| `pto.get_block_idx_x()` | `pto.get_block_idx_x` | `i32` |
| `pto.get_block_idx_y()` | `pto.get_block_idx_y` | `i32` |
| `pto.get_block_idx_z()` | `pto.get_block_idx_z` | `i32` |
| `pto.get_veccoreid()` | `pto.get_veccoreid` | `i32` |
| `pto.get_clock32()` | `pto.get_clock32` | `i32` |
| `pto.get_clock64()` | `pto.get_clock64` | `i64` |
| `pto.get_laneid()` | `pto.get_laneid` | `i32` |
| `pto.get_lanemask_eq()` | `pto.get_lanemask_eq` | `i32` |
| `pto.get_lanemask_le()` | `pto.get_lanemask_le` | `i32` |
| `pto.get_lanemask_lt()` | `pto.get_lanemask_lt` | `i32` |
| `pto.get_lanemask_ge()` | `pto.get_lanemask_ge` | `i32` |
| `pto.get_lanemask_gt()` | `pto.get_lanemask_gt` | `i32` |

### 5.4 Launch API

#### Signature

```python
pto.simt_launch(
    body: pto.SubkernelTemplate,
    *args,
    dims: tuple[int | Scalar, int | Scalar, int | Scalar] = (1, 1, 1),
) -> None
```

`dims` uses `(dim_x, dim_y, dim_z)` order. This matches the textual
`pto.simt_launch @body<<<x, y, z>>>(...)` order and the common launch-site
mental model.

The existing low-level API keeps its backend order:

```python
pto.store_vfsimt_info(dim_z, dim_y, dim_x) -> None
```

This asymmetry is intentional:

- `store_vfsimt_info` is a direct wrapper over the backend operation and should
  not rename or reorder operands.
- `simt_launch` is launch-site sugar and should match the IR sugar order
  `x, y, z`.

#### Example

```python
from ptodsl import pto


@pto.simt
def write_tid(dst: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    tid = pto.get_tid_x()
    idx = pto.index_cast(tid)
    pto.store(tid, dst, idx)


@pto.jit(target="a5")
def kernel(dst: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    pto.simt_launch(write_tid, dst, dims=(32, 1, 1))
```

Expected source-level IR shape for Batch 1:

```mlir
%dim_x = arith.constant 32 : i32
%dim_y = arith.constant 1 : i32
%dim_z = arith.constant 1 : i32
pto.simt_launch @write_tid<<<%dim_x, %dim_y, %dim_z>>>(%dst)
  : (!pto.ptr<i32, ub>) -> ()
```

PTO-DSL emits VPTO `pto.simt_launch` directly. The existing backend
`vpto-expand-wrapper-ops` pass expands it to `pto.store_vfsimt_info + func.call`.

### 5.5 Helper Specialization and Symbol Naming

Each `@pto.simt` body is lowered through a generated `func.func` marked with
`pto.simt_entry`. The generated helper symbol is an implementation detail, not
the public subkernel name. PTO-DSL currently uses symbols of the form:

```text
<subkernel-symbol>__simt_<N>
```

The helper specialization key includes:

- the authored subkernel symbol name;
- positional argument MLIR types;
- static keyword argument values.

This prevents two invalid reuse cases:

- the same SIMT body launched with different pointer or scalar argument types;
- the same SIMT body launched with different static keyword arguments that
  change the traced body.

`pto.simt_launch(...)` must reference the actual generated helper symbol, not
the authored subkernel symbol. Direct `@pto.simt` calls also reuse the same
specialized helper path, with default launch dimensions `(1, 1, 1)`.

Keyword arguments passed to `pto.simt_launch` are treated as static values and
must be hashable or structurally representable for the specialization key.
Runtime SSA values must be passed positionally so they become helper function
arguments. This avoids capturing values from the enclosing entry function into
the generated SIMT helper body.

Because generated SIMT helper symbols are internal specialization names, APIs
that require stable `func.func` symbols must resolve authored `@pto.simt`
helpers to a materialized helper symbol before emitting IR. In particular,
`pto.import_reserved_buffer(peer_func=simt_helper)` resolves to the generated
helper symbol when the helper has exactly one materialized specialization. If
the helper has not been called/launched yet, or if multiple specializations
exist, PTO-DSL raises a frontend error and requires the caller to pass an
explicit peer function symbol.

### 5.6 `@pto.simt` Decorator Attributes

SIMT entry functions may carry an optional VPTO attribute:

- `pto.simt_max_threads`

PTO-DSL exposes it through `@pto.simt`:

```python
@pto.simt(max_threads=256)
def body(...):
    ...
```

Lowering:

```mlir
func.func @body(...) attributes {
  pto.simt_entry,
  pto.simt_max_threads = 256 : i32
}
```

Lowering attaches the attribute to the generated specialized helper function,
not to the authored Python symbol. Omitting the argument emits no explicit
attribute and lets the backend default (1024) apply.

Validation:

- values must be Python integers known at trace time;
- values must be positive;
- these attributes must only be attached to functions that are already marked
  `pto.simt_entry`.
- inline `with pto.simt():` scopes do not generate `pto.simt_entry` helper
  functions, so they do not accept these attributes.

These attributes are part of the implemented SIMT entry surface. They only
describe the resource envelope; the actual workitem count still comes from
`pto.store_vfsimt_info` or `pto.simt_launch`.

### 5.7 Query API Behavior

All query APIs are nullary wrappers and return a wrapped MLIR SSA value.

Implementation pattern:

```python
def get_laneid():
    return wrap_surface_value(_pto.GetLaneIdOp().result)
```

No Python-side context check is required for the first version. The backend
already knows which operations are legal in `pto.simt_entry` when applicable.
Adding a frontend context check can be considered later if it improves error
messages without hiding backend semantics.

### 5.8 Type Handling for Launch Dimensions

Launch dimensions are VPTO `i32` operands. PTO-DSL should accept:

- Python integer literals;
- PTO scalar values that are already `i32`;
- index-like runtime values when they can be explicitly cast to `i32`.

Proposed normalization rule:

- Python `int` is materialized as signless `i32` constant.
- A runtime scalar with type `i32` is accepted unchanged.
- A runtime scalar with type `index` may be cast to `i32` if existing PTO-DSL
  scalar casting helpers provide a clear path.
- Other types should raise a clear Python `TypeError`.

The implementation should not silently accept `i64` or arbitrary integers by
truncation.

### 5.9 Interaction With Existing `@pto.simt` Calls

Current code can call a SIMT subkernel directly:

```python
write_tid(dst)
```

Today that direct call lowers to launch dimensions `(1, 1, 1)`.

To preserve compatibility, direct `SubkernelTemplate.__call__` behavior should
remain valid and keep its current default launch dimensions. `pto.simt_launch`
is the explicit launch-dimension surface for new code.

Future ergonomic options:

```python
write_tid.launch(dst, dims=(32, 1, 1))
```

This method is not required for Batch 1. If added, it should call the same
lowering path as `pto.simt_launch(...)` and should not create a second semantic
route.

### 5.10 Implementation Notes

Frontend files touched by the implemented surface:

- `ptodsl/ptodsl/_ops.py`
  - SIMT query, launch, collective, memory, atomic, math, convert, sync, and
    state wrappers;
  - enum/cache/rounding/saturation normalization for SIMT attrs.
- `ptodsl/ptodsl/pto.py`
  - exported SIMT wrappers.
- `ptodsl/ptodsl/_tracing/session.py`
  - reusable helper lowering for direct SIMT calls and explicit
    `pto.simt_launch`;
  - SIMT helper specialization by argument types and static kwargs;
  - SIMT entry resource attribute emission;
  - actual helper-symbol targeting for `pto.simt_launch`.
- `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
  - documented the SIMT API surface.
- `ptodsl/tests/support/docs_fragment_fixtures.py`
  - declares stable peer functions for docs snippets that use
    `pto.import_reserved_buffer(peer_func=...)` where the snippet needs a
    fixed peer symbol independent of SIMT helper specialization.
- `ptodsl/tests/test_jit_compile.py`
  - compile smoke tests for launch/query wrappers, full SIMT micro-op surface,
    invalid frontend argument combinations, and SIMT helper specialization.

Backend files are not touched by this PTO-DSL frontend surface.

### 5.11 Test Plan

Minimum Python/frontend tests:

1. Existing direct `@pto.simt` call still emits `pto.store_vfsimt_info` and a
   reusable `pto.simt_entry` helper specialization.
2. `pto.simt_launch(body, dst, dims=(32, 1, 1))` emits either:
   - `pto.simt_launch @body<<<...>>>`, or
   - an equivalent `pto.store_vfsimt_info` with dimensions reordered to
     `z, y, x` followed by `func.call @body`.
3. All query wrappers compile inside a SIMT body and emit the expected op names.
4. `get_clock64()` returns an `i64` value; all other query wrappers in Batch 1
   return `i32`.
5. Invalid launch dimensions raise Python errors before backend verification
   when the type is clearly unsupported.
6. The same `@pto.simt` body launched with different argument types produces
   distinct helper functions.
7. The same `@pto.simt` body launched with different static keyword arguments
   produces distinct helper functions and distinct traced bodies.
8. `pto.simt_launch` callee attributes reference the actual generated helper
   symbols.
9. `@pto.simt(max_threads=...)` emits `pto.simt_max_threads` on the generated
   helper function. The register budget (`simt-max-registers`) is automatically
   derived from `max_threads` by the backend according to the hardware resource
   partition and is no longer a configurable attribute.

Suggested lit/frontend assertions:

- `func.func @body(...) attributes {pto.simt_entry}`
- `pto.get_tid_x`
- `pto.get_block_dim_x`
- `pto.get_grid_dim_x`
- `pto.get_block_idx_x`
- `pto.get_veccoreid`
- `pto.get_clock32`
- `pto.get_clock64`
- `pto.get_laneid`
- `pto.get_lanemask_lt`
- explicit launch dimensions are present in the generated IR.

Runtime/ST validation is not required for the first frontend API PR unless a
later implementation changes runtime behavior.
