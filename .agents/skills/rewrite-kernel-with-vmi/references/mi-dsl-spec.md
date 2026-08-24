# PTODSL MI And Non-VMI Index

Use this file as a navigation index for non-VMI parts of an AscendC kernel
rewrite. The source of truth is `ptodsl/docs/user_guide/`.

## Kernel Entry And ABI

Read `03-kernel-entry-and-subkernels.md` for:

- `@pto.jit` entry vs module roles.
- `backend="vpto"` and `mode="explicit"`.
- Pointer-first host ABI: `pto.ptr(dtype, "gm")`.
- Runtime scalar parameters before `*`.
- Compile-time constants after `*` as `pto.const_expr`.
- `.compile(...).mlir_text()` usage.

Use normal PTODSL imports in generated DSL:

```python
from ptodsl import pto
```

Add `scalar` or other PTODSL imports only when the code actually needs them. Do
not add standalone-script bootstrapping that scans `Path(__file__).parents`,
finds a local `ptodsl` checkout, or calls `sys.path.insert(...)`. If a test or
manual run cannot import PTODSL, fix the execution environment instead: install
the package, invoke the repository's runner, or set `PYTHONPATH` outside the DSL
file.

Prefer `pto.const_expr` for C++ template parameters and dtype-generic variants.
One PTODSL function can select pointer casts, dtypes, tile shapes, VMI result
types, and store paths at compile time instead of creating many wrapper/probe
functions that only differ by template value.

Default full-kernel rewrite entry:

```python
@pto.jit(
    name="...",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def kernel(..., *, CONST: pto.const_expr = ...):
    ...
```

Example:

```python
@pto.jit(...)
def kernel(y_addr: pto.i64, *, OUT_DTYPE: pto.const_expr = pto.f32):
    y = pto.castptr(y_addr, pto.ptr(OUT_DTYPE, "ub"))
    if OUT_DTYPE is pto.f32:
        ...
    elif OUT_DTYPE is pto.bf16:
        ...
```

## Types, Buffers, Views

Read `04-type-system-and-buffer.md` for:

- Scalar annotations: `pto.i32`, `pto.ui32`, `pto.i64`, `pto.f16`, `pto.f32`.
- Pointer types and memory spaces: `pto.ptr(dtype, "gm")`,
  `pto.ptr(dtype, "ub")`, `pto.MemorySpace.*`.
- `pto.make_tensor_view`, `pto.partition_view`, `.as_ptr()`.
- `pto.alloc_tile`, tile shape and `valid_shape`.
- `pto.castptr` when raw storage pointer spelling differs from semantic dtype.

## Control Flow And Scalar/Pointer Math

Read `05-control-flow.md` and `06-scalar-and-pointer-ops.md` for:

- Python-native `for` / `if` control flow, which PTODSL rewrites into
  device-side structured control flow when bounds or conditions are runtime PTO
  values.
- `pto.const_expr` and `pto.static_range(...)` for intentional trace-time
  specialization and unrolling.
- Scalar casts, selects, index math, pointer casts, pointer offsets.
- Keeping loop bounds and offset formulas symbolic.

Do not use `pto.for_` / `pto.if_` in new rewrites. Those explicit context-manager
forms exist in older examples and compatibility surfaces, but this skill should
prefer native Python control flow. When hardware-loop lowering is important,
preserve source-like runtime loop bounds and avoid turning loop math into Python
constants.

## Helper Function Policy

Write VMI and MI instruction sequences inline by default. Avoid small helpers
that wrap only a few operations, because they obscure the rewrite and can add
unhelpful call boundaries in generated IR. Introduce a helper only when:

- The same nontrivial sequence is reused many times.
- A long block becomes materially easier to audit by naming the source-level
  abstraction.
- A real PTODSL sub-kernel boundary is required, such as Cube/SIMT ownership.

## Data Movement: MTE And Tile Movement

Read `07-data-movement-ops.md` for:

- Tile-level `pto.tile.load` / `pto.tile.store` in auto-mode style code.
- Explicit mode DMA:
  - `pto.mte_gm_ub(gm_src, ub_dst, l2_cache_ctl, len_burst, nburst=..., loops=..., pad=...)`
  - `pto.mte_ub_gm(ub_src, gm_dst, len_burst, nburst=..., loops=...)`
  - `pto.mte_ub_ub(ub_src, ub_dst, len_burst, nburst=...)`
  - `pto.mte_ub_l1(...)`
- Shorthands `pto.mte_load` and `pto.mte_store` when they match the canonical
  grouped-DMA shape.

Stride units matter:

- GM/UB `mte_gm_ub` and `mte_ub_gm`: byte strides.
- UB/UB `mte_ub_ub`: 32B units for gaps/lengths.
- VMI `vload/vstore`: element offsets and element strides.

## Legacy Vector Compute And Masks

Read `08-compute-operations.md` and `09-predicate-and-mask-ops.md` for:

- Existing top-level vector helpers such as `pto.vadd`, `pto.vlds`, `pto.vsts`.
- Legacy masks such as `pto.make_mask`, `pto.mask_b16`, `pto.mask_b32`.

For this skill, use `pto.vmi` for newly translated pure compute regions unless
the user explicitly asks to keep legacy vector helpers.

## Synchronization

Read `10-sync-ops.md` for:

- `pto.set_flag(pipe_from, pipe_to, event_id=...)`
- `pto.wait_flag(pipe_from, pipe_to, event_id=...)`
- `pto.pipe_barrier(pto.Pipe.ALL)`
- `pto.mem_bar(pto.BarrierType.VV_ALL)` and other barrier types.
- `pto.get_buf` / `pto.rls_buf` for double buffering.
- `pto.set_cross_block` / `pto.wait_cross_block`.
- `pto.set_intra_block` / `pto.wait_intra_block`.

Common explicit-mode pairs:

- DMA load before vector compute:
  `set_flag(MTE2, V)` then `wait_flag(MTE2, V)`.
- Vector compute before DMA store:
  `set_flag(V, MTE3)` then `wait_flag(V, MTE3)`.
- Store-to-load ordering in UB:
  `pto.mem_bar(pto.BarrierType.VST_VLD)`.

In `mode="explicit"`, do not rely on compiler-inserted synchronization unless
the user requests an auto-sync rewrite.

## SIMT Micro-Ops

Read `13-simt-micro-ops.md` when the AscendC compute region uses SIMT-like
scalar lane programming rather than vector SIMD. Preserve SIMT control flow and
only translate to VMI when the algorithm is actually independent SIMD lanes.

## Examples To Inspect

- `ptodsl/examples/tadd_launch.py`: simple PTODSL entry/compile/launch shape.
- `ptodsl/examples/flash_attention/flash_attention_cv_split.py`: mixed
  orchestration, modules, sync, and explicit kernel structure.
- `ptodsl/examples/mixed_backend_kernel_module.py`: entry/module composition.
