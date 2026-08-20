---
name: rewrite-kernel-with-vmi
description: Rewrite a complete AscendC/CANN kernel into an equivalent PTODSL Python implementation that mixes VMI compute with MI/MTE/sync orchestration. Use when given AscendC kernel code or CCE-style device code and preserve the kernel ABI, split compute from movement/synchronization, translate pure vector/SIMT compute regions into pto.vmi semantics, translate non-compute regions into PTODSL micro-instructions, and validate the result with kernel.compile().mlir_text() plus the PTOAS VMI path.
---

# Rewrite Kernel With VMI

Rewrite the source kernel by preserving observable semantics first, then choosing
the clean PTODSL spelling. Do not mechanically replay physical register
choreography when a logical VMI expression captures the same algorithm.

## Workflow

1. Read the complete source kernel and all helpers/macros needed to understand
   its ABI, loop bounds, offsets, memory movement, synchronization, and compute.
2. Read references only as needed:
   - [vmi-dsl-spec.md](references/vmi-dsl-spec.md) for current PTODSL `pto.vmi`
     API spelling, including `vload` / `vstore`, `create_mask`, `vci`, `vbrc`,
     and common AscendC SIMD to VMI patterns.
   - [mi-dsl-spec.md](references/mi-dsl-spec.md) for PTODSL non-VMI user-guide
     navigation: kernel entry, buffers, control flow, MTE, sync, masks, SIMT.
   - [vmi-mlir-spec.md](references/vmi-mlir-spec.md) when VMI IR semantics,
     layout, mask, or PTOAS validation details matter.
3. Extract the PTODSL function interface:
   - Convert each host-visible kernel argument one-for-one.
   - Convert C++ template parameters to keyword-only `pto.const_expr` arguments.
   - For template/generic variants, prefer one PTODSL function with
     `pto.const_expr` parameters that select dtype, constants, and specialized
     branches at compile time. Avoid generating many outer wrapper/probe
     functions that only differ by dtype or template value.
   - Use this default entry unless the source requires otherwise:

```python
@pto.jit(
    name="<kernel_name>",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def <kernel_name>(..., *, CONST: pto.const_expr = ...):
    ...
```

Compile-time selection example:

```python
@pto.jit(...)
def kernel(x: pto.ptr(pto.f8e5m2, "ub"), y_addr: pto.i64, *, OUT_DTYPE: pto.const_expr = pto.f32):
    y = pto.castptr(y_addr, pto.ptr(OUT_DTYPE, "ub"))
    if OUT_DTYPE is pto.f32:
        ...
    elif OUT_DTYPE is pto.bf16:
        ...
```

4. Split the source body into regions:
   - Compute regions: pure vector/SIMT arithmetic, compare/select, conversion,
     reduction, rearrange, or math. `membar` may stay in or near a compute
     region when it only orders vector-visible UB effects.
   - Non-compute regions: GM/UB/L1 movement, tile allocation, pointer/view
     setup, `set_flag`/`wait_flag`, cross/intra flag sync, buffer handoff,
     pipeline barriers, and host/core indexing.
5. Before writing PTODSL for each nontrivial compute region, show the user a
   VMI rewrite design and wait for confirmation when the interaction allows it.
   If the user is unavailable or explicitly asked for an end-to-end conversion,
   continue with clearly stated assumptions.
6. Implement the PTODSL Python file:
   - Use `from ptodsl import pto` and add `scalar` only when needed.
   - Keep imports environment-agnostic. Do not add `Path(__file__)` parent
     walks, local repo discovery, or `sys.path.insert(...)` blocks to find the
     `ptodsl` package; configure installation, runner paths, or `PYTHONPATH`
     outside the generated DSL file instead.
   - Keep the source ABI recognizable.
   - Write VMI/MI operations inline by default. Create small helper functions
     only when a nontrivial instruction sequence is reused many times or when a
     named helper preserves an important source-level abstraction.
   - Use native Python `for` / `if` control flow. Use `pto.const_expr` and
     `pto.static_range` for intentional compile-time specialization or
     unrolling. Do not use `pto.for_` / `pto.if_` in new rewrites.
   - Translate non-compute regions with PTODSL MI/MTE/sync operations.
   - Translate compute regions with `pto.vmi` logical vectors.
7. Validate and iterate:
   - First run or provide a compile helper that calls
     `kernel.compile(...).mlir_text()`.
   - Then pass the emitted MLIR to PTOAS with the VMI path:
     `ptoas --pto-arch=a5 --pto-backend=vpto <mlir_or_pto> -o /dev/null`.
   - When the goal is to inspect lowered VMI output locally and no CANN/toolchain
     environment is needed, prefer the bundled helper script:
     `scripts/compile_vmi_to_vpto.sh <input.vmi.pto> [output.mi.pto]`.
     It wraps `ptoas --pto-backend=vpto --emit-vpto` and is the
     default path for generating reviewable lowered artifacts from emitted
     `vmi.pto` files.
   - Fix PTODSL trace errors before PTOAS errors. Fix semantic mismatches before
     layout/lowering workarounds.

## VMI Rewrite Design Gate

For every nontrivial compute region, present this compact review artifact before
coding the region:

```text
Compute region: <source lines / logical name>
Inputs:
  - <UB pointer or scalar>, offset/stride formula, dtype, shape
Outputs:
  - <UB pointer>, offset/stride formula, dtype, shape
Semantic algorithm:
  - <elementwise/reduction/rearrange description independent of hardware parts>
Physical-only source details to collapse:
  - <PART_*, EVEN/ODD, PK*, vintlv/vdintlv trees, temporary register halves>
VMI plan:
  for off in ...:
      mask = pto.vmi.create_mask(active_lanes, size=lanes)   # or size=lanes, group=groups
      x = pto.vmi.vload(..., size=lanes)
      y = <logical compute>
      pto.vmi.vstore(y, ..., mask)
Lane choice:
  - lanes=<N>, dtype=<T>, reason=<full vreg/tail/per-row/grouping>
Assumptions:
  - <tail, alignment, contiguous access, dtype reinterpretation, unsupported op>
```

Ask the user to confirm or correct this design when the algorithm is ambiguous,
when the source uses heavy physical packing/interleave, or when multiple VMI
forms are plausible. If there is no reply and progress is requested, implement
the stated assumptions and call them out in the result.

## Translation Rules

- Preserve the boundary contract. Do not drop source arguments merely because a
  first rewrite does not use them yet.
- Do not embed local import bootstrapping in generated DSL code. The file should
  import PTODSL normally, e.g. `from ptodsl import pto`, without walking parent
  directories or mutating `sys.path` to locate a workspace checkout.
- Convert common source types as follows:
  - `__gm__ half*` / semantic f16 storage -> `pto.ptr(pto.f16, "gm")`
  - `__gm__ float*` -> `pto.ptr(pto.f32, "gm")`
  - `int32_t` -> `pto.i32`
  - `uint32_t` -> `pto.ui32`
  - `int64_t` / addresses and byte counters -> `pto.i64` unless source intent is
    clearly index-like.
- If a C++ boundary uses raw storage (`uint16_t*`, `uint8_t*`) but the semantic
  element type is `f16`, `bf16`, fp8, or packed f4, keep the safest ABI spelling
  and use `pto.castptr` internally when needed.
- Preserve templates and generic dtype choices with `pto.const_expr` whenever
  possible. Use Python compile-time `if` branches and dtype variables to choose
  pointer casts, VMI lane/dtype choices, conversion targets, and store paths. Only add
  separate wrapper/probe functions when the ABI truly differs or the test
  harness explicitly requires separate entry symbols.
- Use the current VMI surface shape: `vload(..., size=...)`, `vstore(..., mask=...)`,
  `vci(..., size=...)`, `vbrc(..., size=...)`, and `create_mask(..., size=..., group=...)`.
  Do not use the retired `create_group_mask` helper or legacy `result_type`
  spellings in new rewrites.
- Keep source offset and stride formulas symbolic. Do not replace expressions
  such as `vlForHalfNumber * 2` with a constant unless the source is already
  specialized and the user asked for specialization.
- Avoid small one-off helpers around VMI/MI instruction sequences. Inline direct
  PTODSL operations unless the code is long, reused repeatedly, or the source
  abstraction is important enough to preserve by name.
- Use Python-native control flow for both dynamic device-side branches/loops and
  ordinary structured code. Use `pto.static_range(...)` only for trace-time
  static loops driven by Python values or `pto.const_expr` parameters. Do not
  write `with pto.for_(...)` or `with pto.if_(...)` in new kernel rewrites.
- Use explicit `mode="explicit"` orchestration for full kernel rewrites. Do not
  rely on auto-inserted sync unless the user requests an auto-mode rewrite.
- Prefer `pto.vmi.vload` from UB, logical compute, then `pto.vmi.vstore` to UB.
  GM movement belongs to MTE/tile movement outside the VMI compute region.
- Use `pto.vmi.create_mask(active, size=lanes)` for dynamic tails. Put masks on
  compute and store; do not assume a masked load is legal on every backend.
  For grouped tails, use `create_mask(..., size=lanes, group=...)`.
- Collapse physical-only details such as `PART_P0`, `PART_EVEN`, `PART_ODD`,
  packed store modes, and interleave trees when they only describe hardware
  lowering of one logical vector.
- Do not introduce UB store+reload round trips just to satisfy a lowering issue
  unless the user accepts the performance tradeoff.
- For TileLang input, treat it as future scope unless the user explicitly asks.
  Preserve the same VMI design-gate workflow and translate the logical parallel
  loop body rather than physicalizing it early.

## Optimization Tips

- When the logical algorithm is "an `N x VL` vector where each `VL` chunk is
  multiplied by the same `VL`-lane scale vector", prefer one widened/grouped
  VMI expression over a scalarized inner chunk loop.
  - Good PTODSL spelling when legal on the current backend:

```python
wide_x = pto.vmi.vcvt(x, pto.f32)  # e.g. N*VL lanes
scale_wide = pto.vmi.vload(scale_ptr, scale_off, size=N * VL, stride=0, group=N)
wide_y = pto.vmi.vmul(wide_x, scale_wide, full_mask)
```

  - This pattern is often better than:
    1. spilling the widened `N x VL` value to UB,
    2. reloading `VL` chunks in a Python loop,
    3. multiplying each chunk by the same `VL` scale vector,
    4. storing chunk-by-chunk.
  - In practice, this zero-stride grouped `vload` is a useful way to express
    "repeat the same `VL` vector across `N` groups" directly in VMI, keeping
    the computation at full-vector width.
  - Still validate with both `kernel.compile(...).mlir_text()` and PTOAS VMI
    lowering, because legality depends on the source dtype, lane count, and
    backend support for the grouped load shape.

## Validation Checklist

Before finishing a rewrite, verify:

- Function parameters and template/constexpr parameters match the source ABI.
- Template/generic variants are represented by `pto.const_expr` compile-time
  selection unless separate entry symbols are explicitly needed.
- Pointer memory spaces and semantic dtypes are justified.
- GM/UB/L1 movement sizes, offsets, strides, and padding match the source.
- Sync ordering between MTE, Vector, Cube/SIMT, and stores is preserved.
- New VMI/MI code is not hidden behind one-off helper functions.
- Generated DSL code has no manual `ptodsl` path lookup, `Path(__file__)`
  parent search, or `sys.path` mutation for local package discovery.
- Control flow uses native Python `for` / `if`, with `pto.static_range` only for
  intentional compile-time loops.
- Every VMI compute region has a user-visible design note or stated assumptions.
- Tail masks and lane counts match the processed element count.
- `kernel.compile(...).mlir_text()` succeeds or the remaining trace error is
  reported with the exact failing construct.
- PTOAS validation uses `--pto-backend=vpto`; do not diagnose VMI legality from
  an EmitC invocation.

Numeric NPU or simulator validation is optional unless the user requests it.
