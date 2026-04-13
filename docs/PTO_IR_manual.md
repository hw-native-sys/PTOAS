# PTO IR Reference

- **Version:** `v 0.1`
- **Date:** `2026-02-14`
- **Author:** `Wenbo Sun`

## 1. Overview

The **PTO Dialect** (`pto`) is an MLIR dialect for expressing tile-based computations targeting Ascend NPU hardware. It is part of the PTOAS (PTO Assembler & Optimizer) compiler toolchain.

- **Dialect name:** `pto`
- **Source:** `include/PTO/IR/`

### PTO IR Level Model

PTO IR is organized as a hierarchical, multi-level IR stack and intentionally exposes multiple abstraction levels to external users and frameworks.

- **Level-1 (SSA-centric IR):** `pto.tile` is an SSA value; PTO-AS is responsible for buffer allocation and storage planning during lowering.
- **Level-2 (DPS tile-buffer IR):** `pto.tile_buf` is represented in destination-passing style (DPS), i.e., as explicit buffer objects rather than SSA value semantics.
- **Level-3 (Low-level scheduling IR):** pipeline/event synchronization is explicit and user-managed, enabling direct control over execution ordering and inter-op dependencies.

These levels are lowered progressively from Level-1 to Level-3, serving distinct optimization and control requirements across different users and integrations. **This PTO IR API document focuses on Level-2 and Level-3 interfaces.** *The Level-1 public interface is still under active design and will be specified in a future revision.*

### Hardware Memory Hierarchy

```
GM (Global Memory)
|- MAT (L1 Cache)
|  |- LEFT  (L0A - left matrix buffer)
|  |- RIGHT (L0B - right matrix buffer)
|  |- ACC   (L0C - accumulator)
|  `- BIAS  (bias buffer)
`- VEC (UB  - unified buffer)
```

## 1.1 Rationale

For the Level-2/Level-3 profiles documented here, PTO IR models tiles as buffers rather than SSA values. A `pto.tile_buf` denotes a storage object with an explicit lifetime, not a pure value. This design intentionally decouples allocation/tiling from pipeline scheduling: buffer allocation is NP-hard, and pipeline scheduling is also NP-hard. Coupling both problems in a single compiler pass is intractable in practice. Therefore, PTO IR requires users or higher-level frameworks to manage buffer reuse explicitly via `pto.alloc_tile`, while PTO AS passes focus on scheduling and pipeline orchestration.

**Example (explicit buffer lifetime):**

```mlir
%a0 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%a1 = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
pto.tload ins(%pv0 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a0 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
pto.tload ins(%pv1 : !pto.partition_tensor_view<16x16xf16>)
          outs(%a1 : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

## 2. Type System

### 2.1 Element Types

Element types describe the primitive scalar values that can be stored in tensors/tiles; by themselves they do not form a value. They define how a sequence of bits is interpreted and the number of bits required to represent the value. This is distinct from any storage size implied by tensor layout.

Common element categories include:

- **Integers**: signless integers such as `i1/i8/i16/i32`. Signedness is not encoded in the type; it is selected by operation semantics or attributes where required.
- **Floating-point**: IEEE floating-point types such as `f16/f32`. Some targets may also support additional formats (e.g., `bf16` or low-precision exponent/mantissa formats) with stricter constraints.
- **Index-like**: index values may appear as scalar operands in certain operations (e.g., offsets, sizes, or scalar comparisons).

Element type constraints are operation-specific:

- **Shape/type consistency**: most elementwise ops require all operands and results to have the same element type.
- **Numeric domain**: reductions, math ops, and division typically restrict element types to floating-point or a subset of integer types.
- **Bitwise ops**: require integer element types.
- **Conversions**: `pto.tcvt` defines explicit element type changes and is controlled by `RoundMode` when converting between numeric domains.

In addition, memory layout and address space do not change the element type semantics; they only affect placement and access patterns.

### 2.2 `!pto.ptr<elementType>`

A pointer to global memory.

| Parameter | Type | Description |
|-----------|------|-------------|
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element type pointed to |

**Syntax:** `!pto.ptr<f16>`

---

### 2.3 `!pto.tensor_view<d0 x d1 x elementType>`

A descriptor for a global memory tensor. Does not own data - represents a view with shape and stride information.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Tensor shape `[d0, d1]` (each dim may be `?` for dynamic) |
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element data type |

**Syntax:** `!pto.tensor_view<1024x512xf16>`

---

### 2.4 `!pto.partition_tensor_view<d0 x d1 x elementType>`

A logical partition (slice) of a `tensor_view`. Holds shape and stride information for a tile-sized region but does not own data.

| Parameter | Type | Description |
|-----------|------|-------------|
| `shape` | `ArrayRef<i64>` | Partition shape `[d0, d1]` |
| `elementType` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element data type |

**Syntax:** `!pto.partition_tensor_view<16x16xf16>`

---

### 2.5 `!pto.tile_buf<loc=..., dtype=..., rows=..., cols=..., ...>`

`pto.tile_buf` represents a local scratchpad memory tile buffer with explicit placement, shape, valid region, and layout/fractal metadata. Based on formats used in `PTOAS/test`, the canonical textual form is a key-value list.

| Parameter | Type | Description |
|-----------|------|-------------|
| `loc` | keyword (`vec/mat/left/right/acc/bias`) | Local memory domain (`vec` maps to UB; use `vec` in textual IR) |
| `dtype` | `element-type(i1/i8/i16/i32/f16/f32/bf16...)` | Element data type |
| `rows` | `int64` | Physical row count |
| `cols` | `int64` | Physical column count |
| `v_row` | `int64` or `?` | Valid row count |
| `v_col` | `int64` or `?` | Valid column count |
| `blayout` | `BLayout` mnemonic | Base layout (`row_major` / `col_major`) |
| `slayout` | `SLayout` mnemonic | Secondary layout (`none_box` / `row_major` / `col_major`) |
| `fractal` | `int32` | Fractal size |
| `pad` | `PadValue` mnemonic or integer literal | Padding policy/value selector (tests commonly use `pad=0`) |

Here, `?` denotes a dynamic symbol resolved at runtime.

**Syntax:**
```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

## 3. Enums & Attributes

### 3.1 AddressSpace

Defines the physical storage location of a buffer in the Ascend NPU memory hierarchy. This affects which operations are legal on the buffer and how data movement is scheduled (e.g., GM <-> UB, L1 <-> L0).

| Value | Int | Mnemonic | Hardware Mapping |
|-------|-----|----------|-----------------|
| `Zero` | 0 | `zero` | Default (unspecified) |
| `GM` | 1 | `gm` | Global Memory |
| `MAT` | 2 | `mat` | L1 Cache |
| `LEFT` | 3 | `left` | L0A (left matrix buffer) |
| `RIGHT` | 4 | `right` | L0B (right matrix buffer) |
| `ACC` | 5 | `acc` | L0C (accumulator) |
| `VEC` | 6 | `vec` | UB (unified buffer) |
| `BIAS` | 7 | `bias` | Bias buffer |

**Attribute syntax:** `loc=<mnemonic>` (for example, `loc=vec`)

---

### 3.2 PipeEventKind

Defines intra-core pipeline synchronization event kinds in PTO IR, used to express dependencies between pipelines (for example, in [`pto.record_event`](#ptorecord_event) and [`pto.wait_event`](#ptowait_event)).

| Value | Int | Description |
|-------|-----|-------------|
| `EVENT_LOAD_FROM_GM` | 0 | Load from GM |
| `EVENT_STORE_FROM_ACC` | 1 | Store from accumulator |
| `EVENT_STORE_FROM_VEC` | 2 | Store from vector/UB |
| `EVENT_MOVE_MAT_TO_LEFT` | 3 | Move: MAT -> LEFT |
| `EVENT_MOVE_MAT_TO_SCALAR` | 4 | Move: MAT -> scalar |
| `EVENT_MOVE_MAT_TO_BIAS` | 5 | Move: MAT -> BIAS |
| `EVENT_MOVE_MAT_TO_VEC` | 6 | Move: MAT -> VEC |
| `EVENT_MOVE_VEC_TO_MAT` | 7 | Move: VEC -> MAT |
| `EVENT_COMPUTE_MATMUL` | 8 | Matrix multiplication |
| `EVENT_COMPUTE_VEC` | 9 | Vector operation |
| `EVENT_VEC_WAITPOINT` | 10 | Vector wait event |

**Attribute syntax:** `#pto.pipe_event_type<EVENT_LOAD_FROM_GM>`

---

### 3.3 EVENT (Hardware Event IDs)

8 hardware event IDs for synchronization primitives.

| Value | Int |
|-------|-----|
| `EVENT_ID0` - `EVENT_ID7` | 0 - 7 |

**Attribute syntax:** `#pto.event<EVENT_ID0>`

---

### 3.4 Tile Buf config

Composite attribute and component enums for tile buffer configuration.

| Parameter | Type | Description |
|-----------|------|-------------|
| `bLayout` | `BLayoutAttr` | Base layout (RowMajor / ColMajor) |
| `sLayout` | `SLayoutAttr` | Secondary layout (NoneBox / RowMajor / ColMajor) |
| `sFractalSize` | `IntegerAttr (i32)` | Secondary fractal size |
| `pad` | `PadValueAttr` | Pad value policy |

**Syntax:** `#pto.tile_buf_config<row_major, none_box, 16, zero>`

**BLayout** (Base layout):

| Value | Int | Mnemonic |
|-------|-----|----------|
| `RowMajor` | 0 | `row_major` |
| `ColMajor` | 1 | `col_major` |

**SLayout** (Secondary layout):

| Value | Int | Mnemonic |
|-------|-----|----------|
| `NoneBox` | 0 | `none_box` |
| `RowMajor` | 1 | `row_major` |
| `ColMajor` | 2 | `col_major` |

**PadValue** (Pad value policy):

| Value | Int | Mnemonic |
|-------|-----|----------|
| `Null` | 0 | `null` |
| `Zero` | 1 | `zero` |
| `Max` | 2 | `max` |
| `Min` | 3 | `min` |

---

### 3.5 Layout

Global tensor layout inference for [`tensor_view` (Section 2.3)](#23-ptotensor_viewd0-x-d1-x-elementtype)/[`partition_tensor_view` (Section 2.4)](#24-ptopartition_tensor_viewd0-x-d1-x-elementtype). Tile buffers additionally use **Tile Buf config** (see 3.4) to describe physical/fractal layout.

| Value | Int | Mnemonic | Description |
|-------|-----|----------|-------------|
| `ND` | 0 | `nd` | Row-major (Normal-Dimension) |
| `DN` | 1 | `dn` | Column-major (Dimension-Normal) |
| `NZ` | 2 | `nz` | Fractal/blocked layout |

**Attribute syntax:** `#pto.layout<nd>`

---

## 4. Operations Reference

In addition to the `pto.*` operations documented below, PTOAS also accepts a limited set of commonly used third-party MLIR dialect operations as part of the input IR and lowering pipeline.

- **`func`**
  - `func.func`
  - `func.return`
  - `func.call`
- **`arith`**
  - constants and casts such as `arith.constant`, `arith.constant_index`, `arith.index_cast`, `arith.index_castui`, `arith.bitcast`, `arith.extf`, `arith.truncf`, `arith.extsi`, `arith.extui`, `arith.trunci`, `arith.sitofp`, `arith.uitofp`, `arith.fptosi`, and `arith.fptoui`
  - integer and floating-point arithmetic such as `arith.addi`, `arith.subi`, `arith.muli`, `arith.divsi`, `arith.divui`, `arith.remsi`, `arith.remui`, `arith.addf`, `arith.subf`, `arith.mulf`, `arith.divf`, `arith.remf`, and `arith.negf`
  - bitwise and shift operations such as `arith.andi`, `arith.ori`, `arith.xori`, `arith.shli`, `arith.shrsi`, and `arith.shrui`
  - comparisons and selection such as `arith.cmpi`, `arith.cmpf`, and `arith.select`
  - selected extended and min/max operations such as `arith.addui_extended`, `arith.mulsi_extended`, `arith.mului_extended`, `arith.ceildivsi`, `arith.ceildivui`, `arith.floordivsi`, `arith.maxsi`, `arith.minsi`, `arith.maxui`, `arith.minui`, `arith.maximumf`, `arith.minimumf`, `arith.maxnumf`, and `arith.minnumf`
- **`scf`**
  - `scf.for`
  - `scf.if`
  - `scf.yield`
  - PTOAS also handles several structured-control-flow forms by lowering them through `cf`, including `scf.execute_region`, `scf.while`, `scf.index_switch`, and `scf.condition`

NOTE: These third-party ops are supported only to the extent required by PTOAS front-end construction, analysis, and lowering. PTOAS does not imply full support for every operation in these dialects.

### 4.1 Pointer & View Operations

##### `pto.addptr` - Add Element Offset to Pointer

**Summary:** Computes a new pointer by adding an element offset to the base pointer.

**Semantics:**

```
result = ptr + offset   // offset is in elements, not bytes
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `!pto.ptr<elementType>` | Base pointer |
| `offset` | `index` | Element offset (not byte offset) |

**Results:** `!pto.ptr<elementType>` — the same pointer type as the input.

**Constraints & Verification:**

- result type must match the input pointer type
- The operation is pure (no side effects)

**Hardware Mapping:**

- No hardware pipeline (pointer arithmetic only)

**Basic Example:**

```mlir
%ptr_off = pto.addptr %base, %offset : !pto.ptr<f32> -> !pto.ptr<f32>
```

##### `pto.make_tensor_view` - Create Tensor View

**Summary:** Constructs a global tensor view from a pointer, declaring the physical base and strides (no allocation, no data movement).

**Semantics:**

```
result = tensor_view(ptr, shape, strides, layout)
```

This operation defines the physical "base" and stride rules for global memory. It is the reference view for all subsequent partitioning, and it does not move any data.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `AnyType` | Source pointer |
| `shape` | `Variadic<Index>` | Dynamic shape dimensions |
| `strides` | `Variadic<Index>` | Dynamic strides |
| `layout` | `LayoutAttr` (optional) | ND/DN/NZ layout hint |

**Results:** `!pto.tensor_view<...>`

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - `ptr` must be `!pto.ptr<...>` and its element type must match the result element type
  - `shape` and `strides` operand counts must match the tensor_view rank
  - If `layout` is provided with static shapes/strides, it must be consistent with inferred layout

**Notes:**

- Stride patterns may allow the compiler to infer hardware layout hints (e.g., `layout = nz`) to guide later DMA operations.

**Hardware Mapping:**

- No hardware pipeline (metadata/view construction only)

**Basic Example:**

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1] : !pto.tensor_view<?x?xf32>
```

---

##### `pto.get_tensor_view_dim` - Get Tensor View Dimension Size

**Summary:** Returns the size of a given dimension of a logical tensor view.

**Semantics:**

```mlir
dim = get_tensor_view_dim(tv_or_mr, dim_index)
```

This op is primarily defined on `!pto.tensor_view`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `tensor_view` | `!pto.tensor_view<...>` | Logical tensor view |
| `dim_index` | `index` | Dimension index (0-based) |

**Results:** `index` — the runtime size of the requested dimension.

**Notes:**

- Commonly used to drive `partition_view` sizes when the tensor_view shape is dynamic.

**Basic Example:**

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%w = pto.get_tensor_view_dim %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
%pv = pto.partition_view %tv,
       offsets = [%c0, %c0], sizes = [%h, %w]
       : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
```

---

##### `pto.partition_view` - Partition Tensor View

**Summary:** Creates a logical window on a tensor_view using offsets and sizes, producing a `partition_tensor_view`.

**Semantics:**

```
result = source[offsets, sizes]
```

This op captures both static and dynamic shapes. It represents a logical slice without moving data.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `TensorViewType` | Input tensor view |
| `offsets` | `Variadic<Index>` | Dynamic offsets |
| `sizes` | `Variadic<Index>` | Dynamic sizes |

**Results:** `!pto.partition_tensor_view<...>`

**Constraints & Verification:**

- `offsets`/`sizes` counts must match the rank of `source`

**Notes:**

- Pointer arithmetic is modeled as `BasePtr + Offset`, and the logical shape is determined by `sizes`.

**Hardware Mapping:**

- No hardware pipeline (metadata/view construction only)

**Basic Example:**

```mlir
%pv = pto.partition_view %tv, offsets=[%off0, %off1], sizes=[%s0, %s1]
       : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

---

##### `pto.alloc_tile` - Allocate Tile Buffer

**Summary:** Declares the lifetime of a tile buffer. Each `alloc_tile` produces an independent tile buffer instance.

**Semantics:**

```
result = alloc_tile(base_addr, valid_row, valid_col)   // operands are optional
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `base_addr` | `Optional<I64>` | Optional start address for the tile buffer |
| `valid_row` | `Optional<Index>` | Dynamic valid row count |
| `valid_col` | `Optional<Index>` | Dynamic valid column count |

**Results:** `!pto.tile_buf<...>`

**Constraints & Verification:**

- The operation has a custom verifier that checks:
  - If result `v_row`/`v_col` are dynamic (`?`), the corresponding operands must be present
  - If result `v_row`/`v_col` are static, the corresponding operands must be absent
- If `base_addr` is omitted, the address is assigned by the compiler

**Hardware Mapping:**

- No hardware pipeline (allocation/metadata op)

**Basic Example:**

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
%tb3 = pto.alloc_tile addr = %ad : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

##### `pto.subset` - Subview Tile View

**Summary:** Create a strided view from a parent tile. The result tile buffer is a logical subset of the input tile buffer.

**Semantics:**

```
result = source[offsets] with static sizes
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `pto.tile_buf` | Parent tile buffer |
| `offsets` | `Variadic<Index>` | Runtime dynamic offsets [i, j] |
| `sizes` | `I64ArrayAttr` | Static shape [rows, cols] |

**Results:** `pto.tile_buf`

**Constraints & Verification:**

- The verifier derives boxed-vs-non-boxed behavior from `source`'s tile config (`blayout`, `slayout`, `fractal`) and element type.
- For non-boxed layouts (`slayout=none_box`), no additional subset-specific structural checks are enforced.
- For boxed layouts (`slayout != none_box`):
  - The tile layout must be one of the subset layouts supported by the current implementation; otherwise verification fails.
  - `sizes` must be present, must have length 2, and both subset sizes must be positive.
  - The subset sizes must be multiples of the inferred inner boxed shape.
  - `offsets` must have length 2.
  - If an offset is compile-time constant, it must be non-negative and must be a multiple of the inferred inner boxed shape in that dimension.
  - The source tile shape must be statically known.
  - For boxed row-major tiles, the subset must keep the full source column extent, and the column offset must be the constant `0`.
  - For boxed col-major tiles, the subset must keep the full source row extent, and the row offset must be the constant `0`.
- The inferred result type uses:
  - `shape = sizes`
  - the same element type and address space as `source`
  - the same tile config as `source`
  - a `valid_shape` derived from the parent `valid_shape` and constant offsets when possible, otherwise dynamic in that dimension

**Hardware Mapping:**

- No hardware pipeline (view construction only)

**Basic Example:**

```mlir
%sub = pto.subset %src[%i, %j] sizes [32, 32] : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64, v_row=64, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

##### `pto.set_validshape` - Update Dynamic Tile Valid Row/Col In Place

**Summary:** Updates runtime valid row/col metadata directly on an existing dynamic `pto.tile_buf`.

**Semantics:**

```
set_validshape(source, valid_row, valid_col)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `source` | `pto.tile_buf` | Dynamic tile buffer whose runtime valid shape will be updated |
| `valid_row` | `Index` | Runtime valid row count |
| `valid_col` | `Index` | Runtime valid column count |

**Results:** None

**Constraints & Verification:**

- `source` must be a rank-2 `pto.tile_buf`
- `source` must have dynamic valid shape:
  - `v_row = ?`
  - `v_col = ?`
- User-authored PTO IR must use the `pto.tile_buf` form; any memref form seen
  later in the pipeline is compiler-internal lowering state only
- If `valid_row` / `valid_col` are compile-time constants, they must be non-negative and not exceed the tile's static shape bounds

**Hardware Mapping:**

- No hardware pipeline (metadata update op only)
- Lowers in `PTOToEmitC` to updates of the tile's runtime valid-shape fields

**Basic Example:**

```mlir
%src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
pto.set_validshape %src, %vr, %vc
  : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

### 4.2 Buffer-ID Token Operations (A5)

The following operations implement a **buffer-id based ordering model** for the A5 architecture: acquire and release a buffer-id token by high-level sync op type (the op type is mapped to a concrete pipe internally), so that operations guarded by the same buffer-id execute in program order across mapped pipes. They lower to the CCEC builtins `get_buf` and `rls_buf`.

##### `pto.get_buf` - Acquire Buffer-ID Token (A5)

**Summary:** Acquires a buffer-id token for a sync op type (`pipe_event_type` / `sync_op_type`). Used in a buffer-id based ordering model: operations on the mapped pipe that share the same buffer-id are enforced to execute in program order relative to other mapped pipes using the same buffer-id.

**Semantics:**

```
get_buf(op_type, buf_id [, mode])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `PipeEventTypeAttr` / `SyncOpTypeAttr` | High-level sync op type (mapped to concrete pipe) |
| `buf_id` | `I32Attr` | Buffer ID (token identifier) |
| `mode` | `I32Attr` (default: 0) | Optional mode (attribute) |

**Results:** None.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Intended for **A5**; lowered to the CCEC builtin intrinsic `get_buf`

**Basic Example:**

```mlir
pto.get_buf [#pto.pipe_event_type<TVEC>, 0]
pto.get_buf [#pto.pipe_event_type<TMATMUL>, 1] { mode = 0 }
```

---

##### `pto.rls_buf` - Release Buffer-ID Token (A5)

**Summary:** Releases a previously acquired buffer-id token for a sync op type. Used in conjunction with `pto.get_buf`: after operations that were ordered under the same buffer-id complete, `rls_buf` releases the token for that mapped pipe and buffer-id.

**Semantics:**

```
rls_buf(op_type, buf_id [, mode])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `PipeEventTypeAttr` / `SyncOpTypeAttr` | High-level sync op type (mapped to concrete pipe) |
| `buf_id` | `I32Attr` | Buffer ID (must match a prior `pto.get_buf`) |
| `mode` | `I32Attr` (default: 0) | Optional mode (attribute) |

**Results:** None.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Intended for **A5**; lowered to the CCEC builtin intrinsic `rls_buf`

**Basic Example:**

```mlir
pto.get_buf [#pto.pipe_event_type<TVEC>, 0]
// ... operations under buffer-id 0 ...
pto.rls_buf [#pto.pipe_event_type<TVEC>, 0]
pto.rls_buf [#pto.pipe_event_type<TMATMUL>, 1] { mode = 0 }
```

---

### 4.3 DMA Data Movement Operations

#### PadMode

Padding mode for load operations.

| Value | Int | Description |
|-------|-----|-------------|
| `PadNull` | 0 | No padding |
| `PadFirstElem` | 1 | Pad using the first element |
| `PadValue` | 2 | Pad using a specified value |

---

##### `pto.tload` - Load Partition View to Tile

**Summary:** Physical DMA transfer from a global partition view into a local tile buffer.

**Semantics:**

```
For each element (i, j) in the tile valid region:
    dst[i, j] = src[i, j]
```

`partition_tensor_view` and `tile_buf` are both 2-D in this IR profile. `pto.tload` moves data from the global logical view into the local physical tile buffer.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `PartitionTensorViewType` | Source partition view |
| `dst` | `pto.tile_buf` | Destination tile buffer |
| `pad_mode` | `PadModeAttr` (optional) | Padding mode |
| `pad_value` | `AnyType` (optional) | Padding value |
| `left_padding_num` | `Index` (optional) | Left padding count |
| `right_padding_num` | `Index` (optional) | Right padding count |
| `init_out_buffer` | `BoolAttr` (default: false) | Initialize output buffer |
| `init_condition` | `AnyType` (optional) | Init condition |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `i64`, `f16`, `bf16`, `f32`.
  - The destination tile must use `loc=vec` or `loc=mat`.
  - The destination tile element type and source partition element type must have the same bitwidth.
  - Runtime: all source partition extents and the destination valid region must be positive.
- **Implementation checks (A5)**
  - The destination tile element size must be `1`, `2`, `4`, or `8` bytes, and must match the source partition element size.
  - For `i64`, the destination tile `pad` must be `null` or `zero`.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`, GM -> UB)

**Basic Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.tprefetch` - Prefetch Partition View into Tile

**Summary:** Prefetches a GM-backed partition view into a temporary local tile buffer. This maps to PTO-ISA `TPREFETCH(dst, src)` and, unlike most PTO intrinsics, does not add implicit wait-event synchronization in the C++ wrapper.

**Semantics:**

```
TPREFETCH(dst, src)
```

The detailed caching / hint behavior is target-defined by PTO-ISA. In PTOAS the
op is modeled as writing the prefetched data into `dst`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.partition_tensor_view` or lowered GM memref | Source global view |
| `dst` | `pto.tile_buf` or lowered local memref | Destination local tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `src` must be a partition view before lowering, or the corresponding lowered ranked memref form after `PTOViewToMemref`.
- `dst` must be a tile buffer before lowering, or the corresponding lowered ranked memref form after `PTOViewToMemref`.
- `dst` must use `loc=vec` or `loc=mat`.
- Static source extents and static destination valid extents must be positive when known.
- `src` and `dst` element types must have the same element size in bytes.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`, GM -> local tile)

**Basic Example:**

```mlir
pto.tprefetch ins(%pv : !pto.partition_tensor_view<16x16xf16>)
              outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                    v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

##### `pto.tpack` - Pack a Wider Vec Tile into a Narrower Vec Tile

**Summary:** A5-only vector packing operation that narrows a source VEC tile into
a destination VEC tile of the same valid shape.

**Semantics:**

```
TPACK(dst, src)
```

Supported packing directions follow PTO-ISA:
- `b32 -> b16`
- `b32 -> b8`
- `b16 -> b8`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source VEC tile with wider element type |
| `dst` | `pto.tile_buf` | Destination VEC tile with narrower element type |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `pto.tpack` is only supported on **A5** targets.
- `src` and `dst` must both be VEC tiles (`loc=vec`) with row-major layout.
- `src` and `dst` must have the same `valid_shape`.
- Supported element-size pairs are exactly:
  - `4 -> 2` bytes
  - `4 -> 1` bytes
  - `2 -> 1` bytes

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tpack ins(%src : !pto.tile_buf<loc=vec, dtype=i32, rows=128, cols=128,
              v_row=128, v_col=128, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=128, cols=128,
               v_row=128, v_col=128, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.tstore` - Store Tile to Partition View

**Summary:** Stores a 2-D tile buffer back to a 2-D partition view. Supports phase/atomic/relu/pre-quant controls that lower to the corresponding `TSTORE` template overload family.

**Semantics:**

```
For each element (i, j) in the tile valid region:
    dst[i, j] = src[i, j]
```

**Arguments:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `src` | `pto.tile_buf` | `NA` |Source tile buffer |
| `dst` | `PartitionTensorViewType` | `NA` | Destination partition view |
| `preQuantScalar` | `i64` (optional) | `NA` |Optional scalar used by pre-quantized `acc` store forms |
| `stPhase` | `#pto<st_phase ...>` | `unspecified` | Store phase selector (`unspecified/partial/final`) |
| `atomicType` | `#pto<atomic_type ...>` | `atomic_none` | Atomic mode (`atomic_none/atomic_add`) |
| `reluPreMode` | `#pto<relu_pre_mode ...>` | `no_relu` | ReLU pre-processing mode (`no_relu/normal_relu`) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- Common checks:
  - `src` must be `!pto.tile_buf`, `dst` must be `!pto.partition_tensor_view`.
  - Static `dst` shape dims and static `src` valid-shape dims must be positive.
  - If `preQuantScalar` is present, `src` must be `loc=acc`.
  - If `reluPreMode != no_relu`, `src` must be `loc=acc`.
- A2/A3 checks:
  - `src.loc` must be one of `vec/mat/acc`.
  - For `loc=vec` or `loc=mat`:
    - `preQuantScalar` is not allowed.
    - `src` element type must be one of `i8/i16/i32/i64/f16/bf16/f32`.
    - `src`/`dst` element bitwidth must match.
  - For `loc=acc`:
    - `src` element type must be `i32` or `f32`.
    - Without `preQuantScalar`: `dst` element type must be `i32/f32/f16/bf16`.
    - With `preQuantScalar`:
      - `src=i32` -> `dst=i8(ui8)/f16`
      - `src=f32` -> `dst=i8(ui8)`
    - Static/runtime column bound checks on `src`: `1 <= cols <= 4095` and `1 <= valid_shape[1] <= 4095` (when static).
- A5 checks:
  - `src.loc` must be `vec` or `acc` (A5 does not support `mat` here).
  - For `loc=vec`:
    - `preQuantScalar` is not allowed.
    - `src` element type must be one of `i8/i16/i32/i64/f16/bf16/f32`.
    - `src`/`dst` element bitwidth must match.
  - For `loc=acc`:
    - `src` element type must be `i32` or `f32`.
    - Without `preQuantScalar`: `dst` element type must be `i32/f32/f16/bf16`.
    - With `preQuantScalar`:
      - `src=i32` -> `dst=i8(ui8)/f16/bf16`
      - `src=f32` -> `dst=i8(ui8)/f16/bf16/f32`

**Type Note (PTO IR):**

- PTO IR uses signless integers. There is no distinct unsigned integer type in verifier rules; documentation strings like `ui8` are represented by signless `i8` in IR type checks.

**Hardware Mapping:**

- `src=loc=acc`: uses **PIPE_FIX** path.
- `src=loc=vec` or `src=loc=mat`: uses **PIPE_MTE3** path.

**Basic Example:**

```mlir
// 1) TSTORE(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)

// 2) TSTORE<STPhase::Final>(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>}

// 3) TSTORE<TileData, GlobalData, AtomicType::AtomicAdd>(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {atomicType = #pto<atomic_type atomic_add>}

// 4) TSTORE<STPhase::Final, TileData, GlobalData, AtomicType::AtomicAdd>(dst, src)
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>, atomicType = #pto<atomic_type atomic_add>}

// 5) TSTORE<TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}

// 6) TSTORE<STPhase::Final, TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {stPhase = #pto<st_phase final>, atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}

// 7) TSTORE<TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src, preQuantScalar)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, %pq : i64)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}

// 8) TSTORE<STPhase::Final, TileData, GlobalData, AtomicType::AtomicAdd, ReluPreMode::NormalRelu>(dst, src, preQuantScalar)
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32, v_row=32, v_col=32, blayout=col_major, slayout=row_major, fractal=1024, pad=0>, %pq : i64)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {stPhase = #pto<st_phase final>, atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}
```

---

##### `pto.load_scalar` - Load Single Scalar Element

**Summary:** Loads a single scalar element from a pointer at the given offset.

**Semantics:**

```
value = ptr[offset]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `ptr` | `!pto.ptr<...>` | Source pointer |
| `offset` | `index` | Element offset |

**Results:** `AnyType` — the element type of the pointed-to memory.

**Constraints & Verification:**

- The operation has a custom verifier
- `ptr` element type must match the result type

**Hardware Mapping:**

- Scalar load from global

**Basic Example:**

```mlir
%val = pto.load_scalar %ptr[%offset] : !pto.ptr<f32> -> f32
```

---

##### `pto.store_scalar` - Store Single Scalar Element

**Summary:** Stores a single scalar element to a pointer at the given offset.

**Semantics:**

```
ptr[offset] = value
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `value` | `AnyType` | Value to store |
| `ptr` | `!pto.ptr<...>` | Destination pointer |
| `offset` | `index` | Element offset |

**Results:** None.

**Constraints & Verification:**

- The operation has a custom verifier
- `value` type must match the element type of `ptr`

**Hardware Mapping:**

- Scalar store to global memory space.

**Basic Example:**

```mlir
pto.store_scalar %val, %ptr[%offset] : !pto.ptr<f32>, f32
```

---

##### `pto.tmov` - Tile Move Between Local Domains

**Summary:** Moves data between local memory domains (for example `mat/acc/vec/bias/scaling`) using tile buffers, and supports the same optional parameter families as the `TMOV/TMOV_FP` APIs in `pto-isa`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |
| `fp` | `pto.tile_buf` | Optional scaling tile (`loc=scaling`) used by fp-quant/dequant TMOV forms |
| `preQuantScalar` | `i64` | Optional scalar pre-quant parameter used by scalar-quant TMOV forms |
| `accToVecMode` | `pto.acc_to_vec_mode` | Optional acc-to-vec mode template parameter |
| `reluPreMode` | `pto.relu_pre_mode` | Optional relu mode template parameter, default is `NoRelu` |

**Results:** None. Writes into `dst` via DPS pattern.

**Supported PTO IR Forms:**

- `pto.tmov ins(%src) outs(%dst)`
  - maps to `TMOV(dst, src)`
- `pto.tmov ins(%src) outs(%dst) {reluPreMode = ...}`
  - maps to `TMOV<..., ReluPreMode>(dst, src)`
- `pto.tmov ins(%src) outs(%dst) {accToVecMode = ..., reluPreMode = ...}`
  - maps to `TMOV<..., AccToVecMode, ReluPreMode>(dst, src)`
- `pto.tmov ins(%src, %fp) outs(%dst) {reluPreMode = ...}`
  - maps to `TMOV_FP<..., FpTileData, ReluPreMode>(dst, src, fp)`
- `pto.tmov ins(%src, %fp) outs(%dst) {accToVecMode = ..., reluPreMode = ...}`
  - maps to `TMOV<..., FpTileData, AccToVecMode, ReluPreMode>(dst, src, fp)`
- `pto.tmov` with `preQuantScalar`
  - maps to `TMOV<..., ReluPreMode>(dst, src, preQuantScalar)`
  - or `TMOV<..., AccToVecMode, ReluPreMode>(dst, src, preQuantScalar)`

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Static tile shapes must match: `src.rows == dst.rows` and `src.cols == dst.cols`.
  - Supported location pairs (compile-time checked):
    - `loc=mat -> loc=left/right/bias/scaling`
    - `loc=vec -> loc=vec`
    - `loc=acc -> loc=mat`
    - `loc=acc -> loc=vec`
  - `accToVecMode` is only valid for `loc=acc -> loc=vec`.
  - When `fp` or `preQuantScalar` is present, only single-mode acc-to-vec forms are legal.
  - `reluPreMode` / `fp` / `preQuantScalar` forms require `loc=acc` source.
  - For `loc=acc -> loc=mat`, additional fractal and dtype constraints apply (for example `acc` uses accumulator-style layout, `mat` uses `fractal=512`, and only selected dtype conversions are legal).
- **Implementation checks (A5)**
  - For `loc=mat -> *`, static tile shapes must match; for some `loc=vec` moves, the effective copy size is the min of the source and destination valid regions.
  - Supported location pairs include (target-dependent):
    - `loc=mat -> loc=left/right/bias/scaling/scale`
    - `loc=vec -> loc=vec` and `loc=vec -> loc=mat`
    - `loc=acc -> loc=vec` and `loc=acc -> loc=mat`
  - `accToVecMode` is only valid for `loc=acc -> loc=vec`.
  - When `fp` or `preQuantScalar` is present, only single-mode acc-to-vec forms are legal.
  - `reluPreMode` / `fp` / `preQuantScalar` forms require `loc=acc` source.
  - `loc=mat -> loc=left/right` has additional target-specific fractal and dtype constraints.
  - `loc=acc -> loc=vec/mat` has additional target-specific fractal, dtype, and alignment constraints.
  - `loc=mat -> loc=scale` has additional target-specific fractal and dtype constraints.

**Hardware Mapping:**

- `vec -> vec` executes on **PIPE_V**
- `mat -> left/right/bias/scaling` executes on **PIPE_MTE1**
- `acc -> mat/vec` executes on **PIPE_FIX**

**Basic Example:**

```mlir
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.ttrans` - Transpose Tile

**Summary:** Transposes a tile buffer, using a temporary buffer (tmp is required, TBD).

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[j, i]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary buffer |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Source and destination tile element type must match.
  - The source tile must use `blayout=row_major`.
  - Element size must be `1`, `2`, or `4` bytes.
  - Supported element types are restricted per element width:
    - 4 bytes: `i32`, `f32`
    - 2 bytes: `i16`, `f16`, `bf16`
    - 1 byte: `i8`
  - The transpose domain is taken from the source tile valid region.
- **Implementation checks (A5)**
  - Source and destination tile element sizes must match.
  - 32-byte alignment constraints are enforced on the major dimension of both input and output (for `blayout=row_major`, check `cols * sizeof(T) % 32 == 0`; for `blayout=col_major`, check `rows * sizeof(T) % 32 == 0`).
  - Supported element types are restricted per element width:
    - 4 bytes: `i32`, `f32`
    - 2 bytes: `i16`, `f16`, `bf16`
    - 1 byte: `i8`
  - The implementation operates over the static tile shape (`rows/cols`) and does not consult the valid region.
- **Temporary tile**:
  - The C++ API requires `tmp`, but some implementations may not use it.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.ttrans ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
           outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

### 4.4 Matrix Compute Operations

##### `pto.tmatmul` - Matrix Multiply (Tile World)

**Summary:** Matrix multiplication producing an accumulator tile.

**Semantics:**

```
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix (L0A) |
| `rhs` | `pto.tile_buf` | Right matrix (L0B) |
| `dst` | `pto.tile_buf` | Destination (L0C accumulator) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m/k/n` (taken from `lhs valid row`, `lhs valid column`, `rhs valid column`) must be in `[1, 4095]`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - Runtime: `m/k/n` (taken from `lhs valid row`, `lhs valid column`, `rhs valid column`) must be in `[1, 4095]`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul ins(%a, %b : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>,
                          !pto.tile_buf<loc=right, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
            outs(%c : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
```

---

##### `pto.tmatmul.acc` - Matrix Multiply with Accumulation

**Summary:** Matrix multiplication with accumulation (`C = C_in + A * B`).

**Semantics:**

```
For each (i, j):
    dst[i, j] = acc_in[i, j] + sum_k lhs[i, k] * rhs[k, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `acc_in` | `pto.tile_buf` | Previous accumulator value |
| `lhs` | `pto.tile_buf` | Left matrix |
| `rhs` | `pto.tile_buf` | Right matrix |
| `dst` | `pto.tile_buf` | Destination accumulator |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- All constraints from `pto.tmatmul` apply to the `(Destination accumulator, Left matrix, Right matrix)` triple.
- **A2/A3 and A5 notes:**
  - `lhs valid row`, `lhs valid column`, and `rhs valid column` for `m/k/n`.
  - `acc_in Matrix` is not validated by explicit assertions in the current implementations (target-defined behavior).

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.acc ins(%c_in, %a, %b : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>,
                               !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>,
                               !pto.tile_buf<loc=right, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
               outs(%c_out : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
```

---

##### `pto.tmatmul.bias` - Matrix Multiply with Bias

**Summary:** Matrix multiplication with bias addition (`C = A * B + bias`).

**Semantics:**

```
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j] + bias[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix |
| `rhs` | `pto.tile_buf` | Right matrix |
| `bias` | `pto.tile_buf` | Bias tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- All constraints from `pto.tmatmul` apply to the `(Destination accumulator, Left matrix, Right matrix)` triple.
- **A2/A3 bias constraints:**
  - `bias` element type must match `dst` element type.
  - `bias` must use `loc=bias` and `rows=1`.
- **A5 bias constraints:**
  - `bias` element type must match `dst` element type.
  - `bias` must use `loc=bias`, `rows=1`, and `blayout=row_major`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.bias ins(%a, %b, %bias : !pto.tile_buf<loc=left, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=row_major, fractal=512, pad=0>,
                                   !pto.tile_buf<loc=right, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>,
                                   !pto.tile_buf<loc=bias, dtype=f32, rows=1, cols=16, v_row=1, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
                outs(%c : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)
```

---

##### `pto.tmatmul.mx` - Mixed-Precision Matrix Multiply

**Summary:** Matrix multiplication with additional scaling tiles for mixed-precision/quantized matmul.

**Semantics:**

```
For each (i, j):
    dst[i, j] = sum_k lhs[i, k] * rhs[k, j]
// scaling tiles configure target-defined quantization behavior
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix |
| `lhs_scale` | `pto.tile_buf` | Left scaling tile |
| `rhs` | `pto.tile_buf` | Right matrix |
| `rhs_scale` | `pto.tile_buf` | Right scaling tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A5)**
  - `m/k/n` are taken from `lhs valid row`, `lhs valid column`, and `rhs valid column`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                               !pto.tile_buf<...>, !pto.tile_buf<...>)
               outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tmatmul.mx.acc` - Mixed-Precision Matmul with Accumulation

**Summary:** Mixed-precision matrix multiplication with accumulation.

**Semantics:**

```
dst = acc_in + (lhs * rhs)   // scaling tiles configure target-defined behavior
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `acc_in` | `pto.tile_buf` | Accumulator input |
| `lhs` | `pto.tile_buf` | Left matrix |
| `lhs_scale` | `pto.tile_buf` | Left scaling tile |
| `rhs` | `pto.tile_buf` | Right matrix |
| `rhs_scale` | `pto.tile_buf` | Right scaling tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A5)**
  - `m/k/n` are taken from `lhs valid row`, `lhs valid column`, and `rhs valid column`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                      !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                   outs(%c_out : !pto.tile_buf<...>)
```

---

##### `pto.tmatmul.mx.bias` - Mixed-Precision Matmul with Bias

**Summary:** Mixed-precision matrix multiplication with bias addition.

**Semantics:**

```
dst = (lhs * rhs) + bias   // scaling tiles configure target-defined behavior
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Left matrix |
| `lhs_scale` | `pto.tile_buf` | Left scaling tile |
| `rhs` | `pto.tile_buf` | Right matrix |
| `rhs_scale` | `pto.tile_buf` | Right scaling tile |
| `bias` | `pto.tile_buf` | Bias tile |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A5)**
  - `m/k/n` are taken from `lhs valid row`, `lhs valid column`, and `rhs valid column`.
- **Bias form**:
  - `bias` must use element type `f32`, `loc=bias`, and `rows=1` (the current implementation enforces this with compile-time checks).

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tmatmul.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                            !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                    outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv` - Matrix-Vector Multiply

**Summary:** General matrix-vector multiplication.

**Semantics:**

```
For each row i:
    dst[i, 0] = sum_j lhs[i, j] * rhs[j, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Matrix |
| `rhs` | `pto.tile_buf` | Vector |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[1, 4095]`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - No explicit runtime range checks on `m/k/n` are enforced in `TMATMUL_IMPL` on this target.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[1, 4095]`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.acc` - Matrix-Vector Multiply with Accumulation

**Summary:** Matrix-vector multiplication with accumulation.

**Semantics:**

```
dst = acc_in + (lhs * rhs)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `acc_in` | `pto.tile_buf` | Accumulator input |
| `lhs` | `pto.tile_buf` | Matrix |
| `rhs` | `pto.tile_buf` | Vector |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**


- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[1, 4095]`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - No explicit runtime range checks on `m/k/n` are enforced in `TMATMUL_IMPL` on this target.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[1, 4095]`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.acc ins(%c_in, %a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%c_out : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.bias` - Matrix-Vector Multiply with Bias

**Summary:** Matrix-vector multiplication with bias addition.

**Semantics:**

```
dst = (lhs * rhs) + bias
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `pto.tile_buf` | Matrix |
| `rhs` | `pto.tile_buf` | Vector |
| `bias` | `pto.tile_buf` | Bias vector |
| `dst` | `pto.tile_buf` | Destination |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Supported `(dst element type, lhs element type, rhs element type)` triples:
    - `(i32, i8, i8)`
    - `(f32, f16, f16)`
    - `(f32, f32, f32)`
    - `(f32, bf16, bf16)`
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - Tile locations: `lhs.loc=left`, `rhs.loc=right`, `dst.loc=acc`.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[1, 4095]`.
  - Bias checks:
    - The bias tile element type must exactly match the result tile element type.
    - The bias tile must be configured as a single row.
    - The bias tile must use `loc=bias`.
- **Implementation checks (A5)**
  - The destination element type must be `i32` or `f32`.
    - If the destination element type is `i32`, the lhs and rhs element types must both be `i8`.
    - If the destination element type is `f32`, the lhs/rhs element types support `f16`, `bf16`, `f32`, and selected fp8 pairs (target-defined).
  - Shape constraints: `lhs.rows == dst.rows`, `lhs.cols == rhs.rows`, and `rhs.cols == dst.cols`.
  - PTO-visible layout/fractal constraints:
    - `lhs.loc=left`, `lhs.blayout=col_major`, `lhs.slayout=row_major`
    - `rhs.loc=right`, `rhs.blayout=row_major`, `rhs.slayout=col_major`
    - `dst.loc=acc`, `dst.blayout=col_major`, `dst.slayout=row_major`
  - No explicit runtime range checks on `m/k/n` are enforced in `TMATMUL_IMPL` on this target.
  - Runtime: `m` must be `1`; `k/n` (taken from `rhs valid row`, `rhs valid column`) must be in `[1, 4095]`.
  - Bias checks:
    - The bias tile element type must exactly match the result tile element type.
    - The bias tile must be configured as a single row.
    - The bias tile must use `loc=bias`.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.bias ins(%a, %b, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
              outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.mx` - Mixed-Precision Matrix-Vector Multiply

**Summary:** Mixed-precision GEMV with explicit A/B scaling tiles.

**Semantics:**

```
dst = gemv(a, b)   // quantization/mixed-precision behavior is target-defined
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `a` | `pto.tile_buf` | Matrix tile (`loc=left`) |
| `a_scale` | `pto.tile_buf` | Scale tile associated with `a` |
| `b` | `pto.tile_buf` | Vector tile (`loc=right`) |
| `b_scale` | `pto.tile_buf` | Scale tile associated with `b` |
| `dst` | `pto.tile_buf` | Destination accumulator tile (`loc=acc`) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `a/b/dst` reuse the same GEMV shape/location checks as `pto.tgemv`.
- `a_scale` and `b_scale` must be valid tile buffers.

**Hardware Mapping:**

- Executes on the **Matrix pipeline** (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.mx ins(%a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                            !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%c : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.mx.acc` - Mixed-Precision GEMV with Accumulation

**Summary:** Mixed-precision GEMV accumulation form using scale tiles.

**Semantics:**

```
dst = c_in + gemv(a, b)
```

**Arguments:** `c_in, a, a_scale, b, b_scale, dst`

**Hardware Mapping:** Matrix pipeline (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.mx.acc ins(%c_in, %a, %a_scale, %b, %b_scale : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                        !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                 outs(%c_out : !pto.tile_buf<...>)
```

---

##### `pto.tgemv.mx.bias` - Mixed-Precision GEMV with Bias

**Summary:** Mixed-precision GEMV bias form using scale tiles.

**Semantics:**

```
dst = gemv(a, b) + bias
```

**Arguments:** `a, a_scale, b, b_scale, bias, dst`

**Hardware Mapping:** Matrix pipeline (`PIPE_M`)

**Basic Example:**

```mlir
pto.tgemv.mx.bias ins(%a, %a_scale, %b, %b_scale, %bias : !pto.tile_buf<...>, !pto.tile_buf<...>,
                                                            !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
                  outs(%c : !pto.tile_buf<...>)
```

---

### 4.5 Vector Arithmetic Operations

All vector arithmetic operations execute on the **Vector pipeline** (`PIPE_V`) and use `ins`/`outs` with tile buffers in the **VEC (UB)** memory space.

#### Binary Tile-Tile Operations

| Op | Semantics |
|----|----------|
| `pto.tadd` | `dst[i,j] = src0[i,j] + src1[i,j]` |
| `pto.tsub` | `dst[i,j] = src0[i,j] - src1[i,j]` |
| `pto.tmul` | `dst[i,j] = src0[i,j] * src1[i,j]` |
| `pto.tdiv` | `dst[i,j] = src0[i,j] / src1[i,j]` |
| `pto.tmax` | `dst[i,j] = max(src0[i,j], src1[i,j])` |
| `pto.tmin` | `dst[i,j] = min(src0[i,j], src1[i,j])` |
| `pto.trem` | `dst[i,j] = fmod(src0[i,j], src1[i,j])` |
| `pto.tpartadd` | Partial elementwise add |
| `pto.tpartmax` | Partial elementwise max |
| `pto.tpartmin` | Partial elementwise min |
| `pto.tpartmul` | Partial elementwise mul |
| `pto.tprelu` | `dst[i,j] = src0[i,j] > 0 ? src0[i,j] : src1[i,j] * src0[i,j]` |

---

##### `pto.tadd` - Elementwise Add of Two Tiles

**Summary:** Adds two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `f32`, `i16`, `f16`, `bf16`, `i8`.
  - Tile must use row-major layout (`blayout=row_major`).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space
- Implements `OpPipeInterface`

**Basic Example:**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tsub` - Elementwise Subtract of Two Tiles

**Summary:** Subtracts two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Minuend tile buffer |
| `src1` | `pto.tile_buf` | Subtrahend tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsub ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tmul` - Elementwise Multiply of Two Tiles

**Summary:** Multiplies two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `f32`, `i16`, `f16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tdiv` - Elementwise Division of Two Tiles

**Summary:** Divides two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, j]
```

Division-by-zero behavior is target-defined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `tmp` | `pto.tile_buf` | Temporary tile buffer required by the ISA API |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tdiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `f32`, `i16`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Division-by-zero**:
  - Behavior is target-defined.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tdiv ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tmax` - Elementwise Maximum of Two Tiles

**Summary:** Computes the element-wise maximum of two tiles.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tmin` - Elementwise Minimum of Two Tiles

**Summary:** Computes the element-wise minimum of two tiles.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src0`, `src1` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.trem` - Elementwise Remainder of Two Tiles

**Summary:** Computes the element-wise floating-point remainder of two tiles.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Dividend tile buffer |
| `src1` | `pto.tile_buf` | Divisor tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trem ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.
- **Implementation checks (A2A3)**
  - `src0/src1/dst` element type must match, and must be `i32` or `f32`.
  - `tmp` element type must match `dst`.
  - `src0/src1/tmp/dst` must use row-major layout (`blayout=row_major`).
  - `src0/src1/dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row and `tmp.validCol >= dst.validCol`.
- **Implementation checks (A5)**
  - `src0/src1/dst` element type must match, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - `tmp` element type must match `dst`.
  - `src0/src1/tmp/dst` must use row-major layout (`blayout=row_major`).
  - `src0/src1/dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row and `tmp.validCol >= dst.validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trem ins(%a, %b, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
             v_row=1, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tpartadd` - Partial Elementwise Add

**Summary:** Partial elementwise add with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] + src1[i, j]
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

When `src0` and `src1` have different valid regions, the behavior in non-overlapping areas is implementation-defined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tpartadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Only certain partial-validity patterns are handled (e.g., one source equal to `dst` while the other is smaller by valid-rows or valid-cols); other patterns are not supported (target-defined behavior).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tpartmax` - Partial Elementwise Max

**Summary:** Partial elementwise max with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = max(src0[i, j], src1[i, j])
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other input's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Requires `src0` and `src1` valid region to be `<= dst` valid region in both dimensions; other patterns are not supported (target-defined behavior).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tpartmin` - Partial Elementwise Min

**Summary:** Partial elementwise min with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = min(src0[i, j], src1[i, j])
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other input's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Requires `src0` and `src1` valid region to be `<= dst` valid region in both dimensions; other patterns are not supported (target-defined behavior).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tpartmul` - Partial Elementwise Mul

**Summary:** Partial elementwise mul with implementation-defined handling of mismatched valid regions.

**Semantics:**

```
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] * src1[i, j]
```

The valid region is the intersection of each tile's valid rectangle defined by `v_row`/`v_col`; elements outside a tile's valid rectangle are padding/undefined.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - The implementation requires at least one input's valid region to match `dst`'s valid region, and the other input's valid region not greater than `dst`'s valid region (otherwise it asserts).
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`.
  - Requires `src0` and `src1` valid region to be `<= dst` valid region in both dimensions; other patterns are not supported (target-defined behavior).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tpartmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

##### `pto.tprelu` - Parametric ReLU with Per-Element Slope

**Summary:** Applies the Parametric ReLU activation function with a per-element slope tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] > 0 ? src0[i, j] : src1[i, j] * src0[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer (input activations) |
| `src1` | `pto.tile_buf` | Slope tile buffer (per-element negative slopes) |
| `tmp` | `pto.tile_buf` | New temporary source tile buffer for A2/A3. This only a placehold parameter in A5, see examples|
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tprelu ins(<src0>, <src1> : <src0_type>, <src1_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst/src0/src1` element types must be identical, and must be one of: `f16`, `f32`.
  - `tmp` element types must be `u8`.
  - All three tiles must use row-major layout (`blayout=row_major`).
  - For `src0` `src1`: `src valid row == dst valid row` and `src valid column == dst valid column`.
  - For A3, 2 source Tile, destination Tile, temporary space must in different memory range without overlapping.
- **Implementation checks (A5)**
  - `dst/src0/src1` element types must be identical and must be one of: `f16`, `f32`.
  - All three tiles must use row-major layout (`blayout=row_major`).
    - For `src0` `src1`: `src valid row == dst valid row` and `src valid column == dst valid column`.


**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
// A2/A3
pto.tprelu ins(%a, %slopes, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
// A5 tmp reused out %c
pto.tprelu ins(%a, %slopes, %c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

#### Tile-Scalar Operations

| Op | Semantics |
|----|----------|
| `pto.tadds` | `dst[i,j] = src[i,j] + scalar` |
| `pto.tsubs` | `dst[i,j] = src[i,j] - scalar` |
| `pto.tmuls` | `dst[i,j] = src[i,j] * scalar` |
| `pto.tdivs` | `dst[i,j] = src[i,j] / scalar` (or `scalar / src[i,j]`) |
| `pto.tmaxs` | `dst[i,j] = max(src[i,j], scalar)` |
| `pto.tmins` | `dst[i,j] = min(src[i,j], scalar)` |
| `pto.trems` | `dst[i,j] = fmod(src[i,j], scalar)` |

---

##### `pto.tadds` - Elementwise Add Scalar to Tile

**Summary:** Adds a scalar value to every element of a tile buffer.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] + scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer containing the input data |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value to add to each element |
| `dst` | `pto.tile_buf` | Destination tile buffer for the result |

**Results:** None. The operation writes results into `dst` following the Destination-Passing Style (DPS) pattern.

**Assembly Format:**

```
pto.tadds ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
  
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (Unified Buffer / UB)** memory space (`AddressSpace::VEC`)
- The source and destination tile buffers should reside in `VEC` memory (loaded via `tload` from Global Memory)

**Basic Example:**

```mlir
pto.tadds ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tsubs` - Elementwise Subtract Scalar from Tile

**Summary:** Subtracts a scalar value from every element of a tile buffer.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] - scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value to subtract |
| `dst` | `pto.tile_buf` | Destination tile buffer for the result |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsubs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tmuls` - Elementwise Multiply Tile by Scalar

**Summary:** Multiplies every element of a tile buffer by a scalar value.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] * scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar multiplier |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmuls ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmuls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tdivs` - Elementwise Division with Scalar

**Summary:** Divides every element of a tile buffer by a scalar, or divides a scalar by every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] / scalar    (default)
    dst[i, j] = scalar / src[i, j]    (reverse mode)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `AnyType` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar divisor (or dividend in reverse mode) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
// Tile / scalar
pto.tdivs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)

// Scalar / tile (reverse mode)
pto.tdivs ins(<scalar>, <src> : <scalar_type>, <src_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **A2/A3 constraints (both overloads):**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **A5 constraints (both overloads):**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Division-by-zero**:
  - Behavior is target-defined; on A5 the tile/scalar form maps to multiply-by-reciprocal and uses `1/0 -> +inf` for `scalar == 0`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
// tile / scalar
pto.tdivs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)

// scalar / tile (reverse mode)
pto.tdivs ins(%s, %a : f32, !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tmaxs` - Elementwise Max of Tile and Scalar

**Summary:** Computes the element-wise maximum between a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src[i, j], scalar)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmaxs ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **A2/A3 constraints (both overloads):**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid column == dst valid column`.
- **A5 constraints (both overloads):**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmaxs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tmins` - Elementwise Min of Tile and Scalar

**Summary:** Computes the element-wise minimum between a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src[i, j], scalar)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tmins ins(<src>, <scalar> : <src_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tmins ins(%a, %s : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.trems` - Elementwise Remainder with Scalar

**Summary:** Computes the element-wise floating-point remainder of a tile divided by a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = fmod(src[i, j], scalar)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar divisor |
| `tmp` | `pto.tile_buf` | Temporary tile buffer required by the ISA API |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trems ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.
- **Implementation checks (A2A3)**
  - `src/dst` element type must match, and must be `i32` or `f32`.
  - `scalar` type must match the tile element type.
  - `tmp` element type must match `dst`.
  - `src/tmp/dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row and `tmp.validCol >= dst.validCol`.
- **Implementation checks (A5)**
  - `src/dst` element type must match, and must be one of: `i32`, `i16`, `f16`, `f32`.
  - `scalar` type must match the tile element type.
  - `tmp` element type must match `dst`.
  - `src/tmp/dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same `validRow/validCol`.
  - `tmp` must provide at least `1` valid row and `tmp.validCol >= dst.validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trems ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32,
              !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=32,
              v_row=1, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
              v_row=32, v_col=32, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

#### Ternary Operations

| Op | Semantics |
|----|----------|
| `pto.taddc` | `dst = src0 + src1 + src2` |
| `pto.tsubc` | `dst = src0 - src1 + src2` |
| `pto.taddsc` | `dst = src0 + scalar + src1` |
| `pto.tsubsc` | `dst = src0 - scalar + src1` |

---

##### `pto.taddc` - Elementwise Ternary Add of Tiles

**Summary:** Adds three tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j] + src2[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `src2` | `pto.tile_buf` | Third source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.taddc ins(<src0>, <src1>, <src2> : <type0>, <type1>, <type2>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.taddc ins(%a, %b, %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tsubc` - Elementwise Ternary Subtract-Add

**Summary:** Computes `src0 - src1 + src2` element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j] + src2[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Subtrahend tile buffer |
| `src2` | `pto.tile_buf` | Addend tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubc ins(<src0>, <src1>, <src2> : <type0>, <type1>, <type2>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsubc ins(%a, %b, %c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.taddsc` - Fused Add-Scalar-Add

**Summary:** Computes `src0 + scalar + src1` element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + scalar + src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.taddsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.taddsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tsubsc` - Fused Subtract-Scalar-Add

**Summary:** Computes `src0 - scalar + src1` element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - scalar + src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `scalar` | `ScalarType` (signless integer / float) | Scalar value |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsubsc ins(<src0>, <scalar>, <src1> : <type0>, <scalar_type>, <type1>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- The implementation uses `dst valid row` / `dst valid column` as the iteration domain.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsubsc ins(%a, %s, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f32,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

#### Unary Operations

| Op | Semantics |
|----|----------|
| `pto.tabs` | `dst[i,j] = abs(src[i,j])` |
| `pto.tneg` | `dst[i,j] = -src[i,j]` |
| `pto.texp` | `dst[i,j] = exp(src[i,j])` |
| `pto.tlog` | `dst[i,j] = ln(src[i,j])` |
| `pto.tsqrt` | `dst[i,j] = sqrt(src[i,j])` |
| `pto.trsqrt` | `dst[i,j] = 1/sqrt(src[i,j])` |
| `pto.trecip` | `dst[i,j] = 1/src[i,j]` |
| `pto.trelu` | `dst[i,j] = max(0, src[i,j])` |
| `pto.tlrelu` | `dst[i,j] = src[i,j] > 0 ? src[i,j] : slope * src[i,j]` |

---

##### `pto.tabs` - Elementwise Absolute Value

**Summary:** Computes the absolute value of every element in a tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = |src[i, j]|
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tabs ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Implements `OpPipeInterface`
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tabs ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tneg` - Elementwise Negation

**Summary:** Negates every element in a tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = -src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tneg ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `int`, `i16`, `f16`, `f16`, `f32`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`, `bf16`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tneg ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.texp` - Elementwise Exponential

**Summary:** Computes the exponential function for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = exp(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.texp ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.texp ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tlog` - Elementwise Natural Logarithm

**Summary:** Computes the natural logarithm for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = ln(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tlog ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.
- **Domain / NaN**:
  - Domain behavior (e.g., `log(<=0)`) is target-defined.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tlog ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tsqrt` - Elementwise Square Root

**Summary:** Computes the square root for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = sqrt(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsqrt ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.
- **Domain / NaN**:
  - Behavior is target-defined (e.g., for negative inputs).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.ttri` - Fill Triangular Tile Region

**Summary:** Fills a VEC tile using the PTO-ISA `TTRI` triangular pattern.

**Semantics:**

```
TTRI(dst, diagonal)
```

`upperOrLower=0` selects the lower-triangular form and `upperOrLower=1`
selects the upper-triangular form. The exact per-element fill pattern follows
the target PTO-ISA implementation.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `diagonal` | integer SSA value | Runtime diagonal selector |
| `dst` | `pto.tile_buf` | Destination vector tile |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `upperOrLower` | `I32Attr` (default: `0`) | `0` for lower triangular, `1` for upper triangular |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```mlir
pto.ttri ins(%diag {upperOrLower = 1 : i32} : i32)
         outs(%dst : !pto.tile_buf<...>)
```

**Constraints & Verification:**

- `dst` must be a VEC tile (`loc=vec`) whose valid shape stays within the static tile shape.
- `dst` must use `blayout=row_major`.
- `diagonal` must have an integer type.
- `upperOrLower` must be either `0` or `1`.
- Supported element types:
  - A2/A3: `f16`, `f32`, `i16`, `i32`, `u16`, `u32`
  - A5: `f16`, `f32`, `bf16`, `i8`, `i16`, `i32`, `u8`, `u16`, `u32`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.ttri ins(%diag : i32)
         outs(%lower : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
               v_row=32, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)

pto.ttri ins(%diag {upperOrLower = 1 : i32} : i32)
         outs(%upper : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
               v_row=32, v_col=32, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.trsqrt` - Elementwise Reciprocal Square Root

**Summary:** Computes the reciprocal square root for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = 1.0 / sqrt(src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trsqrt ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - `src` and `dst` must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src` and `dst` must have the same valid region;
  - Tiles must use `blayout=row_major`.
- **Domain / NaN**:
  - Behavior is target-defined (e.g., for `src == 0` or negative inputs).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.trecip` - Elementwise Reciprocal

**Summary:** Computes the reciprocal for every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = 1.0 / src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trecip ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**
  - Tile element type must be one of: `f32` or `f16`;
  - Tile must use `loc=vec`;
  - Valid bounds: `valid row <= rows` and `valid column <= cols`;
  - Runtime: `src valid row == dst valid row` and `src valid column == dst valid column`;
  - Tile must use row-major layout (`blayout=row_major`).
  - A3's TRECIP instruction does not support setting the source Tile and destination Tile to the same memory.
- **Domain / NaN**:
  - Division-by-zero behavior is target-defined; the CPU simulator asserts in debug builds.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trecip ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

##### `pto.trelu` - Elementwise ReLU

**Summary:** Applies the Rectified Linear Unit activation function to every element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(0, src[i, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trelu ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `f16`, `f32`, `i32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `f16`, `f32`, `i32`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trelu ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tlrelu` - Leaky ReLU with Scalar Slope

**Summary:** Applies the Leaky ReLU activation function with a scalar slope parameter.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] > 0 ? src[i, j] : slope * src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `slope` | `F32` | Negative slope coefficient |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tlrelu ins(<src>, <slope> : <src_type>, <slope_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `0 < valid row <= rows` and `0 < valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `f16`, `f32`.
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - Runtime: `src` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tlrelu ins(%a, %slope : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>, f32)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### 4.6 Reduction Operations

Reduce along rows or columns of a tile. All execute on the **Vector pipeline** (`PIPE_V`).

| Op | Semantics |
|----|----------|
| `pto.trowsum` | `dst[i,0] = sum_j src[i,j]` |
| `pto.trowmax` | `dst[i,0] = max_j src[i,j]` |
| `pto.trowmin` | `dst[i,0] = min_j src[i,j]` (requires tmp) |
| `pto.tcolsum` | `dst[0,j] = sum_i src[i,j]` (requires tmp, optional isBinary) |
| `pto.tcolmax` | `dst[0,j] = max_i src[i,j]` |
| `pto.tcolmin` | `dst[0,j] = min_i src[i,j]` |

---

##### `pto.trowsum` - Row-wise Sum Reduction

**Summary:** Reduces each row by summing across columns.

**Semantics:**

```
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowsum ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**

- **Implementation checks (A2A3)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - Tile layout of `dst`:
    - **Recommended**: a DN-style 1D column vector tile (`cols=1`, `blayout=col_major`)
    - **Legacy**: an ND-style 2D tile with `valid column == 1`
  - Data types: `f16` or `f32`.
  - Element type consistency: `src_type == dst_type`.
  - Valid checks:
    - `src valid column != 0` and `src valid row != 0`.
    - `src valid row == dst valid row` (the output valid row must match the input valid row).
- **Implementation checks (A5)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - Tile layout of `dst`:
    - **Recommended**: a DN-style 1D column vector tile (`cols=1`, `blayout=col_major`)
    - **Legacy**: an ND-style 2D tile with `valid column == 1`
  - Data types: `f16` or `f32`.
  - Element type consistency: `src_type == dst_type`.
  - Valid checks:
    - `src valid column != 0` and `src valid row != 0`.
    - `src valid row == dst valid row` (the output valid row must match the input valid row).
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowsum ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowmax` - Row-wise Max Reduction

**Summary:** Reduces each row by taking the maximum across columns.

**Semantics:**

```
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**

- **Implementation checks (A2A3)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - Tile layout of `dst`:
    - **Recommended**: a DN-style 1D column vector tile (`cols=1`, `blayout=col_major`)
    - **Legacy**: an ND-style 2D tile with `valid column == 1`
  - Data types: `f16` or `f32`.
  - Element type consistency: `src_type == dst_type`.
  - Runtime valid checks:
    - `src valid column != 0` and `src valid row != 0`.
    - `src valid row == dst valid row` (the output valid row must match the input valid row).
- **Implementation checks (A5)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - Tile layout of `dst`:
    - **Recommended**: a DN-style 1D column vector tile (`cols=1`, `blayout=col_major`)
    - **Legacy**: an ND-style 2D tile with `valid column == 1`
  - Data types: `f16` or `f32`.
  - Element type consistency: `src_type == dst_type`.
  - Runtime valid checks:
    - `src valid column != 0` and `src valid row != 0`.
    - `src valid row == dst valid row` (the output valid row must match the input valid row).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.trowmin` - Row-wise Min Reduction

**Summary:** Reduces each row by taking the minimum across columns. Requires a temporary buffer.

**Semantics:**

```
For each row i:
    dst[i, 0] = min over j of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Temporary buffer (required for intermediate computation) |
| `dst` | `pto.tile_buf` | Destination tile buffer (column vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**

- **Implementation checks (A2A3)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - Tile layout of `dst`:
    - **Recommended**: a DN-style 1D column vector tile (`cols=1`, `blayout=col_major`)
    - **Legacy**: an ND-style 2D tile with `valid column == 1`
  - Data types: `f16` or `f32`.
  - Element type consistency: `src_type == dst_type`.
  - Runtime valid checks:
    - `src valid column != 0` and `src valid row != 0`.
    - `src valid row == dst valid row` (the output valid row must match the input valid row).
- **Implementation checks (A5)**
  - `src` and `dst` must use `loc=vec`.
  - `src` must use ND-style tile layout (`blayout=row_major`, `slayout=none_box`).
  - Tile layout of `dst`:
    - **Recommended**: a DN-style 1D column vector tile (`cols=1`, `blayout=col_major`)
    - **Legacy**: an ND-style 2D tile with `valid column == 1`
  - Data types: `f16` or `f32`.
  - Element type consistency: `src_type == dst_type`.
  - Runtime valid checks:
    - `src valid column != 0` and `src valid row != 0`.
    - `src valid row == dst valid row` (the output valid row must match the input valid row).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.tcolsum` - Column-wise Sum Reduction

**Summary:** Reduces each column by summing across rows. Requires a temporary buffer.

**Semantics:**

```
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `tmp` | `pto.tile_buf` | Temporary buffer (required for intermediate computation) |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |
| `isBinary` | `BoolAttr` (default: `false`) | Use binary reduction tree |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolsum ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>) isBinary = false
```

**Constraints & Verification:**

- **Implementation checks (A2A3):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i16`, `i32`, and `dst_type == tmp_type == src_type`.
- `src valid column == dst valid column`;
- **Implementation checks (A5):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i8`, `i16`, `i32`,`bf16`, and `dst_type == tmp_type == src_type`.
- `src valid row` and `src valid column` must be non-zero; `src valid column == dst valid column` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolsum ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>) isBinary = false
```

---

##### `pto.tcolmax` - Column-wise Max Reduction

**Summary:** Reduces each column by taking the maximum across rows.

**Semantics:**

```
For each column j:
    dst[0, j] = max over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i16`, `i32`, and `dst_type == tmp_type == src_type`.
- `src valid column == dst valid column`;
- **Implementation checks (A5):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i8`, `i16`, `i32`,`bf16`, and `dst_type == tmp_type == src_type`.
- `src valid row` and `src valid column` must be non-zero; `src valid column == dst valid column` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

##### `pto.tcolmin` - Column-wise Min Reduction

**Summary:** Reduces each column by taking the minimum across rows.

**Semantics:**

```
For each column j:
    dst[0, j] = min over i of src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer (row vector) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolmin ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i16`, `i32`, and `dst_type == tmp_type == src_type`.
- `src valid column == dst valid column`;
- **Implementation checks (A5):**
- `src`, `tmp`, and `dst` must use `loc=vec`.
- All tiles must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `src_type` must be one of `f16`, `f32`, `i8`, `i16`, `i32`,`bf16`, and `dst_type == tmp_type == src_type`.
- `src valid row` and `src valid column` must be non-zero; `src valid column == dst valid column` is required.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolmin ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### 4.7 Broadcast Operations

Broadcast values across rows or columns. All execute on the **Vector pipeline** (`PIPE_V`).

| Op | Semantics |
|----|----------|
| `pto.trowexpand` | Broadcast `src[i,0]` across row `i` |
| `pto.tcolexpand` | Broadcast `src[0,j]` across column `j` |
| `pto.tcolexpandmul` | `dst[i,j] = src0[i,j] * src1[0,j]` |
| `pto.tcolexpanddiv` | `dst[i,j] = src0[i,j] / src1[0,j]` |
| `pto.tcolexpandsub` | `dst[i,j] = src0[i,j] - src1[0,j]` |
| `pto.tcolexpandmax` | `dst[i,j] = max(src0[i,j], src1[0,j])` |
| `pto.tcolexpandmin` | `dst[i,j] = min(src0[i,j], src1[0,j])` |
| `pto.trowexpandmul` | `dst[i,j] = src0[i,j] * src1[i,0]` |
| `pto.trowexpanddiv` | `dst[i,j] = src0[i,j] / src1[i,0]` |
| `pto.trowexpandsub` | `dst[i,j] = src0[i,j] - src1[i,0]` |
| `pto.trowexpandadd` | `dst[i,j] = src0[i,j] + src1[i,0]` |
| `pto.texpands` | Broadcast scalar to all elements of dst |

---

##### `pto.trowexpand` - Row-wise Broadcast

**Summary:** Broadcasts the first element of each source row across the entire destination row.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer (column vector) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **NPU constraints:**

- `src` and `dst` must use `loc=vec`.
- `src` must `slayout=none_box`.
- `dst` must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `dst_type == src_type`
- Data type: A2/A3/A5 element types must be one of: `i8` or `i16` or `i32` or `f16` or `bf16` or `f32`.
- requires `src valid row == dst valid row` and requires `src valid row != 0 && src valid column != 0 && dst valid row != 0 && dst valid column != 0`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

##### `pto.tcolexpand` - Column-wise Broadcast

**Summary:** Broadcasts the first element of each source column across the entire destination column.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer (row vector) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- `src` and `dst` must use `loc=vec`.
- Both `src` and `dst` must use ND-style layout (`blayout=row_major`, `slayout=none_box`).
- `dst_type == src_type`
- Data type: A2/A3/A5 element types must be one of: `i8` or `i16` or `i32` or `f16` or `bf16` or `f32`.
- requires `src valid column == dst valid column`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                   v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

##### `pto.tcolexpandmul` - Column-wise Broadcast Multiply

**Summary:** Multiplies each element of `src0` by a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type and the type must be `f16` or `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]`.
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpanddiv` - Column-wise Broadcast Divide

**Summary:** Divides each element of `src0` by a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector (divisor) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type and the type must be `f16` or `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]` (one scalar per destination column).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpandsub` - Column-wise Broadcast Subtract

**Summary:** Subtracts a per-column scalar from `src1` from each element of `src0`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[0, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector (subtrahend) |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type and the type must be `f16` or `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]` (one scalar per destination column).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpandmax` - Column-wise Broadcast Max

**Summary:** Takes the elementwise maximum of `src0` and a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[0, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type and the type must be `f16` or `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.tcolexpandmin` - Column-wise Broadcast Min

**Summary:** Takes the elementwise minimum of `src0` and a per-column scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[0, j])
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-column scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcolexpandmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `src0`, `src1`, `dst` must share the same element type and the type must be `f16` or `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0`, `src1`, `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[1]` must equal `dst valid_shape[1]`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandmul` - Row-wise Broadcast Multiply

**Summary:** Multiplies each row of `src0` by a per-row scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector |
| `tmp` | `Optional<pto.tile_buf>` | Optional scratch tile used by the tmp-taking pto-isa overload |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandmul ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - The shared element type must be one of: `f16`, `f32`.
  - `dst` must use row-major layout (`blayout=row_major`).
 
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandmul ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpanddiv` - Row-wise Broadcast Divide

**Summary:** Divides each row of `src0` by a per-row scalar from `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector (divisor) |
| `tmp` | `Optional<pto.tile_buf>` | Optional scratch tile used by the tmp-taking pto-isa overload |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpanddiv ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - The shared element type must be one of: `f16`, `f32`.
  - `dst` must use row-major layout (`blayout=row_major`).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandsub` - Row-wise Broadcast Subtract

**Summary:** Subtracts a per-row scalar from `src1` from each row of `src0`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector (subtrahend) |
| `tmp` | `Optional<pto.tile_buf>` | Optional scratch tile used by the tmp-taking pto-isa overload |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandsub ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - The shared element type must be one of: `f16`, `f32`.
  - `dst` must use row-major layout (`blayout=row_major`).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandsub ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.trowexpandadd` - Row-wise Broadcast Add

**Summary:** Adds a per-row scalar from `src1` to each row of `src0`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, 0]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer |
| `src1` | `pto.tile_buf` | Per-row scalar vector |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.trowexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks**:
  - `dst`, `src0`, and `src1` must have the same element type.
  - The shared element type must be one of: `f16`, `f32`.
  - `src0` and `dst` must have the same shape and the same `valid_shape`.
  - `src0` and `dst` must use row-major layout (`blayout=row_major`).
  - `src1 valid_shape[0]` must equal `dst valid_shape[0]`.
  - If `src1` is row-major: `src1 valid_shape[1] == 32 / sizeof(dtype)` (`16` for `f16`, `8` for `f32`).
  - If `src1` is not row-major: `src1 valid_shape[1] == 1`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

##### `pto.texpands` - Broadcast Scalar to Tile

**Summary:** Broadcasts a scalar value to all elements of a destination tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `scalar` | `ScalarType` (signless integer / float) | Scalar value to broadcast |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.texpands ins(<scalar> : <scalar_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i32`, `i16`, `f16`, `f32`.
  - Tile must use `loc=vec` or `loc=mat`.
  - If `loc=vec`:
   - Tile must use row-major layout (`blayout=row_major`).
   - Valid bounds: `valid row <= rows` and `valid column <= cols`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i8`, `i16`, `i32`, `f16`, `f32`.
  - Tile must use `loc=vec` or `loc=mat`.
  - If `loc=vec`:
   - Valid bounds: `valid row <= rows` and `valid column <= cols`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space
- Has `MemWrite` memory effect

**Basic Example:**

```mlir
pto.texpands ins(%scalar : f32)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### 4.8 Compare & Select Operations

#### CmpMode

Comparison modes for `pto.tcmp` / `pto.tcmps`.

| Value | Int | Mnemonic |
|-------|-----|----------|
| `EQ` | 0 | `equal` |
| `NE` | 1 | `not_equal` |
| `LT` | 2 | `less_than` |
| `LE` | 3 | `less_equal` |
| `GT` | 4 | `greater_than` |
| `GE` | 5 | `greater_equal` |

**Attribute syntax:** `#pto<cmp less_than>`

---

#### `pto.tcmp`

**Summary:** Compares two tiles element-wise and writes a packed predicate mask.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = (src0[i, j] <cmpMode> src1[i, j]) ? 1 : 0
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First operand |
| `src1` | `pto.tile_buf` | Second operand |
| `dst` | `pto.tile_buf` | Destination mask |
| `cmpMode` | `CmpModeAttr` (optional) | Comparison mode (EQ/NE/LT/LE/GT/GE) |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tcmp ins(<src0>, <src1> {cmpMode = <mode>} : <type0>, <type1>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - Input type must be one of: `i32`, `f16`, `f32`.
  - Output type must be `i8`.
  - `src0/src1/dst` must use `loc=vec`.
  - Valid bounds: `src valid row <= src.rows` and `src valid column <= src.cols`.
  - `src0` and `dst` must have the same valid region: 
      `src0 valid row == src1 valid row == dst valid row` and `src0 valid column == src1 valid column`.
- **Implementation checks (A5)**:
  - Input type must be one of: `i32`, `i16`, `i8`, `f32`, `f16`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcmp ins(%a, %b {cmpMode = #pto<cmp less_than>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### `pto.tcmps`

**Summary:** Compares a tile against a scalar value element-wise.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = (src[i, j] <cmpMode> scalar) ? 1 : 0
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Tile operand |
| `scalar` | `ScalarType` (signless integer / float)| Scalar value to compare against |
| `cmpMode` | `CmpModeAttr` (default: EQ) | Comparison mode |
| `dst` | `pto.tile_buf` | Destination mask |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - Input type must be one of: `i32`, `f16`, `f32`, `i16`.
  - `src` and `dst` must use `loc=vec`.
  - Static valid bounds: `src valid row <= src.rows` and `src valid column <= src.cols`.
  - `src` and `dst` must have the same valid row.
- **Implementation checks (A5)**:
  - Input type must be one of: `i32`, `f16`, `f32`, `i16`, `i8`.
  - `src` and `dst` must use `loc=vec`.
  - Static valid bounds: `src valid row <= src.rows`, `src valid column <= src.cols`, `dst valid row <= dst.rows`, and `dst valid column <= dst.cols`.
  - `src` and `dst` must have the same valid row.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tcmps ins(%a, %s {cmpMode = #pto<cmp less_than>} :
              !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, f16)
          outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

#### `pto.tsel`

**Summary:** Selects between two tiles using a mask tile (per-element selection).

**Semantics:**

```
For each element (i, j):
    dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `mask` | `pto.tile_buf` | Predicate mask |
| `src0` | `pto.tile_buf` | Value when mask is true |
| `src1` | `pto.tile_buf` | Value when mask is false |
| `tmp` | `pto.tile_buf` | Temporary scratch tile required by the current DPS form |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsel ins(<mask>, <src0>, <src1>, <tmp> : <mask_type>, <type0>, <type1>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be a 16-bit or 32-bit type supported by PTO IR: `i16`, `i32`, `f16`, or `f32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
- **Implementation checks (A5)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit type supported by PTO IR: `i8`, `i16`, `i32`, `f16`, or `f32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsel ins(%mask, %a, %b, %tmp : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### `pto.tsels`

**Summary:** Selects between a source tile and a scalar using a mask tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = mask[i, j] ? src[i, j] : scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `mask` | `pto.tile_buf` | Mask tile (select predicate carrier) |
| `src` | `pto.tile_buf` | Source tile |
| `tmp` | `pto.tile_buf` | Temporary scratch tile required by the current DPS form |
| `scalar` | `ScalarType` | Scalar value selected when the mask bit is false |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsels ins(<mask>, <src>, <tmp>, <scalar> : <mask_type>, <src_type>, <tmp_type>, <scalar_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src` and `dst` must have the same element type.
  - The shared element type must be a 16-bit or 32-bit type supported by PTO IR: `i16`, `i32`, `f16`, or `f32`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src` and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit type supported by PTO IR: `i8`, `i16`, `i32`, `f16`, or `f32`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### 4.9 Bitwise Operations

All bitwise operations execute on the **Vector pipeline** (`PIPE_V`) and operate on data in the **VEC (UB)** memory space.

#### Binary Tile-Tile Bitwise

| Op | Semantics |
|----|----------|
| `pto.tand` | `dst = src0 & src1` |
| `pto.tor` | `dst = or(src0, src1)` |
| `pto.txor` | `dst = src0 ^ src1` |
| `pto.tshl` | `dst = src0 << src1` |
| `pto.tshr` | `dst = src0 >> src1` |

---

##### `pto.tand` - Elementwise Bitwise AND

**Summary:** Computes the bitwise AND of two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] & src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tand ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tor` - Elementwise Bitwise OR

**Summary:** Computes the bitwise OR of two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] | src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tor ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>,
            !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

##### `pto.txor` - Elementwise Bitwise XOR

**Summary:** Computes the bitwise XOR of two tiles element-by-element.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] ^ src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile buffer |
| `src1` | `pto.tile_buf` | Second source tile buffer |
| `tmp` | `pto.tile_buf` | New temporary source tile buffer for A2/A3. This only a placehold parameter in A5, see examples|
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src0`, `src1`, `tmp` and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, `src1`, , `tmp` and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
  - `tmp` and `dst` must have the same valid region: `tmp valid row == dst valid row` and `tmp valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1`, and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
// A2/A3
pto.txor ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
// A5: Reuse %dst which is not actually used by A5.
pto.txor ins(%src0, %src1, %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tshl` - Elementwise Shift Left

**Summary:** Shifts each element of `src0` left by the corresponding element of `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] << src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer (values to shift) |
| `src1` | `pto.tile_buf` | Shift amount tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshl ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.tshr` - Elementwise Shift Right

**Summary:** Shifts each element of `src0` right by the corresponding element of `src1`.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src0[i, j] >> src1[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | Source tile buffer (values to shift) |
| `src1` | `pto.tile_buf` | Shift amount tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`, `src1` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - `src0`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  - `src1` and `dst` must have the same valid region: `src1 valid row == dst valid row` and `src1 valid column == dst valid column`.
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshr ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### Unary Bitwise

##### `pto.tnot` - Elementwise Bitwise NOT

**Summary:** Computes the bitwise NOT of every element in a tile.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = ~src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tnot ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

## Constraints

- **Implementation checks (A2A3)**
  - Tile element type must be one of: `i16`.
  - `src.Dtype == dst.Dtype`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` tiles should have the same `validRow/validCol`.
- **Implementation checks (A5)**
  - Tile element type must be one of: `i32`, `i16`, `i8`.
  - `src.Dtype == dst.Dtype`.
  - Tile must use row-major layout (`blayout=row_major`).
  - Tile must use `loc=vec`.
  - Valid bounds: `valid row <= rows` and `valid column <= cols`.
  - `src` and `dst` tiles should have the same `validRow/validCol`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

#### Tile-Scalar Bitwise

| Op | Semantics |
|----|----------|
| `pto.tands` | `dst = src & scalar` |
| `pto.tors` | `dst = or(src, scalar)` |
| `pto.txors` | `dst = src ^ scalar` |
| `pto.tshls` | `dst = src << scalar` |
| `pto.tshrs` | `dst = src >> scalar` |

---

##### `pto.tands` - Bitwise AND with Scalar

**Summary:** Computes the bitwise AND of a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] & scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- Setting the source Tile and destination Tile to the same memory is **Unsupported**.
- **Implementation checks (A2A3)**:
  - `src` and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0` and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0`and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tands ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tors` - Bitwise OR with Scalar

**Summary:** Computes the bitwise OR of a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] | scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Scalar value |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

  - Setting the source Tile and destination Tile to the same memory is **Unsupported**.
- **Implementation checks (A2A3)**:
  - `src0` and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src0` and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src0`and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src0` and `dst` must use row-major layout (`blayout=row_major`).
  - `src0` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
  
**Unsupported**.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tors ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>, i32)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

##### `pto.txors` - Bitwise XOR with Scalar

**Summary:** Computes the bitwise XOR of a tile and a scalar.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] ^ scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Scalar value |
| `tmp` | `pto.tile_buf` | Temporary scratch tile; required by the PTO IR DPS form |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.txors ins(<src>, <scalar>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
          outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

 - Setting the source Tile and destination Tile to the same memory is **Unsupported**.
- **Implementation checks (A2A3)**:
  - `src` and `dst` must have the same element type.
  - The shared element type must be an 8-bit or 16-bit signless integer type supported by PTO IR: `i8`, `i16`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.
  - The DPS form takes a `tmp` scratch tile. On A2/A3 it is used for calculation; on A5 codegen may ignore it, but the PTO IR operand is still required.
- **Implementation checks (A5)**:
  - `sr0`and `dst` must have the same element type.
  - The shared element type must be an 8-bit, 16-bit, or 32-bit signless integer type supported by PTO IR: `i8`, `i16`, `i32`.
  - `src` and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src valid row == dst valid row` and `src valid column == dst valid column`.
  
**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.txors ins(%a, %s, %tmp : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32,
              !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tshls` - Shift Left by Scalar

**Summary:** Shifts each element of a tile left by a scalar amount.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] << scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Shift amount |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshls ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

##### `pto.tshrs` - Shift Right by Scalar

**Summary:** Shifts each element of a tile right by a scalar amount.

**Semantics:**

```
For each element (i, j):
    dst[i, j] = src[i, j] >> scalar
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer |
| `scalar` | `AnySignlessInteger` | Shift amount |
| `dst` | `pto.tile_buf` | Destination tile buffer |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `src1`, and `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.
- **Implementation checks (A5)**:
  - `src`, `dst` must have the same element type.
  - The shared element type must be one of: `i8`, `i16`, `i32`.
  - src and dst tiles must be `loc=vec`.
  - `src`, `dst` must use row-major layout (`blayout=row_major`).
  - `src` and `dst` must have the same valid region: `src0 valid row == dst valid row` and `src0 valid column == dst valid column`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tshrs ins(%a, %s : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>, i32)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### 4.10 Data Rearrangement Operations

#### MaskPattern

Predefined mask patterns for gather operations.

| Value | Int | Pattern |
|-------|-----|---------|
| `P0101` | 1 | Alternating 0-1-0-1 |
| `P1010` | 2 | Alternating 1-0-1-0 |
| `P0001` | 3 | 0-0-0-1 |
| `P0010` | 4 | 0-0-1-0 |
| `P0100` | 5 | 0-1-0-0 |
| `P1000` | 6 | 1-0-0-0 |
| `P1111` | 7 | All ones |

---

##### `pto.tconcat` - Concatenate Tiles (Column-wise)

**Summary:** Concatenates two source tiles along the column dimension into a destination tile.

**Semantics:**

Let \(R\) be `dst` valid rows, \(C_0\) be `src0` valid columns, and \(C_1\) be `src1` valid columns. For each row \(i\):

\[
dst[i, 0:C_0) = src0[i, 0:C_0)
\]
\[
dst[i, C_0:C_0+C_1) = src1[i, 0:C_1)
\]

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src0` | `pto.tile_buf` | First source tile (first segment) |
| `src1` | `pto.tile_buf` | Second tile (second segment) |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tconcat ins(<src0>, <src1> : <src0_type>, <src1_type>)
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2A3)**:
- `src0`, `src1`, and `dst` must have the same element type, and must be one of : `i8/i16/i32/f16/f32/bf16`
- TileType of src and dst tiles must be `loc=vec`
- The total concatenated valid columns must fit in `dst` capacity:
  - `src0.valid_col + src1.valid_col <= dst.cols` (checked when these values are statically known).
- `dst valid row = src0/src1 valid row`, 
- **Implementation checks (A5)**:
- `src0`, `src1`, and `dst` must have the same element type, and must be one of : `i8/i16/i32/f16/f32/bf16`.
- All tiles must `blayout=row_major`
- TileType of src and dst tiles must be `loc=vec`
- The total concatenated valid columns must fit in `dst` capacity:
  - `src0.valid_col + src1.valid_col <= dst.cols` (checked when these values are statically known).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Operates on data in the **VEC (UB)** memory space

**Basic Example:**

```mlir
pto.tconcat ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tgather` - Gather/Select Elements

**Summary:** Gathers elements from a source tile using one of three PTO-ISA-compatible forms:

- index gather: `src + indices + tmp -> dst`
- compare gather: `src + kValue + tmp -> dst + cdst`
- mask-pattern gather: `src + maskPattern -> dst`

**Semantics:**

```
Index form:
    dst[i, j] = src[indices[i, j]]

Compare form:
    dst stores gathered indices that satisfy the scalar compare
    cdst stores the per-row selected-count / compact-count result

Mask form:
    dst[i, j] = src[...] according to mask pattern
```

**Arguments / Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Main destination tile |
| `cdst` | `Optional<pto.tile_buf>` | Secondary destination tile used only by compare form |
| `indices` | `Optional<pto.tile_buf>` | Index tile used only by index form |
| `tmp` | `Optional<pto.tile_buf>` | Temporary tile used by index form and compare form |
| `kValue` | `Optional<scalar>` | Scalar compare value used only by compare form |
| `maskPattern` | `Optional<MaskPatternAttr>` | Mask pattern used only by mask form |
| `cmpMode` | `Optional<CmpModeAttr>` | Compare mode used only by compare form; defaults to `eq` when omitted |
| `offset` | `Optional<i32>` | Compare-form gather base index offset; defaults to `0` when omitted |

**Results:** None. Writes into DPS destinations.

- Index form writes `dst`
- Compare form writes both `dst` and `cdst`
- Mask form writes `dst`

Note: the compare-form C++ API is spelled as `TGATHER(dst, src, k_value, cdst, tmp)`, but in PTO IR the writable operands are grouped under `outs(...)`, so `cdst` appears in `outs(...)` rather than `ins(...)`.

**Assembly Format:**

```mlir
// index + tmp
pto.tgather ins(%src, %indices, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)

// compare + tmp
pto.tgather ins(%src, %kValue, %tmp : !pto.tile_buf<...>, <scalar_type>, !pto.tile_buf<...>)
           outs(%dst, %cdst : !pto.tile_buf<...>, !pto.tile_buf<...>)
           {cmpMode = #pto.cmp<eq|gt>, offset = <i32>}

// mask pattern
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<Pxxxx>} : !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

**Constraints & Verification:**

- Exactly one of the following forms must be used:
  - index form: `indices` and `tmp`
  - compare form: `kValue`, `tmp`, and `cdst`
  - mask form: `maskPattern`
- **Index gather: implementation checks (A2/A3)**:
  - `src` and `dst` element types must match and be one of `i16/i32/f16/f32`.
  - `indices` element type must be `i32`.
  - `tmp` element type must match `indices`.
- **Index gather: implementation checks (A5)**:
  - `src` and `dst` element types must match and be one of `i8/i16/i32/f16/f32`.
  - `indices` element type must be `i16` or `i32`.
- **Compare gather: implementation checks (A2/A3)**:
  - `dst` element type must be `i32`.
  - `src` element type must be `f16/f32`, or `i32` when `cmpMode=eq`.
  - `cmpMode` must be `eq` or `gt`.
  - `src`, `dst` must all be `loc=vec`.
- **Compare gather: implementation checks (A5)**:
  - `dst` element type must be `i32`.
  - `src` element type must be one of `i16/i32/f16/f32`.
  - `cmpMode` must be `eq` or `gt`.
  - `src`, `dst` must all be `loc=vec`.
- **Mask-pattern gather: implementation checks (A2/A3)**:
  - Source element size must be `2` or `4` bytes.
  - `src` and `dst` must both use `loc=vec` and `blayout=row_major`.
  - `src` and `dst` element sizes must match.
- **Mask-pattern gather: implementation checks (A5)**:
  - Source element size must be `1`, `2`, or `4` bytes.
  - `src` and `dst` must both use `loc=vec` and `blayout=row_major`.
  - `src`/`dst` element type must be `i8`, `i16`, `i32`, `f16`, `bf16`, `f32`, or fp8-like supported gather types.
  - `src` and `dst` element sizes must match.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Examples:**

```mlir
// index + tmp
pto.tgather ins(%src, %idx, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)

// compare + tmp
pto.tgather ins(%src, %k, %tmp : !pto.tile_buf<...>, f16, !pto.tile_buf<...>)
           outs(%dst, %cdst : !pto.tile_buf<...>, !pto.tile_buf<...>)
           {offset = 7 : i32}

// mask pattern
pto.tgather ins(%src, {maskPattern = #pto.mask_pattern<P1111>} : !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tgatherb` - Gather by Byte Offsets

**Summary:** Gathers elements using per-element byte offsets.

**Semantics:**

```
dst[i, j] = src[byte_offsets[i, j]]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `offsets` | `pto.tile_buf` | Byte offset tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst` must use row-major layout (`blayout=row_major`).
  - `dst` element size must be `1`, `2`, or `4` bytes.
- **Implementation checks (A5)**
  - Destination element size must be `1`, `2`, or `4` bytes.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tgatherb ins(%src, %offs : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tscatter` - Scatter Rows

**Summary:** Scatters rows from a source tile into a destination tile using per-row indices.

**Semantics:**

```
dst[row_index[i], j] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indexes` | `pto.tile_buf` | Row index tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst`, `src`, and `indexes` must all use `loc=vec`.
  - `dst`/`src` element type must be one of: `i32`, `i16`, `i8`, `f16`, `f32`, `bf16`.
  - `indexes` element type must be one of: `i16`, `i32`.
  - No bounds checks are enforced on `indexes` values.
  - Valid bounds: `dst.valid_shape[i] <= dst.shape[i]`, `src.valid_shape[i] <= src.shape[i]`, and `indexes.valid_shape[i] <= indexes.shape[i]` for each dimension `i`.
  - `dst` and `src` must have the same element type.
  - When `dst` element size is 4 bytes, `indexes` element size must also be 4 bytes.
  - When `dst` element size is 2 bytes, `indexes` element size must also be 2 bytes.
  - When `dst` element size is 1 byte, `indexes` element size must be 2 bytes.
- **Implementation checks (A5)**
  - `dst`, `src`, and `indexes` must all use `loc=vec`.
  - `dst`/`src` element type must be one of: `i32`, `i16`, `i8`, `f16`, `f32`, `bf16`.
  - `indexes` element type must be one of: `i16`, `i32`.
  - No bounds checks are enforced on `indexes` values.
  - Valid bounds: `dst.valid_shape[i] <= dst.shape[i]`, `src.valid_shape[i] <= src.shape[i]`, and `indexes.valid_shape[i] <= indexes.shape[i]` for each dimension `i`.
  - `dst` and `src` must have the same element type.
  - When `dst` element size is 4 bytes, `indexes` element size must also be 4 bytes.
  - When `dst` element size is 2 bytes, `indexes` element size must also be 2 bytes.
  - When `dst` element size is 1 byte, `indexes` element size must be 2 bytes.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.mgather` - Gather-Load from Global Memory

**Summary:** Loads elements from global memory into a tile using per-element indices.

**Semantics:**

```
dst[i, j] = mem[idx[i, j]]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `mem` | `AnyMemRef/pto.tile_buf` | Source memory |
| `idx` | `pto.tile_buf` | Index tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `src.data()`.
- No bounds checks are enforced on `indexes` by the CPU simulator.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE2`)

**Basic Example:**

```mlir
pto.mgather ins(%mem, %idx : memref<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.mscatter` - Scatter-Store to Global Memory

**Summary:** Stores elements from a tile into global memory using per-element indices.

**Semantics:**

```
mem[idx[i, j]] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `idx` | `pto.tile_buf` | Index tile |
| `mem` | `AnyMemRef/pto.tile_buf` | Destination memory |

**Results:** None. Writes into `mem` via DPS pattern.

**Constraints & Verification:**

- Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `dst.data()`.
- No bounds checks are enforced on `indexes` by the CPU simulator.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE3`)

**Basic Example:**

```mlir
pto.mscatter ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%mem : memref<...>)
```

---

##### `pto.treshape` - Reinterpret Tile Shape/Layout

**Summary:** Reinterprets a tile buffer with a new shape/layout (no data movement).

**Semantics:**

```
dst = reinterpret(src)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (different shape) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Tile location must match**: `src.loc == dst.loc`.
- **Total byte size must match**: `sizeof(srcElem) * drcNumel == sizeof(dstElem) * dstNumel`.
- **No boxed/non-boxed conversion**:
  - cannot reshape between `SLayout::NoneBox` and boxed layouts.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.treshape ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tinsert` - Insert Sub-Tile Window

**Summary:** Inserts a source tile into a destination tile at a given row/col offset.

**Semantics:**

```
dst[i + indexRow, j + indexCol] = src[i, j]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indexRow` | `Index` | Destination row offset |
| `indexCol` | `Index` | Destination column offset |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Lowers to **`TINSERT(dst, src, indexRow, indexCol)`**
- Uses target data-movement pipeline (MTE1 by default; A5 UB->L1 path uses MTE3)

**Basic Example:**

```mlir
pto.tinsert ins(%src, %row, %col : !pto.tile_buf<...>, index, index) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.textract` - Extract Sub-Tile Window

**Summary:** Extracts a sub-tile window from a source tile into a destination tile.

**Semantics:**

```
dst[i, j] = src[i + indexRow, j + indexCol]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `indexRow` | `Index` | Starting row |
| `indexCol` | `Index` | Starting column |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - `dst` element type must match `src` element type and must be one of: `i8`, `f16`, `bf16`, `f32`.
  - Source layout/fractal must satisfy one of the target-supported combinations: `slayout=col_major` with `blayout=row_major`, or `slayout=row_major`.
  - Runtime bounds checks:
    - `indexRow + dst.rows <= src.rows`
    - `indexCol + dst.cols <= src.cols`
  - `dst` must use `loc=left` or `loc=right` with a target-supported fractal configuration.
- **Implementation checks (A5)**
  - `dst` element type must match `src` element type and must be one of the target-supported fp8/fp16/bf16/f32 families listed here.
  - Source layout/fractal must satisfy the target-supported combinations for `left`/`right`/scaling destinations; in PTO IR terms this is expressed through the `blayout`/`slayout`/`fractal` tuple.
  - Destination supports `Mat -> Left/Right/Scale` and also supports `Vec -> Mat` for specific tile locations.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.textract ins(%src[%row, %col] : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tfillpad` - Fill Padding Region

**Summary:** Copies `src` into `dst` and fills padded elements using `dst`'s PadVal.

**Semantics:**

```
For valid elements: dst = src
For padded elements: dst = PadVal(dst)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (with pad config) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `dst.pad` must not be `null`.
- `src` and `dst` element sizes must match, and the element size must be `1`, `2`, or `4` bytes.
- `dst.rows/cols` must match `src.rows/cols`.
- `dst.rows >= src.rows` and `dst.cols >= src.cols`.
- For `mat` tiles, the current implementation only supports `blayout=col_major`, `slayout=row_major`, and `pad=zero`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tfillpad_expand` - Fill Padding Region With Expand

**Summary:** Copies `src` into `dst` and fills padded elements using `dst`'s PadVal, allowing `dst` to be larger than `src`.

**Semantics:**

```
For valid elements: dst = src
For padded elements: dst = PadVal(dst)
Constraint: dst.rows >= src.rows and dst.cols >= src.cols
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (with pad config, may be larger) |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- The operation has a custom verifier

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tfillpad_expand ins(%src : !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

---

### 4.11 Sorting Operations

##### `pto.tsort32` - Sort Fixed 32-Element Blocks

**Summary:** Sorts fixed-size 32-element blocks using an explicit input index tile.

**Semantics:**

```
dst = sort(src, idx)
idx = permutation indices for the sort
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Input value tile |
| `idx` | `pto.tile_buf` | Input index tile permuted together with `src` |
| `tmp` | `Optional<pto.tile_buf>` | Optional scratch tile for the tmp-taking DPS overload |
| `dst` | `pto.tile_buf` | Output tile storing sorted value-index pairs |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
pto.tsort32 ins(<src>, <idx>[, <tmp>] : <src_type>, <idx_type>[, <tmp_type>])
           outs(<dst> : <dst_type>)
```

**Constraints & Verification:**

- **Implementation checks (A2/A3/A5)**
  - `dst` element type must be `f16` or `f32`.
  - `src` element type must match `dst` element type.
  - `idx` element type must be `u32`.
  - `src`, `dst`, and `idx` must all use `loc=vec` and `blayout=row_major`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tsort32 ins(%src, %idx : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)

# Optional scratch form:
pto.tsort32 ins(%src, %idx, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tmrgsort` - Merge Sort

**Summary:** Performs merge sort on one or more sorted lists (implementation-defined layout).

**Semantics:**

```
dst = merge_sort(src, blockLen)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` / `src0..src3` | PTO shaped-like type | Input tile(s); format2 supports 2 to 4 sources |
| `blockLen` | `AnyInteger` operand | Block length for format1 |
| `dst` | PTO shaped-like type | Output tile |
| `tmp` | PTO shaped-like type | Temporary output tile for format2 |
| `excuted` | `vector<4xi16>` | Output vector written by format2 |

**Results:** None. Writes into `dst` via DPS pattern.

**Assembly Format:**

```
  - `pto.tmrgsort` has two accepted forms:
    - format1: `ins(src, blockLen : src_type, blockLen_type) outs(dst : dst_type)`
    - format2: `ins(src0, src1[, src2[, src3]] {exhausted = <bool>} : src0_type, src1_type[, src2_type[, src3_type]]) outs(dst, tmp, excuted : dst_type, tmp_type, vector<4xi16>)`
```


**Constraints & Verification:**

- **A2/A3 and Implementation checks (A5)**
  - Element type must be `f16` or `f32` and must match across `dst/tmp/src*` tiles.
  - All tiles must use `loc=vec`, `blayout=row_major`, and `rows == 1` (the list is stored in a single row).
- **Single-list variant (`TMRGSORT(dst, src, blockLen)`)**:
  - `blockLen` must be a multiple of 64 (as checked by the implementation).
  - `src valid column` must be an integer multiple of `blockLen * 4`.
  - `repeatTimes = src valid column / (blockLen * 4)` must be in `[1, 255]`.
- **Multi-list variants**:
  - Accepts 2-way, 3-way, and 4-way merge forms.
  - `dst` and `tmp` must have the same element type and shape.
  - Every `src` must have the same element type as `dst/tmp`.
  - `excuted` must be `vector<4xi16>`.
  - PTOAS maps these forms to the following `pto-isa` APIs:
    - `pto.tmrgsort ins(src0, src1, ...) outs(dst, tmp, excuted)` -> `TMRGSORT(dst, excuted, tmp, src0, src1)`
    - `pto.tmrgsort ins(src0, src1, src2, ...) outs(dst, tmp, excuted)` -> `TMRGSORT(dst, excuted, tmp, src0, src1, src2)`
    - `pto.tmrgsort ins(src0, src1, src2, src3, ...) outs(dst, tmp, excuted)` -> `TMRGSORT(dst, excuted, tmp, src0, src1, src2, src3)`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
// format1
pto.tmrgsort ins(%src, %blockLen : !pto.tile_buf<...>, i32)
             outs(%dst : !pto.tile_buf<...>)

// format2
pto.tmrgsort ins(%src0, %src1 {exhausted = false} :
                 !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst2, %tmp2, %excuted :
                 !pto.tile_buf<...>, !pto.tile_buf<...>, vector<4xi16>)

pto.tmrgsort ins(%src0, %src1, %src2 {exhausted = true} :
                 !pto.tile_buf<...>, !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst3, %tmp3, %excuted :
                 !pto.tile_buf<...>, !pto.tile_buf<...>, vector<4xi16>)

pto.tmrgsort ins(%src0, %src1, %src2, %src3 {exhausted = false} :
                 !pto.tile_buf<...>, !pto.tile_buf<...>,
                 !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst4, %tmp4, %excuted :
                 !pto.tile_buf<...>, !pto.tile_buf<...>, vector<4xi16>)
```

---

### 4.12 Type Conversion

#### RoundMode

Rounding modes for type conversion (`pto.tcvt`) operations.

| Value | Int | Description |
|-------|-----|-------------|
| `NONE` | 0 | No rounding |
| `RINT` | 1 | Round to nearest integer |
| `ROUND` | 2 | Round f16 away from zero |
| `FLOOR` | 3 | Round toward negative infinity |
| `CEIL` | 4 | Round toward positive infinity |
| `TRUNC` | 5 | Truncate toward zero |
| `ODD` | 6 | Round to odd |
| `CAST_RINT` | 7 | Cast with round-to-nearest (default) |

**Attribute syntax:** `#pto<round_mode FLOOR>`

---

##### `pto.tcvt` - Elementwise Type Conversion

**Summary:** Converts each element to a new type with a specified rounding mode.

**Semantics:**

```
dst[i, j] = cast(src[i, j], rmode)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `dst` | `pto.tile_buf` | Destination tile (different element type) |
| `rmode` | `RoundModeAttr` (default: `CAST_RINT`) | Rounding mode |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `dst` and `src` must be compatible in shape/valid region as required by the implementation.
- **A2/A3 and A5 notes:**
  - The current implementation does not add extra compile-time or runtime checks for the type pair; unsupported conversions are target-defined.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>} : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

### 4.13 Integer Sequence Generation Operations

##### `pto.tci` - Contiguous Integer Sequence

**Summary:** Generates a contiguous integer sequence into a destination tile.

**Semantics:**

```
dst[i, j] = S + linear_index(i, j)   // or descending if requested
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `S` | `Integer` | Starting value |
| `dst` | `pto.tile_buf` | Destination tile |
| `descending` | `BoolAttr` (default: false) | Generate descending sequence |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2/A3/A5)**
  - Tile element type must be exactly the same type as the `S`.
  - `dst/scalar` element types must be identical, and must be one of: `i32`, `i16`.
  - `dst.cols != 1`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tci ins(%start : i32) outs(%dst : !pto.tile_buf<...>)
```

---

### 4.14 Scalar Element Access

##### `pto.tgetval` - Read Single Element

**Summary:** Reads a single element from a tile at a linear offset.

**Semantics:**

```
result = src[offset]
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `offset` | `Index` | Linear element offset |

**Results:** Scalar value (`ScalarType`)

**Constraints & Verification:**

- `src` must be a `!pto.tile_buf` or a `memref`.
- `src` must use `loc=vec`.
- If `src` uses `loc=mat`, the current verifier rejects it explicitly because scalar reads from mat tiles are not supported.
- Result type must exactly match the element type of `src`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`) when operating on tile_buf

**Basic Example:**

```mlir
%val = pto.tgetval ins(%src, %off : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>, index) outs : f16
```

---

##### `pto.tsetval` - Write Single Element

**Summary:** Writes a scalar value into a tile at a linear offset.

**Semantics:**

```
dst[offset] = val
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | `pto.tile_buf` | Destination tile |
| `offset` | `Index` | Linear element offset |
| `val` | `ScalarType` (signless integer / float) | Scalar value to write |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

If `dst` is a shaped type, `val` must have exactly the same type as `dst`'s element type.
- The current verifier does not add extra checks on `offset`.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`) when operating on tile_buf

**Basic Example:**

```mlir
pto.tsetval ins(%off, %val : index, f16) outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

### 4.15 MX Quantized Operations

##### `pto.tmov.fp` - Move/Convert with Scaling Tile

**Summary:** Legacy dedicated fp-TMOV op. New code should prefer `pto.tmov` with an `fp` operand, which lowers to the same `TMOV_FP` / fp-parameterized `TMOV` APIs.

**Semantics:**

```
dst[i, j] = Convert(src[i, j]; fp)   // target-defined quantization/dequantization
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile |
| `fp` | `pto.tile_buf` | Scaling (fp) tile |
| `dst` | `pto.tile_buf` | Destination tile |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Src data type only support `f32` or `i32`.
  - `fp` must use `loc=scaling`.
  - Source TileType only support `loc=acc`.
  - Destination TileType only support `loc=mat`.
  - Destination SFractalSize only support fractalABSize(512).
  - Src layout format should be (Blayout: ColMajor, Slayout: RowMajor).
  - Dst layout format should be (Blayout: ColMajor, Slayout: RowMajor).
- **Implementation checks (A5)**
  - Src data type only support `f32` or `i32`.
  - `fp` must use `loc=scaling`.
  - Src layout format should be (Blayout: ColMajor, Slayout: RowMajor).

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`) for accumulator conversion

**Basic Example:**

```mlir
pto.tmov.fp ins(%acc, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
```

---

##### `pto.tquant` - Quantize Tile with Scaling Tile

**Summary:** Quantizes `f32` source tile elements into a lower-precision integer format using a scaling (`fp`) tile. The quantization mode is controlled by the `quant_type` attribute.

**Semantics:**

```
dst[i, j] = Quantize(src[i, j]; fp, quant_type)
```

- `INT8_SYM`: symmetric quantization; `dst` element type must be `i8`.
- `INT8_ASYM`: asymmetric quantization; `dst` element type must be `ui8`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source tile (`f32`) |
| `fp` | `pto.tile_buf` | Scaling parameter tile |
| `dst` | `pto.tile_buf` | Destination tile (`i8` for SYM, `ui8` for ASYM) |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `quant_type` | `#pto.quant_type` | `INT8_SYM` or `INT8_ASYM` |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- `src` element type must be `f32`.
- `dst` element type must be `i8` (`INT8_SYM`) or `ui8` (`INT8_ASYM`).
- A2/A3: `src` and `dst` must use row-major layout.

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)

**Basic Example:**

```mlir
pto.tquant ins(%src, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%dst : !pto.tile_buf<...>)
           {quant_type = #pto<quant_type INT8_SYM>}
```

---

##### `pto.tstore_fp` - Store Accumulator with Scaling

**Summary:** Stores an accumulator tile into global memory using a scaling (`fp`) tile.

**Semantics:**

```
dst[...] = Convert(src[i, j]; fp)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Source accumulator tile |
| `fp` | `pto.tile_buf` | Scaling tile |
| `dst` | `PartitionTensorViewType` | Destination memory |

**Results:** None. Writes into `dst` via DPS pattern.

**Constraints & Verification:**

- **Implementation checks (A2A3)**
  - Source TileType only suport `loc==acc`
  - Source dtype must be `i32` or `f32`.
  - Shape constraints: `1 <= cols <= 4095`;
  - Runtime: `1 <= src valid column <= 4095`.
  - `fp` is used to configure scaling/FPC state; no separate PTO-visible static constraint is enforced on its shape.
- **Implementation checks (A5)**
  - Source TileType only suport `loc==acc`
  - `fp` is used to configure scaling/FPC state; no separate PTO-visible static constraint is enforced on its shape.

**Hardware Mapping:**

- Executes on the **DMA pipeline** (`PIPE_MTE3`)

**Basic Example:**

```mlir
pto.tstore_fp ins(%acc, %fp : !pto.tile_buf<...>, !pto.tile_buf<...>)
             outs(%dst : memref<...>)
```

---

### 4.16 Synchronization Operations

##### `pto.barrier`

**Summary:** Inserts an intra-pipeline memory barrier.

**Semantics:**

```
barrier(pipe)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `pipe` | `PipeAttr` | Pipeline to barrier |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Pipeline barrier for the specified pipe

**Basic Example:**

```mlir
pto.barrier #pto.pipe<PIPE_V>
```

---

##### `pto.barrier_sync`

**Summary:** High-level barrier that specifies a `SyncOpType` instead of a concrete PIPE. The lowering pass maps the op type to the corresponding hardware pipe and emits `pto.barrier`.

**Semantics:**

```
barrier_sync(op_type)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `op_type` | `SyncOpTypeAttr` | High-level sync endpoint (e.g. `TLOAD`, `TSTORE_ACC`, `TMATMUL`, `TVEC`) |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond type consistency

**Hardware Mapping:**

- Pipeline barrier for the specified operation

**Basic Example:**

```mlir
pto.barrier_sync [<TMATMUL>]
pto.barrier_sync [<TVEC>]
```

---

##### `pto.record_event`

**Summary:** Records an event for synchronization between producer and consumer operation classes.

**Semantics:**

```
record_event(src_op, dst_op, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src_op` | `PipeEventKindAttr` | Source operation type |
| `dst_op` | `PipeEventKindAttr` | Destination operation type |
| `event_id` | `EventAttr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Lowered to pipe/event synchronization primitives

**Basic Example:**

```mlir
pto.record_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

---

##### `pto.wait_event`

**Summary:** Waits for a recorded event between producer and consumer operation classes.

**Semantics:**

```
wait_event(src_op, dst_op, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src_op` | `PipeEventKindAttr` | Source operation type |
| `dst_op` | `PipeEventKindAttr` | Destination operation type |
| `event_id` | `EventAttr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Lowered to pipe/event synchronization primitives

**Basic Example:**

```mlir
pto.wait_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

---

#### Cross-Core Synchronization

##### `pto.sync.set`

**Summary:** Sets a synchronization signal between cube and vector cores.

**Semantics:**

```
sync.set(pipe, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `pipe` | `PipeAttr` | Pipeline stage |
| `event_id` | `I32Attr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Cross-core synchronization signal

**Basic Example:**

```mlir
pto.sync.set #pto.pipe<PIPE_M>, 0
```

---

##### `pto.sync.wait`

**Summary:** Waits for a synchronization signal between cube and vector cores.

**Semantics:**

```
sync.wait(pipe, event_id)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `pipe` | `PipeAttr` | Pipeline stage |
| `event_id` | `I32Attr` | Event ID |

**Results:** None.

**Constraints & Verification:**

- No custom verifier beyond attribute validity

**Hardware Mapping:**

- Cross-core synchronization signal

**Basic Example:**

```mlir
pto.sync.wait #pto.pipe<PIPE_V>, 0
```

---

### 4.17 CV-Related Operations

##### `#pto.kernel_kind<cube>` - Cube Kernel Function Attribute

**Summary:** Marks a `func.func` as a Cube-side kernel function.

**Semantics:**

```mlir
func.func @cube_kernel(...) attributes {pto.kernel_kind = #pto.kernel_kind<cube>}
```

The attribute declares that the function is executed in Cube kernel context.
PTOAS uses this attribute to validate Cube-only frontend operations and to
recognize the function as a Cube participant in Cube/Vector communication.

**Attachment Site:** `func.func` attribute.

**Constraints & Verification:**

- Applies to kernel functions only
- Must not conflict with Vector-only frontend operations

**Basic Example:**

```mlir
func.func @cube_kernel()
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  // Cube-only operation
  pto.tmatmul ins(...) outs(...)
  return
}
```

---

##### `#pto.kernel_kind<vector>` - Vector Kernel Function Attribute

**Summary:** Marks a `func.func` as a Vector-side kernel function.

**Semantics:**

```mlir
func.func @vector_kernel(...) attributes {pto.kernel_kind = #pto.kernel_kind<vector>}
```

The attribute declares that the function is executed in Vector kernel context.
PTOAS uses this attribute to validate Vector-only frontend operations and to
recognize the function as a Vector participant in Cube/Vector communication.

**Attachment Site:** `func.func` attribute.

**Constraints & Verification:**

- Applies to kernel functions only
- Must not conflict with Cube-only frontend operations

**Basic Example:**

```mlir
func.func @vector_kernel()
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  // Vector-only operation
  pto.tadd ins(...) outs(...)
  return
}
```

---

##### `pto.section.cube` - Core-Specific Section (Cube)

**Summary:** Marks a region of code that should be emitted only for cube cores.

**Semantics:**

```
section.cube { ... }  // lowered to #if defined(CUBE) ... #endif
```

**Arguments:** None.

**Results:** None.

**Constraints & Verification:**

- The operation has `SingleBlock` and `NoTerminator` traits

**Hardware Mapping:**

- Compile-time control (lowered to preprocessor guards)

**Basic Example:**

```mlir
pto.section.cube {
  // Cube-core-only operations
  pto.tmatmul ins(...) outs(...)
}
```

---

##### `pto.section.vector` - Core-Specific Section (Vector)

**Summary:** Marks a region of code that should be emitted only for vector cores.

**Semantics:**

```
section.vector { ... }  // lowered to #if defined(VECTOR) ... #endif
```

**Arguments:** None.

**Results:** None.

**Constraints & Verification:**

- The operation has `SingleBlock` and `NoTerminator` traits

**Hardware Mapping:**

- Compile-time control (lowered to preprocessor guards)

**Basic Example:**

```mlir
pto.section.vector {
  // Vector-core-only operations
  pto.tadd ins(...) outs(...)
}
```

---

### 4.18 Frontend Pipe Communication Operations

PTOAS exposes a frontend-facing pipe communication interface for Cube/Vector
FIFO-style tile exchange. These operations are intended for frontend/framework
generated IR. The detailed design document is:

- `docs/designs/ptoas-tpush-tpop-design.md`

#### Common Notes

- `dir_mask` uses the current directional encoding:
  - `1`: C2V
  - `2`: V2C
  - `3`: both directions at frontend level
- `slot_size` is expressed in bytes and uses the pre-split logical tile size.
- `split` is a compile-time attribute, not a runtime SSA operand.
- `split = 0/1/2` corresponds to `TILE_NO_SPLIT`, `TILE_UP_DOWN`, and
  `TILE_LEFT_RIGHT`.
- `pto.tpop_from_aic` and `pto.tpop_from_aiv` are result-valued frontend ops.
- A single function currently models at most one logical C2V pipe and one
  logical V2C pipe.

##### `pto.reserve_buffer` - Reserve Local Consumer FIFO Buffer

**Summary:** Declares a local reserved FIFO buffer region for the consumer side
of one frontend logical pipe. The valid way to write this op depends on
whether the active PTOAS compilation flow enables local address planning.

**Syntax:**

```mlir
%buf = pto.reserve_buffer {
  name = "c2v_fifo",
  size = 8192,
  location = #pto.address_space<vec>,
  auto = true
} -> i32
```

When the address is already fixed in the input IR:

```mlir
%buf = pto.reserve_buffer {
  name = "c2v_fifo",
  size = 8192,
  location = #pto.address_space<vec>,
  auto = false,
  base = 4096
} -> i32
```

**Arguments:**

- `name`: string attribute identifying the logical reserved buffer
- `size`: reserved buffer size in bytes
- `location`: local address-space attribute, typically `vec` or `mat`
- `auto`: boolean allocation-mode flag in textual IR
- `base`: optional explicit local base address

**Results:** `i32` local base address value.

**Constraints & Verification:**

- At most one `pto.reserve_buffer` is expected in one function
- `location` must be a supported local address space
- Op-level verification requires:
  - `auto = false` must provide `base`
  - `auto = true` must not provide `base`
- Pipeline compatibility requires:
  - if the active compilation flow enables local address planning: write
    `auto = true` and omit `base`; `PlanMemory` assigns the address and
    `pto-resolve-reserved-buffers` materializes it
  - if the active compilation flow skips local address planning: write
    `auto = false` with explicit `base`; `pto-resolve-reserved-buffers` only
    propagates the pre-resolved address

##### `pto.import_reserved_buffer` - Import Peer Reserved FIFO Buffer

**Summary:** Imports the resolved local FIFO base address from the peer
function's reserved buffer declaration.

**Syntax:**

```mlir
%buf = pto.import_reserved_buffer {
  name = "c2v_fifo",
  peer_func = @vector_kernel
} -> i32
```

**Arguments:**

- `name`: reserved-buffer name in the peer function
- `peer_func`: peer `func.func` symbol

**Results:** `i32` imported local base address value.

**Constraints & Verification:**

- At most one `pto.import_reserved_buffer` is expected in one function
- `peer_func` must contain a matching `pto.reserve_buffer`
- The imported address is resolved by `pto-resolve-reserved-buffers`
  - from the peer `reserve_buffer.base` filled by `PlanMemory` when the active
    compilation flow enables local address planning
  - from the peer's explicit `base` when the active compilation flow skips
    local address planning

##### `pto.aic_initialize_pipe` - Frontend Cube Pipe Initialization

**Summary:** Frontend pipe initialization op used in Cube kernels.

**Syntax:**

```mlir
// A2/A3 (with GM slot buffer):
pto.aic_initialize_pipe {dir_mask = 1, slot_size = 1024}
  (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
   c2v_consumer_buf = %c2v_import : i32,
   v2c_consumer_buf = %c0_i32 : i32)

// A5 (without GM slot buffer):
pto.aic_initialize_pipe {dir_mask = 1, slot_size = 1024}
  (c2v_consumer_buf = %c2v_import : i32,
   v2c_consumer_buf = %c0_i32 : i32)
```

**Arguments:**

- `dir_mask`: communication direction encoding
- `slot_size`: logical slot size in bytes
- `gm_slot_buffer`: optional GM pointer (`!pto.ptr<T>`), required on A2/A3, omitted on A5
- `c2v_consumer_buf`: C2V consumer local base address
- `v2c_consumer_buf`: V2C consumer local base address

**Results:** None.

**Constraints & Verification:**

- Must appear in Cube kernels
- At most one `pto.aic_initialize_pipe` is expected in one Cube function

##### `pto.aiv_initialize_pipe` - Frontend Vector Pipe Initialization

**Summary:** Frontend pipe initialization op used in Vector kernels.

**Syntax:**

```mlir
// A2/A3 (with GM slot buffer):
pto.aiv_initialize_pipe {dir_mask = 1, slot_size = 1024}
  (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
   c2v_consumer_buf = %c2v_local : i32,
   v2c_consumer_buf = %c0_i32 : i32)

// A5 (without GM slot buffer):
pto.aiv_initialize_pipe {dir_mask = 1, slot_size = 1024}
  (c2v_consumer_buf = %c2v_local : i32,
   v2c_consumer_buf = %c0_i32 : i32)
```

**Arguments:** Same operand and attribute structure as
`pto.aic_initialize_pipe`.

**Results:** None.

**Constraints & Verification:**

- Must appear in Vector kernels
- At most one `pto.aiv_initialize_pipe` is expected in one Vector function

##### `pto.tpush_to_aiv` - Frontend C2V Producer Push

**Summary:** Pushes one tile from a Cube kernel to the C2V logical pipe.

**Syntax:**

```mlir
pto.tpush_to_aiv(%tile : !pto.tile_buf<...>) {split = 1}
```

**Arguments:**

- one tile operand
- compile-time `split` attribute

**Results:** None.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents the producer side of a C2V transfer

##### `pto.tpush_to_aic` - Frontend V2C Producer Push

**Summary:** Pushes one tile from a Vector kernel to the V2C logical pipe.

**Syntax:**

```mlir
pto.tpush_to_aic(%tile : !pto.tile_buf<...>) {split = 1}
```

**Arguments:**

- one tile operand
- compile-time `split` attribute

**Results:** None.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents the producer side of a V2C transfer

##### `pto.tpop_from_aic` - Frontend C2V Consumer Pop

**Summary:** Pops one tile from a C2V logical pipe in a Vector kernel.

**Syntax:**

```mlir
%tile = pto.tpop_from_aic {split = 1} -> !pto.tile_buf<...>
```

**Arguments:** compile-time `split` attribute.

**Results:** one `!pto.tile_buf<...>` result tile.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents the consumer side of a C2V transfer

##### `pto.tpop_from_aiv` - Frontend V2C Consumer Pop

**Summary:** Pops one tile from a V2C logical pipe in a Cube kernel.

**Syntax:**

```mlir
%tile = pto.tpop_from_aiv {split = 1} -> !pto.tile_buf<...>
```

**Arguments:** compile-time `split` attribute.

**Results:** one `!pto.tile_buf<...>` result tile.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents the consumer side of a V2C transfer

##### `pto.tfree_from_aic` - Frontend C2V Consumer Free

**Summary:** Releases the current C2V consumer slot in a Vector kernel.

**Syntax:**

```mlir
pto.tfree_from_aic {split = 1}
```

**Arguments:** compile-time `split` attribute.

**Results:** None.

**Constraints & Verification:**

- Must appear in Vector kernels
- Represents the consumer free side of a C2V transfer

##### `pto.tfree_from_aiv` - Frontend V2C Consumer Free

**Summary:** Releases the current V2C consumer slot in a Cube kernel.

**Syntax:**

```mlir
pto.tfree_from_aiv {split = 1}
```

**Arguments:** compile-time `split` attribute.

**Results:** None.

**Constraints & Verification:**

- Must appear in Cube kernels
- Represents the consumer free side of a V2C transfer

---

### 4.19 Runtime Intrinsics

##### `pto.get_block_idx`

**Summary:** Returns the current block (core) index.

**Semantics:**

```
result = block_idx()
```

**Arguments:** None.

**Results:** `i64` block index in `[0, BlockNum-1]`.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%idx = pto.get_block_idx
```

---

##### `pto.get_subblock_idx`

**Summary:** Returns the current sub-block (vector core) index.

**Semantics:**

```
result = subblock_idx()
```

**Arguments:** None.

**Results:** `i64` sub-block index.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%idx = pto.get_subblock_idx
```

---

##### `pto.get_block_num`

**Summary:** Returns the total number of blocks (cores).

**Semantics:**

```
result = block_num()
```

**Arguments:** None.

**Results:** `i64` total block count.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%num = pto.get_block_num
```

---

##### `pto.get_subblock_num`

**Summary:** Returns the total number of vector cores (sub-blocks).

**Semantics:**

```
result = subblock_num()
```

**Arguments:** None.

**Results:** `i64` total sub-block count.

**Constraints & Verification:**

- `Pure` (no side effects)

**Hardware Mapping:**

- Runtime intrinsic (no pipeline)

**Basic Example:**

```mlir
%num = pto.get_subblock_num
```

### 4.20 Debug Operations

##### `pto.tprint` - Print Tile

**Summary:** Prints the contents of a tile for debugging.

**Semantics:**

```
print(src)
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | `pto.tile_buf` | Tile to print |

**Results:** None.

**Constraints & Verification:**

- **Supported element type**:
  - Floating-point: `f32`, `f16`
  - Signless integers (by bitwidth): `i8`, `i16`, `i32`
- **For Tiles**: only `loc=vec` tiles are printable.
- **For GlobalTensor**: Layout must be one of `Layout::ND`, `Layout::DN`, or `Layout::NZ`.

## Behavior
- **Mandatory Compilation Flag**:

  On A2/A3/A5 devices, `TPRINT` uses `cce::printf` to emit output via the device-to-host debug channel. **You must enable the CCE option `-D_DEBUG --cce-enable-print`**.

- **Buffer Limitation:**

  The internal print buffer of `cce::printf` is limited in size. If the output exceeds this buffer, a warning message such as `"Warning: out of bound! try best to print"` may appear, and **only partial data will be printed**.

- **Synchronization**:

  Automatically inserts a `pipe_barrier(PIPE_ALL)` before printing to ensure all prior operations complete and data is consistent.

- **Formatting**:

  - Floating-point values: printed as `%6.2f`
  - Integer values: printed as `%6d`
  - For `GlobalTensor`, due to data size and buffer limitations, only elements within its logical shape (defined by `Shape`) are printed.
  - For `tile_buf`, elements outside `valid_shape` are still printed and are marked with a `|` separator when partial validity is specified.

**Hardware Mapping:**

- Debug/diagnostic intrinsic (implementation-defined)

**Basic Example:**

```mlir
pto.tprint ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

---

##### `pto.print` - Print Scalar with Format String

**Summary:** Prints a scalar value using a compile-time format string (host-visible debug output).

**Semantics:**

```c
printf(format, scalar);
```

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `format` | `StrAttr` | Compile-time format string (e.g. `"%+08.3f"`); must be a literal attribute |
| `ScalarType` (signless integer / float) | Numeric value to print |

**Results:** None.

**Constraints & Verification:**

- `format` is a string attribute; it is not a pointer operand.
- `scalar` must be a numeric type (index / signless integer / f32).
- The op is side-effecting (marked with `MemWrite`) to prevent CSE from removing it.

**Hardware Mapping:**

- Lowered to a call to a debug printing routine (e.g. `cce::printf`) in the generated C++.

**Basic Example:**

```mlir
// Print a single f32 with fixed width/precision.
pto.print ins("%+08.3f", %v : f32)
```

---

##### `pto.trap` - Trap / Abort Execution

**Summary:** Unconditionally aborts execution at runtime. Intended for assertions and debug-only fail-fast paths.

**Semantics:**

```c
trap(); // does not return
```

**Arguments:** None.

**Results:** None.

**Constraints & Verification:**

- May be used anywhere; terminates the current kernel or program as implementation-defined.
- Typically combined with `pto.print` or higher-level assertions for diagnostics.

**Hardware Mapping:**

- Lowered to a device-specific trap/abort intrinsic in the generated C++ (e.g. `TRAP()` or equivalent).

**Basic Example:**

```mlir
// Debug-only guard, e.g. in a lowered assertion.
pto.trap
```

---

### 4.21 Communication Operations

This section documents PTO communication primitives. PTOAS currently exposes:

- Synchronous point-to-point ops: `pto.tput`, `pto.tget`
- Synchronous signal ops: `pto.tnotify`, `pto.twait`, `pto.ttest`
- Synchronous collective ops: `pto.tbroadcast`, `pto.comm_tgather`, `pto.comm_tscatter`, `pto.treduce`
- Asynchronous communication/session ops: `pto.build_async_session`, `pto.tput_async`, `pto.tget_async`, `pto.wait_async_event`, `pto.test_async_event`

##### `pto.build_async_session` - Create Async DMA Session

**Summary:** Creates an async DMA session handle used by `pto.tput_async` and `pto.tget_async`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `scratch` | `pto.tile_buf` / local memref | Local scratch/staging buffer used by the async runtime |
| `workspace` | `!pto.ptr<...>` / GM memref | Global workspace backing the async session |
| `sync_id` | optional `i32` attr | Session synchronization ID |
| `block_bytes` | optional `i64` attr | Communication block size in bytes |
| `comm_block_offset` | optional `i64` attr | Per-block GM offset in bytes |
| `queue_num` | optional `i32` attr | Queue count hint |
| `channel_group_idx` | optional `i64` attr | Communication channel-group selector |

**Results:** `!pto.async_session`

**Constraints & Verification:**

- `scratch` must be tile-like local storage.
- `workspace` must be a GM pointer/memref.
- Optional attrs are forwarded as session configuration and must use the declared integer types.

**Basic Example:**

```mlir
%session = pto.build_async_session(%scratch, %workspace : !pto.tile_buf<loc=vec, dtype=i8, rows=1, cols=256, v_row=1, v_col=256, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.ptr<i8>) {sync_id = 0 : i32} -> !pto.async_session
```

---

##### `pto.tput_async` - Asynchronous Remote Write

**Summary:** Starts an asynchronous remote write from local GM to remote GM and returns an async event handle.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Remote destination buffer |
| `src` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Local source buffer |
| `session` | `!pto.async_session` | Async DMA session |

**Results:** `!pto.async_event`

**Constraints & Verification:**

- `dst` / `src` must be GM-shaped values with identical element type and static shape.
- Current lowering only supports flat contiguous logical-1D transfers for async GM operands.
- `session` must come from `pto.build_async_session`.

**Basic Example:**

```mlir
%event = pto.tput_async(%dst, %src, %session : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.async_session) -> !pto.async_event
```

---

##### `pto.tget_async` - Asynchronous Remote Read

**Summary:** Starts an asynchronous remote read from remote GM to local GM and returns an async event handle.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Local destination buffer |
| `src` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Remote source buffer |
| `session` | `!pto.async_session` | Async DMA session |

**Results:** `!pto.async_event`

**Constraints & Verification:**

- Same operand constraints as `pto.tput_async`.
- `session` must be compatible with the transfer workspace and staging configuration.

**Basic Example:**

```mlir
%event = pto.tget_async(%dst, %src, %session : !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.async_session) -> !pto.async_event
```

---

##### `pto.wait_async_event` / `pto.test_async_event` - Async Event Completion

**Summary:** Consume an async event produced by `pto.tput_async` / `pto.tget_async`.

**Arguments:**

| Op | Operands | Result | Description |
|----|----------|--------|-------------|
| `pto.wait_async_event` | `event`, `session` | `i1` | Blocking wait for completion |
| `pto.test_async_event` | `event`, `session` | `i1` | Non-blocking completion test |

**Constraints & Verification:**

- `event` must have type `!pto.async_event`.
- `session` must have type `!pto.async_session`.
- The event/session pair is expected to come from the same async communication flow.

**Basic Example:**

```mlir
%done0 = pto.wait_async_event(%event0, %session : !pto.async_event, !pto.async_session) -> i1
%done1 = pto.test_async_event(%event1, %session : !pto.async_event, !pto.async_session) -> i1
```

---

##### `pto.tput` - Synchronous Remote Write

**Summary:** Lowers to `pto::comm::TPUT(...)` and copies data from local GM to remote GM through a VEC staging tile.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Remote destination buffer |
| `src` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Local source buffer |
| `ping` | `pto.tile_buf` / local VEC memref | Required staging tile |
| `pong` | `pto.tile_buf` / local VEC memref | Optional second staging tile for ping-pong transfer |
| `atomicType` | `#pto.atomic_type<...>` | Atomic mode, default `atomic_none` |

**Constraints & Verification:**

- `dst` / `src` must be GM-shaped values with positive static shapes.
- `dst` and `src` must have the same element type and static shape.
- `ping` / `pong` must be local VEC tile-like values whose element type matches `src`.

**Basic Example:**

```mlir
pto.tput %dst, %src, %ping {atomicType = #pto.atomic_type<atomic_none>} :
  !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>

pto.tput %dst, %src, %ping, %pong {atomicType = #pto.atomic_type<atomic_add>} :
  !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

##### `pto.tget` - Synchronous Remote Read

**Summary:** Lowers to `pto::comm::TGET(...)` and copies data from remote GM to local GM through a VEC staging tile.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Local destination buffer |
| `src` | GM memref / `pto.tensor_view` / `pto.partition_tensor_view` | Remote source buffer |
| `ping` | `pto.tile_buf` / local VEC memref | Required staging tile |
| `pong` | `pto.tile_buf` / local VEC memref | Optional second staging tile for ping-pong transfer |

**Constraints & Verification:**

- Same GM/global-like and staging constraints as `pto.tput`.
- `dst` and `src` must have the same element type and static shape.

**Basic Example:**

```mlir
pto.tget %dst, %src, %ping :
  !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

---

##### `pto.tnotify` / `pto.twait` / `pto.ttest` - Communication Signal Ops

**Summary:** Lower to `pto::comm::TNOTIFY/TWAIT/TTEST` for GM `i32` signal buffers.

**Arguments:**

| Op | Operands | Attributes | Result |
|----|----------|------------|--------|
| `pto.tnotify` | `signal`, `value` | `notifyOp = #pto.notify_op<atomic_add/set>` | none |
| `pto.twait` | `signal`, `cmpValue` | `cmp = #pto.wait_cmp<eq/ne/gt/ge/lt/le>` | none |
| `pto.ttest` | `signal`, `cmpValue` | `cmp = #pto.wait_cmp<eq/ne/gt/ge/lt/le>` | `i1` |

**Constraints & Verification:**

- `signal` must be a GM-shaped value with element type `i32`.
- `value` / `cmpValue` must be signless integer scalars.

**Basic Example:**

```mlir
pto.tnotify %sig, %v {notifyOp = #pto.notify_op<set>} : !pto.partition_tensor_view<1xi32>, i32
pto.twait %sig, %v {cmp = #pto.wait_cmp<ge>} : !pto.partition_tensor_view<1xi32>, i32
%ok = pto.ttest %sig, %v {cmp = #pto.wait_cmp<eq>} : !pto.partition_tensor_view<1xi32>, i32 -> i1
```

---

##### `pto.tbroadcast` - Collective Broadcast

**Summary:** Lowers to `pto::comm::TBROADCAST(...)`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `src` | GM-shaped value | Root source buffer |
| `ping` / `pong` | local VEC tile-like values | Staging tiles |
| `group` | variadic GM-shaped values | Parallel group members |
| `root` | `i32` attr | Root rank index inside `group` |

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `src` must have the same type as each `group` member.
- `root` must be in range `[0, group.size)`.

**Basic Example:**

```mlir
pto.tbroadcast %src, %ping, %g0, %g1, %g2 {root = 1, operandSegmentSizes = array<i32: 1, 1, 0, 3>} :
  !pto.partition_tensor_view<128xf32>, !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, v_row=1, v_col=128, blayout=row_major, slayout=none_box, fractal=512, pad=0>, !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>, !pto.partition_tensor_view<128xf32>
```

---

##### `pto.comm_tgather` - Collective Gather

**Summary:** Communication collective that lowers to `pto::comm::TGATHER(...)`. This op is distinct from tile-level `pto.tgather`.

**Arguments:** `dst`, `ping`, optional `pong`, variadic `group`, `root`

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `dst` element type must match the group element type.
- `ping` / `pong` must be local VEC tile-like values with matching element type.

---

##### `pto.comm_tscatter` - Collective Scatter

**Summary:** Communication collective that lowers to `pto::comm::TSCATTER(...)`. This op is distinct from tile-level `pto.tscatter`.

**Arguments:** `src`, `ping`, optional `pong`, variadic `group`, `root`

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `src` element type must match the group element type.
- `ping` / `pong` must be local VEC tile-like values with matching element type.

---

##### `pto.treduce` - Collective Reduce

**Summary:** Lowers to `pto::comm::TREDUCE(...)`.

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `dst` | GM-shaped value | Root destination buffer |
| `acc` | local VEC tile-like value | Accumulation tile |
| `recvPing` / `recvPong` | local VEC tile-like values | Receive staging tiles |
| `group` | variadic GM-shaped values | Parallel group members |
| `reduceOp` | `#pto.reduce_op<sum/max/min>` | Reduction mode |
| `root` | `i32` attr | Root rank index inside `group` |

**Constraints & Verification:**

- `group` must be non-empty and all members must have identical types.
- `dst` element type must match the group element type.
- `acc` and `recvPing` / `recvPong` must be local VEC tile-like values whose element type matches `dst`.

---

## 5. Operation Summary Table

| Category | Count | Pipeline |
|----------|-------|----------|
| Pointer/View | 5 | - |
| DMA Data Movement | 4 | MTE2/MTE3/V |
| Matrix Compute | 9 | M (Cube) |
| Vector Arithmetic & Math | 31 | V (Vector) |
| Reduction | 6 | V |
| Broadcast | 6 | V |
| Compare & Select | 4 | V |
| Bitwise | 11 | V |
| Data Rearrangement | 8 | V |
| Sorting | 2 | V |
| Type Conversion | 1 | V |
| Integer Sequence Generation | 1 | V |
| Scalar Element Access | 2 | V |
| MX Quantized | 3 | M/V |
| Synchronization | 5 | - |
| CV-Related | 2 | - |
| Runtime Intrinsics | 4 | - (Pure) |
| Debug | 3 | - |

**Total: 107 operations**
