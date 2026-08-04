# 18. Special Scalar Operations

> **Category:** PTO scalar query, pointer/address, and scalar-memory operations
> **Dialect:** `pto`

Special Scalar operations provide the PTO-specific scalar facilities used
around vector and tile code. They query the current kernel execution instance,
construct and adjust typed pointers, access one scalar element through the
scalar pipeline, and perform ordinary AICore GM accesses that bypass the local
L1 data cache.

This group does not include shared scalar arithmetic, which remains in
[Arith](14-shared-arith.md), or SIMT workitem operations, which remain in
[SIMT Ops](17-simt.md). An operation with a scalar operand but a vector result,
such as `pto.vadds`, belongs to [Vec-Scalar Ops](08-vec-scalar-ops.md).

---

## Operation Summary

| Family | Operations | Purpose |
|--------|------------|---------|
| Kernel execution queries | `pto.get_block_idx`, `pto.get_subblock_idx`, `pto.get_block_num`, `pto.get_subblock_num` | Query the block or subblock identity and launch extent visible to the current kernel instance |
| Typed pointer/address operations | `pto.castptr`, `pto.addptr` | Construct, reinterpret, and offset `!pto.ptr` values |
| Scalar-pipeline memory | `pto.load_scalar`, `pto.store_scalar` | Read or write one element through the general scalar-memory interface |
| AICore scalar GM L1-bypass | `pto.ld_dev`, `pto.st_dev` | Read or write one integer GM element while bypassing the local L1 data cache |

---

## Common Pointer and Offset Rules

Memory operations in this chapter use the typed pointer form
`!pto.ptr<T, space>`:

- `T` is the element type stored at the pointed-to location;
- `space` identifies the memory space, such as `gm` or `ub`;
- a pointer carries an address, element type, and memory-space interpretation,
  but no tensor shape or stride metadata;
- every `%offset` operand in this chapter has type `index` and is measured in
  elements of `T`, not bytes.

For a base address `base`, element type `T`, and element offset `offset`, the
effective byte address is:

```text
effective_address = base + offset * sizeof(T)
```

For example, offset `3` on `!pto.ptr<i32, gm>` selects the element beginning
12 bytes after the base address.

---

## Kernel Execution Query Operations

These nullary, side-effect-free operations expose the block-level execution
state visible to the current PTO kernel instance. They return `i64` values and
do not perform memory access, synchronization, tiling, or work partitioning by
themselves.

They are distinct from the `pto.get_tid_*`, `pto.get_block_idx_*`, and related
SIMT workitem queries documented in [SIMT Ops](17-simt.md).

### `pto.get_block_idx`

- **Purpose:** Return the linear block index of the current kernel instance.
- **Syntax:**

  ```mlir
  %block = pto.get_block_idx
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` block index.
- **Semantics:** The result identifies the current block in the range
  `[0, block_num)`, where `block_num` is the value returned by
  `pto.get_block_num` for the same launch.

```text
block = current_block_index
0 <= block < block_num
```

### `pto.get_subblock_idx`

- **Purpose:** Return the subblock index visible to the current kernel
  instance.
- **Syntax:**

  ```mlir
  %subblock = pto.get_subblock_idx
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` subblock index.
- **Semantics:** The result identifies the current subblock in the range
  `[0, subblock_num)`, where `subblock_num` is returned by
  `pto.get_subblock_num` for the same execution instance.

```text
subblock = current_subblock_index
0 <= subblock < subblock_num
```

### `pto.get_block_num`

- **Purpose:** Return the total number of blocks in the current kernel launch.
- **Syntax:**

  ```mlir
  %block_num = pto.get_block_num
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` block count.
- **Semantics:** The result is the launch-wide block count used to interpret
  `pto.get_block_idx`.

### `pto.get_subblock_num`

- **Purpose:** Return the number of subblocks visible to the current execution
  instance.
- **Syntax:**

  ```mlir
  %subblock_num = pto.get_subblock_num
  ```

- **Operands and attributes:** None.
- **Result:** One `i64` subblock count.
- **Semantics:** The result is the subblock count used to interpret
  `pto.get_subblock_idx`.

### Block Partitioning Example

The following example assigns a disjoint 2048-element GM window to each
block:

```mlir
%block = pto.get_block_idx
%block_num = pto.get_block_num
%block_len = arith.constant 2048 : index
%block_as_index = arith.index_cast %block : i64 to index
%block_offset = arith.muli %block_as_index, %block_len : index
%block_in = pto.addptr %gm_in, %block_offset
  : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
%block_out = pto.addptr %gm_out, %block_offset
  : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

The query operations report the launch state; the surrounding arithmetic and
pointer operations define the actual partitioning policy.

---

## Typed Pointer and Address Operations

### `pto.castptr`

- **Purpose:** Explicitly convert between an integer address, a typed PTO
  pointer, or a memref base address without moving data.
- **Syntax:**

  ```mlir
  %result = pto.castptr %input : input-type -> result-type
  ```

- **Operands:** `%input` is an integer, a memref, or `!pto.ptr<T, space>`.
- **Result:** An integer or `!pto.ptr<T, space>` according to the selected form.
- **Attributes:** None.
- **Legal forms:**

  | Input | Result | Meaning |
  |-------|--------|---------|
  | integer | `!pto.ptr<T, space>` | Interpret the integer as an address in `space` |
  | `!pto.ptr<T, space>` | integer | Expose the pointer address as an integer |
  | `!pto.ptr<S, space>` | `!pto.ptr<T, space>` | Reinterpret the element type while preserving the address and memory space |
  | `memref<..., space>` | `!pto.ptr<T, space>` | Extract the aligned base address and represent it as a PTO pointer |

- **Constraints:** Integer-to-integer and memref-to-integer forms are invalid.
  Pointer-to-pointer casts must preserve the PTO memory space. A memref with an
  explicit PTO memory space must be cast to the same space. The operation does
  not dereference the address and does not change the referenced bytes.

```text
result.address = input.address
result.space = requested_space
result.element_type = requested_element_type
```

Examples:

```mlir
%gm_i32 = pto.castptr %addr : i64 -> !pto.ptr<i32, gm>
%gm_i8 = pto.castptr %gm_i32
  : !pto.ptr<i32, gm> -> !pto.ptr<i8, gm>
%addr_again = pto.castptr %gm_i8 : !pto.ptr<i8, gm> -> i64
```

### `pto.addptr`

- **Purpose:** Produce a pointer displaced from a typed base pointer.
- **Syntax:**

  ```mlir
  %result = pto.addptr %ptr, %offset
    : !pto.ptr<T, space> -> !pto.ptr<T, space>
  ```

- **Operands:** `%ptr` is `!pto.ptr<T, space>` and `%offset` is `index`.
- **Result:** A pointer with exactly the same element type and memory space as
  `%ptr`.
- **Attributes:** None.
- **Semantics:** `%offset` is a signed element displacement. Positive values
  advance the pointer and negative values move it toward lower addresses.

```text
result.address = ptr.address + offset * sizeof(T)
result.element_type = T
result.space = space
```

- **Constraints:** The result type must exactly match the input pointer type.
  `pto.addptr` does not access memory and does not perform bounds checking.

Example:

```mlir
%c16 = arith.constant 16 : index
%tail = pto.addptr %base, %c16
  : !pto.ptr<f32, gm> -> !pto.ptr<f32, gm>
```

`%tail` points 16 `f32` elements, or 64 bytes, after `%base`.

---

## Scalar-Pipeline Memory Operations

`pto.load_scalar` and `pto.store_scalar` access one element through the general
scalar-memory interface. The pointer element type and memory space determine
the accessed value type and storage domain.

### `pto.load_scalar`

- **Purpose:** Read one scalar element from a typed PTO pointer.
- **Syntax:**

  ```mlir
  %value = pto.load_scalar %ptr[%offset]
    : !pto.ptr<T, space> -> T
  ```

- **Operands:** `%ptr` is `!pto.ptr<T, space>` and `%offset` is an element
  offset of type `index`.
- **Result:** One scalar value of type `T`.
- **Attributes:** None.
- **Semantics:** Read the element at `ptr + offset` through the scalar pipeline.

```text
value = memory[ptr.address + offset * sizeof(T)] as T
```

- **Constraints:** The result type must exactly match the pointer element type.
  This op returns a scalar, not a `!pto.vreg` value, and has no vector load
  distribution or mask clauses.

### `pto.store_scalar`

- **Purpose:** Write one scalar element through a typed PTO pointer.
- **Syntax:**

  ```mlir
  pto.store_scalar %value, %ptr[%offset]
    : !pto.ptr<T, space>, T
  ```

- **Operands:** `%value` has type `T`, `%ptr` is `!pto.ptr<T, space>`, and
  `%offset` is an element offset of type `index`.
- **Results:** None.
- **Attributes:** None.
- **Semantics:** Write `%value` to the element at `ptr + offset` through the
  scalar pipeline.

```text
memory[ptr.address + offset * sizeof(T)] = value
```

- **Constraints:** `%value` must exactly match the pointer element type. This
  op writes one scalar element and has no vector store distribution or mask
  clauses.

Example round trip in UB:

```mlir
%c7 = arith.constant 7 : index
%value = pto.load_scalar %ub[%c7] : !pto.ptr<i32, ub> -> i32
pto.store_scalar %value, %ub_out[%c7] : !pto.ptr<i32, ub>, i32
```

---

## AICore Scalar GM L1-Bypass Operations

`pto.ld_dev` and `pto.st_dev` are the ordinary AICore scalar GM access pair
for accesses that must bypass the local L1 data cache. They are not SIMT
operations and must not be substituted with `pto.ldg` or `pto.stg`, whose
execution scope and cache-control contract are different.

### Common Contract

- the pointer must be `!pto.ptr<T, gm>`;
- `T` must be one of `i8`, `i16`, `i32`, or `i64`;
- `%offset` has type `index` and is measured in elements of `T`;
- load result and store value types must exactly match `T`;
- no `l1cache` or `l2cache` policy attribute is accepted;
- the op must appear in an ordinary AICore entry function, outside both a
  `pto.simt_entry` function and `pto.section.simt`;
- the supported target profile is A5 with CANN output version 9.0.0 official
  or newer.

Both operations are non-atomic. They do not imply synchronization, memory
ordering, cache invalidation, cache writeback, or an L2 cache policy. Programs
that combine these accesses with another cached path must provide any required
synchronization and cache maintenance separately. Cache behavior beyond the
local L1 data cache is target-defined.

### `pto.ld_dev`

- **Purpose:** Read one integer scalar from GM while bypassing the local L1
  data cache.
- **Syntax:**

  ```mlir
  %value = pto.ld_dev %ptr[%offset] : !pto.ptr<T, gm> -> T
  ```

- **Operands:** `%ptr` is `!pto.ptr<T, gm>` and `%offset` is an element offset
  of type `index`.
- **Result:** One value of type `T` containing exactly the bytes read from GM.
- **Attributes:** None.
- **Semantics:** Read `sizeof(T)` bytes from the selected GM element. No sign
  extension, zero extension, truncation, or numeric conversion is part of the
  observable operation semantics.

```text
address = ptr.address + offset * sizeof(T)
value = GM[address : address + sizeof(T)] as T
```

### `pto.st_dev`

- **Purpose:** Write one integer scalar to GM while bypassing the local L1 data
  cache.
- **Syntax:**

  ```mlir
  pto.st_dev %value, %ptr[%offset] : !pto.ptr<T, gm>, T
  ```

- **Operands:** `%value` has type `T`, `%ptr` is `!pto.ptr<T, gm>`, and
  `%offset` is an element offset of type `index`.
- **Results:** None.
- **Attributes:** None.
- **Semantics:** Write exactly `sizeof(T)` bytes from `%value` to the selected
  GM element.

```text
address = ptr.address + offset * sizeof(T)
GM[address : address + sizeof(T)] = value as bytes
```

### Nonzero-Offset Example

```mlir
%c3 = arith.constant 3 : index
%value = pto.ld_dev %src[%c3] : !pto.ptr<i32, gm> -> i32
pto.st_dev %value, %dst[%c3] : !pto.ptr<i32, gm>, i32
```

Both operations access element 3, which begins 12 bytes after the corresponding
`i32` base address. The load and store bypass the local L1 data cache; they do
not establish ordering with other memory operations.

---

## Choosing a Scalar Memory Operation

| Requirement | Operation family |
|-------------|------------------|
| General typed scalar access through the scalar-memory interface | `pto.load_scalar`, `pto.store_scalar` |
| Ordinary AICore integer GM access that bypasses local L1 | `pto.ld_dev`, `pto.st_dev` |
| SIMT workitem scalar memory access | See [SIMT Ops](17-simt.md) |
