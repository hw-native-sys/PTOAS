# 15. Special Scalar Memory Access

This chapter covers scalar memory operations that are useful when a kernel
needs one element at a time outside a tile transfer. It explains ordinary
`pto.load` / `pto.store`, explicit L1-bypass access, and SIMT GM access through
`pto.ldg` / `pto.stg`.

## 15.1 Choosing a scalar access form

Use the operation that matches both the execution context and the memory
behavior you need:

| Operation | Execution context | Memory/type contract | Use it for |
|-----------|-------------------|----------------------|------------|
| `pto.load` / `pto.store` | Ordinary entry or SIMT scope | One element through a typed pointer or tile-element form; the element type is inferred from the pointer | Scalar access with the normal scalar-memory behavior |
| `pto.ld_dev` / `pto.st_dev` | Ordinary `@pto.jit` entry | GM pointer with an `i8`, `i16`, `i32`, or `i64` element type | Integer GM metadata or control values that must bypass the local L1 data cache |
| `pto.ldg` / `pto.stg` | `@pto.simt` helper or SIMT scope | GM pointer, optional cache controls, and scalar or supported packed element types | SIMT GM access when cache policy or packed values matter |

`offset` is an element offset in all listed surfaces. It is not a byte offset.
For example, offset `3` on an `i32` pointer addresses the fourth `i32`
element, not byte address `base + 3`.

## 15.2 Scalar-pipeline access

### `pto.load(ptr, offset=0) -> ScalarType`

Loads one element from `ptr` and returns a runtime PTO scalar. The result type
is always the element type of `ptr`; no separate result-type argument is
needed.

`ptr` may refer to the memory space appropriate for the normal scalar-pipeline
access.

### `pto.store(value, ptr, offset=0) -> None`

Stores one scalar `value` to `ptr[offset]`. The value must be compatible with
the pointer element type.

### `pto.ld_dev(ptr, offset=0)` and `pto.st_dev(ptr, offset, value)`

These explicit operations access one integer GM element while bypassing the
local L1 data cache. `ptr` must point to `i8`, `i16`, `i32`, or `i64` in GM.

Both operations are ordinary AICore scalar-pipeline operations. They must not
be placed inside a `@pto.simt` helper or SIMT execution scope. The bypass form
only changes the local L1 data-cache behavior; it does not provide atomicity,
memory ordering, synchronization, or an L2 cache policy.

### Example: normal and L1-bypassing scalar access

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"special_scalar_access_probe","compile":{}} -->
```python
from ptodsl import pto


@pto.jit(target="a5", backend="vpto", mode="explicit")
def special_scalar_access_probe(
    src: pto.ptr(pto.i32, "gm"),
    dst: pto.ptr(pto.i32, "gm"),
):
    # Normal scalar-pipeline access.
    value = pto.load(src, 1)
    pto.store(value, dst, 1)

    # Integer GM access that bypasses the local L1 data cache.
    metadata = pto.ld_dev(src, 3)
    pto.st_dev(dst, 3, metadata)
```

## 15.3 Constraints and diagnostics

The following combinations are rejected during compilation:

- `pto.ld_dev` or `pto.st_dev` is used with a non-GM pointer.
- `pto.ld_dev` or `pto.st_dev` is used with a floating-point or unsupported pointer
  element type.
- Either bypass operation is placed in a SIMT execution scope.
- A store value does not match the destination pointer's element type.

Use `pto.castptr` only when you genuinely have an integer address and know its
memory space and element type. For ordinary GM tensors, pass the typed pointer
from the kernel entry directly.

## 15.4 SIMT scalar and GM access

Use `pto.load` and `pto.store` for ordinary one-element accesses inside a SIMT
scope as well as outside one. They execute one logical access for the current
work-item and accept a tile element such as `pto.load(tile[row, col])` or an
explicit typed pointer and element offset.

`pto.ldg` and `pto.stg` are the SIMT GM-specific forms. They provide explicit
L1/L2 cache-policy arguments and support the broader scalar and packed value
types accepted by the SIMT GM interface. They must remain inside a SIMT
execution scope and should be chosen when those cache controls or packed
values are part of the kernel contract.

For a plain integer GM metadata access in an ordinary AICore entry, prefer
`pto.ld_dev` / `pto.st_dev`.
