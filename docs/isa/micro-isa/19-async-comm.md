# 19. Async Communication

> **Category:** Asynchronous GM↔GM engine transfers
> **Pipelines:** SDMA engine kick from an ordinary AICore scalar stream

This group copies a contiguous GM range through the SDMA engine. The kick does
not wait for the engine except where `{soft_put}` is documented below. The op
does not publish a completion record; local drain and cross-rank visibility are
arranged by the caller.

This document describes:

- `pto.session_init`
- `pto.sdma_gm_gm`

There is no `mte_gm_gm`. Synchronous GM↔UB copies remain in
[2. DMA Copy Programming](02-dma-copy.md).

These ops must sit in an ordinary AICore entry function. They are illegal
inside `pto.simt_entry` functions and `pto.section.simt`.

---

## Session

A session cannot be a kernel argument: only `pto.declare_struct` may produce a
`!pto.struct`. The host therefore writes a GM template, and the kernel declares
its own struct and fills it with `pto.session_init`.

The session type is fixed:

```mlir
!pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
```

The template uses one 8-byte slot per field, in the same order. Narrow fields
occupy the low half of their slot. Each core fills its own copy, so a session
is per-core even when the template is shared and read-only.

After the fill, a kernel may retune individual fields with `pto.struct_set`.
The channel group is field 4, which is how a multi-core launch gives each core
its own queue without the host naming the core.

---

## Operation Summary

| Operation | Purpose |
|-----------|---------|
| `pto.session_init` | Copy the host template into a stack-local session struct |
| `pto.sdma_gm_gm` | Kick a contiguous GM→GM copy through the session |

---

### `pto.session_init`

- **Purpose:** Fill `session` in place from the host-written GM template.
- **Syntax:**

  ```mlir
  pto.session_init %sess, %sess_gm
    : !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>,
      !pto.ptr<i8, gm>
  ```

- **Operands:**

  | Operand | Type | Description |
  |---------|------|-------------|
  | `%sess` | the 13-field session struct | Destination; written in place, no result |
  | `%sess_gm` | `!pto.ptr<T, gm>` | Base of the host template |

- **Results:** None.
- **Constraints:**
  - `%sess` must use the session struct type above.
  - `%sess_gm` must be a GM pointer.
  - Must be outside SIMT entry functions and `pto.section.simt`.
  - Must be inside an ordinary AICore `pto.kernel` function.
- **Semantics:** Copy each template slot into the corresponding struct field.
  The caller keeps using the value `pto.declare_struct` produced.

```text
for i in 0 .. 13:
  session[i] = template_slot[i]
```

- **Example:**

  ```mlir
  %sess = pto.declare_struct
    -> !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
  pto.session_init %sess, %sess_gm
    : !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>,
      !pto.ptr<i8, gm>
  ```

---

### `pto.sdma_gm_gm`

- **Purpose:** Copy `%nbytes` contiguous bytes from `%src` to `%dst` through
  the SDMA engine attached to `%sess`.
- **Syntax:**

  ```mlir
  pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
    {block_bytes = $block}? {channel_idx = $ch}? {soft_put}?
    : !pto.ptr<T, gm>, !pto.ptr<U, gm>, i64,
      !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
  ```

- **Operands and attributes:**

  | Name | Type | Description |
  |------|------|-------------|
  | `%dst` | `!pto.ptr<T, gm>` | Destination range; may name peer memory by address |
  | `%src` | `!pto.ptr<U, gm>` | Source range; may name peer memory by address |
  | `%nbytes` | `i64` | Contiguous byte count |
  | `session(%sess)` | the 13-field session struct | Required session |
  | `block_bytes` | optional `i64` attr | Split size in bytes; omitted uses the session value |
  | `channel_idx` | optional `i64` attr | Channel group for this kick; omitted uses the session value |
  | `soft_put` | optional unit attr | A5 remote-write completion path; ignored on A2/A3 |

- **Results:** None.
- **Constraints:**
  - `%dst` and `%src` must be GM pointers. Element types need not match; the
    transfer is counted in bytes.
  - `%sess` must use the session struct type above.
  - There is no stride or burst model.
  - `block_bytes`, when present, must be positive and a multiple of 64.
  - `channel_idx`, when present, must be less than 40.
  - Must be outside SIMT entry functions and `pto.section.simt`.
  - Must be inside an ordinary AICore `pto.kernel` function.
- **Semantics:** Post a copy of `%nbytes` bytes from `%src` to `%dst`. The
  session supplies the engine connection, the default split, the channel group,
  and the service class. Either pointer may address peer memory; peer-ness is
  the numeric address, not a pointer attribute.

  Without `{soft_put}` the kick does not wait for the engine. Returning from
  the kernel does not mean the destination is visible. The caller observes
  completion by an agreed host-side check or a later sync object.

  `{soft_put}` is for a remote write on A5. That generation's engine does not
  perform a remote write, so this attr makes the copy complete before the op
  returns. A2/A3 ignore it and still post to the engine.

```text
if soft_put and target is A5:
  copy nbytes bytes from src to dst   # finished when the op returns
else:
  post the copy to the session's engine
  return without waiting
```

- **Example (local copy):**

  ```mlir
  %sess = pto.declare_struct
    -> !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
  pto.session_init %sess, %sess_gm
    : !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>,
      !pto.ptr<i8, gm>
  pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64,
      !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
  ```

- **Example (A5 remote write):**

  ```mlir
  pto.sdma_gm_gm %dst, %src, %nbytes session(%sess) {soft_put}
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64,
      !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
  ```

- **Example (per-core channel after init):**

  ```mlir
  %bid = pto.get_block_idx
  %bid32 = arith.trunci %bid : i64 to i32
  pto.session_init %sess, %sess_gm
    : !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>,
      !pto.ptr<i8, gm>
  pto.struct_set %sess[4], %bid32
    : !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>, i32
  pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64,
      !pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>
  ```

---

## PTODSL

PTODSL explicit mode exposes the same two operations as `pto.session_init` and
`pto.sdma_gm_gm`. The session type is `pto.async_session_type()`. A session still
cannot be a kernel argument: the host writes the GM template, and the kernel
declares its own struct then fills it. See
[7.7 GM↔GM SDMA](../../../ptodsl/docs/user_guide/07-data-movement-ops.md#77-gmgm-sdma-ptosession_init-and-ptosdma_gm_gm)
in the PTODSL user guide.
