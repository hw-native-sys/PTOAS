# 19. Async Communication

> **Category:** Asynchronous GM↔GM engine transfers
> **Pipelines:** SDMA engine kick from an ordinary AICore scalar stream

This group copies a contiguous GM range through the SDMA engine. The kick does
not wait for the engine except where `{soft_put}` is documented below; it hands
back a handle the caller waits on instead.

This document describes:

- `pto.session_init`
- `pto.sdma_gm_gm`

There is no `mte_gm_gm`. Synchronous GM↔UB copies remain in
[2. DMA Copy Programming](02-dma-copy.md).

These ops must sit in an ordinary AICore entry function. They are illegal
inside `pto.simt_entry` functions and `pto.section.simt`.

---

## Session

A transfer needs an engine connection, a queue, and a service class. Those
travel together as a session:

```mlir
!pto.dma_session<sdma>
```

The engine is part of the type, so `pto.sdma_gm_gm` accepts only an SDMA
session. `urma` and `rdma` name the other engines; the ops that post on them are
not in this group yet.

A session cannot be a kernel argument, because what backs one is not something
an argument can carry. The host instead writes a template into GM and the kernel
turns it into a session with `pto.session_init`. The template is one 8-byte slot
per field, narrow fields in the low half, in the order `SessionField` gives in
`PTO/Support/AsyncSessionABI.h` -- the one place that order is defined.

The session is opaque: there is nothing to declare beforehand and no field to
read back. Two consequences are worth stating, because they replace things an
open struct would have allowed:

- **Retuning a session is per post, not per session.** `block_bytes` and
  `channel_idx` are attributes on `pto.sdma_gm_gm`, which is what gives a
  multi-core launch its own queue per core without the host naming the core.
- **Where a transfer lands is the caller's pointer arithmetic.** A session is one
  configuration shared by every post that uses it, so a per-transfer offset has
  no place in it. Displace `%src` and `%dst` with `pto.addptr` instead.

---

## Operation Summary

| Operation | Purpose |
|-----------|---------|
| `pto.session_init` | Build a session from the host-written GM template |
| `pto.sdma_gm_gm` | Kick a contiguous GM→GM copy through the session |

---

### `pto.session_init`

- **Purpose:** Produce a session configured by the host-written GM template.
- **Syntax:**

  ```mlir
  %sess = pto.session_init %sess_gm
    : !pto.ptr<i8, gm> -> !pto.dma_session<sdma>
  ```

- **Operands:**

  | Operand | Type | Description |
  |---------|------|-------------|
  | `%sess_gm` | `!pto.ptr<T, gm>` | Base of the host template |

- **Results:**

  | Result | Type | Description |
  |--------|------|-------------|
  | `%sess` | `!pto.dma_session<E>` | Session for engine `E` |

- **Constraints:**
  - `%sess_gm` must be a GM pointer.
  - Must be outside SIMT entry functions and `pto.section.simt`.
  - Must be inside an ordinary AICore `pto.kernel` function.
- **Semantics:** Read the template and yield a session carrying what it
  described. Each core that runs this gets its own session, even when the
  template is shared and read-only.

---

### `pto.sdma_gm_gm`

- **Purpose:** Copy `%nbytes` contiguous bytes from `%src` to `%dst` through
  the SDMA engine `%sess` names.
- **Syntax:**

  ```mlir
  %rec = pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
    {block_bytes = $block}? {channel_idx = $ch}? {soft_put}?
    : !pto.ptr<T, gm>, !pto.ptr<U, gm>, i64, !pto.dma_session<sdma>
      -> !pto.ptr<i64, gm>
  ```

- **Operands and attributes:**

  | Name | Type | Description |
  |------|------|-------------|
  | `%dst` | `!pto.ptr<T, gm>` | Destination range; may name peer memory by address |
  | `%src` | `!pto.ptr<U, gm>` | Source range; may name peer memory by address |
  | `%nbytes` | `i64` | Contiguous byte count |
  | `session(%sess)` | `!pto.dma_session<sdma>` | Required session |
  | `block_bytes` | optional `i64` attr | Split size in bytes; omitted uses the session value |
  | `channel_idx` | optional `i64` attr | Channel group for this kick; omitted uses the session value |
  | `soft_put` | optional unit attr | A5 remote-write completion path; ignored on A2/A3 |

- **Results:**

  | Result | Type | Description |
  |--------|------|-------------|
  | `%rec` | `!pto.ptr<i64, gm>` | Handle to wait on; null when the copy is already complete |

- **Constraints:**
  - `%dst` and `%src` must be GM pointers. Element types need not match; the
    transfer is counted in bytes.
  - There is no stride or burst model.
  - `block_bytes`, when present, must be positive and a multiple of 64.
  - `channel_idx`, when present, must be less than 40.
  - Must be outside SIMT entry functions and `pto.section.simt`.
  - Must be inside an ordinary AICore `pto.kernel` function.
- **Semantics:** Post a copy of `%nbytes` bytes from `%src` to `%dst`. The
  session supplies the engine connection, the default split, the channel group,
  and the service class. Either pointer may address peer memory; peer-ness is
  the numeric address, not a pointer attribute.

  Without `{soft_put}` the kick does not wait for the engine. Returning from the
  kernel does not mean the destination is visible; `%rec` is what says when it
  is. The handle names the channel the post went to, and the channel is complete
  when it has drained -- so a wait is per channel, not per post: a later post on
  the same channel is waited for as well. That is never early, and posts on a
  channel retire in order. `PTO/Support/AsyncSessionABI.h` gives the test a
  caller applies to the handle; no op in this group consumes it yet.

  `{soft_put}` is for a remote write on A5. That generation's engine does not
  perform a remote write, so this attr makes the copy complete before the op
  returns, and `%rec` is null to say there is nothing left to wait for. A2/A3
  ignore the attr and still post to the engine.

```text
if soft_put and target is A5:
  copy nbytes bytes from src to dst   # finished when the op returns
  return null
else:
  post the copy to the session's engine
  return the channel it went to, without waiting
```

- **Example (local copy):**

  ```mlir
  %sess = pto.session_init %sess_gm
    : !pto.ptr<i8, gm> -> !pto.dma_session<sdma>
  %rec = pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, !pto.dma_session<sdma>
      -> !pto.ptr<i64, gm>
  ```

- **Example (A5 remote write):**

  ```mlir
  %rec = pto.sdma_gm_gm %dst, %src, %nbytes session(%sess) {soft_put}
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, !pto.dma_session<sdma>
      -> !pto.ptr<i64, gm>
  ```

- **Example (per-core channel):**

  ```mlir
  %rec = pto.sdma_gm_gm %dst, %src, %nbytes session(%sess) {channel_idx = 2 : i64}
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, !pto.dma_session<sdma>
      -> !pto.ptr<i64, gm>
  ```

- **Example (per-core slice of one buffer):**

  Every core loads the same template and moves its own window, so what differs
  between them is two pointers. The endpoints are `i8`, so an element offset is
  a byte offset.

  ```mlir
  %bid = pto.get_block_idx
  %byte_off = arith.muli %bid, %chunk : i64
  %off = arith.index_castui %byte_off : i64 to index
  %core_src = pto.addptr %src, %off : !pto.ptr<i8, gm> -> !pto.ptr<i8, gm>
  %core_dst = pto.addptr %dst, %off : !pto.ptr<i8, gm> -> !pto.ptr<i8, gm>
  %rec = pto.sdma_gm_gm %core_dst, %core_src, %chunk session(%sess)
    : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, !pto.dma_session<sdma>
      -> !pto.ptr<i64, gm>
  ```

---

## PTODSL

PTODSL explicit mode exposes the same two operations as `pto.session_init` and
`pto.sdma_gm_gm`, with `pto.dma_session_type(engine)` to name the session type.
`pto.session_init` returns the session and `pto.sdma_gm_gm` returns the handle.
See
[7.7 GM↔GM SDMA](../../../ptodsl/docs/user_guide/07-data-movement-ops.md#77-gmgm-sdma-ptosession_init-and-ptosdma_gm_gm)
in the PTODSL user guide.
