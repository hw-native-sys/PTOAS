# 18. Async Communication

> **Category:** Asynchronous cross-rank / GM↔GM engine transfers
> **Pipelines:** SDMA / URMA / RDMA (engine kick from AI core scalar stream)

Asynchronous GM↔GM copies use DMA-class engines. Kick does not block the
scalar stream. An optional result is the CQ completion-record address used to
observe local source reuse. Cross-rank visibility is not implied by kick
completion; publish a sync object or use a fused form in
[19. Communication with Notify](19-comm-with-notify.md).

Synchronous remote UB↔GM remains on existing MTE ops in
[2. DMA Copy Programming](02-dma-copy.md). There is no `mte_gm_gm`.

This document describes:

- `pto.session_config`
- `pto.sdma_gm_gm`
- `pto.urma_gm_gm`
- `pto.rdma_gm_gm`
- `pto.sdma_gm_l2c`

Place async kicks and session wiring inside `pto.comm_scope`
(outside `pto.vecscope`).

---

## Remote Pointers

There is no dedicated remapping op. The caller computes the peer address from
`CommDeviceContext.windowsIn[]` (same-offset remap) and materializes a typed
pointer that carries `#pto.remote`:

```text
remote_addr = windowsIn[peer] + (local_addr - windowsIn[myRank])
```

```mlir
%local_i64 = pto.castptr %local : !pto.ptr<f16, gm> -> i64
%off = arith.subi %local_i64, %my_base : i64
%remote_i64 = arith.addi %peer_base, %off : i64
%dst = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
```

`#pto.mr<rma>` is retained or attached the same way when the window is a
registered RMA MR. Transfer ops require the `#pto.remote` (and, for URMA/RDMA,
`#pto.mr<rma>`) attributes on peer pointers; they do not take a separate peer
operand.

---

## Session

`!pto.async_session` is a stateful engine handle. Host builds it and passes it
into the kernel. Transfer ops that use a session treat it as a linear / token
resource (`%sess` consumed and `%sess'` produced, or an explicit write effect).
Do not rebuild a session in device code.

### `pto.session_config`

- **syntax:**
```mlir
%sess2 = pto.session_config %sess
  { block_bytes = ..., channel_idx = ..., qos = ... }
  : !pto.async_session -> !pto.async_session
```
- **semantics:** Update L2 execution defaults on the session. Omitted fields
  keep the current session value. Subsequent transfer ops that omit the same
  fields inherit these defaults, then fall back to platform defaults.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%sess` | `!pto.async_session` | Input session |
| `block_bytes` | optional attr | Preferred split / block size for engine posts |
| `channel_idx` | optional attr | Channel selection within the session |
| `qos` | optional attr | Quality-of-service hint |
| result | `!pto.async_session` | Updated session |

**Constraints:**

- Only L2 execution knobs are legal here. Resource-sizing parameters belong to
  host-side session construction.
- Place inside `pto.comm_scope`.

**Example:**

```mlir
%sess2 = pto.session_config %sess
  { block_bytes = 4096, channel_idx = 0 }
  : !pto.async_session -> !pto.async_session
```

---

## Async GM↔GM

GM↔GM transfers are contiguous `nbytes` only (no stride / nburst model).
Optional result is a CQ completion-record pointer for local completion only.
Kick failure returns a null pointer; there is no implicit MTE fallback.

There is no `*_gm_gm_list`. Aggregate with contiguous layout, staging, or
multiple kicks.

### `pto.sdma_gm_gm`

- **syntax:**
```mlir
%cq = pto.sdma_gm_gm %dst, %src, %nbytes
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** Kick an asynchronous SDMA copy of `%nbytes` contiguous bytes
  from `%src` to `%dst`. Does not block the scalar stream. The optional result
  addresses this transfer's CQ completion record.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm[, #pto.remote]>` | Destination; peer side uses `#pto.remote` |
| `%src` | `!pto.ptr<T, gm[, #pto.remote]>` | Source |
| `%nbytes` | `i64` | Contiguous byte count |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides; else session → platform default |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Continuous `nbytes` only; no src/dst stride.
- Peer pointers must carry `#pto.remote`.
- Multi-channel / multi-SQE expansion of one kick collapses to one CQ record
  when a result is produced.
- Local completion is observed by polling the CQ pointer (`dcci` + `ldg`).
  Cross-rank visibility requires a separate sync publish or a fused notify op.

**Example:**

```mlir
pto.comm_scope {
  %dst = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
  %cq = pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
    -> !pto.ptr<i64, gm>
}
```

---

### `pto.urma_gm_gm`

- **syntax:**
```mlir
%cq = pto.urma_gm_gm %dst, %src, %nbytes
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** Kick an asynchronous URMA copy of `%nbytes` contiguous bytes
  from `%src` to `%dst`. Same completion-record rules as `pto.sdma_gm_gm`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm, #pto.mr<rma>, #pto.remote>` | Remote registered destination |
| `%src` | `!pto.ptr<T, gm, #pto.mr<rma>>` | Local registered source (normally without `#pto.remote`) |
| `%nbytes` | `i64` | Contiguous byte count |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Both `%src` and `%dst` must carry `#pto.mr<rma>`.
- Peer side must also carry `#pto.remote`.
- No separate peer operand; token lookup uses peer identity associated with the
  remote pointer and session workspace.
- Continuous `nbytes` only; same CQ / visibility rules as `pto.sdma_gm_gm`.

**Example:**

```mlir
%dst = pto.castptr %remote_i64
  : i64 -> !pto.ptr<f16, gm, #pto.mr<rma>, #pto.remote>
%cq = pto.urma_gm_gm %dst, %src, %nbytes session(%sess)
  -> !pto.ptr<i64, gm>
```

---

### `pto.rdma_gm_gm`

- **syntax:**
```mlir
%cq = pto.rdma_gm_gm %dst, %src, %nbytes
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** Kick an asynchronous RDMA copy of `%nbytes` contiguous bytes
  from `%src` to `%dst`. Pointer and completion rules match `pto.urma_gm_gm`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm, #pto.mr<rma>, #pto.remote>` | Remote registered destination |
| `%src` | `!pto.ptr<T, gm, #pto.mr<rma>>` | Local registered source |
| `%nbytes` | `i64` | Contiguous byte count |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Same `#pto.mr<rma>` / `#pto.remote` rules as `pto.urma_gm_gm`.
- Continuous `nbytes` only; same CQ / visibility rules as `pto.sdma_gm_gm`.

**Example:**

```mlir
%dst = pto.castptr %remote_i64
  : i64 -> !pto.ptr<f16, gm, #pto.mr<rma>, #pto.remote>
%cq = pto.rdma_gm_gm %dst, %src, %nbytes session(%sess)
  -> !pto.ptr<i64, gm>
```

---

### `pto.sdma_gm_l2c`

- **syntax:**
```mlir
%cq = pto.sdma_gm_l2c %dst, %src, %nbytes
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** Kick an asynchronous SDMA prefetch of `%nbytes` contiguous
  bytes from local GM `%src` into local L2C view `%dst`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | L2C view / handle | Local L2C destination |
| `%src` | `!pto.ptr<T, gm>` | Local GM source (no `#pto.remote`) |
| `%nbytes` | `i64` | Contiguous byte count |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Single-device only; `%src` must not carry `#pto.remote`.
- Continuous `nbytes` only.

**Example:**

```mlir
%cq = pto.sdma_gm_l2c %l2c_dst, %gm_src, %nbytes session(%sess)
  -> !pto.ptr<i64, gm>
```

---

## Typical Usage

```mlir
%dst = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
%cq = pto.sdma_gm_gm %dst, %src, %nbytes session(%sess) -> !pto.ptr<i64, gm>
scf.while : () -> () {
  pto.dcci %cq "SINGLE_CACHE_LINE" : !pto.ptr<i64, gm>
  %v = pto.ldg %cq[%c0] l1cache(uncache) : !pto.ptr<i64, gm> -> i64
  %pend = arith.cmpi eq, %v, %c0 : i64
  scf.condition(%pend)
} do { }
// then publish remote signal/counter, or use a fused notify op instead
```
