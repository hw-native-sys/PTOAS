# 19. Communication with Notify

> **Category:** Fused async GM↔GM transfer + cross-rank sync publish
> **Pipelines:** SDMA / URMA / RDMA

Fused forms append sync-object publish to an async GM↔GM transfer in one
engine transaction. Peer observation of the sync object implies this
transfer's payload is visible. Base transfer shape, session rules, and CQ
completion-record rules match [18. Async Communication](18-async-comm.md).

Independent sync publish uses ordinary GM stores / `atomic_add` — no dedicated
notify/wait ops. Only the fused forms below add mnemonics. MTE has no fused
`mte_*_signal` / `mte_*_counter`.

This document describes:

- `pto.sdma_gm_gm_signal` / `pto.sdma_gm_gm_counter`
- `pto.urma_gm_gm_signal` / `pto.urma_gm_gm_counter`
- `pto.rdma_gm_gm_signal` / `pto.rdma_gm_gm_counter`

Place these ops inside `pto.comm_scope` (outside `pto.vecscope`).

---

## SDMA Fused Forms

### `pto.sdma_gm_gm_signal`

- **syntax:**
```mlir
%cq = pto.sdma_gm_gm_signal %dst, %src, %nbytes, %sig, %val
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** Kick SDMA copy of `%nbytes` bytes from `%src` to `%dst`, then
  set remote signal `%sig` to `%val` in the same transaction. Peer observation
  of `%sig` implies this payload is visible. Optional CQ result is for local
  source reuse only.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm[, #pto.remote]>` | Destination; peer side uses `#pto.remote` |
| `%src` | `!pto.ptr<T, gm[, #pto.remote]>` | Source |
| `%nbytes` | `i64` | Contiguous byte count |
| `%sig` | `!pto.ptr<i32, gm, #pto.remote>` | Peer signal location |
| `%val` | `i32` | Value written to the signal |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Transfer pointer rules match `pto.sdma_gm_gm`.
- `%sig` must be remote (`#pto.remote`).
- Continuous `nbytes` only.
- No separate post-E2 store sequence is required for this transfer's payload.

**Example:**

```mlir
pto.comm_scope {
  %dst = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
  %sig = pto.castptr %sig_remote_i64 : i64 -> !pto.ptr<i32, gm, #pto.remote>
  %cq = pto.sdma_gm_gm_signal %dst, %src, %nbytes, %sig, %c1
    session(%sess) -> !pto.ptr<i64, gm>
}
```

---

### `pto.sdma_gm_gm_counter`

- **syntax:**
```mlir
%cq = pto.sdma_gm_gm_counter %dst, %src, %nbytes, %ctr, %delta
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** Kick SDMA copy of `%nbytes` bytes from `%src` to `%dst`, then
  atomically add `%delta` to remote counter `%ctr` in the same transaction.
  Peer observation of the updated counter implies this payload is visible.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm[, #pto.remote]>` | Destination; peer side uses `#pto.remote` |
| `%src` | `!pto.ptr<T, gm[, #pto.remote]>` | Source |
| `%nbytes` | `i64` | Contiguous byte count |
| `%ctr` | `!pto.ptr<i32, gm, #pto.remote>` | Peer counter location |
| `%delta` | `i32` | Atomic increment |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Transfer pointer rules match `pto.sdma_gm_gm`.
- `%ctr` must be remote (`#pto.remote`).
- Continuous `nbytes` only.
- Use counter (not signal) when multiple writers may publish to the same sync
  location.

**Example:**

```mlir
%dst = pto.castptr %remote_i64 : i64 -> !pto.ptr<f16, gm, #pto.remote>
%ctr = pto.castptr %ctr_remote_i64 : i64 -> !pto.ptr<i32, gm, #pto.remote>
%cq = pto.sdma_gm_gm_counter %dst, %src, %nbytes, %ctr, %c1
  session(%sess) -> !pto.ptr<i64, gm>
```

---

## URMA Fused Forms

### `pto.urma_gm_gm_signal`

- **syntax:**
```mlir
%cq = pto.urma_gm_gm_signal %dst, %src, %nbytes, %sig, %val
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** URMA fused copy + remote signal set. Same visibility contract
  as `pto.sdma_gm_gm_signal`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm, #pto.mr<rma>, #pto.remote>` | Remote registered destination |
| `%src` | `!pto.ptr<T, gm, #pto.mr<rma>>` | Local registered source |
| `%nbytes` | `i64` | Contiguous byte count |
| `%sig` | `!pto.ptr<i32, gm, #pto.remote>` | Peer signal location |
| `%val` | `i32` | Value written to the signal |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Transfer pointer rules match `pto.urma_gm_gm`.
- `%sig` must be remote (`#pto.remote`).
- Continuous `nbytes` only.

**Example:**

```mlir
%dst = pto.castptr %remote_i64
  : i64 -> !pto.ptr<f16, gm, #pto.mr<rma>, #pto.remote>
%sig = pto.castptr %sig_remote_i64 : i64 -> !pto.ptr<i32, gm, #pto.remote>
%cq = pto.urma_gm_gm_signal %dst, %src, %nbytes, %sig, %c1
  session(%sess) -> !pto.ptr<i64, gm>
```

---

### `pto.urma_gm_gm_counter`

- **syntax:**
```mlir
%cq = pto.urma_gm_gm_counter %dst, %src, %nbytes, %ctr, %delta
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** URMA fused copy + remote counter atomic add. Same visibility
  contract as `pto.sdma_gm_gm_counter`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm, #pto.mr<rma>, #pto.remote>` | Remote registered destination |
| `%src` | `!pto.ptr<T, gm, #pto.mr<rma>>` | Local registered source |
| `%nbytes` | `i64` | Contiguous byte count |
| `%ctr` | `!pto.ptr<i32, gm, #pto.remote>` | Peer counter location |
| `%delta` | `i32` | Atomic increment |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Transfer pointer rules match `pto.urma_gm_gm`.
- `%ctr` must be remote (`#pto.remote`).
- Continuous `nbytes` only.

**Example:**

```mlir
%dst = pto.castptr %remote_i64
  : i64 -> !pto.ptr<f16, gm, #pto.mr<rma>, #pto.remote>
%ctr = pto.castptr %ctr_remote_i64 : i64 -> !pto.ptr<i32, gm, #pto.remote>
%cq = pto.urma_gm_gm_counter %dst, %src, %nbytes, %ctr, %c1
  session(%sess) -> !pto.ptr<i64, gm>
```

---

## RDMA Fused Forms

### `pto.rdma_gm_gm_signal`

- **syntax:**
```mlir
%cq = pto.rdma_gm_gm_signal %dst, %src, %nbytes, %sig, %val
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** RDMA fused copy + remote signal set. Pointer and visibility
  rules match `pto.urma_gm_gm_signal`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm, #pto.mr<rma>, #pto.remote>` | Remote registered destination |
| `%src` | `!pto.ptr<T, gm, #pto.mr<rma>>` | Local registered source |
| `%nbytes` | `i64` | Contiguous byte count |
| `%sig` | `!pto.ptr<i32, gm, #pto.remote>` | Peer signal location |
| `%val` | `i32` | Value written to the signal |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Transfer pointer rules match `pto.rdma_gm_gm`.
- `%sig` must be remote (`#pto.remote`).
- Continuous `nbytes` only.

**Example:**

```mlir
%dst = pto.castptr %remote_i64
  : i64 -> !pto.ptr<f16, gm, #pto.mr<rma>, #pto.remote>
%sig = pto.castptr %sig_remote_i64 : i64 -> !pto.ptr<i32, gm, #pto.remote>
%cq = pto.rdma_gm_gm_signal %dst, %src, %nbytes, %sig, %c1
  session(%sess) -> !pto.ptr<i64, gm>
```

---

### `pto.rdma_gm_gm_counter`

- **syntax:**
```mlir
%cq = pto.rdma_gm_gm_counter %dst, %src, %nbytes, %ctr, %delta
  session(%sess)?
  { block_bytes?, channel_idx?, qos? }
  -> !pto.ptr<i64, gm>
```
- **semantics:** RDMA fused copy + remote counter atomic add. Pointer and
  visibility rules match `pto.urma_gm_gm_counter`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%dst` | `!pto.ptr<T, gm, #pto.mr<rma>, #pto.remote>` | Remote registered destination |
| `%src` | `!pto.ptr<T, gm, #pto.mr<rma>>` | Local registered source |
| `%nbytes` | `i64` | Contiguous byte count |
| `%ctr` | `!pto.ptr<i32, gm, #pto.remote>` | Peer counter location |
| `%delta` | `i32` | Atomic increment |
| `session` | `!pto.async_session` | Required when returning a CQ record or reusing SQ |
| `block_bytes` / `channel_idx` / `qos` | optional attrs | Per-kick L2 overrides |
| result | `!pto.ptr<i64, gm>?` | CQ record for local completion; null on kick failure |

**Constraints:**

- Transfer pointer rules match `pto.rdma_gm_gm`.
- `%ctr` must be remote (`#pto.remote`).
- Continuous `nbytes` only.

**Example:**

```mlir
%dst = pto.castptr %remote_i64
  : i64 -> !pto.ptr<f16, gm, #pto.mr<rma>, #pto.remote>
%ctr = pto.castptr %ctr_remote_i64 : i64 -> !pto.ptr<i32, gm, #pto.remote>
%cq = pto.rdma_gm_gm_counter %dst, %src, %nbytes, %ctr, %c1
  session(%sess) -> !pto.ptr<i64, gm>
```

---

## Typical Usage

Prefer fused forms when one transfer's completion should wake the peer.

Prefer separate `*_gm_gm` + store / `atomic_add` when one sync publish covers
multiple prior transfers that have already reached local completion.
