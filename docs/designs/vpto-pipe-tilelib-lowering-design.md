# VPTO Pipe PTODSL and TileLib Boundary Design

**Issue:** [#966](https://github.com/hw-native-sys/PTOAS/issues/966)
**Status:** Compiler-side implementation complete; TileLib implementation is a
separate follow-up
**Target:** A5 VPTO backend
**This change owns:** PTODSL pipe surface validation, PTO IR, PTOAS passes,
`ExpandTileOp` contract production, VPTO LLVM lowering, and focused frontend
coverage
**Out of scope for this change:** `ptodsl/tilelib/**`, TileLib template tests,
and TileLib-ST/runtime coverage

## 1. Purpose and Scope

PTODSL already lowers its pipe surface to unified PTO pipe operations. The
VPTO path cannot retain the C++ `TPipe` object used by EmitC, because VPTO
emission needs explicit FIFO state, synchronization, and data movement in PTO
IR before LLVM lowering.

This document records the implementation boundary now present in PTOAS. It is
both the design for the compiler-side implementation and the hand-off contract
for the TileLib owner. It deliberately does not claim that a source tree
without the corresponding TileLib change can execute GM or local split pipes
end to end.

The stable frontend-to-backend flow is:

```text
PTODSL Pipe object
  -> frontend pipe operations
  -> unified PTO pipe operations
  -> infer/validate pipe nosplit configuration
  -> materialize PipeState and terminal tdrain
  -> memory planning and reserved-buffer resolution
  -> pipe metadata/candidate discovery
  -> ExpandTileOp PipeSpec ABI
  -> external PTODSL TileLib template
  -> VPTO LLVM lowering
```

## 2. Goals

- Preserve the existing PTODSL Pipe API and its per-transaction `split` value.
- Make mutable FIFO state explicit for the A5 VPTO path.
- Materialize terminal producer cleanup without deriving its policy from an
  arbitrary `tpush` operation.
- Supply `ExpandTileOp` with complete, resolved pipe metadata and resource
  operands after memory planning.
- Keep the EmitC `TPipe` path unchanged when the VPTO feature gate is off.
- Provide a narrow ABI that an independently developed TileLib implementation
  can consume without depending on opaque `!pto.pipe` values.

## 3. Non-Goals

- Replacing the existing EmitC pipe implementation.
- Adding a parallel PTODSL `pipe.push()` or `pipe.pop()` API, or redesigning
  the existing API. `pto.pipe.c2v(...)`, `pto.pipe.v2c(...)`, and
  `pto.pipe.bidirectional(...)` already return Pipe objects with these methods.
- Modifying `ptodsl/tilelib/**` in this change.
- Claiming GM/global or local split runtime correctness before the TileLib
  implementation and TileLib-ST/simulator coverage are merged.
- Supporting A2/A3 VPTO pipe expansion, `v2c_ctrl`, fixpipe quantization, or
  phase/NZ conversion in this implementation.

## 4. PTODSL Pipe Surface

### 4.1 Existing public API

The existing constructors return a direction-aware Pipe object:

```python
pipe = pto.pipe.c2v(
    id=7,
    gm_slot_tensor=slots,
    slot_size=1024,
)
pipe.init_cube()
pipe.init_simd()
entry = pipe.alloc(split=1)
pipe.push(entry, split=1)
entry = pipe.pop(split=1)
pipe.free(entry, split=1)
```

`v2c(...)` has the same transaction methods. `bidirectional(...)` exposes
`.c2v` and `.v2c` endpoints before a direction-specific transaction is
performed. The Pipe API requires a stable explicit `id`.

The public methods lower directly to the matching frontend operations:

| Pipe method | Lowered frontend operation family |
|---|---|
| `alloc(split)` | `talloc_to_aiv` or `talloc_to_aic` |
| `push(entry, split)` | `tpush_to_aiv` or `tpush_to_aic` |
| `pop(split, ...)` | `tpop_from_aic` or `tpop_from_aiv` |
| `free(entry, split)` | `tfree_from_aic` or `tfree_from_aiv` |

The `split` is an operation property, not a Pipe-object property. A
split-capable pipe may therefore issue one transaction with `split=1` and a
later transaction with `split=2`, subject to the normal pipe configuration
validation.

### 4.2 GM slot-size rule

For a GM-entry pipe, implicit `slot_size` inference is only unambiguous for
`nosplit=True`: in that case one `gm_slot_tensor` shape is one full FIFO slot.
For a split-capable GM pipe, the caller must provide an explicit full-slot
`slot_size`. This prevents a split subregion from being mistaken for the FIFO
slot size during frontend lowering.

## 5. Feature Gate and Compatibility

The compiler-side path is enabled only by:

```text
--enable-pipe-tilelib-expand
```

The driver requires all of the following:

```text
--pto-arch=a5 --pto-backend=vpto --tile-lib-backend=ptodsl
```

With the flag off, frontend pipe lowering and the existing EmitC path retain
their prior behavior. In particular, PipeState is not materialized for that
path. With the flag on, the driver performs the pipe-specific validation,
state materialization, candidate discovery, and expansion preparation
described below.

If the installed TileLib has not yet implemented a legal pipe candidate,
candidate discovery/expansion fails explicitly. PTOAS must not silently fall
back to C++ `TPipe` or to the legacy TileLang implementation.

## 6. PTO IR Contract

### 6.1 Stateful operations

The internal `pto.talloc`, `pto.tpush`, `pto.tpop`, and `pto.tfree` operations
have an optional `pipe_state` operand. Legacy source-authored IR remains valid
without it so the existing EmitC path is source-compatible. Feature-owned VPTO
IR carries exactly this type:

```text
!pto.struct<i32, i32>
```

The verifier rejects another state shape. Once any stateful user of a pipe
has a state, every `talloc`, `tpush`, `tpop`, `tfree`, and `tdrain` user of
that pipe must carry the same SSA state. This prevents partial materialization
or independently allocated counters in hand-authored internal IR. `tpush`,
`tpop`, and `tfree` use operand segments so their existing optional
entry/subblock operands remain unambiguous in assembly form.

The two fields have a fixed ownership contract:

| Field | Name | Meaning |
|---|---|---|
| 0 | `prod_index` | Next producer FIFO position. |
| 1 | `cons_index` | Next consumer FIFO position. |

The state contains no pipe handle, physical address, `flag_base`, or split
mode. Those are immutable pipe configuration or runtime resources and are
provided separately to expansion.

### 6.2 Terminal `tdrain`

`pto.tdrain` is an internal operation, inserted only by the PipeState path:

```text
pto.tdrain(%pipe, %state : !pto.pipe, !pto.struct<i32, i32>) { split = <0|1> }
```

It models producer-side cleanup formerly performed by the lifetime of the
EmitC `TPipe` object. It is generated once before each reachable `func.return`
for a pipe that has at least one producer `tpush`; a pipe without a producer
does not get a drain.

`tdrain` is pipe-level cleanup. Its `split` is derived after pipe
configuration resolution, not from a producer operation:

| Resolved initializer `nosplit` | Materialized `tdrain.split` |
|---|---|
| `true` | `0` |
| `false` | `1` |

As a result, the materialization pass accepts different producer operation
axes, such as `tpush(split=1)` followed by `tpush(split=2)`. The existing
infer/validate pass remains responsible for rejecting illegal `split` and
`nosplit` combinations. The IR verifier additionally requires an authored
`tdrain` to use this exact derived split and rejects a drain when its
initializer has not resolved `nosplit` yet.

### 6.3 Materialization pass

`pto-materialize-pipe-state` runs per `func.func` after
`pto-infer-validate-pipe-init` has resolved `nosplit`.

For each initialized pipe with stateful users, it:

1. Creates one `pto.declare_struct` of type `!pto.struct<i32, i32>`.
2. Initializes both fields to zero with `pto.struct_set`.
3. Attaches that same state to all `talloc`, `tpush`, `tpop`, and `tfree` users
   of the pipe.
4. Marks feature-owned pipe IR for later expansion cleanup.
5. Inserts the pipe-level `tdrain` when the pipe has a producer.

The initializer must dominate every return receiving the inserted drain. The
pass diagnoses a nested/lifetime form it cannot represent instead of moving a
pipe handle across regions.

## 7. Expansion ABI

### 7.1 Separation of values

`ExpandTileOp` separates pipe information into four logical operands:

| Logical operand | Role | Physical helper argument |
|---|---|---|
| entry, when present | tile or mutable GM descriptor used by the transaction | yes |
| PipeSpec | immutable configuration and per-operation `split` | no |
| PipeResources | ordered runtime addresses | one argument per resource |
| PipeState | mutable producer/consumer counters | one struct argument |

`!pto.pipe` never becomes a template helper argument. Passing it through would
leave an opaque pipe dependency in VPTO lowering and defeat expansion.

For `tpush` and `tpop`, the ABI additionally includes the AIV subblock ID.
When the original operation has no ID, PTOAS serializes a scalar `i64` value of
zero so the logical template signature is stable.

### 7.2 PipeSpec producer contract

After `pto-resolve-reserved-buffers`, the compiler derives `PipeWireInfo` from
the pipe initializer and the individual operation. It serializes a `pipe`
operand with this contract:

```json
{
  "kind": "pipe",
  "init_kind": "l2g2l",
  "dir_mask": 2,
  "slot_size": 1024,
  "slot_num": 8,
  "local_slot_num": null,
  "flag_base": 0,
  "nosplit": false,
  "split": 2,
  "resource_names": ["gm_addr"]
}
```

`init_kind` is `l2l` or `l2g2l`. The source direction encoding remains the
PTO frontend encoding (`1`, `2`, or `3`); a TileLib consumer must not assume
that this field has already been converted to an ISA-specific direction.

`split` is copied from each `talloc`, `tpush`, `tpop`, `tfree`, or `tdrain`.
It is part of the specialization identity, so helpers for distinct split
paths cannot be incorrectly reused. `tdrain` consequently receives the
pipe-level value established in Section 6.2.

Missing `flag_base`, an unresolved initializer, an absent PipeState, or an
unsupported `acc_push_epilogue` are diagnosed before a helper is requested.

### 7.3 Resources, state, and entries

`PipeResources` contains the present resources in the exact order specified
by `resource_names`:

```text
gm_addr, local_addr, peer_local_addr
```

An `l2l` pipe may expose local and peer-local resources. An `l2g2l` pipe may
also expose `gm_addr`; absent resources do not consume a helper argument.

`PipeState` is serialized as:

```json
{"kind": "pipe_state", "fields": ["i32", "i32"]}
```

A declared global entry is serialized as `pipe_entry`, rather than a normal
read-only view, because TileLib may need to rebind its caller-owned descriptor.
Its helper ABI is `!pto.tensor_view<...>`. Tile entries retain their normal
tile metadata.

### 7.4 Specialization and cleanup

`ExpandTileOp` builds the specialization key from the operation name, target,
entry metadata, PipeSpec, resource topology/type metadata, PipeState schema,
subblock argument, and context attributes such as `kernel_kind`. It forwards
each operation's `split`; it does not select one split value for the whole
pipe.

After successful template expansion, PTOAS removes only feature-owned unified
pipe operations and initializers whose uses have disappeared. A remaining
feature-owned pipe operation is an expansion error, not an opportunity to
fall back to another backend.

## 8. VPTO LLVM Boundary

PipeState survives helper creation and inlining, so both VPTO LLVM emitters
lower the supported struct subset:

- `pto.declare_struct` becomes function-local storage;
- `pto.struct_get` becomes field address calculation and load;
- `pto.struct_set` becomes field address calculation and store.

The pipe entry path also supports a mutable `!pto.tensor_view` descriptor:
`pto.declare_global`, `pto.tassign`, and `pto.tensor_view_addr` lower through
the descriptor storage rather than leaving a memref bridge or an
`unrealized_conversion_cast` in the VPTO pipeline. This is compiler support
for the TileLib ABI, not an implementation of FIFO address or synchronization
semantics.

A global pipe entry may be a `pto.declare_global` result or a result reached
through one or more `pto.tassign` operations from that declaration. The
rebinding result remains the operand passed to the pipe operation.

## 9. TileLib Handoff

The following work intentionally belongs to the TileLib owner and is not
included in this PR:

- metadata classes, renderer bindings, and candidate constraints consuming
  `pipe`, `pipe_resources`, `pipe_state`, and `pipe_entry` operands;
- A5 templates for `talloc`, `tpush`, `tpop`, `tfree`, and `tdrain`;
- GM FIFO split address offsets and local split/subblock constraints;
- operation-specific synchronization, FIFO index updates, and terminal drain
  behavior;
- daemon/unit coverage and TileLib-ST or simulator end-to-end coverage.

The TileLib implementation must treat PipeSpec as immutable configuration,
consume resources in the declared order, and use the individual operation's
`split` together with the optional subblock value. It must not recover policy
from `!pto.pipe`, infer a new pipe-wide split value, or reinterpret `tdrain`
as an arbitrary producer transaction.

Until that implementation is merged, a full pipeline may fail because no
legal pipe template candidate exists or because a template intentionally
rejects an unsupported configuration. Such a failure is expected at the
component boundary and does not demonstrate a PTOAS pass failure.

## 10. Diagnostics

The compiler provides actionable failures for these boundary violations:

| Condition | Diagnostic direction |
|---|---|
| invalid feature-gate combination | require A5, VPTO, and PTODSL TileLib backend |
| invalid PipeState type | require `!pto.struct<i32, i32>` |
| inconsistent PipeState association | require every stateful user of one pipe to share one state |
| invalid authored `tdrain.split` | require the split derived from resolved `nosplit` |
| unresolved `nosplit` before drain | require pipe-init validation before materialization |
| non-dominating drain lifetime | identify initializer and affected return |
| unresolved `flag_base` | require reserved-buffer resolution before expansion |
| unresolved pipe initializer/state | identify the unified pipe operation |
| unsupported `acc_push_epilogue` | reject before template invocation |
| no legal candidate or leftover feature-owned pipe IR | fail expansion without fallback |

## 11. Validation and Follow-up

### 11.1 Included coverage

The PTODSL surface regression covers explicit full-slot GM construction and
two independent nonzero transactions (`split=1` and `split=2`) through
`alloc`, `push`, `pop`, and `free`. It checks that each operation receives its
own split value rather than a pipe-wide default.

The current compiler build and PTODSL unit suite are the required local checks
for this change:

```text
ninja -C build-local-vpto ptoas PTOPythonModules
ptodsl/tests/test_vector_cube_ops.py -v
```

The current unit suite passes 38 tests. `git diff --check` also passes.

Focused compiler lit coverage additionally verifies PipeState materialization,
the PipeSpec RPC payload (using a test-only mock daemon outside
`ptodsl/tilelib/**`), defaulted and explicit AIV subblock operands, ordered
GM and local/peer-local resource lists, VPTO LLVM lowering for mutable
descriptors, verifier diagnostics, and feature-gate/flag-off EmitC
compatibility. It does not claim that the mock daemon validates a TileLib
template or FIFO runtime behavior.

### 11.2 Required TileLib follow-up coverage

Once TileLib consumes this ABI, it must add its own tests for:

- candidate selection and specialization separation for `split=0`, `1`, and
  `2`;
- GM and local resource order, address offsets, and subblock handling;
- PipeState counter transitions and terminal `tdrain` behavior;
- unsupported direction/quantization diagnostics; and
- end-to-end FIFO correctness on the TileLib-ST/simulator path.

The runtime suite must verify output and synchronization behavior. A compile
or rendered-IR check alone is insufficient for GM/local FIFO semantics.

## 12. Acceptance Boundary

This PTOAS-side work is complete when the frontend preserves split values,
PipeState/`tdrain` materialization obeys resolved `nosplit`, the compiler
produces the documented expansion ABI, and VPTO LLVM can lower the required
state and mutable descriptor forms.

The complete feature is complete only after the separate TileLib change
implements the hand-off contract and its TileLib-ST/simulator coverage proves
the supported configurations. The two milestones must remain distinct in PR
status and release claims.
