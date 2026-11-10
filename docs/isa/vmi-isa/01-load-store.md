# 1. Load / Store

> **Category:** A. **Mask:** load none (A5 loads are unpredicated), store `Pg`.
>
> `vload` and `vstore` access one contiguous vector.


---

## `pto.vmi.vload`

- **semantics:** Load elements of type `T` from UB into a logical vector
  register starting at `%source + %offset` (element offset). This is a
  contiguous stride-1 read:

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = ub[base + offset + i];
  ```

- **syntax:**
  ```mlir
  %result = pto.vmi.vload %source[%offset] : !pto.ptr<T, ub> -> !pto.vmi.vreg<L×T>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group load: C rows of L/C elements, row g at base + g*stride
  %result = pto.vmi.vload %source[%offset], %stride {group = C}
      : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<L×T>
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided load: %block_stride is a dynamic i16 operand (no mask)
  %result = pto.vmi.vload %source[%offset], %block_stride
      : !pto.ptr<T, ub>, i16 -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Loaded logical vector |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | positive integer | *(none)* | Strided group load arity; requires `stride` |
  | `pmode` | `"zero"` | `"zero"` | Inactive-lane behavior (applied at consumer, not on load) |

- **lowering to `pto.mi`:**
  **Short continuous access.** Address alignment and readable memory size
  are separate requirements. Let `A` be the effective byte address
  `source + offset * sizeof(T)` and `P = L * sizeof(T)` be the logical payload.

  - `L = 1` reads one element and requires only element alignment and that
    element to be readable.
  - For `L > 1` and `P <= 32`, a provably 32B-aligned `A` uses bounded block
    access. Otherwise an unaligned stateful access is used, subject to the
    `load-safety` policy below.
  - For `32 < P < 256` with `P` a multiple of 32, a provably 32B-aligned `A`
    uses bounded block access. Otherwise the unaligned stateful access applies.
  - A dynamic offset is accepted by the bounded block path when its effective
    address can be proven aligned. Unknown alignment is not an alignment
    guarantee. With `load-safety=policy` (the default) or `warn`, an unaligned
    raw pointer is accepted without allocation-extent proof; `warn` reports the
    unproven read. With `load-safety=error`, the complete physical read range
    must be proven safe, and a raw pointer alone cannot provide that extent.

  Address proofs retain the existing address-space ABI alignment contract for
  pointer block arguments; callers must satisfy that contract. An arbitrary
  integer cast to a pointer does not establish alignment. A frontend that has
  already ensured alignment may additionally attach
  `pto.alignment = N : i64` to an integer-to-pointer `pto.castptr`, or to a
  pointer function argument, to guarantee that exact address is divisible by
  positive power-of-two N bytes. The producer or caller is responsible for
  the claim; verification rejects malformed or provably contradictory claims.
  This guarantee bypasses pointer-origin tracing, for example after an opaque
  integer projection from AscendNPU-IR. Later pointer and access offsets are
  checked separately. Inputs without the attribute use the existing alignment
  proof logic unchanged. The attribute does not establish a readable allocation
  extent or add a runtime alignment check.

  Limiting an access to one block does not remove its address alignment
  requirement. A consumer mask does not shorten the physical read range.
  Full-carrier loads retain their existing rules. These are physical access
  requirements; the separate PTODSL lane whitelist is `1/2/4/8/64/128/256`.

  **Group mode** (`{group = C}` + `stride`) has two sub-cases, decided by the
  relation between `result.L` and `C`:
  - **Full-group load** (`result.L > C`): each group loads `L/C` elements,
    row-strided tile load: `C·(L/C) = L` elements across `C` rows,
    each row `g` at offset `base + g·stride`.
  - **Slot load** (`result.L == C`): each group loads **1 scalar** into the
    corresponding slot, producing a compact `V<C×T>`. This is the
    dual of group reduce — reduce
    folds lanes into slots, slot load reads those slots back into a vreg.
    `C ∈ {1, 2, 4, 8}`.

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided load.
  Memory is read in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. A5 loads are
  unpredicated, so an implicit all-active mask is applied. Not combinable
  with `group`.

  `B*` suffix is derived from `Ptr<T>` element width: `f32/i32 → B32`, `f16/bf16/i16 → B16`, `i8/fp8 → B8`.

- **examples:**

  ```mlir
  // Contiguous load: UB → vreg
  %v = pto.vmi.vload %ub[%offset] : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64×f32>
  // Slot load: 1 scalar per group → compact V<8×f32> (reads back reduce output)
  %s = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<8×f32>
  // → each of 8 groups loads 1 scalar into its slot
  // Broadcast one loaded scalar across the result lanes
  %scalar = pto.vmi.vload %ub[%offset]
      : !pto.ptr<f32, ub> -> !pto.vmi.vreg<1×f32>
  %vb = pto.vmi.vbrc %scalar
      : !pto.vmi.vreg<1×f32> -> !pto.vmi.vreg<64×f32>
  // Broadcast one loaded scalar per group
  %slots = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<8×f32>
  %grouped = pto.vmi.vbrc %slots {group = 8}
      : !pto.vmi.vreg<8×f32> -> !pto.vmi.vreg<64×f32>
  // Full-group load: 8 rows × 8 elements, stride 64
  %t = pto.vmi.vload %ub[%off], %stride {group = 8}
      : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<64×f32>
  // → 8 rows of 8 elements, row g at base + g*stride
  // Block-strided load: block_stride = 8 (dynamic i16 operand, no mask)
  %vb = pto.vmi.vload %ub[%off], %c8_i16
      : !pto.ptr<f32, ub>, i16 -> !pto.vmi.vreg<64×f32>
  // → block-strided load (block=8), all lanes active

  ```

- **notes:**
  - **A5 loads are unpredicated.** A tail mask associated with a `vload` is
    never lowered as a masked load. It migrates to the consuming compute op or
    to a `vstore`.
  - Continuous loads support effective UB addresses that are not 32B-aligned.
    For sources without an extent, `load-safety=policy` (the default) and
    `load-safety=warn` allow the stateful physical read without a safety proof;
    `warn` also emits a diagnostic. Only `load-safety=error` requires the
    complete physical read range to be proven safe. The physical read may be
    much larger than the logical payload: for example, a 32-byte logical load
    can read a full 64-lane/256-byte carrier. Alignment state is managed
    internally and is not part of the VMI programming model.
  - The `pmode` attribute on `vload` governs the result lane behavior at the
    *consumer*, not on the load itself.

- **attention:**
  - **`{group}` and `%block_stride` are mutually exclusive.**
    Specifying more than one at once is rejected by `pto.as`.
  - **`stride` operand is bound to `{group}`.** It is required with
    `{group = C}` and invalid otherwise; `block_stride` is bound to the
    block-stride mode and invalid otherwise. `vload` has no mask operand in
    any mode (A5 loads are unpredicated).


---

## `pto.vmi.vstore`

- **semantics:** Store elements from a vector register to UB starting at
  `%dest + %offset` (element offset). This is a
  contiguous stride-1 write; only lanes where `mask[i] != 0` are written
  (A5 stores are predicated):

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offset + i] = src[i];
  ```

- **syntax:**
  ```mlir
  pto.vmi.vstore %value, %dest[%offset], %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.mask<L>
  ```
- **syntax (`group`):**
  ```mlir
  // strided group store: C rows of L/C elements, row g at base + g*stride (no mask)
  pto.vmi.vstore %value, %dest[%offset], %stride {group = C}
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, index
  ```
- **syntax (block-stride):**
  ```mlir
  // block-strided store: %block_stride is a dynamic i16 operand (mask required)
  pto.vmi.vstore %value, %dest[%offset], %block_stride, %mask
      : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, i16, !pto.vmi.mask<L>
  ```
- **operands:**

 | Operand | Type | Description |
 |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Vector value to store |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offset` | `index` | Element offset from base |
  | `stride` | `index` | Per-row stride (element units); required with `{group}`, invalid otherwise |
  | `block_stride` | `i16` | 32B-block stride between scattered blocks (block-stride mode); mutually exclusive with `{group}` |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate (variadic: 0 or 1) |

- **results:** *(none)*

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `group` | positive integer | *(none)* | Strided group store arity; requires `stride`; forbids `mask` |

- **lowering to `pto.mi`:**
  **Group mode** (`{group = C}` + `stride`): row-strided tile store. Not combinable with
  `mask` (group stores are unpredicated).

  **Block-stride mode** (`%block_stride` operand): 2D-tile block-strided store.
  Memory is written in 32B blocks with block `blk` at
  `base + blk * block_stride` (scattered access); the internal repeat stride
  defaults to 0. `%block_stride` is a dynamic `i16` operand. An explicit
  `mask` is applied; if absent an implicit all-active mask is used. Not
  combinable with `group`.

  Continuous unmasked stores support effective UB addresses that are not
  32B-aligned, including a final prefix tail. Alignment state and the required
  final flush are managed by lowering and are not part of the VMI programming
  model. Explicit sparse-mask stores still use the predicated store path and
  require their target alignment constraints to be statically provable;
  otherwise the access is unsupported.

- **examples:**

  ```mlir
  // Contiguous store: vreg → UB, masked
  pto.vmi.vstore %v, %ub_out[%offset], %mask : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.mask<64>
  ```

  ```mlir
  // Group (strided) store: 8 rows × 8 elements, stride 64 (no mask)
  pto.vmi.vstore %tile, %ub_out[%off], %stride {group = 8}
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, index
  // → 8 rows of 8 elements, row g at base + g*stride
  // Block-strided store: block_stride = 8 (dynamic i16 operand + mask)
  pto.vmi.vstore %v, %ub_out[%off], %c8_i16, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, i16, !pto.vmi.mask<64>
  // → block-strided store (block=8), governed by mask
  ```


---
