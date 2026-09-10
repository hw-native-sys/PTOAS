# 9. Data Rearrange

> **Category:** A (layout-transparent). **Mask:** `Pg`.
>
> In-register data movement and permutation. No UB access. `vintlv`/`vdintlv`
> are per-lane, dtype-preserving ops that do not change vreg layout — the output
> has the same `L` and `T` as the inputs. Commonly used for real+imaginary and
> value+index interleaving within a single vector register.


---

## `pto.vmi.vintlv`

- **semantics:** Interleave two source vectors by even/odd lanes.

  ```c
  // low  = {lhs[0], rhs[0], lhs[1], rhs[1], ..., lhs[L/2-1], rhs[L/2-1]}
  // high = {lhs[L/2], rhs[L/2], lhs[L/2+1], rhs[L/2+1], ...}
  for (int i = 0; i < L/2; i++) {
      lo[2*i]     = lhs[i];
      lo[2*i + 1] = rhs[i];
      hi[2*i]     = lhs[L/2 + i];
      hi[2*i + 1] = rhs[L/2 + i];
  }
  ```

- **syntax:**
  ```mlir
  %lo, %hi = pto.vmi.vintlv %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.vreg<L×T>` | First source (provides low-half even slots) |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second source (provides low-half odd slots) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `low` | `!pto.vmi.vreg<L×T>` | Even-odd interleaved low half |
  | `high` | `!pto.vmi.vreg<L×T>` | Even-odd interleaved high half |

- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vintlv
  ```
  `#mi = K`, `dep = 1`. Layout-transparent (Category A).

- **example:**
  ```mlir
  %lo, %hi = pto.vmi.vintlv %a, %b, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>
  ```


---

## `pto.vmi.vdintlv`

- **semantics:** Deinterleave a paired-source by even/odd lanes (AoS → SoA).

  ```c
  // even = {lhs[0], lhs[2], ..., lhs[L-2], rhs[0], rhs[2], ..., rhs[L-2]}
  // odd  = {lhs[1], lhs[3], ..., lhs[L-1], rhs[1], rhs[3], ..., rhs[L-1]}
  for (int i = 0; i < L/2; i++) {
      even[i]       = lhs[2*i];      // even-indexed slots of lhs
      even[L/2 + i] = rhs[2*i];      // even-indexed slots of rhs
      odd[i]        = lhs[2*i + 1];  // odd-indexed slots of lhs
      odd[L/2 + i]  = rhs[2*i + 1];  // odd-indexed slots of rhs
  }
  ```

- **syntax:**
  ```mlir
  %even, %odd = pto.vmi.vdintlv %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>
  ```
- **operands:** Same shape as `vintlv`.
- **results:** Same shape as `vintlv` (two `!pto.vmi.vreg<L×T>`).
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vdintlv
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %even, %odd = pto.vmi.vdintlv %x, %y, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>
  ```

- **notes:**
  - `vintlv` and `vdintlv` are inverses: `vdintlv(vintlv(a, b))` recovers `(a, b)`.
  - Both are Category A — they do **not** change vreg layout (parity/half/width
    axes pass through unchanged).
  - Common use cases: real+imaginary interleave, value+index pair manipulation,
    complex number arithmetic.
