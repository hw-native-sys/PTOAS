# 3. Eltwise Compute

> **Category:** A (layout-passthrough). **Mask:** `Pg` (optional governing predicate, except `vselr` which has none).
>
> Pure per-lane ops. Layout passes through unchanged. An operand whose
> cardinality along an axis is 1 becomes a broadcast (replicate-read, never
> expanded to `K` copies). Under the `K ≤ 4` core profile these fan out as
> fully-unrolled straight-line code.

---

## 3.1 Binary Arithmetic

### `pto.vmi.vadd` / `pto.vmi.vsub` / `pto.vmi.vmul`

- **semantics:** Unified fp/int elementwise add / subtract / multiply.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? lhs[i] + rhs[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vadd %lhs, %rhs, %mask {pmode = "zero"} : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.vreg<L×T>` | First operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second operand |
  | `mask` | `!pto.vmi.mask<L>` (variadic) | Governing predicate (0 or 1) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Elementwise result |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vadd / pto.vsub / pto.vmul  (+ mask per reg, ppack/punpack if needed)
  ```
  `#mi = K`, `dep = 1`, util = 100%.

- **example:**
  ```mlir
  // fp32 add with deinterleaved layout
  %sum = pto.vmi.vadd %a, %b
      : !pto.vmi.vreg<128×f32>,
        !pto.vmi.vreg<128×f32>
      -> !pto.vmi.vreg<128×f32>
  // → pto.as: 2 × pto.vadd (EVEN/ODD), each with create_mask all-active mask

  // Masked add with merge mode
  %s = pto.vmi.vadd %a, %b, %mask {pmode = "merge"}
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

### `pto.vmi.vdiv`

- **semantics:** Elementwise floating-point divide.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? lhs[i] / rhs[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vdiv %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `f16`, `f32` only
- **lowering to `pto.mi`:**
  ```
  K × pto.vdiv
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vmax` / `pto.vmi.vmin`

- **semantics:** Elementwise maximum / minimum (unified fp/int).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? max(lhs[i], rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vmax %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vmax / pto.vmin
  ```
  `#mi = K`, `dep = 1`.

---

## 3.2 Unary Arithmetic & Activation

### `pto.vmi.vabs`

- **semantics:** Elementwise absolute value (unified fp/int).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? abs(src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vabs %src, %mask {pmode = "zero"} : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vabs
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vneg`

- **semantics:** Elementwise negate: `0 - x`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? -src[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vneg %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vneg (fp) or K × (vsub 0, src) (int)
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vrelu`

- **semantics:** Elementwise ReLU: `max(0, x)`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? max(0, src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vrelu %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vrelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vexp` / `pto.vmi.vln` / `pto.vmi.vsqrt`

- **semantics:** Elementwise transcendental: exponential, natural logarithm, square root.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? exp(src[i]) : (pmode_merge ? dst_old[i] : 0);   // vexp
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ln(src[i])  : (pmode_merge ? dst_old[i] : 0);   // vln
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? sqrt(src[i]) : (pmode_merge ? dst_old[i] : 0);  // vsqrt
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vexp %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `f16`, `f32` only
- **lowering to `pto.mi`:**
  ```
  K × pto.vexp / pto.vln / pto.vsqrt
  ```
  `#mi = K`, `dep = 1`.

---

## 3.3 Bitwise Ops

### `pto.vmi.vand` / `pto.vmi.vor` / `pto.vmi.vxor`

- **semantics:** Elementwise bitwise AND / OR / XOR. Operands and result are
  vregs by default; will also support mask-typed operands, performing a per-lane
  predicate boolean op and yielding a mask (the data operands themselves are
  masks, distinct from the governing `mask`).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] & rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vand %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32` (integer bitwise)
- **lowering to `pto.mi`:**
  ```
  K × pto.vand / pto.vor / pto.vxor
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vnot`

- **semantics:** Elementwise bitwise NOT. Operand and result are vregs by
  default; will also support a mask-typed operand, performing a per-lane predicate
  complement and yielding a mask (the data operand itself is a mask, distinct
  from the governing `mask`).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ~src[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vnot %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vnot
  ```
  `#mi = K`, `dep = 1`.

---

## 3.4 Shift Ops

### `pto.vmi.vshl` / `pto.vmi.vshr`

- **semantics:** Elementwise left shift (`vshl`) or unsigned right shift (`vshr`). The shift count is per-lane from `rhs`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] << rhs[i]) : (pmode_merge ? dst_old[i] : 0);  // vshl
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] >> rhs[i]) : (pmode_merge ? dst_old[i] : 0);  // vshr (unsigned)
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vshl %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `i8`–`i32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vshl / pto.vshr
  ```
  `#mi = K`, `dep = 1`.

---

## 3.5 Vec-Scalar Ops

Vec-scalar ops broadcast a scalar to all lanes (R6 implicit broadcast). The
scalar type must match the vector element type.

### `pto.vmi.vadds` / `pto.vmi.vmuls` / `pto.vmi.vmaxs` / `pto.vmi.vmins`

- **semantics:** Elementwise vector-scalar add / multiply / max / min.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? src[i] + scalar : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vadds %src, %scalar, %mask {pmode = "merge"} : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Vector operand |
  | `scalar` | `T` | Scalar (implicitly broadcast to all lanes) |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Elementwise result |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vadds / pto.vmuls / pto.vmaxs / pto.vmins
  ```
  `#mi = K`, `dep = 1`. No extra reg for scalar.

- **example:**
  ```mlir
  %scaled = pto.vmi.vmuls %x, %scale, %mask
      : !pto.vmi.vreg<64×f32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

### `pto.vmi.vshls` / `pto.vmi.vshrs`

- **semantics:** Elementwise vector-scalar shift.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] << scalar) : (pmode_merge ? dst_old[i] : 0);  // vshls
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] >> scalar) : (pmode_merge ? dst_old[i] : 0);  // vshrs
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vshls %src, %shift, %mask : !pto.vmi.vreg<L×T>, i16, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **datatypes:** `T` is an integer type from 8 to 32 bits. The uniform shift
  amount is a signless `i16` value independent of `T` and should be in the
  range `[0, bitwidth(T))`. For `vshrs`, the signedness of `T` determines
  whether the right shift is arithmetic or logical.
- **lowering to `pto.mi`:**
  ```
  K × pto.vshls / pto.vshrs
  ```
  `#mi = K`, `dep = 1`.

---

## 3.6 Compare & Select

### `pto.vmi.vcmp`

- **semantics:** Elementwise compare → predicate mask. The `seed` mask is the
  governing predicate `Pg`: where `seed[i] = 0` the result lane is 0 (zeroing);
  where `seed[i] = 1` the comparison is evaluated.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = seed[i] ? cmp(lhs[i], rhs[i]) : 0;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmp %lhs, %rhs, %seed {cmp = "lt"} : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `lhs` | `!pto.vmi.vreg<L×T>` | First operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second operand |
  | `seed` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Predicate mask (same L, granularity derived from T) |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `cmp` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | *(required)* | Comparison mode (fp unordered / integer; integer signedness comes from the element type: `iN` vs `uiN`) |
  | | `oeq`, `one`, `olt`, `ole`, `ogt`, `oge` | | FP ordered forms |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |

- **datatypes:** `i8`/`si8`/`ui8` – `i32`/`si32`/`ui32`, `f16`, `bf16`, `f32`.
  Integer signedness is taken from the element type; signless `iN` is treated
  as signed (equivalent to `siN`).
- **lowering to `pto.mi`:**
  ```
  K × pto.vcmp {cmp_mode}
  ```
  `#mi = K`, `dep = 1`. +1 preg per live mask result.

- **example:**
  ```mlir
  // f32 less-than compare over deinterleaved layout
  %lt = pto.vmi.vcmp %a, %b, %seed {cmp = "lt"}
      : !pto.vmi.vreg<128×f32>,
        !pto.vmi.vreg<128×f32>,
        !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // → pto.as: 2 × pto.vcmp "lt" (EVEN/ODD), each with per-reg seed mask

  // i32 signed greater-than-or-equal (signedness carried by the `i32` element type)
  %ge = pto.vmi.vcmp %a, %b, %seed {cmp = "ge"}
      : !pto.vmi.vreg<128×i32>, !pto.vmi.vreg<128×i32>, !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // si32 signed greater-than-or-equal (signedness carried by the `si32` element type)
  %ge = pto.vmi.vcmp %a, %b, %seed {cmp = "ge"}
      : !pto.vmi.vreg<128×si32>, !pto.vmi.vreg<128×si32>, !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // ui32 unsigned greater-than-or-equal (same `cmp = "ge"`; signedness from `ui32`)
  %uge = pto.vmi.vcmp %ua, %ub, %seed {cmp = "ge"}
      : !pto.vmi.vreg<128×ui32>, !pto.vmi.vreg<128×ui32>, !pto.vmi.mask<128×b32>
      -> !pto.vmi.mask<128×b32>
  // bf16 contiguous equality compare (K=1)
  %eq = pto.vmi.vcmp %a, %b, %seed {cmp = "eq"}
      : !pto.vmi.vreg<128×bf16>, !pto.vmi.vreg<128×bf16>, !pto.vmi.mask<128×b16>
      -> !pto.vmi.mask<128×b16>
  ```

### `pto.vmi.vcmps`

- **semantics:** Elementwise vector-scalar compare → predicate mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = seed[i] ? cmp(src[i], scalar) : 0;
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vcmps %src, %scalar, %seed {cmp = "ge"} : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.vmi.vreg<L×T>` | Vector operand |
  | `scalar` | `T` | Scalar to compare against |
  | `seed` | `!pto.vmi.mask<L>` | Governing predicate (required) |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.mask<L>` | Predicate mask |

- **attributes:** Same `cmp` / `pmode` as `vcmp`.
- **datatypes:** `i8`/`si8`/`ui8` – `i32`/`si32`/`ui32`, `f16`, `bf16`, `f32`.
  Integer signedness is taken from the element type; signless `iN` is treated
  as signed (equivalent to `siN`). The scalar operand's element type must
  match the vector's, so signedness is consistent on both operands.
- **lowering to `pto.mi`:**
  ```
  K × pto.vcmps {cmp_mode}
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %ges = pto.vmi.vcmps %a, %c0, %seed {cmp = "ge"}
      : !pto.vmi.vreg<64×f32>, f32, !pto.vmi.mask<64> -> !pto.vmi.mask<64>
  ```

### `pto.vmi.vsel`

- **semantics:** Per-lane selection driven by a predicate mask.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? true_val[i] : false_val[i];
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vsel %mask, %true_val, %false_val {pmode = "zero"} : !pto.vmi.mask<L>, !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `mask` | `!pto.vmi.mask<L>` | Selector predicate (required) |
  | `true_val` | `!pto.vmi.vreg<L×T>` | Value when mask[i] = 1 |
  | `false_val` | `!pto.vmi.vreg<L×T>` | Value when mask[i] = 0 |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Selected result |

- **attributes:**

  | Attribute | Values | Default | Description |
  |---|---|---|---|
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Result handling when selector inactive: `"merge"` retains `false_value` lanes |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vsel
  ```
  `#mi = K`, `dep = 1`.

- **example:**
  ```mlir
  %out = pto.vmi.vsel %mask, %x, %y {pmode = "zero"}
      : !pto.vmi.mask<256×b16>, !pto.vmi.vreg<256×ui16>, !pto.vmi.vreg<256×ui16>
      -> !pto.vmi.vreg<256×ui16>
  ```

### `pto.vmi.vselr`

- **semantics:** Dynamic lane permutation: `result[i] = source[index[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = src[index[i]];
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vselr %source, %index : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×index_T> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.vmi.vreg<L×T>` | Source vector to permute from |
  | `index` | `!pto.vmi.vreg<L×index_T>` | Per-lane source lane index |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | Permuted result |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi:**
  ```
  K × pto.vselr (+ index reg setup)
  ```
  `#mi = K`, `dep = 1` (+1 for index setup). +1 index vreg.

- **notes:**
  - This is the permute/gather class — it is the register-resident realization
    of a grouped broadcast.
  - `vselr` takes no mask; the index vector encodes the permutation directly.
  - Not A5-native `vselrv2` (that form is not available on A5).

- **example:**
  ```mlir
  %r = pto.vmi.vselr %src, %idx
      : !pto.vmi.vreg<64×f16>, !pto.vmi.vreg<4×i16> -> !pto.vmi.vreg<4×f16>
  ```

---

## 3.7 Carry / Borrow Ops (Not Provided)

Vector carry/borrow arithmetic (e.g. multi-word add-with-carry across
lanes) is **not provided** on the current surface. It will be added directly
as `i64` element-wise ops once the `i64` support plan is finalized and the
hardware path is confirmed. Until then, widening to `i64` scalar emulation
or fusing at the `pto.mi` layer is the workaround.
