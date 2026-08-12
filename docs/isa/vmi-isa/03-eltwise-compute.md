# 3. Eltwise Compute

> **Category:** A (layout-passthrough) for ordinary per-lane operations;
> `vselr` is classified separately as Category C below. **Mask:** `Pg`
> (optional governing predicate, except `vselr` which has none).
>
> Pure per-lane ops. Layout passes through unchanged. An operand whose
> cardinality along an axis is 1 becomes a broadcast (replicate-read, never
> expanded to `K` copies). Under the `K ≤ 4` core profile these fan out as
> fully-unrolled straight-line code.


---

## 3.1 Binary Arithmetic

### `pto.vmi.vadd` / `pto.vmi.vsub`

- **semantics:** Unified fp/int elementwise add / subtract.

  ```c
  for (int i = 0; i < N; i++)
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
  K × pto.vadd / pto.vsub  (+ mask per reg, ppack/punpack if needed)
  ```
  `#mi = K`, `dep = 1`, util = 100%.

### `pto.vmi.vmul`

- **semantics:** Unified floating-point/integer elementwise multiply.
- **syntax:** Same operand, mask, result, and `pmode` model as
  `pto.vmi.vadd` / `pto.vmi.vsub`.
- **datatypes:** `i16`, `i32`, `f16`, `bf16`, `f32`. The A5 vector multiply
  family has no 8-bit integer form.
- **lowering to `pto.mi`:** `K × pto.vmul`.

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

### `pto.vmi.vaddc` / `pto.vmi.vaddcs`

Carry-chain integer adds are exposed as multi-result VMI operations so the
frontend can preserve the hardware carry instruction instead of expanding the
operation into an add/compare/select sequence.

```mlir
%sum, %carry = pto.vmi.vaddc %lhs, %rhs, %mask
    : !pto.vmi.vreg<Lxui32>, !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
    -> !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
%next, %carry2 = pto.vmi.vaddcs %lhs, %rhs, %carry, %mask
    : !pto.vmi.vreg<Lxui32>, !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>, !pto.vmi.mask<L>
    -> !pto.vmi.vreg<Lxui32>, !pto.vmi.mask<L>
```

Both operations require matching 32-bit integer data values. The execution
mask, carry-in (for `vaddcs`), and carry-out use the same logical lane count,
layout, and `b32` physical mask granularity as the data ports. They lower
one-to-N to `pto.vaddc` and `pto.vaddcs` respectively.

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
- **datatypes:** `si8`, `si16`, `si32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vabs (si8/si16/si32/f16/f32)
  K × sign-bit clear (bf16)
  ```
  BF16 has no direct A5 vector-absolute instruction, so VMI implements it by
  clearing each element's sign bit. `dep = 1`.

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
- **datatypes:** `i8`–`i32`, `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vneg
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
- **datatypes:** `si32`, `f16`, `f32`
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
  vregs by default. These ops also accept mask-typed operands, performing a
  per-lane predicate boolean op and yielding a mask. When the operands are
  masks (predicate type), no governing `mask` operand may be given — a mask
  operand would be ambiguous with the predicate data operands themselves.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (lhs[i] & rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  // vreg operands (optional governing mask)
  %r = pto.vmi.vand %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>

  // mask operands (no governing mask)
  %r = pto.vmi.vand %lhs, %rhs : !pto.vmi.mask<L>, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  %r = pto.vmi.vxor %lhs, %rhs : !pto.vmi.mask<L>, !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **datatypes:** `i8`–`i32` (integer bitwise); `pred` (per-lane boolean op)
- **lowering to `pto.mi`:**
  ```
  K × pto.vand / pto.vor / pto.vxor
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vnot`

- **semantics:** Elementwise bitwise NOT. Operand and result are vregs by
  default. This op also accepts a mask-typed operand, performing a per-lane
  predicate complement and yielding a mask. When the operand is a mask
  (predicate type), no governing `mask` operand may be given — a mask operand
  would be ambiguous with the predicate data operand itself.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ~src[i] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  // vreg operand (optional governing mask)
  %r = pto.vmi.vnot %src, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>

  // mask operand (no governing mask)
  %r = pto.vmi.vnot %src : !pto.vmi.mask<L> -> !pto.vmi.mask<L>
  ```
- **datatypes:** `i8`–`i32`; `pred` (predicate complement)
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

### `pto.vmi.vadds` / `pto.vmi.vmaxs` / `pto.vmi.vmins`

- **semantics:** Elementwise vector-scalar add / max / min.

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
  K × pto.vadds / pto.vmaxs / pto.vmins
  ```
  `#mi = K`, `dep = 1`. No extra reg for scalar.

- **example:**
  ```mlir
  %shifted = pto.vmi.vadds %x, %bias, %mask
      : !pto.vmi.vreg<64×f32>, f32, !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```

### `pto.vmi.vmuls`

- **semantics:** Elementwise vector-scalar multiply with the same scalar
  broadcast, mask, and `pmode` model as the other vector-scalar operations.
- **datatypes:** `i16`, `i32`, `f16`, `f32`.
- **lowering to `pto.mi`:** `K × pto.vmuls`.
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
  | `cmp` | `eq`, `ne`, `lt`, `le`, `gt`, `ge` | *(required)* | Comparison mode (fp unordered / integer; integer signedness comes from the element type: `siN` vs `iN`/`uiN`) |
  | | `oeq`, `one`, `olt`, `ole`, `ogt`, `oge` | | FP ordered forms |
  | `pmode` | `"zero"`, `"merge"` | `"zero"` | Inactive-lane behavior |

- **datatypes:** `i8`/`si8`/`ui8` – `i32`/`si32`/`ui32`, `f16`, `bf16`, `f32`.
  Integer signedness is taken from the element type; signless `iN` is treated
  as unsigned (equivalent to `uiN`).
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

  // i32 unsigned greater-than-or-equal (signless integers use unsigned semantics)
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
  as unsigned (equivalent to `uiN`). The scalar operand's element type must
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

- **layout contract:** Category C (contiguous-required). Source, index, and
  result use contiguous layout; an arbitrary input layout is not passed through
  this operation. Compilation may materialize a contiguous representation at
  this boundary. IR that reaches this operation with an assigned
  non-contiguous layout is unsupported.

- **semantics:** Dynamic lane permutation: `result[i] = source[index[i]]`.

  ```c
  for (int i = 0; i < N; i++)
      dst[i] = src[index[i]];
  ```

- **syntax:**
  ```mlir
  %r = pto.vmi.vselr %source, %index : !pto.vmi.vreg<N×T>, !pto.vmi.vreg<N×index_T> -> !pto.vmi.vreg<N×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `source` | `!pto.vmi.vreg<N×T>` | Source vector to select from |
  | `index` | `!pto.vmi.vreg<N×index_T>` | Per-lane source lane index |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<N×T>` | Permuted result |

- **datatypes:** 8-, 16-, and 32-bit integer or floating-point source/result
  elements; `index_T` must be an integer type with the same storage width as
  `T`.
- **constraints:** Source, index, and result have the same lane count. The
  supported lane counts are `N ∈ {64, 128, 256}` for 8-bit elements,
  `N ∈ {64, 128}` for 16-bit elements, and `N = 64` for 32-bit elements.
  Every `index[i]` must identify a valid logical source lane; behavior is
  unspecified for an out-of-range index.

- **notes:**
  - This is the permute/gather class — it is the register-resident realization
    of a grouped broadcast.
  - `vselr` takes no mask; the index vector encodes the permutation directly.
  - `vselrv2` is not available on A5 and does not add other supported shapes.

- **example:**
  ```mlir
  %r = pto.vmi.vselr %src, %idx
      : !pto.vmi.vreg<128×f16>, !pto.vmi.vreg<128×i16> -> !pto.vmi.vreg<128×f16>
  ```


---

## 3.7 Carry / Borrow Ops (Not Provided)

Vector carry/borrow arithmetic (e.g. multi-word add-with-carry across
lanes) is **not provided** on the current surface. It will be added directly
as `i64` element-wise ops once the `i64` support plan is finalized and the
hardware path is confirmed. Until then, widening to `i64` scalar emulation
or fusing at the `pto.mi` layer is the workaround.
