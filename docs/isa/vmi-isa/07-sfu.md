# 7. SFU

> **Category:** A (fused arithmetic, `vmull`), B (`vchist`, `vdhist`), C (gather/scatter).
> **Mask:** `Pg` on all except sort-like ops.
>
> Special-function / domain-accelerator ops. Mixed categories: `vchist`
> produces a `half` axis (B); `vdhist` yields a plain per-bin count (B);
> gather/scatter are Category C tile/permute ops; fused activation/arithmetic
> ops (including `vmull`, whose 64-bit product is split into a pair of `i32`
> results at the VMI surface) are Category A `vreg→vreg`.


---

## 7.1 Fused Arithmetic

### `pto.vmi.vexpdif`

- **semantics:** Fused `exp(x − max)` for softmax numerical stability.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? exp(x[i] - max[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×f32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input (`f16` or `f32`) |
  | `max` | `!pto.vmi.vreg<L×T>` | Subtracted max with the same type as `x` |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×f32>` | `exp(x − max)` (always `f32`) |

- **attributes:** `pmode` (`"zero"` / `"merge"`), default `"zero"`
- **datatypes:** `x` and `max`: matching `f16` or `f32`; result: `f32`
- **lowering to `pto.mi`:**
  ```
  f32: K × pto.vexpdif
  f16: 2K × pto.vexpdif
  ```
  Fuses `vsub` + `vexp`.

- **example:**
  ```mlir
  %e = pto.vmi.vexpdif %x, %max, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.mask<64>
      -> !pto.vmi.vreg<64×f32>
  ```

### `pto.vmi.vaxpy`

- **semantics:** Fused `α·x + y` (scale-add). Single hardware instruction.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (alpha * x[i] + acc[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vaxpy %x, %acc, %alpha, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input vector |
  | `acc` | `!pto.vmi.vreg<L×T>` | Accumulator (`y`) |
  | `alpha` | `T` (float scalar) | Scale factor |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<L×T>` | `α·x + acc` |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vaxpy
  ```
  `#mi = K`, `dep = 1`. Fuses `vmuls` + `vadd`.

### `pto.vmi.vlrelu`

- **semantics:** Leaky ReLU: `y = x > 0 ? x : slope × x`. The slope is a
  scalar shared across all lanes.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (src[i] > 0 ? src[i] : slope * src[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vlrelu %x, %slope, %mask : !pto.vmi.vreg<L×T>, T, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input |
  | `slope` | `T` (float scalar) | Negative-slope multiplier |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vlrelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vprelu`

- **semantics:** Parametric ReLU: `y = max(x, 0) + alpha × min(x, 0)`. The
  `alpha` is a per-lane parameter vector (not a shared scalar).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (max(src[i], 0) + alpha[i] * min(src[i], 0)) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %y = pto.vmi.vprelu %x, %alpha, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `x` | `!pto.vmi.vreg<L×T>` | Input |
  | `alpha` | `!pto.vmi.vreg<L×T>` | Per-lane negative-slope parameter |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `f16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vprelu
  ```
  `#mi = K`, `dep = 1`.

### `pto.vmi.vmull`

- **semantics:** Widening 32-bit × 32-bit → 64-bit integer multiply. At the
  VMI surface the 64-bit product is **split into a pair of `i32` results**:
  `%low` carries the lower 32 bits and `%high` carries the upper 32 bits.
  This matches the shape of `pto.mi.vmull` one-to-one, so no `width` axis is
  introduced at the VMI layer. Signedness is inherited from the inputs
  (`i32 → (i32, i32)` uses arithmetic shift for the high half;
  `ui32 → (ui32, ui32)` uses logical shift).

  ```c
  for (int i = 0; i < L; i++) {
      // signed variant; use uint64_t for the ui32 form
      int64_t r = (int64_t)lhs[i] * (int64_t)rhs[i];
      low [i] = mask[i] ? (int32_t)(r & 0xFFFFFFFF)
                        : (pmode_merge ? low_old [i] : 0);
      high[i] = mask[i] ? (int32_t)(r >> 32)
                        : (pmode_merge ? high_old[i] : 0);
  }
  ```

- **syntax:**
  ```mlir
  %low, %high = pto.vmi.vmull %lhs, %rhs, %mask
      : !pto.vmi.vreg<L×i32>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L>
        -> !pto.vmi.vreg<L×i32>, !pto.vmi.vreg<L×i32>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `a` | `!pto.vmi.vreg<L×i32>` | First operand |
  | `b` | `!pto.vmi.vreg<L×i32>` | Second operand |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `low`  | `!pto.vmi.vreg<L×i32>` | Lower 32 bits of the per-lane 64-bit product |
  | `high` | `!pto.vmi.vreg<L×i32>` | Upper 32 bits of the per-lane 64-bit product (arithmetic shift for `i32`; logical shift for `ui32`) |

- **attributes:**

  | Attribute | Type | Default | Description |
  |---|---|---|---|
  | `pmode` | `StrAttr` (`"zero"` \| `"merge"`) | `"zero"` | Predication mode. `"merge"` preserves the previous `low`/`high` lane values on inactive lanes; on A5 this is **not implemented**  (see [Appendix C](10-appendices.md)). |

- **datatypes:** `i32 → (i32, i32)`, `ui32 → (ui32, ui32)` (both result vregs share the input signedness).
- **lowering to `pto.mi`:**
  ```
  for k in [0, K):
      (low_k, high_k) = pto.mi.vmull(lhs_k, rhs_k, mask_k)
  ```
  `#mi = K`, `dep = 1`. Structurally 1:1 with `pto.mi.vmull`

- **example:**
  ```mlir
  %lo, %hi = pto.vmi.vmull %lhs, %rhs, %mask
      : !pto.vmi.vreg<64×i32>, !pto.vmi.vreg<64×i32>, !pto.vmi.mask<64>
        -> !pto.vmi.vreg<64×i32>, !pto.vmi.vreg<64×i32>
  ```

### `pto.vmi.vmula`

- **semantics:** Fused multiply-add: `acc = acc + lhs × rhs`. Single hardware
  instruction. The accumulator is both an input and output (writes back).

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? (acc[i] + lhs[i] * rhs[i]) : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %lhs, %rhs, %mask : !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.vreg<L×T>, !pto.vmi.mask<L> -> !pto.vmi.vreg<L×T>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc` | `!pto.vmi.vreg<L×T>` | Accumulator (read-modify-write) |
  | `lhs` | `!pto.vmi.vreg<L×T>` | First multiply operand |
  | `rhs` | `!pto.vmi.vreg<L×T>` | Second multiply operand |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vmula
  ```
  `#mi = K`, `dep = 1`. Fuses `vmul` + `vadd`.

- **example:**
  ```mlir
  %acc1 = pto.vmi.vmula %acc, %a, %b, %mask
      : !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>, !pto.vmi.vreg<64×f32>,
        !pto.vmi.mask<64> -> !pto.vmi.vreg<64×f32>
  ```


---

## 7.2 Histogram

### `pto.vmi.vchist`

`N` is the source/mask lane count; PTODSL requires both operands to have the
same `N` in `1/2/4/8/64/128/256`. Other counts, including `96`, raise `ValueError`
before the operation is constructed. Compiler-internal VMI types remain general.
`B` is the accumulator/result bin count: `128` or `256`, independent of `N`.
For example, `N=64, B=128` and `N=64, B=256` are both supported. Raw UB `ui8`
loads at 64/128 lanes use bounded 2/4-block reads at aligned addresses.

- **semantics:** **Cumulative histogram** over 8-bit source lanes
  (interpreted as unsigned). Counts per-bin occurrences over `%src` on top
  of a carry-in accumulator `%acc`. `B=256` returns bins 0–255 using
  Bin_N0 + Bin_N1; `B=128` returns bins 0–127 using Bin_N0 only. The result
  is a logical vector of `B` bins; the low/high split is a physical detail.
  The 128-bin form covers the full source range when samples are `< 128`.

  ```c
  // N source samples; B output bins (128 or 256)
  uint16_t dhist[256] = {0};
  for (int i = 0; i < N; i++)
      if (mask[i])
          dhist[src[i]]++;
  uint16_t chist[B];
  uint16_t cumulative = 0;
  for (int b = 0; b < B; b++) {
      cumulative += dhist[b];
      chist[b] = acc[b] + cumulative;
  }
  // B=128: Bin_N0 only; B=256: Bin_N0 and Bin_N1
  ```

- **syntax:**
  ```mlir
  // output is Bin_N0 + Bin_N1
  %h = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>

  // output is Bin_N0 when the source lanes are known to be < 128
  %h = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<128xui16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc`  | `!pto.vmi.vreg<B×{ui16\|i16}>` | Carry-in accumulator; same shape as `result` (256-bin Bin_N0+Bin_N1, or 128-bin Bin_N0-only). Element type is `ui16` or signless `i16` (interpreted as unsigned). |
  | `src`  | `!pto.vmi.vreg<N×{ui8\|i8}>` | Source lanes to be binned; 8-bit element type is `ui8` or signless `i8` (interpreted as unsigned). |
  | `mask` | `!pto.vmi.mask<N×pred>` | Governing predicate over source lanes. Does not gate `acc`. |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<B×{ui16\|i16}>` | Cumulative counts on top of `acc` (`B=128`: Bin_N0; `B=256`: Bin_N0+Bin_N1). Element type is `ui16` or signless `i16` (interpreted as unsigned). |

- **datatypes:** Source bin index: `ui8` or signless `i8`. Accumulator / result:
  `ui16` or signless `i16`. All are interpreted as
  unsigned; signed types (`si8` / `si16`) are rejected by the verifier.
- **lowering to `pto.mi`:**
  ```
  B=128: chistv2 Bin_N0 accumulator chain
  B=256: chistv2 Bin_N0 and Bin_N1 accumulator chains
  ```
  For `K = ceil(N / 256)` physical source chunks, this emits `K * (B / 128)`
  histogram instructions, excluding masks and memory operations. Thus public
  PTODSL sizes emit one instruction for `B=128`, or two for `B=256`. Bin_N1
  uses global cumulative semantics; no software prefix compensation is needed.

  The source operand must be `contiguous`. Raw UB inputs support one-lane
  scalar loads, aligned single-block short reads, aligned exact 64/128-byte multi-block
  reads, and full 256-byte register reads. The lowering intersects the input
  predicate with logical validity, so padding lanes never contribute.

- **example:**
  ```mlir
  // Cumulative histogram, full 256-bin (Bin_N0 + Bin_N1) output
  %h = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>
  // → two physical chistv2 instructions, one per 128-bin result part

  // N=64 samples, B=128 bins (sample values known to be < 128)
  %h0 = pto.vmi.vchist %acc0, %src64, %mask64
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<64xui8>, !pto.vmi.mask<64xpred>
     -> !pto.vmi.vreg<128xui16>

  // signless i16/i8 also accepted (interpreted as unsigned; acc and result must match)
  %hs = pto.vmi.vchist %acc, %src, %mask
      : !pto.vmi.vreg<256xi16>, !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xi16>
  ```

### `pto.vmi.vdhist`

`N` is the source/mask lane count; PTODSL requires both operands to have the
same `N` in `1/2/4/8/64/128/256`. Other counts, including `96`, raise `ValueError`
before the operation is constructed. Compiler-internal VMI types remain general.
`B` is the accumulator/result bin count: `128` or `256`, independent of `N`.
For example, `N=64, B=128` and `N=64, B=256` are both supported. Raw UB `ui8`
loads at 64/128 lanes use bounded 2/4-block reads at aligned addresses.

- **semantics:** **Distribution histogram** over 8-bit source lanes
  (interpreted as unsigned). Counts per-bin occurrences over `%src` on top
  of a carry-in accumulator `%acc`, yielding a logical vector of `B` per-bin
  counts. `B=128` returns bins 0–127; `B=256` returns all bins 0–255. The
  128-bin form covers the full source range when samples are `< 128`.

  ```c
  // N source samples; B output bins (128 or 256)
  uint16_t dhist[B];
  for (int b = 0; b < B; b++) dhist[b] = acc[b];     // carry-in
  for (int i = 0; i < N; i++)
      if (mask[i] && src[i] < B)
          dhist[src[i]]++;
  ```

- **syntax:**
  ```mlir
  // 256-bin full output
  %d = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>

  // 128-bin output when the source lanes are known to be < 128
  %d = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<128xui16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `acc`  | `!pto.vmi.vreg<B×{ui16\|i16}>` | Carry-in accumulator; same shape as `result` (`B=256`: bins 0–255; `B=128`: bins 0–127). Element type is `ui16` or signless `i16` (interpreted as unsigned). |
  | `src`  | `!pto.vmi.vreg<N×{ui8\|i8}>` | Source lanes to be binned; 8-bit element type is `ui8` or signless `i8` (interpreted as unsigned). |
  | `mask` | `!pto.vmi.mask<N×pred>` | Governing predicate over source lanes. Does not gate `acc`. |

- **results:**

  | Result | Type | Description |
  |---|---|---|
  | `result` | `!pto.vmi.vreg<B×{ui16\|i16}>` | Plain per-bin count vector on top of `acc` (`B=256`: bins 0–255; `B=128`: bins 0–127). Element type is `ui16` or signless `i16` (interpreted as unsigned). |

- **datatypes:** Source bin index: `ui8` or signless `i8`. Accumulator / result:
  `ui16` or signless `i16`. All are interpreted as
  unsigned; signed types (`si8` / `si16`) are rejected by the verifier.
- **lowering to `pto.mi`:**
  ```
  B=128: dhistv2 Bin_N0 accumulator chain
  B=256: dhistv2 Bin_N0 and Bin_N1 accumulator chains
  ```
  For `K = ceil(N / 256)` physical source chunks, this emits `K * (B / 128)`
  histogram instructions, excluding masks and memory operations. Thus public
  PTODSL sizes emit one instruction for `B=128`, or two for `B=256`. The
  source and mask must be contiguous, and lowering intersects the b8 user
  mask with logical validity so padding lanes never contribute.

- **example:**
  ```mlir
  // Distribution histogram, plain per-bin count (256-bin full)
  %d = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<256xui8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xui16>

  // N=64 samples, B=128 bins (sample values known to be < 128)
  %d0 = pto.vmi.vdhist %acc0, %src64, %mask64
      : !pto.vmi.vreg<128xui16>, !pto.vmi.vreg<64xui8>, !pto.vmi.mask<64xpred>
     -> !pto.vmi.vreg<128xui16>

  // signless i16/i8 also accepted (interpreted as unsigned; acc and result must match)
  %ds = pto.vmi.vdhist %acc, %src, %mask
      : !pto.vmi.vreg<256xi16>, !pto.vmi.vreg<256xi8>, !pto.vmi.mask<256xpred>
     -> !pto.vmi.vreg<256xi16>
  ```


---

## 7.3 Gather / Scatter

> **Category C** — contiguous-required. `pto.as` materializes `.contiguous()`
> before these ops if the input layout is non-contiguous.

### `pto.vmi.vgather`

- **semantics:** Indexed gather from UB at B32/B16 granularity. For each active
  lane `i`, load `src[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      dst[i] = mask[i] ? ub[base + offsets[i]] : (pmode_merge ? dst_old[i] : 0);
  ```

- **syntax:**
  ```mlir
  // B32 path
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L×b32> -> !pto.vmi.vreg<L×T>   // T in {i32,ui32,f32}

  // B16 path
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<T, ub>, !pto.vmi.vreg<L×ui16>, !pto.vmi.mask<L×b16> -> !pto.vmi.vreg<L×T>   // T in {i16,ui16,f16,bf16}
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<i8, ub>, !pto.vmi.vreg<L×ui16>, !pto.vmi.mask<L×b16> -> !pto.vmi.vreg<L×i16>
  %g = pto.vmi.vgather %src, %offsets, %mask
      : !pto.ptr<ui8, ub>, !pto.vmi.vreg<L×ui16>, !pto.vmi.mask<L×b16> -> !pto.vmi.vreg<L×ui16>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `src` | `!pto.ptr<T, ub>` | UB base pointer |
  | `offsets` | `!pto.vmi.vreg<L×i32>` or `!pto.vmi.vreg<L×ui16>` | Per-lane element offset |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:** `!pto.vmi.vreg<L×T>`
- **attributes:** `pmode`
- **datatypes:** B32 -- `i32`/`ui32`/`f32`; B16 -- `i16`/`ui16`/`f16`/`bf16`,
  plus `i8`/`ui8` -> `i16`/`ui16` zero-extension.
- **lowering:** B16 -> `K × pto.vgather2`; B32 -> `K × pto.vgather2_bc`.
  A statically all-active mask omits the trailing `vsel`.

### `pto.vmi.vscatter`

- **semantics:** Indexed scatter to UB. For each active lane `i`,
  write `value[i]` to `dest[offsets[i]]`.

  ```c
  for (int i = 0; i < L; i++)
      if (mask[i])
          ub[base + offsets[i]] = value[i];
  ```

- **syntax:**
  ```mlir
  pto.vmi.vscatter %value, %dest, %offsets, %mask : !pto.vmi.vreg<L×T>, !pto.ptr<T, ub>, !pto.vmi.vreg<L×i32>, !pto.vmi.mask<L>
  ```
- **operands:**

  | Operand | Type | Description |
  |---|---|---|
  | `value` | `!pto.vmi.vreg<L×T>` | Values to scatter |
  | `dest` | `!pto.ptr<T, ub>` | UB destination base pointer |
  | `offsets` | `!pto.vmi.vreg<L×i32>` or `!pto.vmi.vreg<L×ui16>` | Per-lane element offset |
  | `mask` | `!pto.vmi.mask<L>` | Governing predicate |

- **results:** *(none)*
- **attributes:** `pmode`
- **datatypes:** `i8`–`i32`, `f16`, `bf16`, `f32`
- **lowering to `pto.mi`:**
  ```
  K × pto.vscatter
  ```
  `#mi = K`, `dep = 1`.

  Value, offsets and mask use unit-stride `contiguous` layouts. The PTODSL
  public logical lane whitelist is `1/2/4/8/64/128/256`; invalid counts such as
  96 are rejected in Python. Internal VMI types remain general.

  B32/B16 support partial chunks by intersecting the user mask with logical
  validity. Full chunks do not need an extra intersection. B8 uses dense data
  and a logical b8 mask. Lowering zero-unpacks each low/high 128-byte half into
  the low bytes of B16 request slots, bitcasts back to the original byte type,
  unpacks the matching predicate half to b16, and masks the last request group.
  The number of physical scatters is `ceil(L/64)` for B32 and `ceil(L/128)`
  for B16/B8. Active indices must be valid and pairwise distinct.

- **example:**
  ```mlir
  pto.vmi.vscatter %v, %dest, %offsets, %mask
      : !pto.vmi.vreg<64×f32>, !pto.ptr<f32, ub>, !pto.vmi.vreg<64×i32>, !pto.vmi.mask<64>
  ```
