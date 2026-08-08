# VMI Predicate / Neutral-Element Fold — Implementation Report

**Date**: 2026-08-08  
**Scope**: Generalize `VMIPredicateFold` + skip compiler-synthesized neutral reduce combines in `VMIToVPTO`.  
**Binary-equal**: folds are algebraically identity (AllTrue demask, AllFalse passthru, `max(x,-inf)=x`, `add(x,0)=x`, `vdhist(acc,*,F)=acc`). Camodel ACL runs of the four R4 reduce kernels all **PASS** `np.allclose` vs reference. Device NPU (`torch.npu`) was not available on this host.

## What shipped

| ID | Rule | Where |
|----|------|--------|
| **R1** | AllTrue demask `create_mask(VL)` on compute | `VMIPredicateFold` |
| **R2** | AllTrue pad `vsel` → true arm | `VMIPredicateFold` (existing + kept) |
| **R3** | AllFalse pad `vsel` → false arm | `VMIPredicateFold` (existing + kept) |
| **R4** | Skip `vadd`/`vmax`/`vmin(reduced, 0/-inf/+inf)` after `vcadd`/`vcmax`/`vcmin` | `VMIToVPTO` |
| **R5** | Unrolled `vmax(-inf,x)` / `vmin(+inf,x)` / `vadd(0,x)` → `x` | `VMIPredicateFold` |
| **R6** | AllFalse `vdhist(acc,src,F) → acc` | `VMIPredicateFold` |
| — | Always-emit pad (drop `need_pad`) | `topk_gate_vmi_w128.py` |

Shared analysis: `VMIMaskUtils` (`IntRange`, `MaskLattice`, affine/vci ranges, `mask_and/or/xor/not`).

Lit green: `vmi_predicate_fold_pad.pto`, `vmi_predicate_fold_general.pto`, all updated `vmi_to_vpto_reduce_*.pto` FileChecks.

---

## Top-3 IR gains per rule

Gain ranking is **static IR cost** (ops removed on the hot path), ordered by corpus hit rate / documented cycle gaps where known.

### R1 — AllTrue demask (largest corpus surface)

Hits: quant `create_mask(VL)` on almost every VF iter; dsl ~362 full-mask sites.

#### #1 `vmul` full-mask (quant scale / swiglu-style)

**Before**
```mlir
%c64 = arith.constant 64 : index
%m = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
%out = pto.vmi.vmul %a, %b, %m
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
    -> !pto.vmi.vreg<64xf32>
```

**After**
```mlir
%out = pto.vmi.vmul %a, %b
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xf32>
```

#### #2 `vadd` full-mask (dsl elementwise)

**Before**
```mlir
%c64 = arith.constant 64 : index
%m = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
%out = pto.vmi.vadd %a, %b, %m
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
    -> !pto.vmi.vreg<64xf32>
```

**After**
```mlir
%out = pto.vmi.vadd %a, %b
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xf32>
```

#### #3 `vmax` full-mask (amax / tree reduce leaves)

**Before**
```mlir
%c64 = arith.constant 64 : index
%m = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
%out = pto.vmi.vmax %a, %b, %m
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
    -> !pto.vmi.vreg<64xf32>
```

**After**
```mlir
%out = pto.vmi.vmax %a, %b
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32> -> !pto.vmi.vreg<64xf32>
```

---

### R2 — AllTrue pad `vsel` (topk when `E % VL == 0` or affine-proven)

Documented in this week’s pad fold; always-emit pad + fold removes Python `need_pad` on w128.

#### #1 `pad_all_true_vci` (E = VL = 64)

**Before**: `vci` + `vbrc(E)` + `vcmp lt` + `vsel(m, score, -inf)`  
**After**: `return %score` (cmp/sel DCE’d)

#### #2 `pad_affine_multipass_all_true` (multipass index affine)

**Before**: loop body `vci`/`vadds`/`vcmp`/`vsel` per iter  
**After**: loop yields identity; pad ops removed

#### #3 `vsel_same_arms` (degenerate pad)

**Before**: `vsel %m, %x, %x`  
**After**: `return %x`

---

### R3 — AllFalse pad `vsel` (tail / empty expert chunk)

#### #1 `pad_all_false_vci` (E = 0)

**Before**: `vcmp` + `vsel(m, score, -inf)`  
**After**: `return %-inf`

#### #2 `pad_vadds_all_false` (index past VL)

**Before**: `vadds` + `vcmp` + `vsel`  
**After**: `return %-inf`

#### #3 topk rem chunk with proven empty window

Same rewrite as #1/#2 when range analysis proves all lanes fail `idx < E`.

---

### R4 — Skip neutral reduce combine (RowMax / SoftmaxGrad / EuclideanNorm)

From [`performance_analysis_0804.md`](../../../pto-vmi/docs/performance_analysis_0804.md): RowMax +33%, SoftmaxGrad +17%, EuclideanNorm +21% attributed in part to DSL `vmax`/`vadd` cleanup after `vcmax`/`vcadd`.

#### #1 `vcadd` f16 (SoftmaxGrad / EuclideanNorm style)

**Before** (old lowering)
```mlir
%init = pto.vdup %c0, %pall : f16, !pto.mask<b16> -> !pto.vreg<128xf16>
%vl1 = pto.pset_b16 "PAT_VL1" : !pto.mask<b16>
%red = pto.vcadd %src, %mask : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
%out = pto.vadd %red, %init, %vl1 : ... -> !pto.vreg<128xf16>   // dead: +0
```

**After**
```mlir
// init may remain dead until later DCE; combine + PAT_VL1 gone
%red = pto.vcadd %src, %mask : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
return %red
```

#### #2 `vcadd` i32 / f32 single-chunk

Same shape as #1: `return %vcadd` with no `vadd(reduced, 0)`.

#### #3 `vcmax` multichunk (RowMax across physical parts)

**Before**: `vcmax` × N → `vmax(reduced, -inf)` then inter-chunk `vmax`  
**After**: seed acc from first `vcmax`; only inter-chunk `vmax` remains (no combine with `-inf`)

```mlir
%3 = pto.vcmax %arg0, %arg2
%4 = pto.vcmax %arg1, %arg3
%5 = pto.vmax %4, %3, %vl1   // inter-chunk only
return %5
```

---

### R5 — Unrolled first-iter neutral splat

Hits: topk `acc_max = -inf` tree; quant `acc = 0` sum trees.

#### #1 `vmax(vbrc(-inf), x)` → `x`

**Before**
```mlir
%neg = pto.vmi.vbrc %neginf : f32 -> !pto.vmi.vreg<64xf32>
%out = pto.vmi.vmax %neg, %x, %m : ...
```

**After**: `return %x`

#### #2 `vadd(vbrc(0), x)` → `x`

**Before**
```mlir
%zero = pto.vmi.vbrc %c0 : f32 -> !pto.vmi.vreg<64xf32>
%out = pto.vmi.vadd %zero, %x, %m : ...
```

**After**: `return %x`

#### #3 `vmin(vbrc(+inf), x)` → `x`

**Before**
```mlir
%pos = pto.vmi.vbrc %posinf : f32 -> !pto.vmi.vreg<64xf32>
%out = pto.vmi.vmin %pos, %x, %m : ...
```

**After**: `return %x`

---

### R6 — AllFalse `vdhist`

Hits: rem/empty histogram updates in group-count style kernels.

#### #1 `vdhist` with `create_mask(0)`

**Before**
```mlir
%m = pto.vmi.create_mask %c0 : index -> !pto.vmi.mask<256xpred>
%out = pto.vmi.vdhist %acc, %src, %m : ...
```

**After**: `return %acc`

#### #2 / #3

Same rewrite whenever the mask lattice proves AllFalse (range-`vcmp` or `mask_and` with AllFalse). Additional call sites collapse identically to #1; corpus density is lower than R1/R4.

---

## Camodel A/B (R4) — Ascend950PR_9599 / CANN 9.1.0-beta.3

Measured 2026-08-08 on `edgexpert-59a6` (aarch64).  
**Before** = stock `ptoas` 0.53; **After** = local `ptoas` 0.56 with neutral-combine skip.  
All cases: `*_real_float_Rows_128_Cols_64.py`, ACL path, `PASS`.

| Kernel | CCE (0804) | DSL before | DSL after | Δ ticks | Δ RVECEX | Dead op removed |
|--------|------------|------------|-----------|---------|----------|-----------------|
| **RowMaxKernel** | 358 | 477 | **358** | −119 (−25%) | 388→257 (−131) | 128× `RV_VMAX` |
| **RowMinKernel** | 358 | 477 | **358** | −119 (−25%) | 388→257 (−131) | 128× `RV_VMIN` |
| **SoftmaxGradKernel** | 692 | 813 | **692** | −121 (−15%) | 772→641 (−131) | 128× `RV_VADD` |
| **EuclideanNormVfKernel** | 523 | 632 | **523** | −109 (−17%) | 644→513 (−131) | 128× `RV_VADD` |

After R4, all four match the CCE `vf_real_execute_time` from [`performance_analysis_0804.md`](../../../pto-vmi/docs/performance_analysis_0804.md) (the prior +17…+33% DSL gaps attributed to neutral `vadd`/`vmax`/`vmin` cleanup).

Small-shape sanity (`RowMaxKernel.case1_float_Rows_2_Cols_64`): vf 74→71, RVECEX 10→5, `RV_VMAX` 2→0.

---

## Camodel A/B (R1–R3, R5–R6) — Ascend950PR_9599 / CANN 9.1.0-beta.3

Measured after rebuilding `libPTOASCompiler.so` (fold was in `pto-test-opt` / `.o` but not linked into the CLI shared lib until 2026-08-08).  
Harness: local `ptoas` 0.56; **Before** = `PTO_FLAGS=--disable-vmi-predicate-fold` (now honored by `ptodsl` `native_build`); **After** = fold on. Full `topk_gate` camodel still blocked on `wait_flag` uncovered-section normalize under 0.56 — pad / peep coverage uses VL=64 micros with the same `vcmp+vsel` / `vmax(-inf)` / `vdhist` shapes.

### R1 — AllTrue demask (dsl real)

| Case | Result | vf_real | Signal |
|------|--------|---------|--------|
| SoftmaxGradKernel real 128×64 | PASS | 813→**692** | Dominated by **R4** (−128 `RV_VADD`); `RV_PSET` 3→1 |
| confusionSoftmaxGradArKernel real half 128×64 | PASS | 673→**553** | Same R4-class −128 `RV_VADD`; `RV_PSET` 3→1 |
| BlkScaleMulKernel real f16 128×128 | PASS | 183→183 | Identical asm (no R1 tick win on this shape) |

R1 alone is typically ADD-pipe / setup; largest corpus surface, smaller tick delta than R4.

### R2 — AllTrue pad `vsel` (E covers lanes)

| Case | Result | vf | Before ops | After ops |
|------|--------|-----|------------|-----------|
| pad E=64 (VL=64) | PASS | 53→**50** | `VCI+VCMP+VSEL` | **0** `RV_VSEL`/`RV_VCMP` (load→store) |
| pad E=256 | PASS | 53→**50** | same | **0** pad vsel/vcmp |
| pad E=384 | PASS | 53→**50** | same | **0** pad vsel/vcmp |

### R3 — AllFalse pad `vsel`

| Case | Result | vf | After behavior |
|------|--------|-----|----------------|
| pad E=300 (idx base ≥ E → AllFalse) | PASS | 53→**49** | `VDUPS(-inf)`+store; **0** vsel/vcmp |
| pad E=100 (same AllFalse class) | PASS | 53→**49** | same |
| pad partial E=48 | PASS | 53→53 | Keeps `VCMP+VSEL` (correct non-fold) |

### R5 — Neutral `vmax(-inf,x)`

| Case | Result | vf | Asm |
|------|--------|-----|-----|
| micro `vmax(vbrc(-inf), x)` | PASS | 52→**50** | Before: `VDUPS+VMAX`; After: **0** both (identity) |
| VFShiftVectorKernel real 128×64 | PASS | 439→439 | No `-inf` seed peep site; `RV_PSET` 2→1 only |

### R6 — AllFalse `vdhist → acc`

| Case | Result | vf | Asm |
|------|--------|-----|-----|
| micro `vdhist(acc,src,create_mask(0))` | PASS | 61→**50** | Before: 1× `RV_DHIST`; After: **0** (acc passthrough) |
| ExpertTokenHistVfKernel case1 | PASS | ~73–75 | No AllFalse site (2× `DHIST` both sides); `PSET` 3→2 |
| IndexStatisticInt32VfKernel real | PASS | ~1600 | Still 128× `DHIST`; `PSET` 5→3 (R1-ish) |

### Env notes

- Rebuild `ninja PTOASCompiler` after editing `VMIPredicateFold` — `python/pto/ptoas.so` can be stale; CLI uses `python/ptoas/mlir/_mlir_libs/libPTOASCompiler.so`.
- `PTO_FLAGS` is forwarded by `ptodsl/_runtime/native_build.py` and included in the compile-config cache key.

## Verification checklist

| Check | Result |
|-------|--------|
| `vmi_predicate_fold_pad.pto` | PASS |
| `vmi_predicate_fold_general.pto` | PASS |
| `vmi_to_vpto_reduce_{addf,addf_f16,addi,minf,*_multichunk}.pto` | PASS |
| Camodel R4 RowMax / RowMin / SoftmaxGrad / EuclideanNorm real 128×64 | PASS + CCE-parity ticks |
| Camodel R1 SoftmaxGrad / CSG / BlkScale | PASS |
| Camodel R2 AllTrue pad E∈{64,256,384} | PASS + 0 pad `VSEL`/`VCMP` |
| Camodel R3 AllFalse pad E∈{300,100} + partial E=48 | PASS + false-arm / keep-mixed |
| Camodel R5 neutral vmax micro (+ VFShiftVector) | PASS + `VDUPS+VMAX` gone on micro |
| Camodel R6 AllFalse vdhist micro (+ ExpertTokenHist / IndexStatistic) | PASS + 0 `DHIST` on AllFalse site |
| Full `topk_gate` camodel | **Blocked** — `wait_flag` uncovered tile section under local 0.56 |
| Device NPU smoke (`test_topk_gate.py`) | **Blocked** — no `torch.npu` here |

## Follow-ups (optional)

1. DCE leftover `vdup(0/-inf)` after R4 skips the combine (init still materialized by LowerUnified; already not in VF body for these cases).
2. Memory AllFalse merge-store erase (Tier B).
3. On-device topk / MoE binary-equal under TileKernels-vmi (`env_npu.sh`, CANN 9.1b3).
4. Unblock full `topk_gate` camodel (`PTONormalizeUncoveredTileSections` / `wait_flag`) for production E shapes.
