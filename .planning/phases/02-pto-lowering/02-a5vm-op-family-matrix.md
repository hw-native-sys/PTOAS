# A5VM Op Family Matrix

Updated: 2026-03-19

Scope:

- Source of truth is the shared pre-backend seam IR.
- Matrix-family ops remain deferred.
- This file is the implementation matrix for the non-matrix expansion waves.

## Wave 1 Summary

Already lowered in current A5VM baseline:

- `pto.tload`
- `pto.tabs`
- `pto.tstore`
- `pto.set_flag`
- `pto.wait_flag`
- `pto.barrier`

Wave-1 seam-only families now identified:

- address / view:
  - `pto.pointer_cast`
  - `pto.bind_tile`
  - `pto.view_semantics`
  - `pto.addptr_trace`
- scalar pointer:
  - `pto.load_scalar`
  - `pto.store_scalar`
- unary:
  - `pto.texp`
  - `pto.tlog`
  - `pto.tsqrt`
  - `pto.trecip`
  - `pto.trelu`
  - `pto.tneg`
  - `pto.tnot`
  - `pto.trsqrt`
  - `pto.tlrelu`
- binary:
  - `pto.tadd`
  - `pto.tand`
  - `pto.tcmp`
  - `pto.tdiv`
  - `pto.tmax`
  - `pto.tmin`
  - `pto.tmul`
  - `pto.tor`
  - `pto.trem`
  - `pto.tsel`
  - `pto.tshl`
  - `pto.tshr`
  - `pto.tsub`
  - `pto.txor`
- vec-scalar / mixed:
  - `pto.tadds`
  - `pto.taddsc`
  - `pto.tands`
  - `pto.tcmps`
  - `pto.tcvt`
  - `pto.tdivs`
  - `pto.tgetval`
  - `pto.tmaxs`
  - `pto.tmins`
  - `pto.tmov`
  - `pto.tmuls`
  - `pto.tors`
  - `pto.trems`
  - `pto.tsels`
  - `pto.tsetval`
  - `pto.tshls`
  - `pto.tshrs`
  - `pto.tsubc`
  - `pto.tsubs`
  - `pto.tsubsc`
  - `pto.textract`
  - `pto.txors`

## Matrix

| Seam PTO op | Samples | EmitC helper/API | A5 PTO implementation | Backend form | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `pto.tload` | `Abs`, `VectorAddition`, many | `TLOAD` | `npu/a5/TLoad.hpp` | `a5vm.copy_gm_to_ubuf` + loop attrs | implemented | Baseline complete |
| `pto.tabs` | `Abs` | `TABS` | `npu/a5/TUnaryOp.hpp` | `a5vm.vlds/vabs/vsts` + vec-scope loop | implemented | Baseline complete |
| `pto.tstore` | `Abs`, `VectorAddition`, many | `TSTORE` | `npu/a5/TStore.hpp` | `a5vm.copy_ubuf_to_gm` + loop attrs | implemented | Baseline complete |
| `pto.texp` | `Exp` | `TEXP` | `npu/a5/TUnaryOp.hpp` | `a5vm.vlds/vexp/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.tlog` | `Log` | `TLOG` | `npu/a5/TUnaryOp.hpp` | `a5vm.vlds/vln/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.tsqrt` | `Sqrt` | `TSQRT` | `npu/a5/TUnaryOp.hpp` | `a5vm.vlds/vsqrt/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.trecip` | `Recip` | `TRECIP` | likely `TUnaryOp` path | `a5vm.vlds/vrec/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.trelu` | `Relu` | `TRELU` | likely `TUnaryOp` path | `a5vm.vlds/vrelu/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.tnot` | `Not` | `TNOT` | likely `TUnaryOp` path | `a5vm.vlds/vnot/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.trsqrt` | `Rsqrt` | `TRSQRT` | `npu/a5/TUnaryOp.hpp` special path | A5VM op sequence, not single op | planned | PTO uses explicit sqrt/div structure |
| `pto.tlrelu` | `Lrelu` | `TLRELU` | scalar-vector path in builtin layer | `a5vm` vec-scalar op + vec-scope loop | planned | Needs scalar operand contract |
| `pto.tadd` | `VectorAddition`, `Subset/vadd_pto_pingpong` | `TADD` | `npu/a5/TAdd.hpp` | `a5vm.vlds/vadd/vsts` + vec-scope loop | implemented | Row-major vec path works for static and current dynamic valid-shape VectorAddition cases |
| `pto.tsub` | `Sub` | `TSUB` | `npu/a5/TSub.hpp` | `a5vm.vlds/vsub/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.tmul` | `Mul` | `TMUL` | `npu/a5/TMul.hpp` | `a5vm.vlds/vmul/vsts` + vec-scope loop | implemented | Static row-major vec path |
| `pto.tdiv` | `Div` | `TDIV` | `npu/a5/TDiv.hpp` | `a5vm.vlds/vdiv/vsts` + vec-scope loop | implemented | Static row-major vec path with current type gate |
| `pto.tcmp` | `Cmp` | `TCMP` | `npu/a5/TCmp.hpp` | `a5vm` compare/select style ops | planned | Needs predicate/result form study |
| `pto.tcvt` | `Cvt`, `Reshape/bitcast_inplace_cvt` | `TCVT` | `npu/a5/TCvt.hpp` | `a5vm` conversion op family | planned | Type-dependent builtin spelling |
| `pto.view_semantics` | `Reshape` | emitc helper-side view code | `npu/a5/TReshape.hpp` | LLVM-lowerable non-a5vm IR | planned | Not a CCE builtin |
| `pto.load_scalar` | `ScalarPtr` | scalar pointer C/C++ | no CCE builtin | LLVM dialect | planned | Not `a5vm` |
| `pto.store_scalar` | `ScalarPtr` | scalar pointer C/C++ | no CCE builtin | LLVM dialect | planned | Not `a5vm` |
| `pto.tsetval` | `TileSetGetValue` | helper-side tile mutation | non-builtin helper path | LLVM-lowerable IR | planned | Need exact emitc mapping |
| `pto.tgetval` | `TileSetGetValue` | helper-side tile read | non-builtin helper path | LLVM-lowerable IR | planned | Need exact emitc mapping |
| `pto.textract` | `Extract` | `TEXTRACT` | `npu/a5/TExtract.hpp` | mixed | deferred | Sample seam also contains `pto.tgemv` |
| `pto.tmov` | `Extract` | `TMOV` | `npu/a5/TMov.hpp` | mixed | deferred | Same reason as `pto.textract` |
| `pto.addptr_trace` | `AddPtr` | debug trace helper | no hardware op | LLVM / comments only | planned | Developer-only trace |

## Current Execution Order

1. Extend unary vec family from the existing `TABS` skeleton:
   `texp`, `tlog`, `tsqrt`, `trecip`, `trelu`, then `tnot`.
2. Add binary vec family skeleton from `TBinOp.hpp`:
   `tadd`, `tsub`, `tmul`, then `tdiv`.
3. Add vec-scalar family skeleton from `TBinSOp.hpp`.
4. Revisit mixed/helper families:
   `view_semantics`, scalar pointer, tile get/set value.

## Current Gaps

- vec-scalar families from `TBinSOp.hpp`
- comparison / selection families
- conversion families
- mixed helper families such as `view_semantics`, tile value get/set, and scalar pointer flows

Current blocker:

- non-`VectorAddition` dynamic-shape samples still expose uncovered lowering semantics outside the unified copy-op operand contract, especially in deferred families and TODO domains.
