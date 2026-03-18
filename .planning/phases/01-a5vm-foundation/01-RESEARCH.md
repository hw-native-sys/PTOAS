# Phase 1: A5VM Foundation - Research

**Researched:** 2026-03-19
**Domain:** Corrected A5-side A5VM primitive dialect bring-up and backend-boundary replanning for PTOAS
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### A5VM identity and namespace
- `a5vm` remains a first-class dialect under the project directory tree, but its C++ namespace must be `mlir::a5vm`, not `mlir::pto::a5vm`.
- `a5vm` should keep the normalized vector type spelling such as `!a5vm.vec<64xf32>`.
- The already-landed namespace and op-surface choices from the first Phase 1 attempt should be treated as superseded by this corrected context.

### Primitive-op direction
- `a5vm` ops should stay as close as practical to CCE builtin naming rather than freezing PTO-interface-shaped pseudo-ops such as `a5vm.load` and `a5vm.store`.
- Phase 1 should define the minimum primitive surface needed by the real PTO-library-aligned lowering shape, including vector-memory and vector-compute primitives such as `vld`, `vabs`, and `vst`, plus any loop-scope or copy-family primitives that later `TLOAD` / `TSTORE` lowering will need.
- Phase 1 must not encode PTO interface semantics directly into A5VM ops when the real hardware-facing primitive is lower-level and differently named.
- General control flow and scalar arithmetic should remain in shared dialects when they are not hardware-facing.

### Backend switching and output seam
- Keep dual backend paths during the correction pass.
- Select backend through an explicit CLI flag rather than a hidden or hardwired mode switch.
- Default CLI behavior should remain compatible with current usage, but new backend selection must be available for developers.
- The final textual HIVM emission seam still belongs at the current `emitc::translateToCpp` boundary, but Phase 1 should not over-specify final intrinsic names before the primitive A5VM surface is corrected.

### Failure and placeholder policy
- When something can be emitted as legal textual IR, do that even if some details remain provisional.
- When a case cannot be emitted legally, output should include explicit unresolved markers or comments instead of silently guessing.
- Placeholder handling should preserve enough context that required intrinsic mappings can be reviewed and confirmed later.

### Replanning Notes
- The previous assumption that the `Abs` path should be represented primarily with `a5vm.load` / `a5vm.abs` / `a5vm.store` is incorrect and must not guide new planning.
- Downstream planning must use the A5-side PTO library implementation plus CCE builtin families as the semantic source of truth, not the currently landed A5VM pseudo-load/store design.
- Replanning does not need to preserve or mirror `a2a3` implementation details; only the A5 PTO path matters for this effort.

### Claude's Discretion
- Exact file split for the corrected A5VM dialect implementation under the existing PTO directory tree
- Exact choice of which primitive ops land in Phase 1 versus wait for Phase 2, as long as they align to the PTO-library-backed lowering shape
- Exact flag names for backend selection and debug controls

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Exact PTO `TLOAD` / `TABS` / `TSTORE` lowering structure belongs to Phase 2, but Phase 1 must stop making assumptions that contradict the PTO library.
- Exact final HIVM intrinsic names remain deferred.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| BACK-01 | Developer can run the existing PTOAS compilation flow with a backend path that replaces the current `emitc` generation slot without requiring a pass-pipeline redesign. | Keep backend selection in `tools/ptoas/ptoas.cpp` at the existing final-emission seam; replace only the obsolete A5VM branch and its fixtures, not the upstream pass pipeline. |
| BACK-02 | Developer can keep ordinary control flow and scalar arithmetic in shared dialects such as `scf` and `arith` while only hardware-facing PTO operations enter the new backend path. | Use `a5vm` only for hardware-facing copy and vector primitives; keep loop structure, scalar arithmetic, and non-hardware control flow in shared MLIR dialects. |
| A5VM-01 | Developer can represent legal `a5vm` vector types whose total width is always exactly 256 bytes under the corrected `mlir::a5vm` dialect namespace. | Keep `!a5vm.vec<...>` and its 256-byte verifier, but correct the dialect `cppNamespace` to `::mlir::a5vm` and remove all `mlir::pto::a5vm` references. |
| A5VM-02 | Developer can represent the `Abs` path with hardware-facing `a5vm` primitive operations whose naming stays close to the CCE builtin layer rather than to pseudo PTO-interface-shaped ops. | Replace pseudo ops with corrected primitive families: copy-family GM/UB movement plus vector register compute ops named after the A5 builtin layer (`vlds`/`vabs`/`vsts` family, not PTO-shaped `load`/`store`). |
| A5VM-03 | Developer can represent the `Abs` vector compute path with corrected `a5vm` vector primitives such as `vld`, `vabs`, and `vst` whose operand/result types are legal `a5vm` vector types. | Model the `Abs` compute kernel after A5 `TUnaryOp.hpp`: UB source and destination plus register-vector ops for load/store around `vabs`, with legal `!a5vm.vec<...>` values on the compute edges. |
| A5VM-04 | Developer can represent the `Abs` memory-movement path with corrected `a5vm` primitives that are suitable for PTO-library-aligned GM/UB transfer lowering. | Add explicit GM<->UB copy-family primitives that preserve layout, valid rows/cols, burst/stride, and padding metadata instead of flattening memory movement into fake vector `load`/`store` ops. |
</phase_requirements>

## Summary

The old Phase 1 research is superseded. The A5 PTO headers show that `Abs` is not structured as high-level `load -> abs -> store`. `TLOAD` is a GM-to-UB copy family with layout-specific branching and explicit burst/stride/valid-region parameters. `TABS` is a UB register loop that repeatedly performs `vlds`, `vabs`, and `vsts` inside vector scope. `TSTORE` is a UB-to-GM copy family with its own layout- and tile-domain-specific branching. Phase 1 therefore needs a corrected hardware-facing primitive surface, not PTO-shaped pseudo-ops.

The minimum corrected Phase 1 surface for the `Abs` path is: a legal 256-byte `!a5vm.vec<...>` type under `mlir::a5vm`; one GM-to-UB copy primitive family; one UB-to-GM copy primitive family; and the vector-register compute trio for the unary body. That is enough to plan Phase 2 PTO-library-faithful lowering without locking in final intrinsic spellings or overbuilding the full lowering framework now.

Planning should explicitly replace, not extend, the wrong work already in tree. The existing `a5vm.load` / `a5vm.abs` / `a5vm.store` contracts, the `::mlir::pto::a5vm` namespace, and every Phase 1/2 fixture that encodes those assumptions should be treated as invalid baseline. The clean plan is to rewrite Wave 0 contracts first, then re-land the dialect and backend boundary against the corrected primitive direction.

**Primary recommendation:** Plan Phase 1 around `mlir::a5vm`, `!a5vm.vec<...>`, GM/UB copy primitives, and UB vector-register primitives (`vlds`/`vabs`/`vsts` family), while keeping loops and scalar logic in shared dialects and leaving PTO semantic lowering to Phase 2.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LLVM | 19.1.7 | Core compiler support and build integration | Verified from the local toolchain config already used by this repo. |
| MLIR | 19.1.7 | Dialect, type, op, parser, and pass infrastructure | The repo is already structured as an out-of-tree MLIR project and should keep using first-class dialect/TableGen patterns. |
| Ascend CANN PTO headers | 8.5.0 | Semantic source of truth for A5 PTO instruction behavior | The corrected phase is explicitly constrained to A5 PTO semantics from the installed CANN headers. |
| Ascend CCE vector intrinsics headers | 15.0.5 toolchain headers under CANN 8.5.0 | Primitive naming and operand-shape reference | These headers expose the builtin families that should drive corrected `a5vm` op naming. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| MLIR TableGen | 19.1.7 | Generate dialect/type/op declarations and definitions | Use for the corrected dialect surface; do not hand-roll a separate registry pattern. |
| Bash + `FileCheck` | workspace LLVM tools | Fast fixture verification for parser/printer/backend contracts | Use for corrected Wave 0 and per-task validation. |
| CTest | CMake/CTest from the build | Broader regression pass already used by the repo | Run at wave merge or phase gate, not for every tiny dialect iteration. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `a5vm` copy-family + vector-register primitives | Keep `a5vm.load` / `a5vm.abs` / `a5vm.store` and reinterpret them later | This preserves the wrong abstraction and makes Phase 2 lowering misleading. |
| `vlds` / `vabs` / `vsts`-aligned naming | Generic names like `vector_load` / `vector_store` | Easier to read at first glance, but farther from the A5 builtin source of truth. |
| Clean replacement of old fixtures and dialect ops | Compatibility aliases for obsolete op names | Aliases would keep invalid semantics alive and invite plan drift. |

**Version verification:**
- LLVM `19.1.7` verified in `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake`
- MLIR `19.1.7` verified in `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake`
- CANN references verified under `/usr/local/Ascend/cann-8.5.0/...`

## Architecture Patterns

### Recommended Project Structure
```text
include/PTO/IR/
├── A5VM.h
├── A5VMDialect.h
├── A5VMDialect.td
├── A5VMTypes.td
└── A5VMOps.td

lib/PTO/IR/
└── A5VM.cpp

lib/PTO/Transforms/
├── PTOToA5VM.cpp
└── A5VMTextEmitter.cpp

test/phase1/
├── a5vm_vec_type.mlir
├── a5vm_copy_gm_to_ubuf_op.mlir
├── a5vm_vabs_kernel_shape.mlir
├── a5vm_copy_ubuf_to_gm_op.mlir
├── a5vm_backend_switch.mlir
├── a5vm_shared_dialects.mlir
└── run_phase1_checks.sh
```

### Pattern 1: Model A5 `Abs` as Copy + UB Vector Register Ops
**What:** Separate memory movement from vector computation, following the A5 PTO implementation shape.
**When to use:** For the corrected Phase 1 primitive contract and for all future PTO-to-A5VM lowerings.
**Example:**
```mlir
%src = a5vm.copy_gm_to_ubuf %gm[%base]
  {layout = "nd", valid_rows = 32 : i64, valid_cols = 32 : i64,
   burst_len = 128 : i64, burst_count = 1 : i64,
   gm_stride = 32 : i64, ub_stride = 64 : i64, ub_pad = false}
  : memref<?xf32> -> memref<64xf32, #a5vm.ub>

scf.for %i = %c0 to %c1 step %c1 {
  %r0 = a5vm.vlds %src[%c0] : memref<64xf32, #a5vm.ub> -> !a5vm.vec<64xf32>
  %r1 = a5vm.vabs %r0 : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
  a5vm.vsts %r1, %dst[%c0] {dist = "norm"} : !a5vm.vec<64xf32>, memref<64xf32, #a5vm.ub>
}

a5vm.copy_ubuf_to_gm %dst[%base], %gm_out
  {layout = "nd", valid_rows = 32 : i64, valid_cols = 32 : i64,
   burst_len = 128 : i64, burst_count = 1 : i64,
   gm_stride = 32 : i64, ub_stride = 64 : i64}
  : memref<64xf32, #a5vm.ub>, memref<?xf32>
```
Source: A5 `TLoad.hpp`, `TUnaryOp.hpp`, `TStore.hpp`

### Pattern 2: Keep Shared Dialects for Looping and Scalar Bookkeeping
**What:** Use `scf`, `arith`, `func`, and `memref` for non-hardware control flow and scalar values.
**When to use:** Always, unless the operation itself is a hardware-facing A5 primitive.
**Example:**
```text
PTO semantic IR
   |
   | Phase 2 lowering
   v
func/scf/arith/memref + a5vm.copy_* + a5vm.vlds/vabs/vsts
   |
   | Phase 3 text emission
   v
textual HIVM at the existing final seam
```
Source: project requirements and current `tools/ptoas/ptoas.cpp` boundary

### Pattern 3: Make Memory-Movement Primitives Explicitly Metadata-Rich
**What:** Carry the transfer facts that A5 `TLOAD` / `TSTORE` actually branch on: layout, valid rows/cols, burst counts, burst length, strides, and padding/quantization controls where applicable.
**When to use:** On copy-family ops only.
**Example:**
```c++
template <typename TileData, typename GlobalData>
void TLoad(..., int gShape0, int gShape1, int gShape2, int gShape3, int gShape4,
           int gStride0, int gStride1, int gStride2, int gStride3, int gStride4,
           int validRow, int validCol);
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp`

### Pattern 4: Keep Phase 1 Primitive, Not Semantic
**What:** Define only the primitive surface needed so Phase 2 can lower PTO semantics faithfully later.
**When to use:** Throughout Phase 1.
**Example:**
- `a5vm.copy_gm_to_ubuf`: primitive data movement
- `a5vm.vlds`: UB-to-register load
- `a5vm.vabs`: register unary compute
- `a5vm.vsts`: register-to-UB store
- `a5vm.copy_ubuf_to_gm`: primitive data movement

Do not add:
- `a5vm.tload`
- `a5vm.tabs`
- `a5vm.tstore`
- PTO-layout-specific mega-ops that already encode Phase 2 semantics

### Anti-Patterns to Avoid
- **Keeping compatibility aliases for `a5vm.load` and `a5vm.store`:** This defeats the correction.
- **Collapsing GM/UB transfer into vector result/value ops:** A5 `TLOAD` / `TSTORE` are not single-vector load/store semantics.
- **Adding a custom `a5vm.for` or scalar arithmetic ops in Phase 1:** BACK-02 explicitly keeps those in shared dialects.
- **Naming ops after final HIVM spellings now:** Phase 1 should not guess final intrinsic names.
- **Mirroring `a2a3` behavior:** The correction scope is A5 only.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dialect boilerplate | Handwritten op/type registration and parser glue | MLIR TableGen plus one `A5VM.cpp` implementation file | Matches the repo structure and keeps replacement contained. |
| PTO semantics in Phase 1 | A custom pseudo-semantic A5VM layer | Primitive copy/register ops only | Prevents Phase 1 from freezing the wrong abstraction. |
| General control flow in `a5vm` | Custom loop/scope arithmetic ops | `scf`/`arith`/`func` | BACK-02 already requires this split. |
| Transfer parameter inference in the emitter | Recomputing burst/stride/layout from ad hoc clues | Explicit attrs/operands on copy-family ops | A5 `TLOAD` / `TSTORE` branch on these values directly. |
| Legacy fixture migration | Incremental edits that leave old contracts half alive | Replace obsolete fixtures in one Wave 0 rewrite | Mixed old/new contracts would make planning incoherent. |

**Key insight:** The deceptively hard problem here is not `vabs`; it is preserving the real A5 transfer structure without prematurely lowering PTO semantics. Hand-rolled shortcuts around that boundary will force a rewrite in Phase 2.

## Common Pitfalls

### Pitfall 1: Treating `TLOAD` as a Vector Load
**What goes wrong:** Planning assumes a single vector result op is enough.
**Why it happens:** The `Abs` sample is small, so the GM-to-UB copy work is easy to overlook.
**How to avoid:** Model `TLOAD` as copy-family primitives that preserve layout/stride/valid-region facts.
**Warning signs:** An op shaped like `%v = a5vm.load %gm[...] : memref -> !a5vm.vec<...>`.

### Pitfall 2: Treating `TABS` as a Single Primitive Operation
**What goes wrong:** Phase 1 skips the UB register movement that the A5 implementation actually performs.
**Why it happens:** `TABS` is conceptually simple, but A5 implements it as `vlds` + `vabs` + `vsts` inside repeated vector loops.
**How to avoid:** Make the compute contract explicitly register-based and leave loop structure in shared dialects.
**Warning signs:** No `vlds`/`vsts` family in the planned primitive set.

### Pitfall 3: Keeping the Wrong Namespace Alive
**What goes wrong:** New code compiles, but the dialect still lands in `mlir::pto::a5vm`.
**Why it happens:** The current `A5VMDialect.td` and `A5VM.cpp` already hardcode the obsolete namespace.
**How to avoid:** Treat namespace correction as a replacement task, not a follow-up cleanup.
**Warning signs:** `using namespace mlir::pto::a5vm;` or `cppNamespace = "::mlir::pto::a5vm"`.

### Pitfall 4: Freezing Old Fixtures as “Close Enough”
**What goes wrong:** Wave 0 passes while locking the wrong op surface into plans and reviews.
**Why it happens:** The wrong fixtures already exist and seem convenient to reuse.
**How to avoid:** Rewrite Phase 1 and affected Phase 2 fixtures before implementation planning.
**Warning signs:** Any Phase 1 fixture named around `a5vm_load_op`, `a5vm_abs_op`, or `a5vm_store_op` without corrected semantics.

### Pitfall 5: Overreaching into Full Phase 2 Lowering
**What goes wrong:** Phase 1 starts encoding PTO-specific layout branching and tile-domain behavior directly in the dialect.
**Why it happens:** The source headers expose many cases and it is tempting to land them early.
**How to avoid:** Keep Phase 1 to the primitive surface needed by the `Abs` path only.
**Warning signs:** A large matrix of ND/DN/NZ-specific semantic ops or `TSTORE` quantization specializations inside Phase 1.

### Pitfall 6: Choosing Wrapper Names Without Checking the A5 Call Sites
**What goes wrong:** Ops are named after generic wrappers while the real A5-side implementation uses different primitive families.
**Why it happens:** The builtin header exposes both higher-level wrappers (`vload` / `vstore`) and lower-level register ops (`vlds` / `vsts`).
**How to avoid:** Use the A5 PTO implementation as the deciding source. For `Abs`, the actual compute path uses `vlds` / `vabs` / `vsts`.
**Warning signs:** The plan justifies names from intuition rather than from `TUnaryOp.hpp`.

## Code Examples

Verified patterns from primary sources:

### A5 `TABS` Inner Shape
```c++
for (uint16_t i = 0; i < repeatTimes; ++i) {
  pReg = CreatePredicate<T>(sReg);
  vlds(srcReg, src, i * nRepeatElem, NORM);
  Op::UnaryInstr(dstReg, srcReg, pReg);
  vsts(dstReg, dst, i * nRepeatElem, distValue, pReg);
}
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp`

### A5 `TABS` Primitive Binding
```c++
template <typename T>
struct AbsOp {
  PTO_INTERNAL static void UnaryInstr(RegTensor<T> &dstReg, RegTensor<T> &srcReg, MaskReg &pReg) {
    vabs(dstReg, srcReg, pReg, MODE_ZEROING);
  }
};
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp`

### A5 `TLOAD` Copy Primitive Shape
```c++
copy_gm_to_ubuf_align_v2(reinterpret_cast<__ubuf__ uint32_t *>(dst),
    reinterpret_cast<__gm__ uint32_t *>(src), 0, nBurst, lenBurst,
    0, 0, enableUBPad, 0, gmStride, ubStride);
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp`

### CCE Builtin Wrapper Naming
```c++
CCE_INTRINSIC void vload(vector_f32 &dst, __ubuf__ float *base, ...);
CCE_INTRINSIC void vstore(__ubuf__ float *base, vector_f32 src, uint32_t elementCount, ...);
```
Source: `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PTO-shaped pseudo ops (`a5vm.load`, `a5vm.abs`, `a5vm.store`) | Primitive A5-facing copy + register ops | Corrected on 2026-03-19 planning pass | Phase 1 plans and fixtures must be rewritten before implementation resumes. |
| `::mlir::pto::a5vm` namespace | `::mlir::a5vm` namespace | Corrected on 2026-03-19 planning pass | Existing dialect files must be replaced, not incrementally patched around. |
| Phase 1 fixtures as stable contract | Phase 1 fixtures as obsolete baseline | Corrected on 2026-03-19 planning pass | Validation/Wave 0 must lock the new direction first. |

**Deprecated/outdated:**
- `include/PTO/IR/A5VMDialect.td` current `cppNamespace = "::mlir::pto::a5vm"` setting: superseded.
- `include/PTO/IR/A5VMOps.td` current `A5VM_LoadOp` / `A5VM_AbsOp` / `A5VM_StoreOp`: superseded.
- `test/phase1/a5vm_load_op.mlir`, `test/phase1/a5vm_abs_op.mlir`, `test/phase1/a5vm_store_op.mlir`: superseded.
- `test/phase2/tload_contract_trace.mlir`, `test/phase2/tabs_precheck.mlir`, `test/phase2/tstore_branch_shape.mlir`, `test/phase2/unary_template_shape.mlir` as currently written: require follow-on correction because they still lock obsolete Phase 1 primitives.

## Open Questions

1. **Should Phase 1 use `vlds` / `vsts` exact names or `vld` / `vst` family names?**
   - What we know: A5 `TUnaryOp.hpp` uses `vlds` / `vsts` directly, while the builtin header also exposes higher-level `vload` / `vstore` wrappers.
   - What's unclear: Whether the project wants exact call-site names or family-normalized names in MLIR.
   - Recommendation: Use `vlds` / `vabs` / `vsts` for the `Abs` compute path because they match the A5 PTO implementation, and document that choice explicitly in the plan.

2. **Does Phase 1 need a dedicated UB memory-space type/attr in `a5vm`?**
   - What we know: Copy-family ops need to distinguish GM and UB endpoints, and the compute ops operate on UB-backed buffers plus vector registers.
   - What's unclear: Whether that should be represented through memref memory spaces, a small dialect attr, or by reusing existing PTO address-space attrs.
   - Recommendation: Reuse existing memref/address-space machinery if it already models GM/UB cleanly; do not invent a new type family unless required by parser/emitter constraints.

3. **How much copy metadata should be first-class in Phase 1?**
   - What we know: `TLOAD` / `TSTORE` branch on layout, valid rows/cols, burst lengths/counts, strides, and sometimes pad/quantization controls.
   - What's unclear: The smallest attribute set that keeps Phase 2 faithful without overcommitting.
   - Recommendation: For `Abs`, lock layout, valid rows, valid cols, burst count, burst length, GM stride, UB stride, and UB-pad flag; leave quantization and ACC/MAT variants for Phase 2.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Bash runner + `FileCheck` fixtures, with `ctest` as broader regression |
| Config file | none |
| Quick run command | `./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir` |
| Full suite command | `bash test/phase1/run_phase1_checks.sh && ctest --test-dir build --output-on-failure` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| BACK-01 | `--pto-backend=a5vm` selects the corrected backend branch at the current final-emission seam | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase1/a5vm_backend_switch.mlir -o - | FileCheck test/phase1/a5vm_backend_switch.mlir` | ❌ Wave 0 |
| BACK-02 | Shared `scf` / `arith` structure stays outside `a5vm` while hardware-facing ops use corrected primitives | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase1/a5vm_shared_dialects.mlir -o /dev/null 2>&1 | FileCheck test/phase1/a5vm_shared_dialects.mlir` | ❌ Wave 0 |
| A5VM-01 | `!a5vm.vec<...>` stays legal only at 256 bytes and the dialect lands in `mlir::a5vm` | unit | `bash -lc 'rg -n \"cppNamespace = \\\"::mlir::a5vm\\\"\" include/PTO/IR/A5VMDialect.td && ./build/tools/ptoas/ptoas test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir'` | ❌ Wave 0 |
| A5VM-02 | Corrected `a5vm` op names stay close to A5/CCE builtin families rather than pseudo PTO names | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_vabs_kernel_shape.mlir -o - | FileCheck test/phase1/a5vm_vabs_kernel_shape.mlir` | ❌ Wave 0 |
| A5VM-03 | `Abs` compute path uses legal vector primitives for register load/compute/store | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_vabs_kernel_shape.mlir -o - | FileCheck test/phase1/a5vm_vabs_kernel_shape.mlir` | ❌ Wave 0 |
| A5VM-04 | `Abs` memory path uses corrected GM/UB transfer primitives with explicit metadata | unit | `./build/tools/ptoas/ptoas test/phase1/a5vm_copy_gm_to_ubuf_op.mlir -o - | FileCheck test/phase1/a5vm_copy_gm_to_ubuf_op.mlir && ./build/tools/ptoas/ptoas test/phase1/a5vm_copy_ubuf_to_gm_op.mlir -o - | FileCheck test/phase1/a5vm_copy_ubuf_to_gm_op.mlir` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `bash test/phase1/run_phase1_checks.sh`
- **Per wave merge:** `bash test/phase1/run_phase1_checks.sh && ctest --test-dir build --output-on-failure`
- **Phase gate:** Full suite green plus direct inspection that no corrected fixture still mentions obsolete `a5vm.load` / `a5vm.store`

### Wave 0 Gaps
- [ ] `test/phase1/a5vm_copy_gm_to_ubuf_op.mlir` — corrected copy-family contract for A5VM-04
- [ ] `test/phase1/a5vm_vabs_kernel_shape.mlir` — corrected register-vector `Abs` contract for A5VM-02 and A5VM-03
- [ ] `test/phase1/a5vm_copy_ubuf_to_gm_op.mlir` — corrected copy-family contract for A5VM-04
- [ ] `test/phase1/a5vm_backend_switch.mlir` — rewrite to use corrected primitives instead of `a5vm.load` / `a5vm.abs` / `a5vm.store`
- [ ] `test/phase1/a5vm_shared_dialects.mlir` — rewrite to keep shared loops/scalars around corrected primitives
- [ ] `test/phase1/run_phase1_checks.sh` — replace obsolete fixture names and add a grep guard against `a5vm.load` / `a5vm.store`
- [ ] `test/phase2/tload_contract_trace.mlir` — follow-on correction because current Phase 2 contract still assumes `a5vm.load`
- [ ] `test/phase2/tstore_branch_shape.mlir` — follow-on correction because current Phase 2 contract still assumes `a5vm.store`
- [ ] `test/phase2/tabs_precheck.mlir` and `test/phase2/unary_template_shape.mlir` — follow-on correction because current Phase 2 unary surface still assumes obsolete Phase 1 naming

## Sources

### Primary (HIGH confidence)
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` - public PTO API entrypoints for `TLOAD`, `TABS`, and `TSTORE`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp` - A5 GM-to-UB transfer structure and layout/stride/valid-region parameters
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp` - A5 unary `Abs` inner loop shape and builtin call sequence
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp` - A5 UB-to-GM transfer structure and tile-domain branching
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` - CCE builtin family names and vector wrapper signatures
- `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake` - local LLVM version
- `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake` - local MLIR version
- `/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp` - current backend boundary and selector seam
- `/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/A5VMDialect.td` - current obsolete namespace setting to replace
- `/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/A5VMOps.td` - current obsolete op surface to replace

### Secondary (MEDIUM confidence)
- `https://mlir.llvm.org/docs/DefiningDialects/` - standard out-of-tree dialect structure guidance, consistent with the repo layout
- `https://mlir.llvm.org/docs/DialectConversion/` - standard guidance for keeping shared dialects while selectively lowering hardware-facing ops

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - versions and stack choice are verified from the local toolchain, repo build files, and installed CANN headers.
- Architecture: HIGH - the corrected primitive split is directly supported by A5 `TLoad` / `TUnaryOp` / `TStore` source code and by the repo’s existing backend boundary.
- Pitfalls: HIGH - the major failure modes are visible both in the installed A5 sources and in the currently landed wrong namespace/op surface.

**Research date:** 2026-03-19
**Valid until:** 2026-04-18
