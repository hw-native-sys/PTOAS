# Phase 2: PTO Lowering - Research

**Researched:** 2026-03-19
**Domain:** Corrected PTO-to-A5VM lowering for the `Abs` path
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### TLOAD semantic preservation
- The previous idea that `TLOAD` should lower into an A5VM pseudo-op analogous to `a5vm.load` is wrong and must not guide new planning.
- `TLOAD` lowering should mirror the PTO library’s GM-to-UB transfer behavior and builtin family selection, specifically the `copy_gm_to_ub*` style operations rather than a vector-load abstraction.
- `TLOAD` lowering should retain the real loop and transfer structure used by the PTO library instead of flattening the operation into a single fake load.
- `pad_mode`, `pad_value`, `left_padding_num`, `right_padding_num`, `init_out_buffer`, and `init_condition` should still be preserved in the lowering contract even if the current `Abs` path does not exercise them fully.
- Layout rules, valid-row/valid-col information, padding/init behavior, source/domain information, strides, and partition-trace information are all important and should not be casually dropped.

### TSTORE branch structure
- `TSTORE` has the same class of issue as `TLOAD`: it should not lower to a fake single A5VM store when the real PTO library uses copy-family movement between UB and GM.
- The lowering contract and code skeleton should still explicitly show `ACC`, `VEC`, and `MAT` source-tile branches, but the `VEC` branch must model the real PTO-library move/writeback structure rather than a pseudo-store abstraction.
- PTO-side layout rules that influence `TSTORE` selection should be part of the lowering input rather than inferred implicitly later.
- Source tile domain, destination layout/shape/stride, valid row/column, and trace metadata are all equally important selection inputs for `TSTORE` lowering.

### TABS alignment standard
- `lowerTABS` should align as closely as practical to the PTO template and implementation decision structure, not only to the observable `abs` result.
- For the `Abs` path, `TABS` should lower through the actual vector primitive sequence: `a5vm.vld`, `a5vm.vabs`, and `a5vm.vst`.
- `TABS` lowering must reflect both the `__VEC_SCOPE__` hardware-loop structure and the inner software loop structure used by the PTO implementation to iterate UB data by vector granularity.
- PTO-side restrictions such as dtype/domain/valid-shape compatibility should be enforced in lowering pre-checks before building `a5vm`.
- `TABS` should become the standard lowering template for future unary ops such as `TNEG` and `TLOG`, but only if that template is built from the real PTO-library control structure.

### Reusable framework boundary
- Phase 2 should still use three strong PTO entrypoints (`lowerTLOAD`, `lowerTABS`, `lowerTSTORE`) backed by shared helper utilities, but those entrypoints must read like PTO-library control decompositions, not like wrappers around invented `a5vm.load/store` pseudo-ops.
- Important PTO decision logic should remain visible at each PTO entrypoint; repeated mechanics and repeated branch detail can move into shared helpers.
- The code should make PTO template-to-lowering correspondence readable when someone inspects it later, including where hardware-loop and software-loop structure come from.
- For current non-`Abs` branches such as `ACC` / `MAT`, the interface entrypoints should contain explicit TODO branches so future completion points are obvious.

### Replanning Notes
- The previously executed Phase 2 implementation is seriously misaligned with the PTO library because it treated `TLOAD` / `TSTORE` as load/store-like pseudo-ops and failed to model the real `TABS` vector loop structure.
- Downstream planning must treat the A5-side PTO library implementation as the source of truth and regard the current landed Phase 2 code as superseded.
- Replanning does not need to preserve or mirror `a2a3` implementation details; only the A5 PTO path matters for this effort.

### Claude's Discretion
- Exact helper names and file split for shared lowering utilities
- Exact representation of loop-scope structure and trace metadata inside the lowering contract
- Exact placeholder form for non-`Abs` `ACC` / `MAT` branches so long as the branch structure and inputs remain visible

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Broader PTO op lowering beyond the `Abs` path remains outside this phase, even though the framework should be designed to support it later.
- Full implementation of `ACC` and `MAT` `TSTORE` branches is deferred; Phase 2 should make the branches explicit and preserve their inputs, but not necessarily complete them for execution.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PTO-01 | Developer can lower PTO `TLOAD` on the `Abs` path into `a5vm` operations using the real PTO-library GM-to-UB copy structure while preserving PTO-side layout, shape, valid-region, stride, and trace decisions needed for backend code selection. | Plan `lowerTLOAD` around A5 `copy_gm_to_ubuf_align_v2` family selection plus `set_loop*_outtoub` programming, not a fake vector load. |
| PTO-02 | Developer can lower PTO `TABS` on the `Abs` path into `a5vm` operations in a way that matches the PTO library’s real vector pipeline and loop structure, including `vld`, `vabs`, `vst`, and the surrounding hardware/software loop semantics. | Plan `lowerTABS` as a unary template that materializes the `__VEC_SCOPE__` region and inner loops around `vld -> vabs -> vst`. |
| PTO-03 | Developer can lower PTO `TSTORE` on the `Abs` path into `a5vm` operations using the real PTO-library UB-to-GM copy structure while preserving the PTO-side source tile domain and destination layout behavior needed for code selection. | Plan `lowerTSTORE` around A5 `copy_ubuf_to_gm_align_v2` family behavior plus `set_loop*_ubtoout` programming, with explicit `VEC` / `ACC` / `MAT` entry branches. |
| PTO-04 | Developer can add new PTO-to-A5VM lowerings through the same PTO-library-aligned framework without changing the backend architecture established for `Abs`. | Keep three PTO entrypoints and a minimal helper layer for contract extraction, branch selection, loop programming, trace attachment, and common unary checks. |
</phase_requirements>

## Summary

The corrected Phase 2 plan should treat the A5 PTO headers as the only semantic source of truth. The current repo state is still aligned to the wrong model: [`lib/PTO/Transforms/PTOToA5VM.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/lib/PTO/Transforms/PTOToA5VM.cpp) lowers `TLOAD`/`TSTORE` into `a5vm.load`/`a5vm.store`, and `TABS` into a single `a5vm.abs`. That shape is explicitly contradicted by the A5 headers in `TLoad.hpp`, `TStore.hpp`, and `TUnaryOp.hpp`.

For the `Abs` path, truthful planning means separating three concerns. `TLOAD` and `TSTORE` are copy-family lowerings with layout-dependent loop programming and GM/UB stride semantics. `TABS` is a vector-compute lowering with a `__VEC_SCOPE__` region and an inner software loop over vector-width chunks using `vlds`, `vabs`, and `vsts`. The lowering contract must preserve PTO-side layout, valid-region, stride, padding/init, and partition-trace information because the PTO library dispatches on exactly those values.

The main planning consequence is that Wave 0 fixtures must stop checking for pseudo `a5vm.load/store/abs` and instead lock the PTO-library decision structure. If the corrected Phase 1 A5VM surface does not yet expose copy-family and vector-scope primitives, Wave 0 for Phase 2 must explicitly extend that surface before implementing the pass.

**Primary recommendation:** Plan Phase 2 around three visible entrypoints, each mirroring the A5 PTO control structure, with fixtures that lock copy-family selection, loop programming, unary vector-loop shape, and `--pto-backend=a5vm` pass wiring before any lowering code is written.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LLVM | 19.1.7 | Pass manager, diagnostics, `FileCheck`, `ctest` | Verified local toolchain for this workspace. |
| MLIR | 19.1.7 | Dialects, rewrites, structured control-flow lowering | Existing PTO compiler architecture is MLIR-native. |
| Ascend CANN PTO A5 headers | 8.5.0 | Authoritative `TLOAD` / `TABS` / `TSTORE` semantics | These headers define the real branch structure and loop programming. |
| Workspace PTO IR | workspace | Source op surface and verifier behavior | `TLoadOp`, `TAbsOp`, and `TStoreOp` already encode the PTO entry surface. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| Corrected Phase 1 `a5vm` dialect | workspace | Hardware-facing primitive target IR | Use for vector primitives, copy-family ops, loop-scope ops, and metadata carriers. |
| MLIR `PatternRewriter` | 19.1.7 | PTO op replacement | Use inside the `pto-to-a5vm` pass only. |
| Bash + `FileCheck` runner | workspace | Wave 0 structural verification | Use for fast contracts before full sample validation. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| A5 PTO headers as semantic source | Current repo `PTOToA5VM.cpp` | Reuses code, but preserves the wrong semantics. |
| Copy-family and vector-scope A5VM primitives | Pseudo `a5vm.load/store/abs` | Easier short term, but wrong for Phase 2 and blocks truthful backend emission. |
| Explicit lowering contracts per PTO family | Reconstruct meaning later in the emitter | Hides dispatch decisions and forces Phase 3 to reverse-engineer Phase 2 mistakes. |

**Version verification:**
```bash
sed -n '1,20p' /data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake
sed -n '1,20p' /data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake
```

**Verified local versions:**
- LLVM `19.1.7`
- MLIR `19.1.7`
- CANN PTO reference tree `8.5.0`

## Architecture Patterns

### Recommended Project Structure
```text
include/PTO/Transforms/
├── Passes.h                   # pass entrypoint
└── A5VMLowering.h             # corrected public lowering contracts

lib/PTO/Transforms/
├── PTOToA5VM.cpp              # pass wiring and rewrite dispatch
└── PTOToA5VMLowering.cpp      # contract extraction + lowerTLOAD/TABS/TSTORE
```

### Pattern 1: Strong PTO Entrypoints
**What:** Keep `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` as the only family entrypoints.
**When to use:** For all Phase 2 lowering work.
**Example:**
```c++
LogicalResult lowerTLOAD(pto::TLoadOp op, PatternRewriter &rewriter);
LogicalResult lowerTABS(pto::TAbsOp op, PatternRewriter &rewriter);
LogicalResult lowerTSTORE(pto::TStoreOp op, PatternRewriter &rewriter);
```
Source: [`include/PTO/Transforms/A5VMLowering.h`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/Transforms/A5VMLowering.h)

### Pattern 2: Contract Extraction Before A5VM Building
**What:** Extract a small, truthful lowering contract from PTO SSA first, then emit A5VM primitives from that contract.
**When to use:** Always, especially for `TLOAD` and `TSTORE`.
**Example:**
```c++
struct A5VMMoveContract {
  Layout layout;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  int64_t validRows;
  int64_t validCols;
  PartitionTrace trace;
  PadInfo pad;
};
```
Source: PTO op surface in [`include/PTO/IR/PTOOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/PTOOps.td) and A5 template parameters in `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp`

### Pattern 3: Mirror the First-Order PTO Branches
**What:** The lowering boundary should expose the same first branch points the PTO library exposes.
**When to use:** `TLOAD` layout branch, `TSTORE` domain branch, unary 1D-vs-2D branch.
**Example:**
```c++
switch (contract.srcDomain) {
case Vec: return lowerVecStore(contract, rewriter);
case Acc: return emitTodo(op, "TSTORE ACC lowering TODO");
case Mat: return emitTodo(op, "TSTORE MAT lowering TODO");
}
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp`

### Pattern 4: Separate Copy Semantics From Vector Compute Semantics
**What:** `TLOAD`/`TSTORE` use copy-family primitives and loop registers. `TABS` uses vector register load/compute/store inside `__VEC_SCOPE__`.
**When to use:** Always. Never unify them into one pseudo vector-op abstraction.
**Example:**
```c++
// TLOAD/TSTORE: build copy-family op sequence.
// TABS: build vec-scope + inner loop around vld/vabs/vst.
```
Source: `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp`, `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp`, `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp`

### Anti-Patterns to Avoid
- **Pseudo-op mirroring:** Do not plan around `a5vm.load`, `a5vm.store`, or a single `a5vm.abs` as the Phase 2 semantic contract.
- **Emitter-side recovery:** Do not defer copy-family or loop-shape reconstruction to Phase 3.
- **A2/A3 contamination:** Do not cite or mirror `a2a3` behavior. The user explicitly rejected that.
- **Abs-only shortcuts:** Do not hardcode just the sample’s visible effect and drop padding, layout, stride, or trace info.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Semantic truth | A custom PTO model disconnected from CANN | A5 CANN PTO headers + PTO ODS/verifiers | The A5 headers already define the real dispatch structure. |
| Memory movement | A fake generic load/store op | Copy-family A5VM ops that mirror GM↔UB movement | Copy-family selection and loop registers are the semantics. |
| Unary control flow | A single `abs` op with attrs | `vec_scope` + `vld` + `vabs` + `vst` + software loop shape | The PTO library does real vector-looping, not scalarized abstraction. |
| Store branching | A hidden default branch | Explicit `VEC` / `ACC` / `MAT` entry branches | Phase 2 must preserve visible future extension points. |

**Key insight:** In corrected Phase 2, the “backend contract” is not a value-level `abs` result. It is the preserved decision structure that tells later stages which copy family, loop registers, tile-domain branch, and vector-loop pattern the PTO library would have used.

## Common Pitfalls

### Pitfall 1: Treating `TLOAD` as a vector load
**What goes wrong:** Planning keeps only base pointer plus valid region and loses layout/stride/padding decisions.
**Why it happens:** The current repo already implements it that way.
**How to avoid:** Lock Wave 0 fixtures around `copy_gm_to_ubuf_align_v2`-style structure and `set_loop*_outtoub` metadata.
**Warning signs:** Fixture still checks `a5vm.load`.

### Pitfall 2: Treating `TSTORE` as the inverse of `TLOAD`
**What goes wrong:** The plan forgets that `TSTORE` branches first on tile domain and has different UB-to-GM loop programming.
**Why it happens:** `Abs` only exercises the `VEC` branch.
**How to avoid:** Require visible `VEC` / `ACC` / `MAT` branches in `lowerTSTORE`.
**Warning signs:** One generic helper with no domain switch.

### Pitfall 3: Flattening `TABS` into one op
**What goes wrong:** The plan drops `__VEC_SCOPE__`, 1D vs 2D shape selection, and the inner software loop.
**Why it happens:** The visible math is only absolute value.
**How to avoid:** Lock a fixture that checks for vec-scope region plus ordered `vld`, `vabs`, `vst`.
**Warning signs:** Fixture only checks `a5vm.vabs`.

### Pitfall 4: Losing valid-region equality semantics
**What goes wrong:** The plan preserves only one tile’s valid shape or invents `validRows == validCols` as a backend rule.
**Why it happens:** The current code uses an incorrect precheck.
**How to avoid:** Follow `TUNARY_IMPL`: require src and dst valid row/col equality, not square shape.
**Warning signs:** Precheck still says “matching valid rows and valid cols”.

### Pitfall 5: Ignoring padding/init operands because `Abs` does not use them
**What goes wrong:** Future `TLOAD` variants require a redesign.
**Why it happens:** The sample path is simple.
**How to avoid:** Preserve pad/init fields in the `TLOAD` contract now, even if the first lowering uses only `PadNull`.
**Warning signs:** Contract no longer contains `pad_mode`, `pad_value`, `left_padding_num`, `right_padding_num`, `init_out_buffer`, `init_condition`.

### Pitfall 6: Planning Phase 2 without checking the A5VM primitive surface
**What goes wrong:** The lowering plan assumes `vld`, `vst`, vec-scope, and copy-family A5VM ops already exist when the current repo still exposes stale pseudo ops in [`include/PTO/IR/A5VMOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/A5VMOps.td).
**Why it happens:** Phase 1 was corrected in planning, but the workspace still contains stale implementation artifacts.
**How to avoid:** Make Wave 0 explicitly verify the corrected A5VM primitive inventory or extend it before Phase 2 lowering work starts.
**Warning signs:** Planner writes tasks assuming current `A5VMOps.td` is already correct.

## Code Examples

Verified patterns from authoritative local sources:

### TLOAD Vec ND/DN Shape
```c++
// Source: /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp
set_loop2_stride_outtoub(...dst_stride..., ...src_stride...);
set_loop1_stride_outtoub(...dst_stride..., ...src_stride...);
set_loop_size_outtoub(loop2 << 21 | loop1);
for (uint32_t i = 0; i < gShape0; ++i)
  copy_gm_to_ubuf_align_v2(... nBurst, lenBurst, ..., gmStride, ubStride);
```

### TABS Unary Template Shape
```c++
// Source: /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp
__VEC_SCOPE__ {
  for (...) {
    vlds(srcReg, src, offset, ...);
    vabs(dstReg, srcReg, pReg, MODE_ZEROING);
    vsts(dstReg, dst, offset, ..., pReg);
  }
}
```

### TSTORE Vec ND/DN Shape
```c++
// Source: /usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp
set_loop_size_ubtoout(...);
set_loop1_stride_ubtoout(...src_stride..., ...dst_stride...);
set_loop2_stride_ubtoout(...src_stride..., ...dst_stride...);
for (uint32_t k = 0; k < gShape0; ++k)
  copy_ubuf_to_gm_align_v2(... nBurst, lenBurst, ..., burstDstStride, burstSrcStride);
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Pseudo `a5vm.load` / `a5vm.abs` / `a5vm.store` | Copy-family plus vector primitive lowering that mirrors A5 PTO headers | Phase correction on 2026-03-19 | Phase 2 fixtures and helper contracts must be rewritten. |
| `TABS` square-shape precheck | `TABS` src/dst valid-shape equality plus dtype/domain/layout checks from A5 unary template | Phase correction on 2026-03-19 | Existing precheck fixture is semantically wrong. |
| A5VM backend as a semantic abstraction | A5VM backend as a hardware-facing primitive layer | Phase 1/2 correction cycle | Phase 2 depends on truthful primitives, not convenience wrappers. |

**Deprecated/outdated:**
- Current [`test/phase2/tload_contract_trace.mlir`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/test/phase2/tload_contract_trace.mlir): outdated because it locks `a5vm.load`.
- Current [`test/phase2/tstore_branch_shape.mlir`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/test/phase2/tstore_branch_shape.mlir): outdated because it locks `a5vm.store`.
- Current [`test/phase2/unary_template_shape.mlir`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/test/phase2/unary_template_shape.mlir): outdated because it checks only one `a5vm.abs`.
- Current [`test/phase2/tabs_precheck.mlir`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/test/phase2/tabs_precheck.mlir): outdated because it encodes an incorrect square-shape rule.

## Wave 0 Fixtures

### Corrected fixtures to create first

| Fixture | What it must lock | Why |
|---------|-------------------|-----|
| `test/phase2/tload_copy_family_shape.mlir` | `lowerTLOAD` emits copy-family A5VM structure for the `Abs` path, preserving layout, shape, strides, valid row/col, pad/init fields, partition trace, and out-to-UB loop programming | Prevents regression back to fake load ops. |
| `test/phase2/tabs_abs_loop_shape.mlir` | `lowerTABS` emits `vec_scope` plus inner loop with ordered `a5vm.vld`, `a5vm.vabs`, `a5vm.vst` | Locks the real unary loop shape. |
| `test/phase2/tstore_copy_family_shape.mlir` | `lowerTSTORE` emits UB-to-GM copy-family structure with destination layout/shape/strides, valid row/col, trace, and UB-to-out loop programming | Prevents regression back to fake store ops. |
| `test/phase2/tstore_domain_todos.mlir` | `ACC` and `MAT` branches remain explicit and produce dedicated TODO diagnostics | Preserves truthful future extension points. |
| `test/phase2/pto_backend_a5vm_wiring.mlir` | `--pto-backend=a5vm` runs PTO-to-A5VM lowering before final backend emission and does not route through EmitC lowering | Locks the backend seam. |

### Fixture details the planner should require

- `TLOAD` fixture must check branch shape, not just attribute presence.
- `TLOAD` fixture must include `pad_mode`, `pad_value`, `left_padding_num`, `right_padding_num`, `init_out_buffer`, and `init_condition` in at least one subcase, even if only one is executable in v1.
- `TABS` fixture must check `vec_scope` and loop-carried indexing metadata or op nesting that proves the software loop exists.
- `TABS` precheck fixture must check:
  `tile domain == vec`, row-major compatibility per unary template, supported dtype set, and src/dst valid-shape equality.
- `TSTORE` fixture must distinguish `VEC` from `ACC` and `MAT`.
- All three fixtures must preserve partition trace, layout, valid-region, and stride information as explicit A5VM attrs or operands.

## Minimal Truthful Helper Layer

### Required contract structs

| Helper | Must contain | Why |
|--------|--------------|-----|
| `A5VMMoveContract` | layout, shape, strides, valid row/col, partition trace, pad/init fields, tile layout/domain | Shared truth for `TLOAD`/`TSTORE`. |
| `A5VMUnaryContract` | src/dst valid row/col, tile layout/domain, element type, loop shape kind (`1D_no_post_update`, `1D_post_update`, `2D`) | Shared truth for `TABS` and future unary ops. |
| `A5VMLoopProgram` | loop sizes, loop1/loop2 stride configs, burst counts, burst lengths, source/destination stride values | Keeps register programming explicit and reusable. |

### Required helper functions

- `extractPartitionTrace(Value)`:
  Must preserve offsets and sizes from `pto.partition_view`.
- `extractMoveContractFromTLoad(TLoadOp)`:
  Must keep all pad/init operands plus source layout/shape/stride.
- `extractMoveContractFromTStore(TStoreOp)`:
  Must keep destination layout/shape/stride and source tile domain.
- `buildOutToUbLoopProgram(...)`:
  For `TLOAD` `set_loop*_outtoub`.
- `buildUbToOutLoopProgram(...)`:
  For `TSTORE` `set_loop*_ubtoout`.
- `buildUnaryLoopContract(...)`:
  Must decide 1D vs 2D shape according to the A5 unary template.
- `emitUnaryPrechecks(...)`:
  Must validate A5 restrictions before creating A5VM ops.

### What the helper layer must not contain

- No generic `lowerLoadLike` / `lowerStoreLike` abstraction that erases copy direction.
- No emitter-specific intrinsic names.
- No a2a3-derived branch selection logic.

## Exact Pass Wiring

### `--pto-backend=a5vm` path

Use the existing seam in [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp):

1. Keep all shared pre-backend passes unchanged up to the backend split.
2. On `PTOBackend::A5VM`, run `createLowerPTOToA5VMPass()` after shared passes and before final text emission.
3. Do not run `createEmitPTOManualPass(...)` or EmitC expression formation on the A5VM branch.
4. Keep the raw-A5VM parse shortcut intact for already-lowered A5VM IR.
5. Run `CSE` after PTO-to-A5VM only if it preserves the fixture-visible loop/copy structure.

### Pass contract

- The pass rewrites only PTO hardware-facing ops for this phase: `TLoadOp`, `TAbsOp`, `TStoreOp`.
- Shared dialect ops remain in `func`, `scf`, `arith`, `memref`.
- The pass must preserve pipe/domain semantics visible from PTO:
  `TLOAD -> PIPE_MTE2`, `TABS -> PIPE_V`, `TSTORE(vec/mat) -> PIPE_MTE3`, `TSTORE(acc) -> PIPE_FIX`.
- The planner should assume the current stale dependent dialect names in [`include/PTO/Transforms/Passes.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/Transforms/Passes.td) need to match corrected Phase 1 `mlir::a5vm` naming before implementation.

## Open Questions

1. **Which corrected A5VM primitives already exist after Phase 1 correction?**
   - What we know: current workspace still exposes stale pseudo ops in [`include/PTO/IR/A5VMOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/A5VMOps.td).
   - What's unclear: whether corrected `vld`, `vabs`, `vst`, vec-scope, and copy-family ops already landed elsewhere or still need Phase 2-adjacent work.
   - Recommendation: make this the first Wave 0 check and extend A5VM if missing.

2. **How should loop programming be represented in A5VM IR?**
   - What we know: A5 PTO uses explicit loop-size and loop-stride register programming for copy families.
   - What's unclear: whether A5VM models those as dedicated ops, structured attrs, or helper ops plus attrs.
   - Recommendation: use dedicated ops or explicit structured attrs that survive to Phase 3 emission; do not hide them in comments.

3. **How much of pad/init behavior should be executable in v1?**
   - What we know: the user requires those inputs preserved in the lowering contract even if `Abs` does not use them fully.
   - What's unclear: whether v1 must only carry them or partially lower them.
   - Recommendation: Wave 0 should at minimum lock preservation, even if full execution is deferred.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Bash runner + `FileCheck` + `ctest` (LLVM 19.1.7 toolchain) |
| Config file | none |
| Quick run command | `./test/phase2/run_phase2_checks.sh` |
| Full suite command | `./test/phase2/run_phase2_checks.sh` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PTO-01 | `TLOAD` preserves copy-family structure, loop programming, layout/stride/trace/pad metadata | structural | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tload_copy_family_shape.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tload_copy_family_shape.mlir` | ❌ Wave 0 |
| PTO-02 | `TABS` emits vec-scope plus `vld -> vabs -> vst` loop shape and rejects unsupported cases | structural | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tabs_abs_loop_shape.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tabs_abs_loop_shape.mlir` | ❌ Wave 0 |
| PTO-03 | `TSTORE` preserves UB-to-GM copy-family structure and explicit `ACC` / `MAT` TODO branches | structural | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/tstore_copy_family_shape.mlir -o /dev/null 2>&1 | FileCheck test/phase2/tstore_copy_family_shape.mlir` | ❌ Wave 0 |
| PTO-04 | Backend wiring keeps the same architecture and only activates PTO-to-A5VM on `--pto-backend=a5vm` | structural | `./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir test/phase2/pto_backend_a5vm_wiring.mlir -o /dev/null 2>&1 | FileCheck test/phase2/pto_backend_a5vm_wiring.mlir` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** the requirement-specific `ptoas | FileCheck` command for the touched contract
- **Per wave merge:** `./test/phase2/run_phase2_checks.sh`
- **Phase gate:** full Phase 2 runner green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `test/phase2/tload_copy_family_shape.mlir` — corrected TLOAD contract
- [ ] `test/phase2/tabs_abs_loop_shape.mlir` — corrected TABS loop-shape contract
- [ ] `test/phase2/tstore_copy_family_shape.mlir` — corrected TSTORE contract
- [ ] `test/phase2/tstore_domain_todos.mlir` — explicit ACC/MAT TODO branch checks
- [ ] `test/phase2/pto_backend_a5vm_wiring.mlir` — backend branch contract
- [ ] Update `test/phase2/run_phase2_checks.sh` to run the corrected fixture set

## Sources

### Primary (HIGH confidence)
- [`include/PTO/IR/PTOOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/PTOOps.td) - checked `TLoadOp`, `TStoreOp`, `TAbsOp` operands, pipes, and assembly surface
- [`tools/ptoas/ptoas.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/tools/ptoas/ptoas.cpp) - checked exact `--pto-backend=a5vm` wiring point
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` - checked public `TLOAD`, `TABS`, `TSTORE` entrypoints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp` - checked GM-to-UB copy-family structure and loop programming
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp` - checked UB-to-GM copy-family structure and `VEC` / `ACC` branching
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp` - checked `__VEC_SCOPE__`, unary loop shape, and `TABS_IMPL`

### Secondary (MEDIUM confidence)
- [`test/compile_cpp/abs_vec_core.cpp`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/test/compile_cpp/abs_vec_core.cpp) - confirms sample-level `TLOAD -> TABS -> TSTORE` shape used for `Abs`
- [`include/PTO/IR/A5VMOps.td`](/data/mouliangyu/projects/github.com/zhangstevenunity/PTOAS/include/PTO/IR/A5VMOps.td) - used only to identify current stale pseudo-op surface that Phase 2 must not follow

### Tertiary (LOW confidence)
- None

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - verified from local workspace toolchain and installed CANN headers
- Architecture: HIGH - derived directly from A5 PTO implementation files and current repo backend seam
- Pitfalls: HIGH - based on direct contradictions between current repo state and A5 PTO headers

**Research date:** 2026-03-19
**Valid until:** 2026-04-18
