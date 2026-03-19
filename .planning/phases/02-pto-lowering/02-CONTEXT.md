# Phase 2: PTO Lowering - Context

**Gathered:** 2026-03-19
**Status:** Ready for replanning

<domain>
## Phase Boundary

Implement PTO-to-A5VM lowering that matches the real PTO library structure for the `Abs` path rather than a pseudo-op approximation. This phase is about re-expressing `TLOAD`, `TABS`, and `TSTORE` in terms that stay faithful to PTO library behavior, including the correct builtin families, hardware/software loop structure, and UB/GM movement semantics.

</domain>

<decisions>
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
- For the `Abs` path, `TABS` should lower through the actual vector primitive sequence already landed in Phase 1: `a5vm.vlds`, `a5vm.vabs`, and `a5vm.vsts`.
- `TABS` lowering must reflect both the `__VEC_SCOPE__` loop-hint structure and the inner software loop structure used by the PTO implementation to iterate UB data by vector granularity.
- `__VEC_SCOPE__` is now confirmed to correspond to `__attribute__((cce_aiv_loop_hint))` and downstream `llvm.loop.aivector_scope` loop metadata, so it must not be represented only as an ordinary software `scf.for`.
- `__VEC_SCOPE__` is not just an attribute attached abstractly somewhere nearby; it is a dummy loop form whose induction variable is annotated with `__attribute__((cce_aiv_loop_hint))`, currently materialized in CCE frontend terms as `for (__attribute__((cce_aiv_loop_hint)) int dummy = 0; dummy <= 0; dummy += 1)`.
- The dummy loop's lower bound, upper bound, and step are not the semantic payload. The payload is that this specific loop owns the AIV scope and is the loop that eventually receives `llvm.loop.aivector_scope` metadata.
- The lowering contract should preserve which loop carries AIV vector-scope semantics separately from any plain software chunking loop that may appear inside or around it.
- PTO-side restrictions such as dtype/domain/valid-shape compatibility should be enforced in lowering pre-checks before building `a5vm`.
- `TABS` should become the standard lowering template for future unary ops such as `TNEG` and `TLOG`, but only if that template is built from the real PTO-library control structure.

### Reusable framework boundary
- Phase 2 should still use three strong PTO entrypoints (`lowerTLOAD`, `lowerTABS`, `lowerTSTORE`) backed by shared helper utilities, but those entrypoints must read like PTO-library control decompositions, not like wrappers around invented `a5vm.load/store` pseudo-ops.
- Important PTO decision logic should remain visible at each PTO entrypoint; repeated mechanics and repeated branch detail can move into shared helpers.
- The code should make PTO template-to-lowering correspondence readable when someone inspects it later, including where hardware-loop and software-loop structure come from.
- For current non-`Abs` branches such as `ACC` / `MAT`, the interface entrypoints should contain explicit TODO branches so future completion points are obvious.

### Replanning Notes
- The previously executed Phase 2 implementation is seriously misaligned with the PTO library because it treated `TLOAD` / `TSTORE` as load/store-like pseudo-ops and failed to model the real `TABS` vector loop structure.
- The previous repair still under-modeled `__VEC_SCOPE__` by degrading it to plain structured loops; that interpretation is now superseded by the confirmed `cce_aiv_loop_hint` / `llvm.loop.aivector_scope` meaning.
- Downstream planning must treat the A5-side PTO library implementation as the source of truth and regard the current landed Phase 2 code as superseded.
- Replanning does not need to preserve or mirror `a2a3` implementation details; only the A5 PTO path matters for this effort.

### Claude's Discretion
- Exact helper names and file split for shared lowering utilities
- Exact representation of loop-scope structure and trace metadata inside the lowering contract
- Exact placeholder form for non-`Abs` `ACC` / `MAT` branches so long as the branch structure and inputs remain visible

</decisions>

<specifics>
## Specific Ideas

- The corrected design principle is: use PTO library implementation structure as the backbone so later extension does not require tearing the framework apart.
- `TLOAD` and `TSTORE` should expose GM/UB movement semantics, not collapse into fake load/store ops.
- `TABS` should explicitly expose the `vlds -> vabs -> vsts` vector pipeline, the AIV-scoped loop corresponding to `__VEC_SCOPE__`, and the surrounding software loop structure.
- When modeling `__VEC_SCOPE__`, the lowering should preserve a loop-shaped carrier for the AIV scope instead of collapsing it into a generic region marker, because downstream semantics attach to the loop itself.
- The user still wants `lowerTLOAD`, `lowerTABS`, and `lowerTSTORE` to feel like clear PTO interface entrypoints, not like thin aliases over one giant opaque lowering engine.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project and prior phase context
- `.planning/PROJECT.md` — project goals, compatibility constraints, and v1 scope
- `.planning/REQUIREMENTS.md` — corrected phase requirements and acceptance boundaries
- `.planning/ROADMAP.md` — corrected Phase 2 boundary and success criteria
- `.planning/STATE.md` — current milestone position
- `.planning/phases/01-a5vm-foundation/01-CONTEXT.md` — corrected Phase 1 backend and A5VM primitive decisions that Phase 2 must build on

### PTO IR definitions
- `include/PTO/IR/PTOOps.td` — `TLoadOp`, `TStoreOp`, and `TAbsOp` definitions, arguments, assembly, and pipe behavior

### Current lowering/backend reference
- `lib/PTO/Transforms/PTOToEmitC.cpp` — current PTO backend lowering sites for `TLOAD`, `TSTORE`, and `TABS`, useful as a reference boundary that the corrected Phase 2 is replacing
- `include/PTO/Transforms/Passes.h` — pass declaration surface for new lowering passes
- `tools/ptoas/ptoas.cpp` — current pipeline structure and backend switch location from Phase 1

### PTO library behavior references
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp` — public PTO instruction template entrypoints
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TLoad.hpp` — A5-side `TLOAD_IMPL` branching and GM-to-UB movement structure
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TStore.hpp` — A5-side `TSTORE_IMPL` branch structure and UB-to-GM movement structure
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a5/TUnaryOp.hpp` — A5-side `TABS_IMPL` behavior, vector primitive sequence, and unary-op restriction pattern
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` — builtin wrapper families that the lowering must eventually feed through `a5vm`
- `.planning/todos/pending/2026-03-19-preserve-vec-scope-as-aiv-loop.md` — confirmed interpretation of `__VEC_SCOPE__` as AIV loop metadata and resulting replanning target

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `include/PTO/IR/PTOOps.td`: already records the PTO-side operation surface and is the source of truth for which operands and attributes exist at the IR level.
- `lib/PTO/Transforms/PTOToEmitC.cpp`: contains the current lowering touchpoints for `TLoadOp`, `TStoreOp`, and `TAbsOp`, useful for identifying where the current backend starts flattening PTO semantics too early.
- `tools/ptoas/ptoas.cpp`: already has the backend-switch boundary introduced in Phase 1, which corrected Phase 2 lowering will plug into.

### Established Patterns
- PTO op semantics are centralized in ODS plus C++ verifiers/implementations, so lowering decisions should track those sources rather than inventing a disconnected contract.
- The repo prefers clear named pass entrypoints and adjacent helper code over scattering compiler logic across unrelated directories.
- Compiler diagnostics are expected to be technical and explicit, which fits the need for strong lowering pre-checks and unsupported-branch messaging.

### Integration Points
- Phase 2 lowering should connect the PTO op layer to the corrected `a5vm` primitive dialect established in Phase 1.
- The main implementation surface is still a lowering pass and helpers under `lib/PTO/Transforms/`, with declarations in `include/PTO/Transforms/Passes.h`.
- Future unary-op and load/store extension should reuse the corrected Phase 2 entrypoint/helper structure instead of bypassing it with ad hoc lowering code.

</code_context>

<deferred>
## Deferred Ideas

- Broader PTO op lowering beyond the `Abs` path remains outside this phase, even though the framework should be designed to support it later.
- Full implementation of `ACC` and `MAT` `TSTORE` branches is deferred; Phase 2 should make the branches explicit and preserve their inputs, but not necessarily complete them for execution.

</deferred>

---
*Phase: 02-pto-lowering*
*Context gathered: 2026-03-19*
