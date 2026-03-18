# PTOAS A5VM Backend

## What This Is

PTOAS is a dialect for tile-level tensor programming that currently relies on the PTO library in `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto` as a backend. This project initializes the work needed to replace the current `emitc` generation path with a PTOAS-native backend built around an `a5vm` dialect, so PTO library semantics can be represented directly in PTOAS and lowered into textual LLVM HIVM intrinsic IR without losing template-instantiation information needed for optimization.

The initial target is a minimal but extensible backend framework that preserves the existing pass pipeline, matches current PTO interface and template behavior, and compiles the `Abs` sample through the new path.

## Core Value

Preserve PTO library semantics and template-driven behavior inside PTOAS so backend lowering retains enough information to enable optimization instead of losing it during library instantiation.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Define an `a5vm` dialect for the CCE-style hardware operations needed by the `Abs` sample, starting with legal fixed-width 256-byte vector types and the minimum ops required on that path.
- [ ] Replace the current `emitc` backend slot with a PTOAS-native lowering path that keeps the existing pass pipeline intact.
- [ ] Add a PTO interface lowering framework that aligns with existing PTO parameter semantics and template behavior, starting with the minimum interfaces needed by `Abs` such as `TLOAD`, `TABS`, and `TSTORE`.
- [ ] Add an `a5vm` lowering framework that emits textual LLVM HIVM intrinsic IR with names and type variants selected from operation/builtin shape.
- [ ] Compile `./test/samples/runop.sh -t Abs` through the new backend path and use that sample as the initial acceptance case.
- [ ] Produce a concrete list of LLVM HIVM intrinsics required by the implemented `Abs` path so the final intrinsic mapping can be confirmed externally.

### Out of Scope

- Broad PTO library reimplementation beyond the minimum interface set required to make the `Abs` sample compile — v1 is intentionally constrained to the smallest viable slice.
- Validation of the emitted LLVM HIVM IR on downstream machines or compilers — current scope ends at producing legal and reasonable IR for external verification.
- Reworking unrelated pass pipeline structure or changing established PTO frontend semantics — the new backend must fit into the existing pipeline and preserve behavior.

## Context

PTOAS describes tile-level tensor programs and currently uses the PTO library as a backend. That backend loses information during template instantiation, which blocks optimizations that depend on preserving higher-level intent. The replacement path will keep PTO-like semantics inside PTOAS by lowering PTO interfaces into a new `a5vm` dialect and then lowering `a5vm` into textual LLVM HIVM intrinsics.

The Ascend toolchain under `/usr/local/Ascend/cann-8.5.0/` contains CCE intrinsic wrappers and the underlying builtin-facing interfaces. The file `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` is a key reference for matching builtin wrapper behavior. For `v1`, the implementation only needs to satisfy operations exercised by the `Abs` sample. Ordinary control flow and scalar arithmetic should continue using shared dialects such as `scf` and `arith`.

The initial concrete example is `TABS`, which uses CCE builtins such as `vld` and `vabs`. `a5vm` vector types must always be fixed at 256 bytes total width. For example, `f32` vectors must be `v64f32`. The final set of LLVM HIVM intrinsic spellings is not fully known yet; the implementation should first establish the lowering framework and then extract the required intrinsic list from the builtins and builtin parameters exercised by the sample.

## Constraints

- **Behavioral compatibility**: PTO interface parameters and template behavior must align with the existing PTO implementation — divergence would invalidate the backend replacement.
- **Pipeline compatibility**: The current pass pipeline must remain intact — only the `emitc` backend position should be replaced.
- **Backend scope**: v1 only needs to cover the minimum operations required by `Abs` — this is a framework-first slice, not a full PTO backend migration.
- **Vector typing**: `a5vm` vector types must be fixed-width 256-byte vectors — element counts must always satisfy that byte-width constraint.
- **Verification environment**: Emitted LLVM HIVM intrinsic IR cannot be fully validated locally — output must still be structurally legal and reasonable for external verification.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Introduce `a5vm` as a dedicated dialect for hardware-facing vector ops | Separates PTO semantic lowering from hardware/intrinsic codegen concerns and creates a reusable codegen boundary | — Pending |
| Replace `emitc` generation at its current pipeline position instead of redesigning the pipeline | Minimizes integration risk and preserves existing frontend and pass behavior | — Pending |
| Scope v1 to the `Abs` sample and only the PTO/library interfaces it actually requires | Keeps the first slice small while still proving the full backend architecture | — Pending |
| Lower `a5vm` to textual LLVM HIVM intrinsic IR before final intrinsic validation | The target intrinsic set is not fully available locally, but the textual path is enough to prove framework correctness and request missing mappings | — Pending |

---
*Last updated: 2026-03-18 after initialization*
