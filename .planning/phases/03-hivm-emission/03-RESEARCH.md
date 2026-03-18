# Phase 3: HIVM Emission - Research

**Researched:** 2026-03-19
**Domain:** A5VM-to-textual-LLVM HIVM emission in PTOAS
**Confidence:** HIGH

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
## Implementation Decisions

### LLVM-like output skeleton
- The default output should aim for a complete LLVM `.ll`-style module, not only a function-body fragment.
- The textual emitter should use LLVM-style types directly in the emitted text, such as `<64 x float>`, rather than carrying A5VM type syntax into the final artifact.
- Control flow and ordinary scalar/vector arithmetic that appear on the implemented path should also be emitted in LLVM-style textual form rather than left in mixed-dialect form.
- When there is a tradeoff between strict local conservatism and closeness to the final downstream-consumed HIVM text shape, Phase 3 should favor the final consumption-oriented form.

### Unresolved intrinsic handling
- When an intrinsic is not yet user-confirmed but the backend still needs to preserve the emitted structure, the main output should use placeholder intrinsic names rather than dropping the call site.
- When only the operation family is known, the emitter should still produce a structured placeholder call whose argument and result structure preserve as much real information as possible.
- Unresolved or provisional emission cases should produce a separate unresolved report artifact in addition to what appears in the main textual output.
- The main emitted text should still read as close as possible to real HIVM/LLVM IR even when placeholders are present; the report artifact is the primary place for explaining missing mappings.

### Intrinsic naming boundary
- Intrinsic-name synthesis should be driven by op kind, element type, vector shape, and variant-selection fields such as load/store or `sx1`-style suffix decisions.
- The information needed for name synthesis may be split between explicit `a5vm` op attributes and emitter-side derivation from operand/type shape; the design does not require every decision input to be stored redundantly on the op itself.
- When name synthesis fails, diagnostics and unresolved reporting should include the operation, the participating naming fields, and which fields are missing or unsupported.
- Naming logic should be organized as shared family-specific builders, with at least separate `load/store` and unary-style paths reusing common string-fragment assembly helpers.

### Developer observability
- By default, normal tool output should contain only the final HIVM textual artifact; debugging and reporting features must be explicitly enabled by developer-facing flags.
- The main HIVM text should continue to use the normal output destination, while debug/trace information should go to `stderr` by default so it does not pollute the emitted IR.
- An explicit `--dump-a5vm-ir` style flag should be supported as part of the Phase 3 developer workflow.
- The unresolved report should include, at minimum, the source op, candidate or placeholder intrinsic name, the naming fields that participated, the fields that were missing, and a useful source location.

### Claude's Discretion
- Exact textual module boilerplate details, so long as the default output remains LLVM `.ll`-like and suitable for downstream verification.
- Exact placeholder intrinsic spelling convention, so long as it is machine-distinguishable and structurally preserves the original call shape.
- Exact split of responsibility between `a5vm` attributes and emitter-side derivation for each individual op family.
- Exact CLI spelling for the new emitter/debug/report flags beyond the requirement that they are explicit and developer-facing.

### Deferred Ideas (OUT OF SCOPE)
## Deferred Ideas

- Exact user-confirmed HIVM intrinsic names for the implemented subset are deferred to the later inventory-validation phase once emitted sites can be enumerated concretely.
- Broader debug-flag surface beyond the minimum explicit A5VM dump and unresolved reporting controls can be expanded later if the first implementation shows gaps.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| HIVM-01 | Developer can lower `a5vm` operations into textual LLVM HIVM intrinsic IR instead of `emitc` C++. | Replace the current `emitc::translateToCpp(...)` branch in `tools/ptoas/ptoas.cpp` with a dedicated A5VM text emission branch that writes `.ll`-style text directly to `-o`. |
| HIVM-02 | Developer can derive intrinsic spellings from operation family, vector type, and variant metadata rather than hardcoding a single `Abs` string path. | Centralize name synthesis in reusable family builders fed by `a5vm` attrs plus type-derived fragments; keep per-family decisions out of the printer. |
| HIVM-03 | Developer can emit structurally legal and reasonable textual IR for the `Abs` path even though final downstream validation happens on another machine. | Emit a complete module, deduplicate declarations, keep placeholder calls parseable, and validate locally with `FileCheck` plus `llvm-as` from the local LLVM 19.1.7 install. |
</phase_requirements>

## Summary

Phase 3 should be planned as a direct replacement of the final EmitC translation slot, not as another lowering pass. The exact seam is already visible in `tools/ptoas/ptoas.cpp`: after the pass manager runs, the current path performs EmitC-specific cleanup (`dropEmptyEmitCExpressions`, `materializeControlFlowOperands`), calls `emitc::translateToCpp(...)`, then rewrites EmitC marker syntax in the resulting C++ string. The A5VM/HIVM branch should fork before those EmitC-only cleanups and write final `.ll`-style text directly. Reusing EmitC post-processing in this phase is the wrong abstraction and will cause churn.

The implementation should be split into four separately plannable slices: `1)` intrinsic naming, `2)` LLVM-like text printing, `3)` unresolved diagnostics and sidecar reporting, and `4)` `ptoas` tool wiring plus flags. That separation matches the user’s locked decisions and prevents one giant emitter file from mixing policy, syntax, and CLI behavior. The printer should consume already-decided naming outputs and typed op information; it should not infer backend semantics that Phase 2 was supposed to preserve on `a5vm` ops.

One critical planning point is placeholder legality. Real LLVM intrinsics live under the reserved `llvm.` namespace; unresolved placeholders should therefore stay structurally LLVM-like but use a machine-distinguishable non-reserved callee prefix such as `@__ptoas_hivm_unresolved.*`, with the intended candidate name recorded in the unresolved report. That keeps the emitted `.ll` parseable by `llvm-as` while still preserving call shape for later intrinsic inventory work.

**Primary recommendation:** Plan Phase 3 around a dedicated A5VM text-emission library with separate naming/reporting helpers and a `ptoas` branch that bypasses all EmitC translation and post-processing.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| LLVM LangRef / assembler | 19.1.7 local toolchain | Defines legal `.ll` surface and enables local parse validation | The repo already builds against LLVM 19.1.7 and the local install includes `llvm-as`. |
| MLIR core IR APIs | 19.1.7 local toolchain | Walk `ModuleOp`, `func.func`, blocks, values, locations, and types | The emitter is a printer over the post-pass MLIR module; no extra framework is needed. |
| PTOTransforms workspace library | workspace | Natural home for A5VM text emission helpers | Existing backend code and pass declarations already live under `include/PTO/Transforms` and `lib/PTO/Transforms`. |
| Ascend CANN vector intrinsic header | 8.5.0 local install | Evidence for naming families like `vld`, `vlds`, `vst`, `vsts`, `vabs` | This is the best local primary source for family and suffix dimensions even when final HIVM spellings remain deferred. |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| FileCheck | LLVM 19.1.7 toolchain | Lock exact text fragments, declarations, placeholders, and diagnostics | Use for all Phase 3 source fixtures. |
| CTest | CMake workspace | Existing top-level test entrypoint | Use as the full-suite command; Phase 3 should add a direct runner script in addition to CTest. |
| `llvm::raw_ostream` / `raw_string_ostream` | LLVM 19.1.7 | Deterministic text emission and sidecar serialization | Use in the printer and unresolved report writer. |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Dedicated text emitter over MLIR | Convert to LLVM dialect first | Too much scope for this phase; the requirement is to replace the EmitC slot, not introduce full LLVM IR lowering. |
| Shared family-specific name builders | Hardcode names in per-op printer cases | Faster for `Abs` only, but directly violates HIVM-02 and guarantees rewrite churn. |
| Parseable unresolved extern placeholders | Fake `@llvm.hivm.*` names for unresolved cases | Looks closer to final form, but risks colliding with LLVM intrinsic namespace and weakens local parse validation. |

**Version verification:**
- LLVM `19.1.7` from `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/llvm/LLVMConfigVersion.cmake`
- MLIR `19.1.7` from `/data/mouliangyu/projects/github.com/llvm/llvm-project/install/lib/cmake/mlir/MLIRConfigVersion.cmake`
- PTOAS project version `0.7` from `build/CMakeCache.txt`
- CANN reference tree `8.5.0` from `/usr/local/Ascend/cann-8.5.0/...`

## Architecture Patterns

### Recommended Project Structure
```text
include/PTO/Transforms/
├── A5VMTextEmitter.h          # public entrypoint + emission options
├── A5VMIntrinsicNaming.h      # naming inputs/results, family builders
└── A5VMUnresolvedReport.h     # report record schema / serializer

lib/PTO/Transforms/
├── A5VMTextEmitter.cpp        # module/function/basic-block printer
├── A5VMIntrinsicNaming.cpp    # load/store/unary name synthesis
└── A5VMUnresolvedReport.cpp   # sidecar writer and diag formatting
```

### Pattern 1: Replace the Exact EmitC Translation Slot
**What:** Branch in `tools/ptoas/ptoas.cpp` at the current final emission point, not earlier in the pipeline.
**When to use:** Immediately; this is the exact integration boundary.
**Example:**
```c++
if (backend == PTOBackend::EmitC) {
  dropEmptyEmitCExpressions(module.get());
  materializeControlFlowOperands(module.get());
  return emitc::translateToCpp(...);
}

if (backend == PTOBackend::A5VM) {
  return translateA5VMModuleToText(*module, outputFile.os(), options, llvm::errs());
}
```
**Why:** The current cleanup and marker-rewrite helpers are EmitC-specific. Reusing them in the A5VM branch would reintroduce the dependency this phase is supposed to remove.

### Pattern 2: Name First, Print Second
**What:** Split intrinsic selection from textual printing.
**When to use:** For every `a5vm` op family.
**Example:**
```c++
struct IntrinsicSelection {
  std::string callee;
  bool resolved;
  llvm::SmallVector<std::string> usedFields;
  llvm::SmallVector<std::string> missingFields;
};

FailureOr<IntrinsicSelection> selectLoadIntrinsic(a5vm::LoadOp op);
FailureOr<IntrinsicSelection> selectUnaryIntrinsic(StringRef family, Type vecTy,
                                                   DictionaryAttr attrs);
```
**Why:** This keeps HIVM-02 testable on its own and gives unresolved reporting structured data instead of scraping printer strings.

### Pattern 3: Emit Parseable LLVM-Like Text, Not Mixed-Dialect Debug Dumps
**What:** Print a full module with `declare` and `define`, LLVM vector types, SSA names, and `call` instructions.
**When to use:** Always for normal `-o` output.
**Example:**
```llvm
; ModuleID = 'abs_kernel_2d'

declare <64 x float> @llvm.hivm.vabs.v64f32(<64 x float>)

define void @abs_kernel_2d(ptr %arg0, ptr %arg1) {
entry:
  %0 = call <64 x float> @llvm.hivm.vld.v64f32(ptr %arg0)
  %1 = call <64 x float> @llvm.hivm.vabs.v64f32(<64 x float> %0)
  call void @llvm.hivm.vst.v64f32(ptr %arg1, <64 x float> %1)
  ret void
}
```
**Why:** The user explicitly wants `.ll`-like output. A printer that leaves `a5vm.load` or `arith.constant` syntax in the final artifact misses the phase boundary.

### Pattern 4: Keep Unresolved Cases Structurally Legal
**What:** Preserve calls in the main output, but route unresolved cases through a non-reserved placeholder callee plus a sidecar record.
**When to use:** Whenever real HIVM naming cannot be finalized.
**Example:**
```llvm
declare <64 x float> @__ptoas_hivm_unresolved.unary.abs.v64f32(<64 x float>)

%1 = call <64 x float> @__ptoas_hivm_unresolved.unary.abs.v64f32(<64 x float> %0)
; A5VM-UNRESOLVED: op=a5vm.abs candidate=llvm.hivm.vabs.v64f32 missing=variant loc=unknown
```
**Why:** This preserves the emitted site for later inventory work and keeps local `llvm-as` validation possible.

### Pattern 5: Report Records Should Be Data-Shaped
**What:** Use one structured record per unresolved or provisional emission case.
**When to use:** Sidecar report and optionally stderr diagnostics.
**Example:**
```json
{
  "op": "a5vm.abs",
  "function": "abs_kernel_2d",
  "placeholder": "__ptoas_hivm_unresolved.unary.abs.v64f32",
  "candidate": "llvm.hivm.vabs.v64f32",
  "used_fields": ["family", "element_type", "lanes"],
  "missing_fields": ["variant"],
  "loc": "unknown"
}
```
**Why:** JSON or line-delimited JSON is easier to diff, grep, and promote into a later intrinsic inventory than free-form prose.

### Anti-Patterns to Avoid
- **EmitC reuse in the A5VM branch:** `dropEmptyEmitCExpressions`, `materializeControlFlowOperands`, and marker rewrites are all wrong-layer logic for this phase.
- **Printer-side semantic recovery:** If the printer must rediscover tile domain or variant details by reading old PTO ops, Phase 2 data preservation failed.
- **One giant emitter file:** Combining naming policy, text printing, sidecar serialization, and CLI wiring in one task will make planning and verification brittle.
- **Fake reserved `llvm.` placeholder names:** This weakens structural validation and makes unresolved cases look more final than they are.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Final output staging | Another custom post-processing pass over C++ or mixed MLIR text | Direct `raw_ostream` emission of final `.ll`-like text | Fewer moving parts and no EmitC dependency. |
| Intrinsic naming | Ad hoc string concatenation in each op printer | Shared family builders plus common type-fragment helpers | Prevents divergence between load/store/unary families. |
| Unresolved diagnostics | Print-only warnings with no machine-readable artifact | Structured sidecar report records plus optional stderr trace | Phase 4 needs emitted-site inventory, not just logs. |
| Structural legality checks | Human inspection only | `llvm-as` parse check against the emitted output | Catches malformed signatures, duplicate declarations, and syntax regressions locally. |

**Key insight:** The deceptive complexity in this phase is not printing text; it is keeping naming policy, syntax policy, and fallback policy from bleeding into each other.

## Common Pitfalls

### Pitfall 1: Branching Too Early in `ptoas`
**What goes wrong:** The plan inserts the new branch before the current shared pass pipeline finishes.
**Why it happens:** It is tempting to think of Phase 3 as “another pass.”
**How to avoid:** Keep the same pre-backend pipeline shape and replace only the current final EmitC translation branch.
**Warning signs:** Tasks propose touching bufferization/planning order or changing unrelated pass topology.

### Pitfall 2: Using Reserved `llvm.` Names for Unresolved Placeholders
**What goes wrong:** The output looks realistic but becomes harder to validate and semantically misleading.
**Why it happens:** The user wants the text close to final HIVM.
**How to avoid:** Use a non-reserved placeholder prefix in emitted calls and record the intended candidate name separately.
**Warning signs:** Placeholder examples start with `@llvm.hivm.` even when the mapping is explicitly unresolved.

### Pitfall 3: Letting the Printer Own Naming Semantics
**What goes wrong:** Load/store/unary cases all synthesize names differently, and unresolved reports become inconsistent.
**Why it happens:** The printer already has op/type access, so it feels convenient.
**How to avoid:** Make intrinsic selection a separately testable helper layer.
**Warning signs:** Printer methods contain long chains of suffix conditionals and repeated type-fragment code.

### Pitfall 4: Emitting Only the `a5vm` Calls but Not Module Scaffolding
**What goes wrong:** The output is inspectable by humans but not by LLVM tools.
**Why it happens:** A function-body fragment is quicker to implement.
**How to avoid:** Emit module header, deduplicated declarations, full `define` blocks, and terminators from the start.
**Warning signs:** Planned fixtures check only for `call @...` lines and never for `define` or `declare`.

### Pitfall 5: Mixing Debug Output Into stdout
**What goes wrong:** The IR file becomes unusable for downstream tools.
**Why it happens:** Debug logging is implemented with opportunistic `errs()`/`outs()` calls.
**How to avoid:** Keep final IR exclusively on the normal output stream and send traces to `stderr` or sidecars behind explicit flags.
**Warning signs:** Plans refer to “annotated output” without separating channels.

### Pitfall 6: Overcommitting to Full LLVM IR Lowering
**What goes wrong:** Phase 3 scope expands into a real LLVM dialect lowering effort.
**Why it happens:** The final artifact is `.ll`-like, so full LLVM IR seems closer to “correct.”
**How to avoid:** Limit the printer to the surviving Phase 2 subset and the `Abs` path; defer full intrinsic ops to `A5VX-01`.
**Warning signs:** Tasks mention LLVM dialect conversion, `LLVM::CallIntrinsicOp`, or broad MLIR-to-LLVM lowering.

## Code Examples

Verified patterns from official and repo-local sources:

### A5VM Emission Entry Point
```c++
struct A5VMEmissionOptions {
  bool dumpA5VMIR = false;
  bool printIntrinsicSelections = false;
  bool allowUnresolved = false;
  std::string unresolvedReportPath;
};

LogicalResult translateA5VMModuleToText(ModuleOp module,
                                        llvm::raw_ostream &irOS,
                                        llvm::raw_ostream &diagOS,
                                        const A5VMEmissionOptions &options);
```
Source: recommended repo-local fit based on `tools/ptoas/ptoas.cpp`, `include/PTO/Transforms/Passes.h`, and the Phase 1 plan contract.

### Family-Specific Naming Boundary
```c++
FailureOr<IntrinsicSelection> selectIntrinsic(Operation *op) {
  if (auto load = dyn_cast<a5vm::LoadOp>(op))
    return selectLoadIntrinsic(load);
  if (auto abs = dyn_cast<a5vm::AbsOp>(op))
    return selectUnaryIntrinsic("abs", abs.getType(), abs->getAttrs());
  if (auto store = dyn_cast<a5vm::StoreOp>(op))
    return selectStoreIntrinsic(store);
  return failure();
}
```
Source: recommended decomposition based on the locked “shared family-specific builders” decision.

### Sidecar-Friendly Unresolved Record
```c++
struct UnresolvedEmissionRecord {
  std::string opName;
  std::string functionName;
  std::string placeholderName;
  std::string candidateName;
  llvm::SmallVector<std::string> usedFields;
  llvm::SmallVector<std::string> missingFields;
  std::string loc;
};
```
Source: required report content from `03-CONTEXT.md`.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `createEmitPTOManualPass(...)` + `emitc::createFormExpressionsPass()` + `emitc::translateToCpp(...)` | Dedicated A5VM textual `.ll` emission branch at the same final tool slot | Planned for Phase 3 in the 2026-03-18 roadmap | Removes EmitC as the final artifact dependency for the implemented subset. |
| Opaque-call naming buried in `PTOToEmitC.cpp` | Shared A5VM family naming helpers | Planned for Phase 3 | Makes intrinsic spelling reusable across more than `Abs`. |
| Human-only inspection of backend text | `FileCheck` plus local `llvm-as` parse validation | Recommended now | Catches malformed `.ll` before remote validation. |

**Deprecated/outdated:**
- EmitC marker rewrites as a backend finishing step: valid only for the legacy C++ path.
- Treating unresolved mappings as comments only: insufficient once emitted-site inventory matters.

## Open Questions

1. **Exact placeholder prefix convention**
   - What we know: it must be machine-distinguishable, preserve call shape, and keep the main output close to LLVM text.
   - What's unclear: whether downstream external tooling prefers a call that looks intrinsic-like or a plainly namespaced extern.
   - Recommendation: plan around a non-reserved placeholder prefix in emitted IR and carry the candidate real HIVM name in the sidecar.

2. **How much shared-op printing is required beyond the `Abs` path**
   - What we know: the user wants control flow and ordinary arithmetic on the implemented path printed in LLVM style, not mixed dialect syntax.
   - What's unclear: which shared ops survive the current A5VM branch after Phase 2 lands.
   - Recommendation: add a task early in Phase 3 to inventory the surviving op subset from `--dump-a5vm-ir`, then keep printer support limited to that set.

3. **Whether the unresolved report should be JSON or plain text**
   - What we know: it needs op, candidate or placeholder name, participating fields, missing fields, and source location.
   - What's unclear: whether a later phase will ingest it programmatically.
   - Recommendation: prefer JSON Lines because it remains grep-friendly and future-proof for inventory tooling.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Source fixtures using `FileCheck` plus local LLVM assembler parsing |
| Config file | none — direct `RUN:` lines and a phase runner script |
| Quick run command | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/<case>.mlir -o - | FileCheck test/phase3/<case>.mlir` |
| Full suite command | `bash test/phase3/run_phase3_checks.sh && ctest --test-dir build --output-on-failure` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| HIVM-01 | A5VM backend writes textual LLVM-like IR instead of EmitC C++ | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/backend_switch.mlir -o - | FileCheck test/phase3/backend_switch.mlir` | ❌ Wave 0 |
| HIVM-02 | Intrinsic spelling derives from family, vector type, and variant metadata | integration | `./build/tools/ptoas/ptoas --pto-backend=a5vm --dump-a5vm-ir test/phase3/intrinsic_naming.mlir -o - 2>&1 | FileCheck test/phase3/intrinsic_naming.mlir` | ❌ Wave 0 |
| HIVM-03 | Emitted `Abs` path is structurally legal and reasonable `.ll` text | integration | `out=$(mktemp) && ./build/tools/ptoas/ptoas --pto-backend=a5vm test/phase3/abs_emit.mlir -o \"$out\" && /data/mouliangyu/projects/github.com/llvm/llvm-project/install/bin/llvm-as \"$out\" -o /dev/null && FileCheck test/phase3/abs_emit.mlir < \"$out\"` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** run the relevant `FileCheck` fixture plus the `llvm-as` parse check if the task touches printing.
- **Per wave merge:** `bash test/phase3/run_phase3_checks.sh`
- **Phase gate:** `bash test/phase3/run_phase3_checks.sh && ctest --test-dir build --output-on-failure`

### Wave 0 Gaps
- [ ] `test/phase3/backend_switch.mlir` — locks the exact tool boundary replacement for HIVM-01
- [ ] `test/phase3/intrinsic_naming.mlir` — locks family/type/variant-based spelling for HIVM-02
- [ ] `test/phase3/unresolved_report.mlir` — locks placeholder call shape plus sidecar contents
- [ ] `test/phase3/abs_emit.mlir` — locks complete module shape and `Abs` path output for HIVM-03
- [ ] `test/phase3/run_phase3_checks.sh` — direct runner for all Phase 3 source fixtures plus `llvm-as`

## Sources

### Primary (HIGH confidence)
- Repo-local integration point: `tools/ptoas/ptoas.cpp` — current pass pipeline, final EmitC translation slot, and EmitC-only cleanup/post-processing.
- Repo-local pass surface: `include/PTO/Transforms/Passes.h` and `include/PTO/Transforms/Passes.td` — existing transform declaration/registration boundary.
- Repo-local legacy backend: `lib/PTO/Transforms/PTOToEmitC.cpp` — evidence of current opaque-call naming and why Phase 3 should not extend it.
- LLVM 19.1.0 LangRef: https://releases.llvm.org/19.1.0/docs/LangRef.html — module/function syntax, call instructions, and intrinsic namespace expectations.
- MLIR Dialect Conversion docs: https://mlir.llvm.org/docs/DialectConversion/ — confirms this phase should not be confused with a full conversion rewrite when only the final emission slot is changing.
- MLIR Defining Dialects docs: https://mlir.llvm.org/docs/DefiningDialects/ — confirms repo-standard separation of dialect/transform concerns.
- CANN intrinsic family reference: `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h` — local primary evidence for `vld`/`vlds`/`vst`/`vsts` family dimensions.

### Secondary (MEDIUM confidence)
- Existing phase plans: `.planning/phases/01-a5vm-foundation/01-03-PLAN.md` and `.planning/phases/02-pto-lowering/02-03-PLAN.md` — useful because they lock prior planner intent, but they are project artifacts rather than implemented code.

### Tertiary (LOW confidence)
- None. The main planning claims are supported by repo-local code plus official LLVM/MLIR references.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - versions were verified locally and the stack is already present in the repo/build.
- Architecture: HIGH - the exact integration boundary and prior phase contracts are visible in repo-local sources.
- Pitfalls: MEDIUM-HIGH - mostly derived from concrete repo seams plus stable LLVM naming/printing constraints.

**Research date:** 2026-03-19
**Valid until:** 2026-04-18
