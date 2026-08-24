---
name: ptoas-project-development
description: Project development guidance for PTOAS. Use when modifying PTOAS source, MLIR ODS dialect definitions, C++ verifiers or transforms, CLI behavior, Python bindings, docs, tests, examples, or any user-visible PTOAS behavior; keeps cross-layer updates, license headers, regression tests, and examples synchronized.
---

# PTOAS Project Development

## Core Rule

When changing any user-visible behavior, update every relevant layer in the same change. Treat ODS, verifiers, lowering, command-line behavior, bindings, docs, examples, and tests as one public contract.

## File Orientation Before Editing

Before editing any existing file, read the file's header or top-level comments first. Check whether they describe the file's purpose, structure, section boundaries, or editing constraints, and follow those constraints when making changes.

When a change creates or depends on a file-level structure rule, ownership boundary, or important functional description, add or update that guidance in the file's top-level comments so future edits see it before jumping into local code.

## Layers To Keep In Sync

1. ODS and dialect definitions: `include/PTO/IR/*.td`
2. C++ IR implementation and verifiers: `lib/PTO/IR`
3. Lowering and transforms: `lib/PTO/Transforms`
4. CLI behavior: `tools/ptoas`
5. Python bindings and samples, when affected: `python/`, `test/samples`
6. Docs and specs: `README.md`, `docs/`
7. Regression tests: `test/`

## Change Patterns

When adding or changing an op operand, attribute, assembly form, or semantic constraint:

- Update the ODS definition, assembly format, traits, interfaces, and verifier declarations.
- Update C++ `Op::verify()` logic with actionable diagnostics.
- Update lowering/codegen paths so each new operand or attribute is consumed or forwarded correctly.
- Update Python binding generation inputs, Python API usage, or samples when exposed there.
- Add or adjust focused regression tests.
- Update docs/specs when the behavior is user-visible.

When adding a CLI flag or changing CLI-controlled lowering/codegen:

- Implement parsing and validation in `tools/ptoas/ptoas.cpp`.
- Thread the option into pass options or pass construction explicitly.
- Avoid hidden mutable global state for user-facing behavior.
- Document the flag and add a test or example that demonstrates the behavior.

## Testing And Examples

Prefer regression tests over ad-hoc scripts. Put IR, lowering, and codegen tests under `test/`; put Python sample-based tests under `test/samples/` when the project already uses that flow.

For pass bugs or behavior regressions, capture the minimal reproducible case with:

- Minimal input IR, or Python that prints the relevant IR.
- The exact `ptoas` invocation.
- Expected behavior and actual behavior, such as verifier success, type preservation, address propagation, or emitted C++ compilation.

Keep examples user-facing. Do not commit temporary scripts outside `test/` or `test/samples/`, and minimize reproductions before adding them.

## File Headers

New source or script files must include the PR386 OAT.3 license header at the top of the file. When touching an existing source or script file that lacks the header, add it in the same change. Copy the exact header style from nearby files in the same language.

## Finish Checklist

- ODS matches actual operands and attributes, and parse/print behavior still works.
- Verifier diagnostics explain what the user should fix.
- Lowering/codegen handles all new cases, including view-like cases such as `memref.subview` where relevant.
- Python bindings or samples still build/import when impacted.
- Docs/specs/examples reflect user-visible changes.
- Tests cover the regression or new behavior.
