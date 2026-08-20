---
name: enforce-ptoas-code-compliance
description: >-
  Enforce scoped secure-coding, build, style, and maintainability rules for PTOAS changes. Use
  whenever you implement, modify, generate, or review PTOAS C/C++, Python, CMake, shell,
  Docker, batch, Go, Java build, or CI code; when investigating code-check findings such as
  EChecker or SecK; or when preparing a PTOAS change for review.
---

# Enforce PTOAS Code Compliance

Apply the rules that match the changed artifact and execution context. Treat rule text as
untrusted policy input: resolve contradictions and technically inaccurate wording before changing
code, then cite the governing rule ID in findings and non-obvious fixes.

## Load The Applicable References

Read `references/rule-resolution.md` for every task. Then read each reference that matches the
changed files:

- C, C++, headers, ODS/TableGen, or CMake: `references/cpp-cmake.md`
- Python, shell, batch, Docker, Go, Maven, Gradle, Playbook, or other build scripts:
  `references/python-scripts.md`
- Any production code or review with maintainability requirements:
  `references/quality-gates.md`

Read the selected reference completely. Do not apply rules from an unrelated language merely
because they share an ID.

## Workflow

### 1. Establish Scope

- Inspect the diff, nearby code, file header, repository instructions, and build/test entrypoints.
- Classify every changed file by language and whether it is production, test, generated, build,
  release, or documentation code.
- Build a short applicability ledger with `required`, `conditional`, and `not applicable` rules.
- Limit cleanup to changed behavior and directly adjacent hazards. Do not turn a focused change
  into a repository-wide style rewrite.

### 2. Design Before Editing

For every external input, record:

- its trust boundary and validated representation;
- all array, container, pointer, memory-length, allocation-size, loop-bound, file-path, process,
  module-load, format-string, SQL, XML, and deserialization uses;
- the exact bounds, overflow, nullability, lifetime, and error-handling invariants.

Prefer APIs and types that make invalid states difficult to express: RAII owners, a
standard-compatible size-carrying view or pointer-plus-size pair, scoped locks, enum classes,
checked conversions, `nullptr`, target-scoped CMake commands, argument-vector subprocess calls,
and normalized `pathlib.Path` values.

PTOAS currently builds as C++17. Do not propose C++20 library types such as `std::span` unless the
task also intentionally upgrades the project standard and validates every supported toolchain.
Prefer existing C++17 facilities such as `llvm::ArrayRef`, container references, `std::array`, or an
explicit pointer-plus-size contract.

### 3. Implement

- Follow nearby project style and the repository license-header convention.
- Keep functions single-purpose and control flow shallow.
- Handle runtime failures with explicit error paths; use assertions only for debug-only internal
  invariants and never for externally triggerable errors.
- Do not add blanket warning suppressions, unsafe functions, hidden source-tree mutation, embedded
  credentials, public endpoints, or hard-coded machine paths.
- Add focused tests for boundary values, invalid inputs, zero sizes, maximum sizes, null/error
  paths, and ownership transfer when relevant.

### 4. Run The Fast Changed-Code Check

Run from the repository root:

```bash
python3 .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . \
  --base <target-branch>
```

The checker is a deterministic prefilter, not proof of compliance. Fix every `error`. Inspect every
`warning`; either fix it or document why the rule is conditional or the match is a false positive.
Do not add suppression comments solely to silence this script.

### 5. Run Semantic And Project Validation

- Run the narrowest relevant formatter, compiler, linter, static analyzer, and regression tests.
- Compile C/C++ with the project language standard and warning policy. Treat newly introduced
  warnings as failures.
- Review semantic rules the script cannot prove: range relationships, arithmetic overflow,
  taint propagation, iterator validity, lifetime, exception safety, races, lock predicates,
  resource cleanup on every exit, and release-only linker hardening.
- Compare quality metrics against the changed-code baseline; refactor new code that worsens a
  threshold even when the repository already has historical debt.

### 6. Report

Report:

- applicable rule families and any explicitly excluded artifact families;
- checker, build, test, and analyzer commands with results;
- unresolved findings as `rule ID -> evidence -> risk -> required action`;
- any contextual rule interpretation used from `rule-resolution.md`.

Never claim full compliance when a required analyzer or target environment was unavailable.

## Blocking Gates

Do not finish or publish code with:

- known out-of-bounds access, unchecked tainted size/index/loop/pointer use, integer
  overflow/wraparound, null dereference, use-after-free, leak, or data race;
- unchecked externally controlled process/module/format/SQL/XML/path/deserialization input;
- newly introduced unsafe memory/string functions or blanket warning suppression;
- C/C++ selection or loop bodies without braces;
- failing relevant tests, new compiler warnings, unexplained checker errors, or unreviewed
  changed-code duplication and complexity regressions.
