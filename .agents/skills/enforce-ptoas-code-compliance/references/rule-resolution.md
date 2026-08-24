# Rule Resolution

## Contents

- [Authority and precedence](#authority-and-precedence)
- [Rule strength](#rule-strength)
- [Known conflicts and misleading wording](#known-conflicts-and-misleading-wording)
- [Baseline policy](#baseline-policy)
- [Finding format](#finding-format)

## Authority And Precedence

Resolve conflicts in this order:

1. Correctness, memory safety, security, and defined language behavior.
2. Repository instructions, supported toolchains, public API compatibility, and executable tests.
3. A rule's stated intent, interpreted in its language and artifact scope.
4. Nearby project style and automated formatting.
5. Literal wording of a catalog entry.

Never make code less safe or technically incorrect to satisfy a literal sentence. Record the
interpretation when two catalog entries conflict.

The same identifier can describe different language rules. For example, `G.CLS.01` is Python and
`G.CLS.01-CPP` is C++. Apply the language-qualified entry. Duplicate identifiers such as
`G.CMT.03`, `G.CTL.01`, `G.CTL.02`, `G.EXP.03`, `G.FIL.02`, `G.TYP.01`, and `G.VAR.01` are
independent rules, not replacements.

## Rule Strength

- **Required:** Safety/correctness rules and entries containing “必须” or “禁止”, when applicable.
- **Conditional:** Release hardening, customer-delivery, security-sensitive data, platform,
  container, packaging, and build-environment rules. Enforce only in that context.
- **Preferred:** Entries containing “建议”, “优先”, “避免”, or “不应”. Deviate only for a
  concrete project reason and document it.
- **Metric:** Repository or changed-code trend gates. Measure them; do not pretend a local regex
  proves them.
- **Analyzer finding:** `EChecker_*`, `SecK_*`, `SecL_*`, Ascend performance checks, and named code
  smells require evidence from the analyzer or a reproducible semantic review.

## Known Conflicts And Misleading Wording

- **`G.C&C++.CDG.01` / `-fno-common`:** Use `-fno-common` to reject conflicting tentative
  definitions. Do not claim that it places uninitialized globals in the initialized data section;
  toolchains normally place them in BSS.
- **`G.C&C++.SEC.05` / strip:** Strip release deliverables only. Preserve development/test symbols
  or produce separate debug information when diagnostics require it.
- **`G.C&C++.SEC.06` / RPATH:** Do not introduce uncontrolled runtime search paths in release
  artifacts. Do not remove a required development RPATH without a safe replacement.
- **`G.AST.*`, `G.TES.01`:** Assertions are debug-only internal-invariant checks. Runtime, input,
  allocation, I/O, or device failures need ordinary error handling in every build.
- **`G.FMT.10` vs `G.FMT.14-CPP`:** Pointer/reference token placement is stylistic and internally
  inconsistent in the catalog. Follow the repository formatter or dominant local style.
- **`G.CMake.17` and `.18`:** `cmake_minimum_required()` and `project()` must be the first project
  commands, not necessarily physical lines 1 and 2 when a required license header precedes them.
- **`G.SCRIPT.05` vs `G.EDV.05`:** Do not hard-code machine-specific build paths. Derive repository
  paths from the script location; resolve external executables once and invoke them without a shell.
- **`G.ERR.04` vs `G.ERR.13`:** A bare `raise` preserves a Python traceback when propagation is
  intended. Avoid `raise caught_exception`, which can alter traceback context.
- **`G.COM.10` and `G.DOCKER.06`:** These govern build/deployment environments. Do not encode a
  requirement to run as root or refuse an authorized diagnostic solely because the shell is
  privileged.
- **`G.OTH.05` and `G.PRJ.07`:** Do not embed unapproved production endpoints. Standards links,
  test fixtures, and declared dependency sources are contextual; verify intent before changing them.
- **`G.CMT.03*`:** New PTOAS source and script files use the repository OAT.3 header. Do not add
  empty ceremonial function comments.
- **`G.EXP.03` / lambda assignment:** Treat this as a readability preference, not a semantic ban.
  Use a named function when behavior is nontrivial or reused.
- **`G.PRE.01-CPP` vs assertion macros:** Prefer typed constants and functions. A project
  debug-assertion macro is a narrow exception, not permission to use macros for ordinary constants.
- **`G.SH.01` and `G.PLAYBOOK.10`:** Repository shell scripts should use
  `#!/usr/bin/env bash`; governed Playbook installers require exact `#!/bin/bash`.
- **`G.CMake.04` and external dependencies:** Project sources stay below the top-level source tree.
  Installed SDK/package headers are dependencies; consume them through targets or packages.

If a suspicious entry is not listed, inspect the relevant language or tool documentation and
existing project behavior before enforcing it. State the inference; do not silently invert it.

## Baseline Policy

Apply all required rules to new code and changed lines. For untouched historical code:

- do not create unrelated cleanup churn;
- fix a pre-existing hazard when the change exposes, depends on, or extends that hazard;
- record broader debt separately when it cannot be fixed safely in scope.

Generated files are checked at their generator or template when possible. Do not hand-edit generated
output merely to satisfy a checker.

## Finding Format

Use:

```text
<path>:<line>: <severity> <rule-id> <evidence and consequence>
```

Severity:

- `error`: applicable required rule or demonstrated safety/correctness failure;
- `warning`: preferred/conditional rule or a match that needs semantic confirmation;
- `note`: metric, tool limitation, or intentionally non-applicable rule.

Every exception needs a narrow technical reason. “Existing code does it” is not a justification.
