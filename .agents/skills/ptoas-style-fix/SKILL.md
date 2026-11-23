---
name: ptoas-style-fix
description: >-
  Fix PTOAS C/C++ CodeCheck, OAT, clang-format, comment, spelling, constant,
  expression, and maintainability findings with a minimal semantics-preserving
  diff. Use when editing PTOAS source for style warnings or reviewing a
  style-fix patch; do not use for unrelated repositories or broad source-tree
  reformatting.
---

# PTOAS Style Fix

Produce the smallest source change that removes the reported trigger without
changing program behavior or creating a new style finding.

## Establish The Source Of Truth

Before editing, read the checked-out revision's `.clang-format`,
`.pre-commit-config.yaml`, `typos.toml`, relevant file-level comments, and
nearby code. Repository configuration and the current source take precedence
over remembered formatting.

Treat warning reports as location hints. Resolve every finding by rule ID,
file, symbol, and source snippet rather than by line number alone: earlier
edits can shift later findings. Re-open the current file before each group of
changes.

Read [references/style-rules.md](references/style-rules.md) completely before fixing C/C++ style findings.

## Editing Principles

- Understand the enclosing function and involved types before changing code.
  Preserve control flow, precedence, evaluation order, short-circuiting,
  integer width and signedness, overload selection, diagnostics, and failure
  behavior.
- Limit edits to reported findings and directly necessary declarations. Do not
  reformat a whole file or clean unrelated historical code.
- Format the smallest complete syntactic unit with a coherent indentation
  baseline. If line-only formatting inherits legacy indentation and produces
  six-space nesting, normalize the complete affected function or block.
- Prefer an existing constant only when its name, type, and meaning match.
  Otherwise introduce a narrowly scoped semantic `constexpr`; do not hide a
  magic value behind an equally opaque name.
- Preserve useful comments. Reword, relocate, or remove a comment only when
  necessary to fix an identified comment or spelling finding.
- For a suspected false positive, first identify the checker's lexical trigger
  and prove the code structure is already correct. Apply a semantics-neutral
  rewrite only when the trigger is certain; never guess from a stale line
  number.

## Changed-Code Review

After each batch, inspect the actual diff and map every changed line to a
reported finding or necessary supporting change. Check the complete affected
function or block, not only the edited line, for:

- spaces-only four-column indentation and no trailing whitespace;
- 120-column limit, operator placement, argument alignment, braces, and blank lines;
- comment spacing, meaningful documentation, and spelling;
- type-correct constants and unchanged expression semantics;
- accidental formatting churn outside the intended region.

Apply the repository formatter to complete affected units and inspect its
output. A second formatter run should make no changes, but idempotence alone is
not proof that nesting or semantics are correct. Use the configured
style/checker when available, and do not weaken formatter, spelling, or scanner
configuration to hide a source finding.
