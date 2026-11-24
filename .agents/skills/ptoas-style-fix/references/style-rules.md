# PTOAS Code Style Rules

## Contents

- [Formatting](#formatting)
- [Braces and multiline conditions](#braces-and-multiline-conditions)
- [Comments and spelling](#comments-and-spelling)
- [Constants](#constants)
- [Expressions and maintainability](#expressions-and-maintainability)
- [Batch warning handling](#batch-warning-handling)

## Formatting

The repository `.clang-format` is authoritative. Its current C/C++ baseline is
Google-derived with `ColumnLimit: 120`, `IndentWidth: 4`, `TabWidth: 4`,
`UseTab: Never`, left-aligned pointer tokens, no include sorting, and custom
brace wrapping.

The configured formatter also requires these details:

- access modifiers are outdented by one indentation level;
- escaped newlines align left, operands align, and trailing comments align with
  one separating space;
- an opening bracket forces multiline content to break, while arguments may be
  bin-packed and may all move to the next line;
- multiline string literals are not forced to break;
- template declarations always break before the declaration;
- ternary operators do not break before `?` or `:`;
- empty functions stay compact, while empty records and namespaces may split;
- short functions may remain on one line, but blocks, case labels, `if`
  statements, and loops may not.

- `G.FMT.02-CPP`: indent with spaces only and use four spaces per level. Tabs
  are forbidden. Inspect the whole function or block because formatting
  isolated changed lines can inherit a legacy two-space baseline and create
  six-space nested code.
- `G.FMT.06-CPP`: when an expression wraps, leave a binary operator at the end
  of the preceding line. Indent the continuation one level or align
  homogeneous operands. For wrapped calls, use a stable argument layout
  consistent with `.clang-format`.
- `G.FMT.11-CPP`: selection and loop bodies require braces, including one-line bodies.
- `G.FMT.17-CPP`: use one blank line to separate logical sections. Remove
  repeated, decorative, or structurally disruptive blank lines, and inspect
  both sides of the reported location.
- Keep one statement per line, consistent line endings, no trailing whitespace, and a final newline.
- Stay within 120 columns unless an indivisible token makes wrapping
  impossible. Do not flatten a condition merely to silence a scanner when
  readability, precedence, or macro behavior would suffer.
- Follow the configured brace layout: control statements, classes, structs,
  enums, namespaces, and unions keep the opening brace on the same line;
  function definitions place it on the following line. Do not compress
  ordinary blocks, `if` statements, loops, or case bodies onto one line.
- Preserve include order because `SortIncludes` is disabled. Use the configured
  left pointer alignment and spacing rather than manually applying a different
  style.
- Apply formatting to a complete affected function, declaration, initializer,
  or block. Review formatter output for unrelated churn before retaining it.

## Braces And Multiline Conditions

`G.FMT.11-CPP` scanners can misread a multiline `if` condition as a
missing-brace violation even when `{` follows the closing parenthesis.

1. Verify matching braces structurally before editing.
2. If braces exist, treat the warning as a lexical false positive rather than adding duplicate braces.
3. Put the condition on one line only when it remains at most 120 columns and is exactly semantics-preserving.
4. Otherwise introduce a clearly named boolean or use a scanner-compatible
   wrapped layout. Preserve grouping, short-circuit order, side effects, and
   evaluation count.
5. If the source anchor is uncertain, leave it unchanged rather than editing the numeric line from an old report.

## Comments And Spelling

- `G.CMT.01-CPP`: place exactly one space between `//` or `///` and the comment text. Never leave a bare `///`.
- `G.CMT.04-CPP`: do not write empty ceremonial function headers. Remove empty
  `///`, `/** */`, `@brief`, `@param`, or `@return` scaffolding, or replace it
  with concise information about behavior, constraints, invariants, or
  rationale.
- When documentation syntax is required, add meaningful documentation after
  `///`; do not add filler solely to satisfy the checker.
- Preserve accurate and useful comments. Move an inline comment to its own line
  or reword it if a line-oriented scanner mistakes it for code. Delete it only
  when it is redundant, obsolete, or misleading.
- Keep one space before a trailing comment and keep aligned trailing comments within the column limit.
- Rephrase words that spelling tools tokenize incorrectly instead of broadening
  `typos.toml`. Prefer ordinary phrases such as “combined with logical AND” and
  “synchronization mismatch”.
- Spelling checks scan documentation, examples, and inline code spans too. Do
  not repeat a rejected token as a negative example; describe the preferred
  wording without reproducing the triggering text.
- Do not add TODO, TBD, or FIXME markers as a style workaround. Keep comments
  synchronized with the code after every edit.

## Constants

For `G.CNS.02-CPP`, remove an unexplained literal only after identifying its domain meaning and type requirements.

- Search the current component for an existing semantic constant first. Generic
  constants such as `kValue2`, `kValue32`, `kValue62`, or `kValue128` are
  suitable only when their type and meaning remain clear at the use site.
- Otherwise add the narrowest well-named `constexpr`, normally at function
  scope or in an anonymous namespace. Names should express concepts such as
  channel count, depth limit, APInt width, inline capacity, lane multiplier,
  or offset span.
- Preserve constructor and overload behavior. Pay special attention to
  `llvm::APInt` widths and signedness, `SmallVector` inline capacity, shifts,
  depth limits, sentinel values, multiplication, and signed/unsigned
  conversions.
- Replace every occurrence that belongs to the reported expression
  consistently, including wrapped continuation lines, but do not sweep
  unrelated equal-valued literals from the file.
- `G.CNS.01-CPP`: use uppercase `L`, not lowercase `l`, for integer suffixes.
- Do not introduce an unused constant or move a local concept to an unnecessarily broad scope.

## Expressions And Maintainability

- `G.EXP.22-CPP`: prove every divisor used by `/` or `%` is nonzero on all
  paths. Place the guard where it dominates the operation and follow the
  function's existing diagnostic or failure convention. Do not silently
  substitute another divisor or result.
- `G.EXP.30-CPP`: add parentheses around the smallest complete subexpression
  whose intended evaluation order would otherwise rely on default operator
  precedence. In a stream insertion chain, for example, write
  `<< (numBaseTypes + 1)` without enclosing or restructuring the surrounding
  `<<` operations.
- Confirm the intended expression tree from the surrounding types and behavior
  before adding parentheses. Clarify the existing semantics without changing
  associativity, short-circuit behavior, overload resolution, implicit
  conversions, or evaluation order and count.
- Do not parenthesize every operand or the entire statement merely to silence
  `G.EXP.30-CPP`. Keep the edit local to the mixed-operator subexpression, then
  re-run formatting and check wrapped operator placement against
  `G.FMT.06-CPP`.
- For bounds, narrowing, or signedness findings, use the types established by
  surrounding APIs and prove the valid range. Avoid casts that merely suppress
  diagnostics.
- For cyclomatic-complexity or oversized-function findings, extract a cohesive
  helper only when necessary. Preserve mutation order, early exits,
  diagnostics, ownership, and source locations. Use a descriptive name and
  narrow parameter list; then inspect both caller and helper for new naming,
  constant, comment, and formatting findings.
- Do not add dead code, commented-out code, blanket warning suppressions, or checker-disable comments as a style fix.

## Batch Warning Handling

For a warning list, maintain a mapping of rule, file, original snippet or
symbol, current anchor, intended rewrite, and review status.

- Group findings by file, but resolve them against current source rather than applying stale numeric line edits.
- After changing a file, search again for every remaining anchor in that file; earlier edits may shift later lines.
- A scanner may report a continuation line instead of the start of the
  triggering construct. Inspect the complete statement and adjacent lines.
- Review the full diff after every file. Each changed line must correspond to a
  reported finding or a directly required supporting declaration.
- If a finding is a confirmed lexical false positive, prefer the smallest
  readable semantics-neutral rewrite. If the trigger cannot be established
  with certainty, do not modify the code.
