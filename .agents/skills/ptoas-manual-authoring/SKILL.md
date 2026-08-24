---
name: ptoas-manual-authoring
description: Write or revise PTOAS user-facing manuals and ISA/spec docs. Use when updating docs under `docs/`, especially op manuals, syntax references, semantic descriptions, constraints, examples, or bundled release docs, and when the user asks for clear semantic documentation without exposing lowering, raw ops, registers, intrinsics, or other implementation details.
---

# PTOAS Manual Authoring

Use this skill when writing user-facing PTOAS manuals, ISA chapters, op
references, release specs, and examples. The goal is an accurate current manual
for users of the IR, not an implementation note.

## Boundary

User-facing manuals describe the operation contract. Do not expose lowering
details unless the user explicitly asks for an implementation design document.

Do not mention:

- raw op layers, bridge ops, or wrapper-to-raw expansion
- legacy or low-level operation names in high-level manuals or in
  `docs/vpto-spec.md` high-level summaries, inventories, and navigation tables
- hardware registers, control-bit numbers, packed instruction fields, or
  intrinsic names
- pass names, lowering helper names, emitter internals, or source file paths
- historical alternatives, removed syntax, migration notes, or stale design
  states that are no longer valid

Implementation details may live in `docs/designs/` or implementation plans, but
not in the stable user manual.

Legacy and low-level operation references may remain only in explicitly legacy
release snapshots or dedicated low-level reference material. Do not promote them
into current high-level op manuals, examples, or `docs/vpto-spec.md` indexes.

## Required Content

For each op or op family, document these items when applicable:

- Purpose: what logical operation the op represents.
- Syntax: complete assembly form, including optional clauses and their order.
- Operands and attributes: names, types, address spaces, units, defaults, and
  whether values are element counts, bytes, strides, flags, modes, or pointers.
- Legal expressions: all accepted keywords, enum values, flags, and mutually
  exclusive forms.
- Constraints: type combinations, address-space requirements, target-profile
  availability, alignment, shape/layout restrictions, and cross-operand rules.
- Detailed semantics: exact logical meaning of each operand and clause.
- Semantic pseudocode: reference-style pseudocode for the observable result.
- Hardware execution logic: describe the user-visible execution behavior, data
  movement, pipeline ordering, synchronization, layout transformation, numeric
  mode, saturation, rounding, accumulation, or broadcasting behavior without
  naming underlying instructions or register fields.
- Examples: minimal but meaningful examples using non-trivial values or
  realistic shapes. Avoid examples that only prove parsing.

## Writing Rules

- Prefer semantic names over hardware-field names.
- State units explicitly. For strides and lengths, say whether they are bytes,
  elements, tiles, blocks, or C0 units.
- State defaults explicitly. If omitting a clause inherits surrounding state,
  say that; if it means disabled, say disabled.
- State invalid combinations as constraints instead of implying them through
  examples.
- Keep the manual canonical. Remove obsolete plans and superseded forms instead
  of preserving them for history.
- When updating `docs/vpto-spec.md`, keep high-level summaries, inventories,
  and chapter op lists aligned with the current semantic surface. Do not list
  legacy implementation ops just because they exist in ODS or old manuals.
- Avoid vague phrases such as "sets parameters", "configures the pipeline", or
  "does the conversion" unless followed by the concrete values, organization,
  and observable effect.
- If behavior is inferred from simulator or hardware validation, write the
  semantic result and note uncertainty only when it still affects users.

## Pseudocode Guidance

Pseudocode should model the observable operation, not the lowering sequence.

Use logical buffers and indices:

```text
for m in 0 .. M:
  for n in 0 .. N:
    dst[m, n] = ...
```

When layout transforms are involved, describe source indexing, destination
indexing, shape interpretation, stride units, and padding or invalid-lane
behavior. When numeric modes are involved, show when rounding, saturation,
conversion, or exceptional-value handling occurs relative to the main
calculation.

## Review Checklist

Before finishing a manual edit:

- The doc answers "what values can I write" and "what do they mean".
- Every optional clause has legal values, default behavior, and constraints.
- The semantic pseudocode matches the prose.
- User-visible hardware behavior is described without leaking instruction or
  register implementation.
- No removed syntax, old方案, TODO design fragments, or dead alternatives remain
  in the stable manual.
- No legacy or low-level op names are introduced into high-level manuals or
  `docs/vpto-spec.md` high-level summaries/inventories.
- Examples are meaningful and consistent with verifier/lowering behavior.
- Related generated or bundled docs are refreshed when this repo expects them
  to be kept in sync.
