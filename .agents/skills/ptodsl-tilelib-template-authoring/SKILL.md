---
name: ptodsl-tilelib-template-authoring
description: Add, port, debug, review, or document PTODSL TileLib templates, template candidate/version selection, InsertTemplateAttributes handoff, ExpandTileOp specialization, or TileLang ST failures involving PTODSL templates.
---

# PTODSL TileLib Template Authoring

Use this skill for PTODSL TileLib template work in this repository.

## Required Reading

Before changing template code or selection logic, read the relevant committed
docs:

- `docs/designs/ptodsl-tilelib-template-selection-design.md` for candidate
  selection, compact candidate attrs, context attrs, and specialization keys.
- `ptodsl/docs/developer_guide/tilelib-template-authoring.md` before adding or
  changing PTODSL template metadata, constraints, callable forms, or template
  bodies.
- `ptodsl/docs/developer_guide/tilelib-debugging-playbook.md` before debugging
  ST failures, wrong-output cases, or PTODSL TileLib IR differences.

Read only the files needed for the current task, but do not skip the design doc
when touching `InsertTemplateAttributes`, `ExpandTileOp`, or template version
selection.

## Workflow

1. Classify the failure first: metadata legality, candidate attr handoff,
   PTODSL tracing, helper specialization, or runtime semantics.
2. Compare against the intended TileLib semantics before changing ST tests.
3. Prefer a focused Python or lit regression for selection/specialization bugs.
4. Ask the user before running heavy smoke or non-smoke ST unless they have
   already requested that validation.
5. When a fix teaches a durable rule, update the design doc or developer guide
   instead of adding more scratch status notes.

## Guardrails

- Do not treat dated pass/fail summaries as current truth without checking the
  logs or rerunning the requested case.
- Do not weaken template constraints just to pass one ST case unless the same
  form is legal for TileLib.
- Include view shape, strides, memory space, and layout in reasoning whenever a
  template bakes `ViewSpec` data into generated helper bodies.
- Keep scratch migration trackers separate from committed design guidance.
