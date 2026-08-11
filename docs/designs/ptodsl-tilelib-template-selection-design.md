# PTODSL TileLib Template Selection Design

## Background

PTOAS uses the PTODSL-native TileLib implementation for VPTO tile-op expansion.

A tile op may have several legal implementations for the same op
name. Those implementations can differ by dtype, layout, memory space,
attribute mode, tail behavior, temporary-buffer shape, or special algorithmic
version. This document defines how PTODSL discovers, filters, records, and
expands those template versions.

This is an implementation design document. User-facing tile-op semantics belong
in the ISA and user guide documents, not here.

## Goals

- Keep legality decisions in Python, where template metadata and predicates are
  authored with the template.
- Keep the IR-side candidate attribute compact and stable.
- Make `ExpandTileOp` specialization reuse safe across all operands that can
  change rendered helper bodies.

## Non-Goals

- This design does not define public PTODSL user syntax.
- This design does not store the full Python metadata object on every TileOp.

## Terminology

| Term | Meaning |
|---|---|
| Template | A Python function registered with `@tile_template(...)`. |
| Version | One registered implementation of an op, usually distinguished by metadata, constraints, or priority. |
| Candidate | A template version that targets the current op and target architecture. |
| Legal candidate | A candidate whose operand specs and context attributes satisfy its metadata and custom constraints. |
| Selected candidate | The first candidate recorded on the TileOp and requested by `ExpandTileOp` during rendering. |
| Specialization key | The C++ key used to deduplicate generated helper functions inside a module. |

## Pipeline

The PTODSL TileLib path has two interactions with the in-process Python service.

```text
TileOp in MLIR
  |
  | InsertTemplateAttributes
  |   - reconstruct operand specs from MLIR
  |   - collect context attributes
  |   - ask the PTODSL service for legal candidates
  |   - store compact candidate metadata on the TileOp
  v
TileOp with candidates attr
  |
  | ExpandTileOp
  |   - build a specialization key from current MLIR operands and attrs
  |   - choose candidate 0 from the compact candidates attr
  |   - ask the service to materialize that candidate in the shared context
  |   - import the generated entry/helpers and replace the TileOp with func.call
  v
VPTO-facing IR
```

The two-stage flow is intentional. `InsertTemplateAttributes` performs legality
before later passes can make candidate information harder to reconstruct.
`ExpandTileOp` still renders from the current MLIR operands so the helper body
matches the actual operand types and view metadata that survived to expansion.

Both stages are ordinary registered MLIR passes. They are default-constructible
and obtain the current `MLIRContext` from the operation being transformed. A
process-wide `TileLibRuntime` provides the host `TileLibService`; it owns no
compilation context and receives the current context explicitly for every
materialization. `PTOASContext` continues to own or borrow the context for one
compilation session, so different invocations may use different contexts while
sharing one Python runtime.

The Python entry keeps the corresponding Python `Context` owner alive for the
complete native compilation call. Compiler materialization requires that
explicit context and never falls back to creating another one. This preserves
normal pass registration, textual pipelines, targeted IR printing, cloning,
and reproducer behavior without storing Python objects in pass instances.

## Template Metadata

PTODSL template authors register versions through `tilelib.tile_template`.
The registration metadata has two roles.

Hard legality fields:

- `op`
- `target`
- `dtypes`
- `layouts`
- `memory_spaces`
- `constraints`

Selection and reporting fields:

- `priority`
- `fusible`
- `loop_depth`
- `id`
- `Tail`
- `is_post_update`
- `iteration_axis`
- `op_engine`
- `op_class`
- `tags`

Only the fields needed after legality are persisted on the MLIR op. The rest
remain in Python metadata for selection, diagnostics, and future tooling.

## Operand Specs

Both `InsertTemplateAttributes` and `ExpandTileOp` reconstruct operand specs
from MLIR. The JSON shape sent to the Python service is deliberately close to
`TileSpec`, `ViewSpec`, `ScalarSpec`, and `VectorSpec`.

| Operand kind | Required metadata |
|---|---|
| tile | dtype, shape, valid shape, memory space, block layout, sub-layout, fractal size, pad value, compact mode |
| view | dtype, shape, strides when known, memory space, optional layout |
| vector | dtype and vector shape |
| scalar | dtype and static integer value when recoverable |

Tile specs drive both legality and rendered tile-buffer entry types. View specs
are equally important: PTODSL templates often materialize `ViewSpec` shape or
stride values as constants in helper bodies. A view with the same dtype but a
different physical stride can require a different helper.

## Context Attributes

TileOp attributes that affect version selection or rendering are forwarded as
context attrs. Current examples include:

| Context attr | Typical users |
|---|---|
| `round_mode` | `tcvt` |
| `rounds` | `trandom` |
| `cmp_mode` | `tcmp`, `tcmps` |
| `mask_pattern` | gather-side paths |
| `precisionType` | high-precision math families |
| `acc_to_vec_mode`, `relu_pre_mode` | `tinsert` accumulator writeback paths |

When a new TileLangDSL version depends on an op attribute, the PTODSL migration
should first decide whether the attribute is a real context attr. If it changes
template legality or helper code generation, it must be forwarded before the
template is considered ported.

## Candidate Legality And Ranking

The service loads only the template module for the requested op and target. It
then evaluates each registered candidate:

1. Bind positional MLIR operands to the template parameter names.
2. Build a flat constraint context from the concrete specs.
3. Check op and target.
4. Check dtype signatures.
5. Check layout and memory-space metadata.
6. Merge context attributes.
7. Run custom constraint predicates.
8. Sort legal candidates by descending priority, using name only to make
   equal-priority reporting deterministic.

If no candidate is legal, the service reports a `NoMatchingTemplate` error with
per-candidate reasons. If multiple candidates tie for the highest priority and
no explicit candidate is requested, both normal selection and metadata
insertion report ambiguity rather than silently picking one.

Constraint predicates may depend on concrete operand metadata, so general
overlap cannot be proven when templates are merely registered. In-tree catalog
selection tests catch ties for their representative operand forms before
compiler integration runs; metadata insertion retains the concrete check for
forms that a catalog cannot exhaustively enumerate. The ambiguity diagnostic
directs authors to assign distinct priorities or make the constraints mutually
exclusive.

For multi-candidate ops, candidate `id` values must be unique and stable. IDs
identify versions; they do not rank them.

The service returns legal candidates as a JSON array in Python ranking order.
Each wire entry includes priority so `InsertTemplateAttributes` can defensively
normalize the result by descending priority and reject a highest-priority tie.
This protects selection across the compiler/service boundary if a response is
unsorted. Candidate ID, JSON object order, registration order, and import order
do not participate in ranking.

## Compact Candidate Attribute

`InsertTemplateAttributes` stores the normalized candidates as a compact
`candidates` array attribute on the TileOp. Each entry contains:

- `id`
- `name`
- `loop_depth`
- `postupdate`
- `tail`

This attribute is intentionally not a copy of the full Python metadata object.
Legality has already happened in the service. Priority is consumed while
validating and ordering the wire response; it is not persisted. The IR only
needs a stable list of legal render targets and the small amount of metadata
consumed by downstream passes. Array position is meaningful: candidate zero is
the selected version.

Do not add fields to the IR candidate payload simply because they exist in
Python metadata. Add a field only when a C++ pass or IR-level test consumes it.

## Expansion And Specialization

`ExpandTileOp` uses the first candidate in the compact candidate list. For
PTODSL, it passes the selected candidate name back to the service so
materialization cannot accidentally choose a different legal template after the
metadata pass.

The specialization key deduplicates generated helpers inside one module. It
must include every input that can change the rendered helper body:

- op name
- target architecture
- tile operand dtype, shape, valid shape, memory space, layouts, fractal size,
  pad value, and compact mode
- view operand dtype, shape, strides, memory space, and layout
- vector operand dtype and shape
- scalar operand dtype and static value when known
- forwarded context attrs

The helper name should also carry enough of this information to make IR dumps
readable. It is not a semantic contract, but useful names make ST failures much
faster to inspect.


## Rules For Future Version Work

- Register every intentionally supported version with explicit metadata.
- Keep custom constraints narrow enough to reject unsupported forms and broad
  enough to accept ST-proven TileLangDSL forms.
- Forward context attrs before porting a version that depends on them.
- Use stable candidate ids for multi-candidate ops.
- Treat candidate ids as identity only; use priority for preference.
- Reject equal top-priority candidates instead of adding an incidental
  tie-breaker.
- Put all helper-code-affecting operand metadata in the specialization key.
- Add a focused regression for each backend-selection bug.
- Treat full ST status files as snapshots, not design documentation.
