# PTODSL TileLib Template Authoring Guide

This guide is for developers adding or changing PTODSL TileLib templates under
`lib/TileOps`.

For the compiler-side design, see
`docs/designs/ptodsl-tilelib-template-selection-design.md`.

## Template Shape

Register a template with `tilelib.tile_template`:

```python
from ._common import same_dtype_signatures

@tilelib.tile_template(
    op="pto.tadd",
    target="a5",
    name="template_tadd",
    dtypes=same_dtype_signatures(3),
    constraints=[
        tilelib.check_memory_space("ub"),
        tilelib.check_layout("row_major"),
    ],
    priority=0,
    loop_depth=2,
    id=0,
    iteration_axis="row",
    op_engine="vector",
    op_class="elementwise",
)
def template_tadd(src0, src1, dst):
    ...
```

The function parameter order is the operand binding contract. The TileLib runtime binds
MLIR operands positionally to these parameter names before evaluating
constraints or rendering. If a TileLangDSL template had multiple callable
forms, either match the ST operand order exactly or register separate PTODSL
versions with different names and constraints.

## Metadata Checklist

Fill in metadata deliberately.

| Field | Use |
|---|---|
| `op` | Full PTO op name, such as `pto.tload` or `pto.tmatmul.mx`. |
| `target` | Currently `a5`. |
| `name` | Stable candidate name used by diagnostics and expansion. |
| `dtypes` | Tuple of legal operand dtype signatures. Empty means unrestricted, so avoid empty unless that is intended. |
| `layouts` | Block layout requirement for tile operands. |
| `memory_spaces` | Tile/view memory-space requirement. One value applies to all matching operands; otherwise provide one per operand. |
| `constraints` | Predicate tuple for shape, valid-shape, attr, or callable-form rules. |
| `priority` | Higher priority wins among legal candidates. |
| `id` | Stable unique id for multi-candidate ops. |
| `loop_depth` | Metadata consumed by the expansion path. |
| `Tail` | Boolean or predicate describing tail behavior. |
| `is_post_update` | Whether this is a post-update form. |
| `iteration_axis`, `op_engine`, `op_class`, `tags` | Classification for docs, debugging, and future tooling. |

Prefer explicit dtype signatures over broad custom predicates. Dtype mismatch
errors are faster to read than opaque constraint failures.

## Operand Specs

Template parameters receive concrete specs built from MLIR operands:

- `TileSpec`: rank-2 tile shape, dtype, memory space, valid shape, layouts,
  fractal size, pad value, and compact mode.
- `ViewSpec`: view shape, dtype, memory space, optional strides, and optional
  layout.
- `ScalarSpec`: scalar dtype plus static integer value when known.
- `VectorSpec`: vector shape and dtype.

Use the spec fields rather than re-deriving them from names. For example,
valid-shape-sensitive templates should read `tile.valid_shape`; view-sensitive
load/store templates should read `view.shape` and `view.strides`.

## Constraint Predicates

Constraint predicates are called by parameter-name matching. A predicate can
ask for values such as:

- `src_dtype`
- `dst_shape`
- `dst_valid_shape`
- `dst_memory_space`
- `dst_config`
- `operand_dtypes`
- `operand_kinds`
- context attrs such as `round_mode`, `cmp_mode`, or `precisionType`

Keep predicates small and named by the rule they enforce. A good predicate
answers one question, such as "is this a row-major vec tile" or "does this
view have a static stride".

Ordinary element-wise 1D candidates should use
`require_elementwise_1d(*operand_names)` and explicitly name every traversed
tile, including temporary operands. It proves a common static logical range on
gap-free row-major/none-box local storage. Predicate-mask and dtype-width-
changing conversion families need additional representation-specific rules.

When porting a legacy TileLangDSL version, first write down which rule made the
legacy version legal. Then encode that rule as metadata or a predicate. Avoid
fixing a single failing case by weakening a predicate beyond the legacy
behavior.

## Context Attributes

Some TileOps need op attributes in addition to operand specs. The compiler
forwards selected attributes as context attrs:

- `round_mode`
- `rounds`
- `cmp_mode`
- `mask_pattern`
- `precisionType`

If a template needs a new op attribute, update the C++ context-attr forwarding
before relying on it in Python. A template that silently assumes a default when
the real op carries a different mode will usually pass simple smoke cases and
fail later in non-smoke ST.

## Candidate Priority And Ids

Use priority only to choose between genuinely overlapping legal candidates.
Do not use priority to hide an overly broad version. If two versions should be
mutually exclusive, fix the constraints.

For multi-candidate ops:

- assign stable `id` values;
- keep ids unique for the op;
- keep names descriptive enough for IR dumps;
- add a lit check when candidate ordering matters.

## Current Elementwise Binary Policy

Simple binary elementwise ops should use the shared
`_elementwise.register_binary` helper unless they need a proven TileLangDSL
version split. For same-dtype arithmetic/min/max families, prefer
`same_dtype_signatures(...)` over hand-written f32-only lists so integer,
unsigned integer, and floating-point coverage stays aligned with TileLangDSL.
For bitwise and shift families, use the shared `INT_DTYPES` group; scalar
shift counts remain `i16`, matching the legacy TileLangDSL callable form.

The straightforward PTODSL elementwise templates intentionally use one
non-post-update candidate and do not set `Tail`.

Do not add `Tail` or post-update variants speculatively. Reintroduce those
versions only with a concrete ST/lit case that needs them and a real selection
rule that matches TileLangDSL behavior.

## Runtime-Safe PTODSL

Template bodies execute under PTODSL tracing. Python values and PTODSL runtime
values are not interchangeable.

Use PTODSL control-flow and scalar APIs when a value is runtime-dependent.
Avoid:

- native Python `if` on a PTODSL runtime value;
- native Python `range` using runtime bounds;
- assigning Python integers into runtime branch state;
- assuming a scalar operand has a compile-time value unless `ScalarSpec.value`
  is present.

This class of bug often appears after selection is fixed: the template becomes
legal, then tracing fails in a larger non-smoke path.

## View And Valid-Shape Rules

Do not assume the logical valid shape is the same as the physical view shape.
ST reductions and row-arg ops often write a `3x1` valid result into a physical
`3x8` destination. Load/store templates must distinguish:

- tile shape;
- tile valid shape;
- view physical shape;
- view strides;
- view layout.

If rendered code materializes a view stride as a constant, that view metadata
also has to be represented in expansion specialization. The compiler design doc
describes the cache rule in detail.

## Porting Workflow

1. Find the TileLangDSL template version and the ST case that requires it.
2. Identify operand order, dtypes, layouts, memory spaces, valid shapes, and
   context attrs.
3. Add or adjust PTODSL metadata and constraints.
4. Render a focused case and inspect the generated IR.
5. Add a Python metadata/render test or a lit expansion test for the new rule.
6. Ask for smoke ST, then non-smoke ST when the case family is broad.

Keep broad parity tables out of the source of truth for a template change. They
are useful for planning, but committed behavior should be represented by
metadata, implementation, and focused regression tests.
