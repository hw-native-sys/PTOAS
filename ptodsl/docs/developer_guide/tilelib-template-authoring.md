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

Packed compare candidates use
`require_predicate_compare_1d(*data_operand_names,
predicate_operand="dst")`. The data operands still need a common contiguous
logical range, while the predicate destination is checked as one bit per
source element with dtype-dependent complete-load/store rounding. Multi-row
flattening additionally requires every source row to end on a complete
predicate-store boundary and the predicate destination row stride to contain
exactly that row's packed bytes; its physical row must retain the target's
32-byte alignment. Do not substitute ordinary same-shape legality for this
rule: padded predicate rows must restart in the 2D fallback.

Packed select candidates use
`require_predicate_select_1d(predicate_operand, *data_operand_names,
temporary_operand=...)`. Data operands follow ordinary continuity and
same-range rules, while the mask is checked in physical bytes because `tsels`
allows i8, i16, and i32 mask storage containers for the same packed bits.
Multi-row masks must have no predicate-row padding. On A5 the select temporary
is unused, so validate that its local tile metadata is complete and supported
without requiring its shape to match the data range.

Width-changing conversion candidates use `require_conversion_1d()`. The rule
proves that source and destination are independently gap-free typed streams;
ordinary conversions have equal element shapes, while packed forms pass an
explicit source-elements-per-destination ratio. Keep unpack/pack distribution
modes, mask dtypes, rounding, saturation, and multi-step instruction sequences
inside the conversion module. Unknown metadata falls back to 2D. If a future
conversion has a tile temporary, extend the family rule to validate it rather
than applying the current source/destination-only predicate unchanged.

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
- `acc_to_vec_mode`
- `relu_pre_mode`

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
- use `priority` rather than ID to express preference;
- avoid equal top-priority candidates for the same concrete operands;
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

Source-backed TileLib template functions use PTODSL's control-flow AST rewrite.
Plain Python `if` and `for ... in range(...)` in a template body or its nested
helpers therefore lower to runtime structured control flow. Prefer that syntax
for ordinary loops:

```python
remained = valid_cols
for col in range(0, valid_cols, lanes):
    mask, remained = pto.make_mask(dtype, remained)
    ...
```

Use `pto.static_range(...)` when a loop should execute during tracing. Keep the
explicit `pto.if_` and `pto.for_` APIs for unsupported or deliberately explicit
control-flow patterns.

Module-level Python helpers are outside the registered template function's
source tree, so their bodies are not rewritten merely because the template
calls them. A module-level helper that contains runtime Python control flow
must opt into `rewrite_jit_function`, as the shared element-wise traversal
cores do, or use the explicit control-flow APIs.

Avoid:

- assigning Python integers into runtime branch state;
- assuming a scalar operand has a compile-time value unless `ScalarSpec.value`
  is present;
- relying on native runtime control-flow rewrite for functions without
  retrievable source.

This class of bug often appears after selection is fixed: the template becomes
legal, then tracing fails in a larger non-smoke path.

## Shared Element-wise Traversal Forms

Ordinary A5 unary, Tile-Tile, Tile-Scalar, and scalar-fill candidates should
use the registration helpers in `templates/a5/_elementwise.py`. Each registrar
accepts `traversal="1d"` or `traversal="2d"`:

- `1d` derives `loop_depth=1`, adds
  `require_elementwise_1d(*operand_names)`, and emits one vector loop over
  `valid_rows * valid_cols`;
- `2d` derives `loop_depth=2` and emits the general row loop plus per-row
  vector loop.

The default remains `2d`, so adding the shared foundation does not silently
change existing operation selection. A migrated operation normally preserves
its existing candidate as the 2D fallback and adds a separately named,
higher-priority 1D candidate:

```python
template_tadd = register_binary(
    op="pto.tadd",
    name="template_tadd",
    vector_op=pto.vadd,
    dtypes=DTYPES,
    traversal="2d",
)

template_tadd_1d = register_binary(
    op="pto.tadd",
    name="template_tadd_1d",
    vector_op=pto.vadd,
    dtypes=DTYPES,
    traversal="1d",
)
```

The shared registrars derive the fallback priority/ID as `0/0` and the
preferred 1D priority/ID as `10/1`. Bespoke family registrars should use
`traversal_metadata(...)` to retain this policy; multi-form families pass their
fallback ID and form count so IDs remain unique. Candidate ID expresses
identity, not preference. Every ID for the same operation must remain unique.
The 1D registrar constraint must name every tile operand that participates in
or constrains the traversal, including temporary TileOp operands.

The first production users of this pattern are the ordinary unary operations
`tabs`, `texp`, `tneg`, `tnot`, `trelu`, `trsqrt`, and `tsqrt`. Their original
candidate names and ID 0 remain the 2D fallbacks; their `_1d` candidates use
ID 1 and higher priority.

The ordinary Tile-Tile operations `tadd`, `tand`, `tmax`, `tmin`, `tmul`,
`tor`, `tshl`, `tshr`, and `tsub` follow the same pattern through
`register_binary`. Even when an operation previously had an equivalent local
row-wise body, as `tmin` did, keep the ordinary computation as the vector
callback and let the shared registrar own both traversal forms.

The ordinary Tile-Scalar operations use `register_scalar_binary` with the same
ID-0 fallback and preferred ID-1 pattern. Flattened legality names only the
traversed `src` and `dst` tiles; a scalar operand has no layout or continuity
requirement. Preserve instruction-specific scalar forms: bitwise and
tile-minus-scalar operations request vector broadcast, while scalar shift
operations keep their `i16` scalar dtype. `tsubs` is expressed as broadcast
plus `vsub` through the shared registrar rather than owning separate loops.

Specialized Tile-Scalar algorithms remain in their family modules and call the
shared traversal emitters. `tdivs` retains its precision handling and distinct
tile-scalar/scalar-tile call forms; `_remainder.py` retains scalar remainder
math; and `tlrelu.py` retains slope coercion. Temporary scalar forms must name
the temporary in `require_elementwise_1d`, even when the generated body does
not access it.

For callable forms distinguished by positional operand kind or order, test
selection through daemon metadata or a compiler lit test. Daemon requests
retain the original positional MLIR operand sequence. A direct registry call
uses a name-keyed mapping that is rebound to each candidate and therefore
cannot by itself prove which positional overload is legal.

Scalar-fill operations use `register_scalar_fill`. Their 1D constraint names
only the destination tile because the scalar has no physical layout. Preserve
the original dtype signatures and let the shared emitter own flattened element
count, mask generation, tail handling, and vector stores.

For bespoke computation, use the lower-level `emit_elementwise_1d` and
`emit_elementwise_2d` chunk callbacks or the family emitters. The flattened
form supplies one linear element offset; do not convert it back into row and
column indices. Predicate and dtype-width-changing conversion operations need
their additional representation-specific legality before using the 1D form.

Keep bespoke algorithms in their operation modules. For example, `tlog` owns
its high-precision subnormal compensation and `trecip` owns its `1 / src`
calculation; both call shared unary traversal emitters. Do not move
operation-specific constants, precision algorithms, temporary calculations, or
instruction sequences into `_elementwise.py`.

The same boundary applies to specialized binary families. `tdiv.py` owns its
precision-dependent division callback, and `_remainder.py` owns fmod/remainder
instruction sequences and their family registrars. These modules may call
shared binary traversal emitters, but their algorithms do not belong in
`_elementwise.py`.

Temporary-operand Tile-Tile forms follow the same registration pattern, but
the 1D constraint must include the temporary explicitly. For example,
`tprelu`, `trem`, and `txor` call their family registrar with `has_tmp=True`;
the registrar passes `src0`, `src1`, `tmp`, and `dst` to
`require_elementwise_1d`. Keep the temporary in that proof even when the
current generated helper does not read it. A non-contiguous or insufficiently
described temporary must independently disqualify the 1D candidate.

The issue-scope acceptance catalog lives in
`ptodsl/tests/test_tilelib_elementwise.py` as
`ELEMENTWISE_SCOPE_BY_FAMILY`. It is the authoritative checklist for the
unary, Tile-Tile, Tile-Scalar, compare, select, conversion, and scalar-fill
operations covered by shared 1D/2D selection. The catalog test requires unique
candidate IDs, a general 2D fallback, and higher-ranked 1D coverage for every
listed operation. There are currently no operation-level 1D exceptions. If an
instruction representation makes all flattened forms illegal in the future,
add a non-empty reviewed reason to `ELEMENTWISE_1D_EXCEPTIONS` and a focused
legality regression; do not remove the operation from the catalog.

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
