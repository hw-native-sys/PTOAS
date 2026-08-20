# VMI MLIR Semantics And Validation Notes

This is a compact rewrite-focused summary of the larger VMI draft. Prefer
`ptodsl/docs/user_guide/14-vmi-virtual-instruction-set.md` for Python API
spelling and `docs/examples/vmi/vmi-design.md` for deeper design rationale.

## Mental Model

VMI is a logical SIMD IR:

- A `!pto.vmi.vreg<LxT>` is a flat logical vector of `L` lanes of element type
  `T`.
- A `!pto.vmi.mask<Lxpred>` gates those logical lanes.
- Physical register count, interleave, deinterleave, pack, part, and layout
  materialization are lowering concerns handled by PTOAS.

Do not translate AscendC physical register choreography directly if a single
logical VMI value has the same observable meaning.

## Logical Vector Size

Physical backing size:

```text
K = ceil(lanes * bitwidth(dtype) / 2048)
```

One physical vector register is 256B / 2048 bits. Typical full-register lane
counts:

- `f32/i32`: 64 lanes.
- `f16/bf16/i16`: 128 lanes.
- `i8/ui8/fp8`: 256 lanes.

VMI also allows compact logical vectors such as `vreg(64, pto.f16)`. Unused
physical lanes are undefined and must be masked out when they may be observed.

## Layout

Common layouts:

- Contiguous: logical lane `i` appears in logical order. This is the default.
- Deinterleaved: even/odd or grouped physical backing created by widening,
  deinterleave loads, or layout-sensitive lowering.

Author PTODSL without explicit layout by default. PTODSL does not expose
explicit layout selection on `pto.vmi.vreg(...)` or `pto.vmi.mask(...)`; use
`ensure_layout` only when debugging layout assignment or when the user asked
for layout-level IR.

## Mask And Tail Semantics

For dynamic tails:

```python
mask = pto.vmi.create_mask(active_lanes, size=lanes)
```

Use the same `lanes` as the gated vector. Put the mask on compute ops and
stores. If the backend cannot predicate a load, load the full vector from a safe
UB range and rely on masked consumers/stores. For grouped tails, use
`create_mask(..., size=lanes, group=...)`.

Group masks represent active elements inside each group and should be used for
group reductions or grouped broadcasts rather than plain prefix tails.

## Semantic Equivalence Rules

Treat these AscendC details as physical-only unless they change observable
logical order or data value:

- `PART_P0`, `PART_P1`, `PART_P2`, `PART_P3`.
- `PART_EVEN`, `PART_ODD`.
- Pack/store modes such as `PK4_B32`.
- `vintlv` / `vdintlv` trees that merely repair physical layout.
- Separate low/high or even/odd temporary registers produced by hardware
  widening/narrowing.

Preserve these details when they are semantic:

- Real element permutation or transposition.
- Gather/scatter offsets.
- Row/column order changes.
- Reduction grouping.
- Saturation, rounding, sign/zero extension mode.
- Tail behavior that changes which output elements are written.

## Rewrite Pattern

Preferred surface VMI shape:

```python
mask = pto.vmi.create_mask(active, size=lanes)
x = pto.vmi.vload(x_ub, x_off, size=lanes)
y = pto.vmi.vcvt(x, to_dtype=compute_dtype)      # if needed
z = pto.vmi.vmul(y, scale, mask)                 # representative compute
out = pto.vmi.vcvt(z, to_dtype=dst_dtype)        # if needed
pto.vmi.vstore(out, y_ub, y_off, mask)
```

Keep offsets symbolic and aligned with the source. Prefer one logical vector per
semantic row/block iteration. Split only when the algorithm, UB layout, or VMI
constraints require it.

## PTOAS Validation

For MLIR emitted by PTODSL, validate through the VMI backend path:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto <input.mlir> -o /dev/null
```

Optional debugging forms:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto <input.mlir> -o -
ptoas --pto-arch=a5 --pto-backend=vpto --emit-pto-ir <input.mlir> -o -
```

Do not diagnose VMI legality from an EmitC invocation. The VMI semantic
pipeline always runs when the effective backend is VPTO.

## Debugging Priorities

1. PTODSL trace error: fix Python DSL syntax, types, missing required kwargs,
   or unsupported wrapper usage.
2. VMI verifier error: fix lane/mask mismatch, dtype mismatch, invalid
   inferred shape, invalid group count, or unsupported conversion.
3. VMI lowering error: check whether the logical form needs a different legal
   VMI expression. Avoid adding store/reload workarounds without user approval.
4. Semantic mismatch: revisit the VMI design gate; most errors come from
   accidentally preserving or dropping a physical packing/interleave detail.
