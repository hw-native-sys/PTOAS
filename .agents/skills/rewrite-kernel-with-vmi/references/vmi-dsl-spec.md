# PTODSL VMI DSL Quick Reference

This is a compact guide derived from
`ptodsl/docs/user_guide/14-vmi-virtual-instruction-set.md`. Use the user guide
as the source of truth when an operation is missing here.

## Core Types

- `pto.vmi.vreg(lanes, dtype)` creates a logical vector type.
  `lanes` must be a multiple of 64. Common full-register choices:
  - `pto.f32` / `pto.i32`: 64 lanes per 256B physical vreg.
  - `pto.f16` / `pto.bf16` / `pto.i16`: 128 lanes per physical vreg.
  - `pto.i8` / `pto.ui8` / fp8: 256 lanes per physical vreg.
- `pto.vmi.mask(lanes)` creates a logical per-lane mask type.
  Its lane count must match the gated vector.

PTODSL does not expose layout selection on `pto.vmi.vreg(...)` or
`pto.vmi.mask(...)`; PTOAS infers layout during lowering.

`dtype` may come from a `pto.const_expr` parameter. Prefer compile-time dtype
selection inside one PTODSL function over many small wrappers that only vary the
VMI element type:

```python
vec_ty = pto.vmi.vreg(lanes, OUT_DTYPE)
vec = pto.vmi.vcvt(src, to_dtype=OUT_DTYPE)
```

## Load And Store

```python
vec = pto.vmi.vload(ub_src, offset, size=128)
even, odd = pto.vmi.vload(ub_src, offset, size=128, dist_mode="dintlv")
wide = pto.vmi.vload(ub_src, offset, size=128, dist_mode="unpack", to_dtype=pto.f32)
pto.vmi.vstore(vec, ub_dst, offset, mask)
pto.vmi.vstore(vec, ub_dst, offset, group=8, stride=row_stride)
```

Use VMI load/store only for UB-resident compute operands/results. Use MTE or
tile operations for GM movement.

Useful options:

- `size` is required for every `vload`.
- `dist_mode=None` or `"continuous"` is the default contiguous load/store form.
- `dist_mode="dintlv"` returns an `(even, odd)` pair.
- `dist_mode="unpack", to_dtype=<dtype>` widens by one adjacent bit-width step.
- `group=...`, `stride=...` select grouped access.
- `block_stride=...`, `repeat_stride=...` select block-strided access.
- `vload` does not take `mask`.
- `group` store does not take `mask`.
- `pmode="zero"` is the default masked-store behavior; `pmode="merge"` preserves
  inactive lanes.

Backend note: prefer putting dynamic tail masks on compute/store. Do not rely on
masked loads unless the current backend explicitly supports the form.

## Masks

```python
mask = pto.vmi.create_mask(active_lanes, size=lanes)
gmask = pto.vmi.create_mask(active_per_group, size=lanes, group=num_groups)
```

Use `create_mask` for prefix-active dynamic tails. Use grouped masks for grouped
reductions or grouped broadcast patterns.

## Index And Broadcast

```python
idx = pto.vmi.vci(pto.i32(0), size=64, order="ASC")
bc = pto.vmi.vbrc(pto.f16(0.0), size=128)
```

Use `vci` for lane-wise index ramps and gather/scatter offsets. Use `vbrc` for
scalar-to-vector or group-to-vector broadcast.

## Elementwise And Scalar Ops

Same-shape vector ops usually infer result type:

```python
y = pto.vmi.vadd(a, b, mask)
y = pto.vmi.vsub(a, b, mask)
y = pto.vmi.vmul(a, b, mask)
y = pto.vmi.vdiv(a, b, mask)
y = pto.vmi.vmax(a, b, mask)
y = pto.vmi.vmin(a, b, mask)
```

Unary ops:

```python
y = pto.vmi.vabs(x, mask)
y = pto.vmi.vneg(x, mask)
y = pto.vmi.vrelu(x, mask)
y = pto.vmi.vexp(x, mask)
y = pto.vmi.vln(x, mask)
y = pto.vmi.vsqrt(x, mask)
```

Vector-scalar ops require a mask:

```python
y = pto.vmi.vadds(x, scalar, mask)
y = pto.vmi.vmuls(x, scalar, mask)
y = pto.vmi.vmaxs(x, scalar, mask)
y = pto.vmi.vmins(x, scalar, mask)
```

Integer/bitwise ops include `vand`, `vor`, `vxor`, `vnot`, `vshl`, `vshr`,
`vshls`, and `vshrs`.

## Compare, Select, Reductions

```python
cmp_mask = pto.vmi.vcmp(lhs, rhs, seed_mask, "lt")
cmp_mask = pto.vmi.vcmps(x, scalar, seed_mask, "ge")
out = pto.vmi.vsel(cmp_mask, true_value, false_value)
```

Reduction result types are inferred:

```python
sum1 = pto.vmi.vcadd(x, mask, reassoc=True)
max1 = pto.vmi.vcmax(x, mask)
sum_g = pto.vmi.vcadd(x, gmask, group=num_groups, reassoc=True)
```

## Conversion And Reinterpretation

```python
wide = pto.vmi.vcvt(x_f16, to_dtype=pto.f32)
narrow = pto.vmi.vcvt(x_f32, to_dtype=pto.f16, rounding="...", saturate=...)
bits = pto.vmi.vinterpret_cast(x, pto.i32)
```

Use `vcvt` for numeric conversion. Use `vinterpret_cast` only for bit-level
reinterpretation.

## Gather, Scatter, Rearrangement

```python
values = pto.vmi.vgather(src_ub, offsets, mask)
pto.vmi.vscatter(values, dst_ub, offsets, mask)
lo, hi = pto.vmi.vintlv(a, b, mask)
even, odd = pto.vmi.vdintlv(a, b, mask)
sel = pto.vmi.vselr(source, index)
```

Only use explicit VMI interleave/deinterleave when it changes the logical data
ordering. If the AscendC source uses interleave only to repair physical register
layout after widening or packing, collapse it into the logical VMI value.

## Common AscendC SIMD To VMI Patterns

- `vlds` from UB -> `pto.vmi.vload`.
- `vsts` to UB -> `pto.vmi.vstore`.
- `vcvt f16/bf16 -> f32` -> `pto.vmi.vcvt(..., to_dtype=pto.f32)`.
- `vcvt f32 -> f16/fp8` -> `pto.vmi.vcvt(..., to_dtype=<dst>)`.
- `vmul`, `vadd`, `vsub`, `vmax`, `vmin` -> corresponding VMI elementwise op.
- `vcmp` + `vsel` -> VMI compare mask plus `vsel`.
- `vintlv`/`vdintlv` trees, `PART_P*`, `PART_EVEN/ODD`, packed store modes:
  usually physical-only lowering details; collapse unless they change the
  logical result order.

## Lane Selection Heuristic

Choose the largest contiguous logical chunk that matches the algorithm and VMI
constraints:

1. Keep dtype equal to the logical element type at that stage.
2. Prefer one full physical-register worth of elements when the algorithm and
   UB layout are contiguous.
3. Use smaller multiples of 64 for row tails or natural row/group widths.
4. For dynamic remainders, use `create_mask(..., size=lanes[, group=...])` and
   keep offsets symbolic.
