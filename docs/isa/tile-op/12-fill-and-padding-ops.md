# 12. Fill and Padding Operations

> **Category:** Tile-local fill, pad, and expansion materialization
> **Pipeline:** PIPE_V

This chapter documents the unified TileLib fill / padding operation. It preserves or materializes valid data and then synthesizes the remaining destination region from the destination tile's padding policy.

The destination tile's `pad` / `pad_value` configuration determines which value is written into the synthesized padding or expansion region.

---

## 12.1 `pto.tfillpad`

- **syntax:**
```mlir
pto.tfillpad ins(%src : !pto.tile_buf<...>)
             outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** PTOAS infers normal, in-place, or expand behavior from the physical tile shapes and the addresses produced by memory planning. Users do not specify a mode.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile. |
| `dst` | `pto.tile_buf` | Destination tile carrying the pad configuration. |
| `padValue` | `#pto.pad_value<...>` (optional) | Explicit MAT `TFILLPAD<PadValue>` argument. |

**Inference Table:**

| Compiler condition | Behavior | PTO-ISA mapping |
|--------------------|----------|-----------------|
| VEC, equal physical shapes, and different or unprovable addresses | Copy valid data from `src`, then fill padding in `dst`. | `TFILLPAD(dst, src)` |
| VEC, equal physical shapes, and identical starting addresses after memory planning | Skip the copy phase and fill padding on shared storage. | `TFILLPAD<pto::TFillPadMode::InPlace>(dst, src)` |
| VEC, every `dst` physical dimension is at least the corresponding `src` dimension, and at least one is larger | Copy `src` into the larger destination and fill the expanded region. | `TFILLPAD<pto::TFillPadMode::Expand>(dst, src)` |
| Supported non-VEC form, regardless of address equality | Use the architecture's normal overload. | `TFILLPAD(dst, src)` |

**Constraints:**

- Source and destination element types must be compatible.
- The destination tile must carry a meaningful pad configuration.
- In-place and expand lowering are VEC-only. Normal lowering also supports the homogeneous MAT overload.
- Expand inference compares physical `shape`, not `valid_shape`.
- When physical shapes are equal, PTOAS compares exact starting addresses after PlanMemory. If equality cannot be proven, it conservatively chooses alias-safe normal lowering, which copies the complete valid region before writing padding.
- MAT always uses Normal lowering, including when source and destination share the same starting address.

**Example:**

```mlir
pto.tfillpad ins(%src : !pto.tile_buf<vec, 8x64xf32, valid=?x?>)
             outs(%dst : !pto.tile_buf<vec, 8x64xf32, pad=1>)

pto.tfillpad ins(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)
             outs(%tile : !pto.tile_buf<vec, 32x32xf32, pad=1>)

pto.tfillpad ins(%src_small : !pto.tile_buf<vec, 4x32xf32>)
             outs(%dst_large : !pto.tile_buf<vec, 8x64xf32, pad=1>)
```
