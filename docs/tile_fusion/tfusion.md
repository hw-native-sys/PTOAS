# `pto.tfusion` Definition (v0.1)

##### `pto.tfusion` - Internal Elementwise Chain Fusion Op

**Summary:**  
`pto.tfusion` is an internal destination-style op used by Tile Fusion to encode a linear chain of binary elementwise ops (`tadd/tsub/tmul/tdiv`) in one operation, with optional kept intermediate outputs.

**Semantics:**

```
Given:
  ops = [op0, op1, ..., opN-1], N >= 2
  srcs size = N + 1
  prev_pos size = N - 1

Stage 0:
  s0 = apply(op0, srcs[0], srcs[1])

Stage i (i >= 1):
  one operand is s(i-1), the other is srcs[i+1]
  prev_pos[i-1] decides where s(i-1) is placed:
    prev_pos[i-1] = 0 -> si = apply(opi, s(i-1), srcs[i+1])
    prev_pos[i-1] = 1 -> si = apply(opi, srcs[i+1], s(i-1))

Outputs:
  dsts[0] stores the final stage result s(N-1)
  dsts[k+1] stores kept stage result s(keep_stage[k])
```

`ops` encoding (v0.1):
- `0 = tadd`
- `1 = tsub`
- `2 = tmul`
- `3 = tdiv`

**Arguments:**

| Name | Type | Description |
|------|------|-------------|
| `srcs` | `Variadic<PTODpsType>` | External source buffers; count must be `ops.size() + 1` |
| `dsts` | `Variadic<PTODpsType>` | Destination buffers (`dsts[0]` final output, `dsts[1..]` kept intermediate outputs) |
| `ops` | `DenseI32ArrayAttr` | Stage op code sequence |
| `prev_pos` | `DenseI32ArrayAttr` | Position of previous stage result for stage `i>=1` (`0` or `1`) |
| `keep_stage` | `DenseI32ArrayAttr` | Stage indices kept in `dsts[1..]` |
| `fusion_kind` | `StringAttr` | v0.1 fixed value: `"elemwise_chain"` |

**Results:** None. Writes into `dsts` via DPS pattern.

**Assembly Format:**

```mlir
pto.tfusion ins(<srcs> : <src_types>)
            outs(<dsts> : <dst_types>)
            ops = <dense_i32_array>,
            prev_pos = <dense_i32_array>,
            keep_stage = <dense_i32_array>,
            fusion_kind = "<string>"
```

**Constraints & Verification:**

- `ops.size() >= 2`
- `srcs.size() == ops.size() + 1`
- `prev_pos.size() == ops.size() - 1`
- `dsts.size() >= 1`
- `keep_stage.size() == dsts.size() - 1`
- `ops` values must be in `[0, 3]`
- `prev_pos` values must be in `{0, 1}`
- `keep_stage` must be strictly increasing, unique, and in `[0, ops.size()-2]`
- `fusion_kind == "elemwise_chain"`
- all `srcs/dsts` must be PTO shaped-like, with identical element type and shape
- if any stage uses `tdiv` (`op code = 3`), element type must be `f16` or `f32`

**Hardware Mapping:**

- Executes on the **Vector pipeline** (`PIPE_V`)
- Memory effects: reads `srcs`, writes `dsts`

**Basic Example:**

```mlir
// ops = [tadd, tmul, tsub]
// stage0: s0 = a + b
// stage1: s1 = s0 * c        (prev_pos[0] = 0)
// stage2: s2 = d - s1        (prev_pos[1] = 1)
// outputs: final=s2 -> out, keep s1 -> mid
pto.tfusion ins(%a, %b, %c, %d : memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>, memref<16x16xf32>)
            outs(%out, %mid : memref<16x16xf32>, memref<16x16xf32>)
            ops = [0, 2, 1],
            prev_pos = [0, 1],
            keep_stage = [1],
            fusion_kind = "elemwise_chain"
```
