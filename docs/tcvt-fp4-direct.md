# A5 direct FP4 E2M1 to FP8 E4M3FN conversion

`pto.tcvt` directly converts `!pto.f4E2M1x2` to `f8E4M3FN` on A5. The destination
must be this exact type; `f8E4M3`, `f8E4M3FNUZ` and `f8E4M3B11FNUZ` are
unsupported for this direct conversion. It lowers to
`TCVT(dst, src, round, saturation)` and preserves all FP4 values, including
negative zero, for all rounding and saturation modes supported by TCVT.

Both tiles must be in `vec`. The source uses `row_major`/`none_box` with
`compact=0` (Null). The destination uses either the same ND layout with Null,
or standard NZ (`blayout=col_major`, `slayout=row_major`, `fractal=512`) with
`compact=0` or `compact=1` (Normal). RowPlusOne is unsupported.

FP4 IR columns and valid columns count **packed bytes**; generated ISA Tile
columns count **logical nibbles**. For example, an IR `32x256x!pto.f4E2M1x2`
converts to `32x512xf8E4M3FN`. Logical valid regions must agree. Physical row
and column padding may differ. Source physical columns must be a multiple of
32 packed bytes (64 logical columns); NZ physical rows must be a multiple of
16. Destination columns must also satisfy the existing Tile layout alignment.
The verifier checks the ISA byte-offset, row-pitch and VF counter limits.

Physical dimensions and packed-axis valid columns must be static. An odd
logical FP4 valid width cannot be represented by packed-byte valid columns.
Dynamic valid rows are supported; callers must supply equal runtime valid
rows within both physical bounds. NZ Normal uses the runtime valid row count
rounded up to 16 as its panel pitch; Null uses the physical destination rows.

Level 2 memory planning uses the complete physical storage shape and keeps
source and destination allocations disjoint. A 32x512 logical conversion uses
8192 bytes for packed FP4 and 16384 bytes for FP8: 24576 bytes total before
any other live buffers. No implicit temporary tile or multihop conversion is
introduced. Existing explicit temporary operands remain source-compatible
but are unnecessary for this pair. A2/A3, FP4 E1M2 and FP8 E5M2 are unsupported.

```sh
ptoas --pto-arch=a5 --pto-level=level2 input.pto -o output.cpp
```

### A5 FP4-to-FP8 padded NZ conversion

Direct E2M1x2-to-E4M3FN TCVT accepts standard Vec NZ destinations with
`compact=row_plus_one` (numeric compact tag 2). Logical Rows/Cols remain
unchanged. The source remains ND with compact=null. Padded maximum panel pitch
and byte offsets are checked in addition to ordinary direct-convert bounds.
The existing RowPlusOne memory planner handles storage: a full 128x256 FP8
destination reserves 33024 bytes after local alignment. This requires an ISA
TCVT implementation supporting RowPlusOne and truthful padded storage sizing;
it does not imply a runtime dependency pin has that support.
