# Linx CPU-sim runtime validation

Run all commands on a remote worker. The input is MLIR text with the `.pto`
extension. No Python bindings or CAPI frontend are required.

The supported profile is static rank-2 local f32 (row-major and CUBE M16/M32/N8). Each GM argument
lowers to a pair `(float *base, int32_t row_stride_elements)` in an `extern "C"`
function. The caller must provide valid buffers, positive row strides at least
as large as the column count, and disjoint output/input buffers. Dynamic valid,
shared tiles, views, in-place TADD, control flow, and other layouts/dtypes are not yet
supported. Unknown operation attributes are rejected.

Use the backend revision and flags in `config/cpu-sim.json`. Set `PTOAS_BIN` to
the compiler built from this worktree, `PTO_ISA_ROOT` to that backend checkout,
and `LINX_OUTPUT_DIR` to a separate remote output directory. From the source root:

```bash
mkdir -p "$LINX_OUTPUT_DIR"
"$PTOAS_BIN" --pto-arch=linx test/linx/Runtime/vadd/kernel.pto \
  -o "$LINX_OUTPUT_DIR/generated.cpp"
clang++ -std=c++20 -D__CPU_SIM -DPTO_CPU_SIM_TARGET_LINX \
  -O2 -ffp-contract=off -pthread \
  -I "$PTO_ISA_ROOT/include" -I "$LINX_OUTPUT_DIR" \
  test/linx/Runtime/vadd/main.cpp -o "$LINX_OUTPUT_DIR/vadd"
"$LINX_OUTPUT_DIR/vadd"
```

The runtime case checks 4096 results across contiguous and padded row strides,
unchanged input buffers, output row gaps, and the output tail guard. The JSON is
an explicit dependency/flag manifest; the commands above consume its documented
values rather than automatically parsing it.

Compiler regression cases are discovered under `test/lit/pto/linx/`, including
parse/print roundtrip, API emission, shape/type errors, capacity errors,
architecture conflicts, and reads before initialization.

For Local matrix multiplication, generate C++ from the same `.pto` input used by
lit, then compile the matrix runner with the backend flags above:

```bash
"$PTOAS_BIN" --pto-arch=linx test/lit/pto/linx/matmul.pto \
  -o "$LINX_OUTPUT_DIR/generated.cpp"
clang++ -std=c++20 -D__CPU_SIM -DPTO_CPU_SIM_TARGET_LINX \
  -O2 -ffp-contract=off -pthread \
  -I "$PTO_ISA_ROOT/include" -I "$LINX_OUTPUT_DIR" \
  test/linx/Runtime/matmul/main.cpp -o "$LINX_OUTPUT_DIR/matmul"
"$LINX_OUTPUT_DIR/matmul"
```

The matrix runner checks M16 (16x8 times 8x8), M32 (32x12 times 12x16),
and partial CELL envelopes (7x5 times 5x11) against scalar f32 multiplication.
A TADD-to-TMATMUL case also checks Left-role propagation and arithmetic.
Every case checks row strides, output gaps/tail guards and unchanged inputs.
The inputs are small binary fractions, so products and sums are exactly
representable; the absolute tolerance is 1e-6.

Softmax validation uses `test/lit/pto/linx/softmax.pto` as input and
`test/linx/Runtime/softmax/main.cpp` as the runner. Generate `generated.cpp` and
compile the runner using the same commands and backend flags as matmul.
It checks row-major 16x128, row-major 7x13, and M16 16x16 against a double
precision stable softmax reference (absolute tolerance 1e-6, row-sum tolerance
2e-6). Inputs cover mixed values, large positive offsets, constant negative
rows and a dominant element. Row gaps, output guards and input preservation
are checked for contiguous and padded strides.
