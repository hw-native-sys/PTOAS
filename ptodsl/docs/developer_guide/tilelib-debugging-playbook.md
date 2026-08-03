# PTODSL TileLib Debugging Playbook

This playbook is for diagnosing PTODSL TileLib failures. It converts the
original migration scratch notes into a reusable workflow.

The main rule: classify the failure before editing templates. A build failure,
a candidate-selection failure, a tracing failure, and a wrong-output failure
usually live in different layers.

## Common Commands

Build PTOAS after C++ or TableGen changes:

```bash
cmake --build build-llvm21 --target PTOASPythonCore
```

Stage PTODSL after Python package changes:

```bash
ninja -C build-llvm21 PTODSLPackage
```

Run one smoke ST:

```bash
python3 test/tilelang_st/script/run_all_st.py \
  -r sim -v a5 \
  -p build-llvm21/tools/ptoas/ptoas \
  -t <tileop> --smoke -j 1
```

Run one non-smoke ST:

```bash
python3 test/tilelang_st/script/run_all_st.py \
  -r sim -v a5 \
  -p build-llvm21/tools/ptoas/ptoas \
  -t <tileop> -j 1
```

Run one named ST case when supported by `run_st.py`:

```bash
python3 test/tilelang_st/script/run_st.py \
  -r sim -v a5 \
  -p build-llvm21/tools/ptoas/ptoas \
  -t <tileop> \
  -c <case_name>
```

## Failure Classification

| Symptom | First place to look |
|---|---|
| `NoMatchingTemplate` | PTODSL metadata, dtype signatures, layouts, memory spaces, constraints |
| custom constraints are not satisfied | predicate inputs and real ST operand form |
| no candidate survives to expansion | candidate attr insertion and attr preservation through passes |
| Python tracing error | template body mixes Python values and PTODSL runtime values |
| build succeeds but compare fails | emitted VPTO, after-expand IR, binary outputs |
| isolated case passes but full test fails | helper specialization cache or stale generated package |

Do not assume every ST failure means the testcase is wrong. First check whether
the PTODSL lowering has enough metadata to represent the testcase behavior.

## Candidate Selection Failures

For `NoMatchingTemplate` or constraint failures:

1. Read the error reason. Dtype-signature failures are usually metadata
   coverage gaps.
2. Check the real operand order in the `.pto` file.
3. Compare the PTODSL template function parameter order to that operand order.
4. Inspect dtype, layout, memory-space, valid-shape, and context attrs.
5. Add the missing version or relax the exact predicate that is too narrow.

Good fixes usually change one of:

- a `dtypes` signature;
- a `memory_spaces` or `layouts` requirement;
- a custom constraint;
- context-attr forwarding;
- callable parameter order or a split template version.

## Candidate Attr And Expansion Failures

If legality appears correct but `ExpandTileOp` cannot expand:

1. Dump after `pto-insert-template-attributes` and confirm the TileOp has a
   non-empty `candidates` attr.
2. Dump after passes that rewrite view/tile operands and confirm the attr is
   still attached.
3. Confirm candidate 0 has a `name` and that the in-process TileLib service can
   materialize that candidate directly.

Useful compiler-only command:

```bash
build-llvm21/tools/ptoas/ptoas \
  --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --enable-insert-sync \
  --mlir-print-ir-after=pto-expand-tile-op \
  --mlir-print-ir-tree-dir=/tmp/<tileop>_after_expand_ptodsl \
  test/tilelang_st/npu/a5/src/st/testcase/<tileop>/<tileop>.pto \
  -o /tmp/<tileop>_ptodsl.vpto
```

Avoid relying on module-scope IR printing unless the local `ptoas` exposes the
needed threading-control flag. Tree-dir dumps are more reliable in this setup.

## Compare Failures

When ST builds and runs but compare fails, inspect the data before changing
template code.

Recommended sequence:

1. Locate `golden.bin` and `output.bin` for the failing case.
2. Load them with the expected dtype.
3. Print the first values and reshape according to the physical destination
   shape, not only the logical valid shape.
4. Decide whether computation is wrong or only writeback/readback layout is
   wrong.
5. Dump after `ExpandTileOp` to see whether the divergence entered during
   template expansion or later lowering.

Compiler-only dump:

```bash
build-llvm21/tools/ptoas/ptoas \
  --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  --enable-insert-sync \
  test/tilelang_st/npu/a5/src/st/testcase/<tileop>/<tileop>.pto \
  -o /tmp/<tileop>_ptodsl.vpto
```

## Case Study: `trowargmin`

The full non-smoke `trowargmin` failure looked like a row-arg semantic bug at
first, but the output data told a different story.

The failing case was:

```text
uint32_float_3x8_3x3480_3x3473
```

The binary output showed:

```text
golden = [1088, 661, 176]
output first 24 = [1088, 661, 176, 0, 0, ...]
output as 3x8 first column = [1088, 0, 0]
```

That proved the row answers were computed correctly, but writeback packed them
as offsets `0, 1, 2` instead of first-column offsets `0, 8, 16`.

The next check compared TileLangDSL and PTODSL VPTO. Both backends stored the
row result in UB at the expected row stride. The divergence was the final store
helper:

- TileLangDSL used a 32-byte GM row stride.
- PTODSL reused a helper with a 4-byte GM row stride.

The after-expand IR already contained the wrong PTODSL helper, so the bug was
not later VPTO lowering. The full `.pto` file had an earlier case with the same
output tile type and a compact `3x1` destination. PTODSL helper caching keyed
view operands only by dtype/layout, so the later physical `3x8` destination
reused the compact helper.

The fix was in the specialization design, not in the row-arg template:

- include view shape in the specialization key;
- include view strides in the specialization key;
- include view memory space and layout;
- include that metadata in helper names for readable IR dumps;
- add a focused lit test with compact and strided `tstore` cases.

This pattern can affect any PTODSL template that bakes `ViewSpec` metadata into
the rendered helper body.
