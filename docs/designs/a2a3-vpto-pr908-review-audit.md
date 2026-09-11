# PR #908 Review Comment Audit

Date: 2026-07-15

This document audits the review comments added to PR #908 against head commit
`2ca10958ae5f9adab666891d231e16a89a986701`. The findings were rechecked after
merging upstream `main` at `f0bdc1ee2` into the feature branch in merge commit
`a81518402`.

The two review rounds contain nine inline comments. Two comments repeat an
earlier finding, leaving seven unique findings. All seven identify real issues;
the PTODSL launcher finding has lower demonstrated impact than its wording
suggests. Testing after the merge also found one additional P1 integration
regression that blocked the complete A2/A3 pipeline. The current PR working
tree resolves all eight findings.

## Validation Results

Tests used `ptoas:dev-cann900-llvm-21-fork`, built from
`docker/Dockerfile.dev`, with LLVM 21, PTO Python bindings, CANN 9.0.0, and an
Ascend 910B2 device.

| Test | Result | Notes |
| --- | --- | --- |
| Full Docker build | PASS | `ptoas`, `ptobc`, and PTO Python bindings built and linked |
| Complete lit suite | PASS | 918 of 918 tests passed after the review fixes |
| Focused A2/A3 UB lit suite | PASS | 37 of 37 tests passed |
| PTODSL runtime toolchain tests | PASS | 5 of 5 tests passed; `test_jit_compile.py` also passed against the source package |
| A3 VPTO hardware E2E | PASS | 392 of 392 numerical cases passed on Ascend 910B2 |

The pre-fix 392 E2E failures had the same root cause:
`InsertTemplateAttributes` queried the PTODSL registry for A3 operations,
beginning with `pto.tload`, but the merged template registry contained only A5
entries. Gating template discovery away from the A2/A3 direct UB pipeline
restored compilation and enabled the complete passing hardware run.

## Summary

| Priority | Finding | Verdict |
| --- | --- | --- |
| P1 | A2/A3 incorrectly enters A5-only PTODSL template discovery | Resolved |
| P1 | GM view offsets and strides are discarded | Resolved; non-unit innermost strides are rejected |
| P1 | Large repeat counts are silently truncated | Resolved |
| P2 | A5 cube emission inherits the vector target CPU | Resolved |
| P2 | CANN900 lowering is globally disabled | Resolved |
| P2 | Tile valid-shape metadata is discarded | Resolved; duplicate comments |
| P2 | Full-mask restoration leaves upper lanes disabled | Resolved |
| P3 | PTODSL launcher compilation does not receive `target_arch` | Resolved; duplicate comments, impact overstated |

## Implemented Resolutions

- Skip PTODSL template discovery for the A2/A3 direct UB lowering pipeline.
- Extract composed memref offsets and physical strides for DMA, expand the
  metadata before emission, and reject unsupported dynamic or non-unit
  innermost strides.
- Preserve tile physical and valid shapes, reject dynamic valid shapes, and use
  physical row strides for partial-width unary, scalar, duplicate, shift, and
  binary lowering.
- Split large count-mode work into repeat-one operations and restore both full
  mask words on every partial/count path.
- Leave A5 `march` unset for per-kernel-kind vector/cube selection and retain
  CANN900 dispatch for non-C220 targets.
- Forward `target_arch` through native launcher compilation and validate the
  exact architecture in the production call chain.

## Findings

### P1: A2/A3 incorrectly enters A5-only PTODSL template discovery

This regression was exposed after merging `main`; it was not one of the inline
review comments against `2ca10958`.

When VPTO input contains tile operations, `compilePTOASModule` resolves PTODSL
expansion options and unconditionally schedules `InsertTemplateAttributes`
(`tools/ptoas/ptoas.cpp:2989-2993` and `3040-3048`). The later VPTO backend
pipeline explicitly lowers A2/A3 directly through `LowerPTOToUBufOps` and
returns before `ExpandTileOp` (`tools/ptoas/ptoas.cpp:2681-2695`). Template
selection is therefore unnecessary on A2/A3.

The merged lazy registry contains only `(a5, op)` entries
(`ptodsl/ptodsl/tilelib/templates/__init__.py:14-123`). It cannot answer an A2
or A3 metadata request. This caused 34 lit failures and all 392 E2E failures;
the latter consistently stopped at `NoMatchingTemplate` for `pto.tload` before
any kernel was emitted or launched.

Recommended correction:

- Do not schedule `InsertTemplateAttributes` for A2/A3 direct UB lowering.
- Avoid starting the PTODSL template daemon for A2/A3 when `ExpandTileOp` will
  not run.
- Add A2 and A3 default-backend tests that use the PTODSL default and prove the
  pipeline reaches `LowerPTOToUBufOps` without template lookup.
- Rerun all 392 hardware cases before treating prior E2E results as current.

### P1: GM view offsets and strides are discarded

Comment: [discussion_r3585509398](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585509398)

`extractDmaMemRefViewInfo` traces a subview or reinterpret cast back to its root
memref, reconstructs row-major strides from the result shape, and fills every
offset with zero (`lib/PTO/Transforms/LowerPTOToUBufOps.cpp:1183-1217`). The
normal pipeline has already converted `pto.partition_view` to `memref.subview`,
so the direct `PartitionViewOp` handling does not protect normal inputs.

A nonzero partition therefore reads or writes the wrong GM address. For
example, partitioning a contiguous `17x32xf32` view at `[1, 0]` should add 128
bytes, but current lowering computes zero. Padded and non-contiguous outer
strides are also replaced with an incorrect contiguous stride.

Recommended correction:

- Extract strided metadata from the actual memref view.
- Preserve its composed linear offset, sizes, and physical strides.
- Reject or separately lower unsupported non-unit innermost strides.
- Add `tload` and `tstore` tests with nonzero offsets and padded row strides.

### P1: Large repeat counts are silently truncated

Comment: [discussion_r3585509393](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585509393)

`modeCount1L` passes `totalRpts` directly to a UB operation
(`lib/PTO/Transforms/LowerPTOToUBufOps.cpp:1500-1508`). For a `16x1024xf32`
tile, this is 256 repeats. The LLVM emitter masks the value with `0xff`
(`lib/PTO/Transforms/VPTOCANN900LLVMEmitterUbuf.cpp`), encoding it as zero. The C220
count-mode design also requires repeat-one operations rather than a multi-repeat
count-mode instruction.

`modeCount2L` has the same latent issue because it emits `colRpts` while count
mode is active. Preserving valid-shape metadata will make that path reachable
for normal allocation-produced tiles.

Post-merge reproduction with the existing `tadd_count_large.pto` and the
legacy TileLang frontend produced `set_mask 16384` followed by
`pto.ub.vadd ... repeat=256`, then restored the mask as `(-1, 0)`. This directly
confirms both the out-of-range repeat and the mask-restoration issue. The
default-frontend lit test currently fails earlier at the template-discovery
blocker.

Recommended correction:

- Split large work into representable repeat-one chunks or loops.
- Apply the same rule to row-wise count mode.
- Verify constant repeat operands instead of silently truncating them.
- Change the existing `tadd_count_large.pto` expectation to require splitting.
- Add boundary tests for 255 and 256 repeats.

### P2: A5 cube emission inherits the vector target CPU

Comment: [discussion_r3585078492](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585078492)

`buildVPTOEmissionOptions` sets a nonempty A5 `march` to `dav-c310-vec`
(`tools/ptoas/ptoas.cpp:2726`). Device option construction treats nonempty
`march` as an explicit override, so a cube module also receives the vector CPU
instead of selecting `dav-c310-cube`. Object emission then passes that CPU to
Bisheng.

Post-merge reproduction with the A5 cube `mad_semantic_vpto_llvm.pto` emitted
both `target-cpu="dav-c310-vec"` and vector target features on the cube
functions, confirming the wrong-target path independently of object emission.

Recommended correction:

- Seed `march` only for A2/A3, leaving A5 empty so existing per-kernel-kind
  defaults select vector or cube.
- Add A5 cube and mixed vector/cube target-attribute tests.
- Add a command-level object-emission test for `dav-c310-cube`.

### P2: CANN900 lowering is globally disabled

Comment: [discussion_r3585078496](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585078496)

At the time of the review, `usesCANN900Lowering` unconditionally returned false
(`lib/PTO/Transforms/VPTOLLVMEmitterDispatcher.cpp:14-25`). That routed every
VPTO target through the Beta1 emitter, including A5 on CANN 9 releases that
previously selected the CANN900 emitter. The two emitters used materially
different intrinsic spellings and contracts. This has been resolved: Beta1
lowering is removed, while CANN900 and C220 use the official pipeline.

The complete lit run confirms this behavior with three A5/CANN 9 failures:
`a5_extra_arith_vpto_llvm.pto`, `issue220_vrelu_i32_vpto_llvm.pto`, and
`vreg_low_precision_memory_vpto_llvm.pto`. All expected CANN900 spellings but
received Beta1 spellings. For example, `issue220` expected
`llvm.hivm.vrelu.x.v64i32` and received `llvm.hivm.vrelu.v64s32.x`.

Recommended correction:

- Route A2/A3 C220 targets through the same official CANN900 emitter pipeline;
  Beta1 lowering is no longer supported.
- Preserve version-based CANN900 selection for A5.
- Test A5 CANN900 and A2/A3 C220 dispatch explicitly.

### P2: Tile valid-shape metadata is discarded

Comments:

- [discussion_r3585078499](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585078499)
- [discussion_r3585509385](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585509385)

The allocation rewrite records only `TileBufType::getShape()` before replacing
the tile with a pointer (`lib/PTO/Transforms/LowerPTOToUBufOps.cpp:229-253`).
Pointer-based shape extraction therefore has no valid shape and substitutes the
physical rows and columns. For the standard allocation path, valid-shape-aware
binary dispatch becomes unreachable and operations process padded, potentially
uninitialized UB elements.

Preserving the metadata alone is insufficient for unary, scalar, duplicate,
and decomposed operations: those helpers flatten `vRows * vCols` and do not
apply the physical row stride when `vCols != cols`.

Post-merge reproduction with the existing physical `4x96`, valid `4x32` test
produced a six-iteration loop with 64-element pointer steps and masks. The
correct valid-region lowering should process four 32-element rows with a
96-element physical row stride. This confirms that the existing test's loose
checks previously accepted the wrong dispatch path.

Recommended correction:

- Store physical shape, valid shape, and any dynamic valid dimensions in the
  pointer metadata.
- Use row-strided lowering for unary, scalar, duplicate, and decomposition
  paths whenever the valid width differs from the physical width.
- Either implement dynamic valid-shape lowering or reject it explicitly.
- Strengthen existing valid-shape tests to check mask counts, loop bounds, and
  pointer offsets rather than only operation presence.

### P2: Full-mask restoration leaves upper lanes disabled

Comment: [discussion_r3585509400](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585509400)

`fullMask` emits `mask0=-1, mask1=0`
(`lib/PTO/Transforms/LowerPTOToUBufOps.cpp:1130-1132`). The UB mask contract
states that full-write mode is `mask0=-1, mask1=-1`
(`include/PTO/IR/VPTOUbOps.td:205`). The current restoration leaves lanes
64-127 disabled for f16/i16 operations. Some dispatch branches also return
after leaving count or partial mask state without calling `fullMask`.

The large-count reproduction emitted `pto.ub.set_mask %c-1, %c0` after
returning to normal mode, directly confirming that the generated IR restores
only the lower mask word. The full f16 behavioral consequence remains untested
because the merged A3 frontend blocks all E2E cases before lowering.

Recommended correction:

- Restore both mask words to `-1`.
- Audit every dispatch exit for consistent count-mode and mask restoration.
- Add a chained partial-mask-to-full-f16 regression test covering upper lanes.

### P3: PTODSL launcher compilation does not receive target_arch

Comments:

- [discussion_r3585078502](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585078502)
- [discussion_r3585509406](https://github.com/hw-native-sys/PTOAS/pull/908#discussion_r3585509406)

The kernel object receives `module_spec.target_arch`, but `_compile_launch_cpp`
calls `_kernel_compile_flags(kernel_kind)` without it
(`ptodsl/ptodsl/_runtime/native_build.py:125-162`). The new helper consequently
uses its A5 default and compiles an A2/A3 launch translation unit with a C310
architecture option.

The mismatch is real, but the actual kernel payload is still emitted by PTOAS
for C220. `launch.cpp` contains host launch plumbing rather than a second kernel
implementation. Historical A3 end-to-end results exercised this path, but the
post-merge suite now fails before launcher compilation because of the template
blocker. This remains an incomplete/toolchain-sensitive integration rather than
a demonstrated wrong-kernel execution bug.

Recommended correction:

- Thread `target_arch` through `_compile_launch_cpp` and
  `_kernel_compile_flags`.
- Include the effective launch architecture in cache configuration.
- Test the generated Bisheng command, not only the helper function.

## Implementation Order

1. Gate PTODSL template discovery away from A2/A3 and restore lit/E2E reachability.
2. Correct DMA view metadata and add address/stride regressions.
3. Preserve valid-shape metadata and make all operation families row-aware.
4. Split invalid count-mode repeat values and add verifier coverage.
5. Restore mask state consistently.
6. Repair A5 target selection and architecture-aware CANN dispatcher behavior.
7. Thread PTODSL launcher architecture through the production call path.
