# PTOAS + VfSim Cost Model Interface Design

This document describes the current interface through which PTOAS integrates the
VfSimulator cost model as a source-level submodule. The current implementation
targets the A5 tile-fusion path: PTOAS generates legal fusion groups, VfSim uses
the selected groups to make cost-model optimization decisions, and VfSim writes
the results back to the same MLIR IR.

## Integration Model

VfSimulator is included in PTOAS as a git submodule:

```text
PTOAS/
  3rdparty/
    VfSimulator/
```

The PTOAS repository records only the submodule commit. The source history of
VfSimulator remains in the VfSimulator repository. Currently,
`PTO_ENABLE_VFSIM_COSTMODEL=ON` supports only build-tree/source-tree development;
install/export use is not yet supported.

## Control Options

| Option | Type | Purpose |
|---|---|---|
| `-DPTO_ENABLE_VFSIM_COSTMODEL=ON` | CMake configure-time option | Builds and links the native VfSim planner. Defaults to `OFF`. |
| `--enable-vfsim-costmodel-optimization` | `ptoas` runtime option | Enables VfSim cost-model optimization, which generates and annotates cost-model decision attributes. |
| `--enable-unroll-after-loop-fusion` | `ptoas` runtime option | Enables the VPTO backend unroll pass, which consumes existing `pto.fusion.row/col_unroll_factor` attributes. |
| `--dump-vfsim-unroll-test` | `ptoas` runtime debugging option | Prints the cycle prediction for each VfSim unroll candidate. It does not control whether optimization is enabled. |

A typical command is:

```bash
ptoas \
  --pto-backend=vpto \
  --pto-arch=a5 \
  --pto-level=level2 \
  --enable-op-fusion \
  --enable-vfsim-costmodel-optimization \
  --enable-unroll-after-loop-fusion \
  input.pto
```

## Build Integration

| File | Purpose |
|---|---|
| `CMakeLists.txt` | Includes `cmake/VfSimulator.cmake`. |
| `cmake/VfSimulator.cmake` | Defines `PTO_ENABLE_VFSIM_COSTMODEL`, validates the submodule, and adds `3rdparty/VfSimulator/native`. |
| `lib/PTO/Transforms/CMakeLists.txt` | Links `vfsim::native_core` and `vfsim::ir_planner` into `PTOTransforms`. |

The main native VfSim targets are:

```text
vfsim::native_core
vfsim::ir_planner
```

## Compilation Pipeline

```text
PreFusionAnalysis
  -> FusionPlan
       - PTOAS generates legal fusion groups
       - PTOAS writes pto.fusion.group_id / pto.fusion.order
       - When --enable-vfsim-costmodel-optimization is enabled, calls the VfSim planner
       - VfSim writes back pto.fusion.row_unroll_factor / pto.fusion.col_unroll_factor
  -> OpScheduling
  -> FusionRegionGen
       - Wraps each tileop group in a pto.fusion_region
       - Promotes consistent row/col unroll attributes to pto.fusion_region
  -> ExpandTileOp / Inline / shape-only fold
  -> PTOLowLevelLoopFusion
  -> PTOUnrollAfterLoopFusion
       - When --enable-unroll-after-loop-fusion is enabled, consumes the row/col unroll attributes
       - Resets a successfully consumed factor to 1
  -> FlattenFusionRegion
```

The EmitC path can run `FusionPlan` and the VfSim planner, but the unroll
attributes are currently consumed only by `PTOUnrollAfterLoopFusion` in the
VPTO backend.

## PTOAS Call Sites

| File | Symbol | Purpose |
|---|---|---|
| `include/PTO/Transforms/Passes.td` | `FusionPlan` and `FusionPlanSearch` options | Defines the `enableVfSimCostmodelOptimization` and `dumpVfSimUnrollTest` pass options. |
| `tools/ptoas/ptoas.cpp` | `enableVfSimCostmodelOptimization` | Defines the user-facing `--enable-vfsim-costmodel-optimization` option. |
| `tools/ptoas/ptoas.cpp` | `enableUnrollAfterLoopFusion` | Defines the user-facing `--enable-unroll-after-loop-fusion` option. |
| `tools/ptoas/ptoas.cpp` | `compilePTOASModule` | Passes the cost-model option to `FusionPlanOptions` and uses the unroll option to insert the VPTO backend pass. |
| `lib/PTO/Transforms/TileFusion/PTOFusionPlan.cpp` | `runVfSimFusionPlanner` | Calls `vfsim::planTileFusionIR`. |
| `lib/PTO/Transforms/TileFusion/PTOFusionPlanSearch.cpp` | `runVfSimFusionPlanner` | Calls `vfsim::planTileFusionIR` after selecting the highest-ranked fusion groups. |
| `lib/PTO/Transforms/TileFusion/PTOFusionRegionGen.cpp` | `getCommonSpanI64Attr` | Validates and promotes the row/col unroll attributes to `pto.fusion_region`. |
| `lib/PTO/Transforms/TileFusion/PTOUnrollAfterLoopFusion.cpp` | `PTOUnrollAfterLoopFusion` | Consumes the row/col unroll attributes on `pto.fusion_region`. |

## VfSim C++ API

VfSim exposes a source-level C++ API to PTOAS:

```cpp
namespace vfsim {

struct PlannerOptions {
  bool dumpCandidates = false;
  unsigned maxUnroll = 8;
};

mlir::LogicalResult planTileFusionIR(
    mlir::Operation *candidateIR,
    const PlannerOptions &options = {});

} // namespace vfsim
```

Interface contract:

| Item | Contract |
|---|---|
| Input IR | An MLIR operation after `FusionPlan` or `FusionPlanSearch`; both paths pass a `func::FuncOp`. |
| Input contents | Tileop-level IR on which PTOAS has already written `pto.fusion.group_id` and `pto.fusion.order`. |
| Output method | VfSim writes `pto.fusion.row_unroll_factor` / `pto.fusion.col_unroll_factor` in place. |
| Return value | `success()` means that the planner completed, found no processable groups, or skipped some groups after warning and falling back. `failure()` means an interface-level error. |
| Allowed modifications | The planner may write attributes only. It must not add, remove, replace, or move operations, or modify operands, results, or types. |

## Input IR

The IR passed from PTOAS to VfSim must already contain:

| Attribute | Meaning |
|---|---|
| `pto.fusion.group_id` | The fusion-group ID selected by PTOAS. |
| `pto.fusion.order` | The execution order of a tileop within the group. |

VfSim reads the following information from the IR:

| Information | Source |
|---|---|
| Group boundaries | `pto.fusion.group_id` |
| Order within a group | `pto.fusion.order` |
| Tileop type | MLIR operation name, for example `pto.tadd` |
| Data dependencies | SSA use-def chains |
| Input and output values | Tileop operands |
| Data type | Operand/result types |
| Shape | Tile-buffer types |
| Template parameters | Tileop attributes |

Example:

```mlir
pto.tadd ins(%b, %c : !pto.tile_buf<vec, 32x128xf32>,
                      !pto.tile_buf<vec, 32x128xf32>)
         outs(%a : !pto.tile_buf<vec, 32x128xf32>)
         {pto.fusion.group_id = 0 : i64,
          pto.fusion.order = 0 : i64}

pto.tmul ins(%a, %b : !pto.tile_buf<vec, 32x128xf32>,
                      !pto.tile_buf<vec, 32x128xf32>)
         outs(%d : !pto.tile_buf<vec, 32x128xf32>)
         {pto.fusion.group_id = 0 : i64,
          pto.fusion.order = 1 : i64}
```

VfSim does not reassess whether the tileops in a group may be fused. It makes
optimization decisions only for groups already selected by PTOAS.

## Output IR

VfSim outputs the same tileop-level IR and represents its optimization plan
through attributes:

| Attribute | Meaning |
|---|---|
| `pto.fusion.row_unroll_factor` | The unroll factor to use when the row loop is the actual innermost loop. |
| `pto.fusion.col_unroll_factor` | The unroll factor to use when the column loop is the actual innermost loop. |

Current unroll semantics:

| Condition | VfSim output |
|---|---|
| Column trip count is 1 | `row_unroll_factor > 1`, `col_unroll_factor = 1` |
| Column trip count is greater than 1 | `row_unroll_factor = 1`, `col_unroll_factor > 1` |

Example:

```mlir
pto.tadd ... {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.order = 0 : i64,
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
}

pto.tmul ... {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.order = 1 : i64,
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
}
```

## Attribute Promotion in RegionGen

`FusionRegionGen` wraps each group in a `pto.fusion_region` and promotes
consistent row/column unroll attributes from the tileops to the region:

```mlir
%0 = pto.fusion_region {
  pto.tadd ...
  pto.tmul ...
  pto.yield(%d) : (!pto.tile_buf<vec, 32x128xf32>) -> ()
} {
  pto.fusion.group_id = 0 : i64,
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
} : !pto.tile_buf<vec, 32x128xf32>
```

Promotion rules:

- For a given unroll attribute, either every member of a group has the
  attribute or none of them do.
- If every member has the attribute, all values must be identical.
- The factor must be a positive integer.
- An error is reported if only some members have the attribute or if their
  values differ.

## VPTO Unroll Consumption

`PTOUnrollAfterLoopFusion` runs after VPTO low-level loop fusion. It reads the
following attributes from `pto.fusion_region`:

```text
pto.fusion.row_unroll_factor
pto.fusion.col_unroll_factor
```

Consumption rules:

- Only the current innermost `scf.for` is unrolled.
- Only constant trip counts are supported, and the trip count must be divisible
  by the factor.
- Under the current convention, `col_unroll_factor` is consumed when a column
  loop exists. If the column loop has been folded, the row loop becomes the
  innermost loop and `row_unroll_factor` is consumed.
- After a factor is successfully consumed, the corresponding attribute on the
  region is reset to `1` to prevent the same factor from being applied again
  during subsequent greedy/walk processing.

Illustration:

```mlir
// Before PTOUnrollAfterLoopFusion
pto.fusion_region {
  scf.for %row = %c0 to %c32 step %c1 {
    scf.for %col = %c0 to %c2 step %c1 {
      pto.vadd ...
      pto.vmul ...
    }
  }
} {
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 2 : i64
}

// After PTOUnrollAfterLoopFusion
pto.fusion_region {
  scf.for %row = %c0 to %c32 step %c1 {
    pto.vadd ...
    pto.vmul ...
    pto.vadd ...
    pto.vmul ...
  }
} {
  pto.fusion.row_unroll_factor = 1 : i64,
  pto.fusion.col_unroll_factor = 1 : i64
}
```

## VfSim Planner Behavior

The native VfSim planner is located in `3rdparty/VfSimulator/native`.

| File | Purpose |
|---|---|
| `IRPlanner.h` | Defines `PlannerOptions` and `planTileFusionIR`. |
| `IRPlanner.cpp` | Collects fusion groups, enumerates unroll candidates, invokes the native `VfInfo` simulator, and writes the attributes back. |
| `TileOpTemplates.h/cpp` | Lowers PTOAS tileop groups to VfSim `VfInfo` micro-op programs. |
| `ParamDB.h/cpp` | Loads ISA, microarchitecture, forwarding, and initiation-interval parameters. |

Candidate enumeration rules:

- The search range is `1..maxUnroll`.
- The current default is `maxUnroll = 8`.
- Only factors that evenly divide the target loop trip count are considered.

Fallback diagnostics:

| Scenario | Behavior |
|---|---|
| Parameter-table loading fails | Prints a warning, falls back for this compilation without writing unroll attributes, and allows PTOAS compilation to continue. |
| A group contains an unsupported tileop/template | Prints a warning and skips that group. |
| Micro-op/data-type parameters are missing, or simulation fails for every candidate | Prints a warning and skips that group. |

`--dump-vfsim-unroll-test` additionally prints predicted candidate results, for
example:

```text
unroll=1 trip=2 dtype=fp32 cycles=278
unroll=2 trip=2 dtype=fp32 cycles=131
```

## Current Template Coverage

The current source-level planner primarily covers elementwise-style tileops.

| TileOp | Micro-op |
|---|---|
| `tadd` | `VADD` |
| `tsub` | `VSUB` |
| `tmul` | `VMUL` |
| `tdiv` | `VDIV` |
| `tmax` | `VMAX` |
| `tmin` | `VMIN` |
| `tabs` | `VABS` |
| `texp` | `VEXP` |
| `tadds` | `VADDS` |
| `tsubs` | `VSUB`; the scalar is converted to a vector through `VBR` |
| `tmuls` | `VMULS` |
| `tdivs` | `VDIV`; the scalar is converted to a vector through `VBR` |
| `tmaxs` | `VMAXS` |
| `tmins` | `VMINS` |
| `tcvt` | `VCVT_F16_TO_F32` or `VCVT_F32_TO_F16` |

Template-selection rules:

- Select the tileop template family based on the operation name.
- Select micro-op parameters and vector lanes based on the data type.
- Derive the row/column loop structure from the shape and valid shape.
- Keep template semantics consistent with the VPTO backend tileop templates.

## Future Direction for Hand-Written VF IR

In addition to the automatic tileop-fusion path, a future extension could allow
developers to provide VF/micro-op IR directly. This path could reuse the
source-level submodule integration, but its input would be developer-authored VF
IR rather than a tileop group.

```text
developer VF IR
  -> VfSim analyzer
  -> VF IR with cost/advice attrs or diagnostic report
```

This path would not select fusion groups. It would evaluate an existing VF
implementation and provide predictions and optimization advice.
