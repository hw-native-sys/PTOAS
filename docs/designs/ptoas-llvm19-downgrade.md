// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTOAS LLVM 19 Downgrade

## Status

PTOAS has been ported from LLVM/MLIR 21.1.8 to LLVM/MLIR 19.1.7. LLVM 19 is
the only supported dependency baseline after this migration; the source does
not attempt to support LLVM 19 and LLVM 21 simultaneously.

The validated LLVM dependency is:

- repository: `vpto-dev/llvm-project`;
- branch: `feature-vpto`;
- commit: `fa2fd1f75d742fec91ecf40d98c9c7961647ec42`;
- version: 19.1.7.

All checked-in development, CI, wheel, Docker, and installed-LLVM instructions
select `feature-vpto`. CMake rejects non-LLVM-19 SDKs with a diagnostic that
names the expected branch.

## Compatibility Policy

The downgrade is implemented in PTOAS rather than by backporting LLVM 21 MLIR
APIs into the LLVM 19 fork. This keeps the dependency branch close to its
existing `feature-vpto` contract and avoids maintaining two conversion,
EmitC, Python binding, and pass-manager implementations.

Generated textual IR and EmitC are not required to be byte-for-byte identical
to LLVM 21 output. Compatibility is judged by preserved source semantics,
types, address calculations, data flow, control-flow structure, diagnostics,
and target artifacts. Test changes must be the smallest adaptation required by
an LLVM 19 API or representation difference; they must not hide a lowering
regression or weaken an operation, region, branch, yield, or terminator
relationship that the test is intended to verify.

## Implemented Changes

### Build, CI, Docker, and Documentation

- The root CMake version gate now requires LLVM major version 19.
- README, installed-LLVM guidance, developer build instructions, Docker files,
  CI, simulator CI, and wheel workflows use `feature-vpto` and LLVM 19 cache
  identities.
- LLVM 21 cache identities are not reused for LLVM 19 builds.

### Python Bindings

LLVM 19's `AddMLIRPython` implementation is pybind11-based and does not expose
LLVM 21's nanobind domain or binding-library selection arguments. PTOAS now:

- builds `_site_initialize_0` with `PybindAdaptors.h` and
  `PYBIND11_MODULE`;
- removes `MLIR_BINDINGS_PYTHON_NB_DOMAIN` and
  `PYTHON_BINDINGS_LIBRARY` arguments;
- removes nanobind from the Python build requirements;
- retains the `ptoas.mlir` package prefix and the package-owned
  `libPTOASCompiler.so` runtime boundary.

This preserves dialect registration and prevents the generated MLIR modules
and `ptoas._core` from using unrelated MLIR runtimes.

### VMI One-to-N Conversion

`VMIToVPTO` is ported to LLVM 19's standalone one-to-N framework from
`mlir/Transforms/OneToNTypeConversion.h`. The implementation uses:

- `OneToNTypeConverter`;
- `OneToNOpConversionPattern` and `OneToNPatternRewriter`;
- explicit one-to-N signature and result mappings;
- `applyPartialOneToNConversion`.

The port retains current VMI functionality, including vector/mask expansion,
control flow, loads and stores, group-slot operations, iota, quant/dequant,
reductions, tail handling, and materialization boundaries.

LLVM 19's one-to-N driver does not guarantee LLVM 21's rewrite scheduling or
constant placement. Tests therefore use SSA-aware checks and DAG checks only
for independent operations whose relative order is not a semantic requirement.
Control-flow operations, block labels, region boundaries, yields, and
terminators remain structurally ordered checks. Values that LLVM 19
legitimately eliminates are kept live only in tests that need to inspect their
lowering.

Reduction lowering does not inspect partially converted VPTO mask producers.
It first emits the conservative physical reduction tree, then the
`vpto-combine-reductions` pass reconstructs the LLVM 21 optimized tree after
mask simplification, LICM, and CSE have stabilized VPTO producers. The combine
applies to the add, max, and min `vc*` and `vcg*` reduction families whenever
all participating reduction masks are the same SSA value or equivalent static
`pset`/`pge` patterns. Dynamic or otherwise non-equivalent masks retain the
conservative tree.

### EmitC Lowering

LLVM 19 lacks `emitc::LValueType`, `emitc::LoadOp`, operand-bearing
`emitc::VerbatimOp`, and the LLVM 21 SCF conversion overloads used by the
previous implementation. PTO-to-EmitC now uses LLVM 19 value types and
conversion adaptors directly.

The converted adaptor normally contains the target EmitC type. Most conversion
patterns therefore consume adaptor operands directly. Pipe operations carrying
GlobalTensor values are the narrow exception: when the converted operand is an
EmitC GlobalTensor opaque type, the pattern peels only that materialization to
preserve the concrete stride specialization used by the C++ pipe template.
Ordinary TileBuf operands and function signatures retain their converted
adaptor types; a GlobalTensor bridge is never lowered generically to a
C-style `emitc.cast`.

LLVM 19's FormExpressions pass runs EmitC's expression rewrite together with
generic operation folding. The generic folding can erase an expression while
the EmitC rewrite is still processing it. LLVM 21 decouples those behaviors and
runs only the expression patterns. PTOAS mirrors that behavior locally: it
wraps C-expression operations and repeatedly folds single-use, side-effect-free
EmitC expressions without invoking generic operation folding. This preserves
the LLVM 21 expression structure, including across modules containing
`emitc.if`, and avoids changing emitted C++ merely because of the downgrade.

### MLIR API Compatibility

The remaining source port covers these LLVM 19 API families:

- `applyPatternsAndFoldGreedily` and LLVM 19 greedy rewrite configuration;
- `createConvertSCFToCFPass`;
- bufferization interfaces and `getBuffer` signatures;
- memref stride/offset utilities;
- `FunctionOpInterface::insertArguments` return behavior;
- LLVM dialect inline-assembly builders;
- LLVM 19 calling-convention syntax in textual LLVM IR;
- VPTO beta.1 and CANN 9.0 LLVM emitters.

LLVM 19's Func inliner does not honor PTOAS's SIMT `no_inline` requirement in
the same way as LLVM 21. PTOAS registers a project inliner interface that
retains standard Func terminator handling while rejecting calls or callables
marked `no_inline`.

### Diagnostic Tests

LLVM 19 may stop module verification after the first invalid operation.
Verifier tests that previously placed several unrelated invalid operations in
one module have been split into one-error test files. This preserves coverage
without relying on verifier traversal or diagnostic accumulation order.

## Validation

The migration is validated in the project-local build tree
`.work/llvm19-compat-audit`, configured as a Release build with Python bindings
enabled against LLVM 19.1.7.

Built targets include:

- `PTOASCompiler` and the `ptoas` CLI;
- `PTOASPythonPackage`;
- `pto-test-opt`;
- `ptobc`.

Completed checks include:

- full LLVM lit regression suite: 1,688 passing tests and 1 unsupported test;
- all focused VMI-to-VPTO tests;
- PTO-to-EmitC and VPTO LLVM emission tests;
- Python import, PTO dialect registration, and IR construction;
- `ptobc` encode/decode followed by PTOAS parsing;
- textual A5 VPTO LLVM IR export;
- Bisheng compilation of emitted A5 LLVM IR to a relocatable device object;
- workflow YAML parsing and source formatting checks.

The object smoke test uses:

```text
--cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto
--emit-vpto-llvm-ir
```

on `test/lit/vpto/aicore_ld_st_dev_vpto_llvm.pto`. The exported textual IR is
compiled by CANN 9.0 Bisheng for `dav-c310-vec` into an ELF relocatable device
object.

## Remaining Release Validation

The source migration and local regression baseline are complete. The following
checks remain release or platform responsibilities because they require the
corresponding builders or runtime environment:

- Linux wheel build, repair, clean-environment install, and import;
- macOS wheel build, repair, install, and import;
- static-PIC LLVM 19 wheel SDK profiles;
- representative simulator or NPU runtime suites;
- PTO-Gym and broader TileLang runtime validation.

These checks should use the same `feature-vpto` dependency baseline. A failure
in a platform packaging or runtime job should not be worked around by restoring
LLVM 21-only source APIs or nanobind requirements.
