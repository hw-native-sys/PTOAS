# Mix-Kernel and Mix-Backend Compile Flow

## Purpose

This document describes the current PTOAS implementation for two related
composition modes:

- **Mix-kernel**: one logical kernel is split into physical Vector and Cube
  execution units for the VPTO backend.
- **Mix-backend**: one outer PTO module contains child modules that are compiled
  by different backends — currently `emitc` and `vpto` — and then linked into a
  single final fatobj.

The implementation boundary is the MLIR child `module`. Each child module is
treated as an independent backend compile unit. PTOAS does not merge the lowered
backend IR back together; instead, it compiles each child to a fatobj and
delegates the final link to BiSheng.

## Architecture Overview

```text
                        PTODSL / PTO input
                               |
                               v
              +----------------------------------+
              | Backend-partitioned MLIR module  |
              | outer module + backend children  |
              +----------------------------------+
                               |
                               v
              +----------------------------------+
              | Resolve target arch and backend  |
              | CLI override > module attrs      |
              +----------------------------------+
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
       +-------------------+       +----------------------+
       | Single backend    |       | Mixed backend mode   |
       | job               |       | child job collection |
       +-------------------+       +----------------------+
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
             +----------------------+              +----------------------+
             | EmitC child compile  |              | VPTO child compile   |
             | unit                 |              | unit                 |
             +----------------------+              +----------------------+
                         |                                       |
                         v                                       v
             +----------------------+              +----------------------+
             | compilePTOASModule   |              | compilePTOASModule   |
             | -> CCE C++ text      |              | -> VPTO object parts |
             +----------------------+              +----------------------+
                         |                                       |
                         v                                       v
             +----------------------+              +----------------------+
             | emitFatobjCCE        |              | emitFatobjLLVM       |
             | -> child fatobj      |              | -> child fatobj      |
             +----------------------+              +----------------------+
                         |                                       |
                         +-------------------+-------------------+
                                             |
                                             v
                                  +----------------------+
                                  | linkFatobjs          |
                                  | BiSheng fatobj link  |
                                  +----------------------+
                                             |
                                             v
                                  +----------------------+
                                  | final fatobj at -o   |
                                  +----------------------+
```

For mix-kernel inside a VPTO child, the VPTO pipeline may further normalize and
split a logical kernel into physical Vector and Cube modules before object
emission:

```text
VPTO child compile unit
  -> VPTO normalization
  -> Vector/Cube section split
  -> Cube LLVM module + Vector LLVM module + optional host stub
  -> emitFatobjLLVM
```

## PTODSL Frontend Design

PTODSL is not a passive textual producer in this flow. The current frontend
encodes most of the mixed-kernel and mixed-backend structure directly in the
emitted MLIR, and PTOAS consumes that structure rather than rediscovering it
from unrelated flat functions.

### PTODSL Compile Boundary

`@pto.jit` is the PTODSL compile boundary. In the current implementation it
emits a backend-partitioned container by default — not a legacy flat
single-function module.

The key authored knobs are:

- `target="a5"`: chooses the target architecture carried by the outer module and
  child modules.
- `backend="emitc"` or `backend="vpto"`: chooses the backend recorded on the
  owning child module.
- `entry=True` or `entry=False`: distinguishes launch entries from kernel
  modules and helpers.
- `kernel_kind="vector"` or `kernel_kind="cube"`: records PTODSL authoring
  intent and native-build defaults. PTODSL does not rely on this alone to model
  mixed kernels; Vector/Cube regions are primarily expressed through section
  authoring.
- `mode="auto"` or `mode="explicit"` and `insert_sync=...`: preserved as child
  compile policy metadata for the downstream flow.

### PTODSL Expression of Mix-Backend

At the PTODSL surface level, mix-backend is expressed by composing `@pto.jit`
artifacts that use different `backend=` settings within a single trace.

The common pattern is:

1. An outer launch entry such as `@pto.jit(entry=True, backend="emitc")`.
2. One or more internal kernel modules such as
   `@pto.jit(entry=False, backend="vpto", mode="explicit")`.
3. The entry calls those kernel modules as normal Python callsites during PTODSL
   tracing.

For example:

```python
@pto.jit(target="a5", entry=False, backend="vpto", mode="explicit")
def scale_row(base_gm: pto.ptr(pto.f32, "gm"), row: pto.i32):
    with pto.tileop():
        ...

@pto.jit(target="a5", backend="emitc")
def entry(x_ptr: pto.ptr(pto.f32, "gm"), o_ptr: pto.ptr(pto.f32, "gm"), rows: pto.i32):
    ...
    scale_row(o_ptr, row)
```

PTODSL lowers this to one outer `module` with separate child modules:

- one child for the EmitC entry/caller side
- one child for the VPTO callee side

Each child carries PTOAS-facing `pto.backend` metadata. The outer container
holds shared state such as `pto.target_arch`.

### PTODSL Expression of Mix-Kernel

At the PTODSL surface level, mix-kernel is expressed as one logical kernel
program. PTODSL does not require the author to decide whether each op or code
segment belongs to the Vector section or the Cube section.

The frontend may contain ordinary tile operations, helper calls, inline scopes,
or decorated subkernels, but these are still authored in terms of the logical
kernel. The Vector/Cube execution ownership is a PTOAS responsibility:

- PTOAS infers the physical section for uncovered code from the operations it
  contains.
- PTOAS materializes the corresponding `pto.section.vector` or
  `pto.section.cube` structure during normalization.
- PTOAS later splits those sections into physical Vector/Cube compile units for
  the VPTO backend.

This keeps the PTODSL programming model independent of the physical sectioning
rules. PTODSL exposes logical helper abstractions such as `@pto.tileop` and
`with pto.tileop():`, but the design does not require users or the frontend to
manually partition every operation into a final section.

### PTODSL IR Codegen Shape

The PTODSL tracer builds a backend-partitioned outer container and allocates one
child module per owning kernel body or kernel-module callee.

The current codegen shape is:

```text
@pto.jit trace
  -> outer module { pto.target_arch = ... }
  -> child module per owning compiled unit
  -> func.func entry / helper definitions inside child modules
  -> helper calls, import declarations, and logical-name metadata
```

Important emitted invariants:

- PTODSL emits `pto.entry` on launch entries.
- PTODSL emits `pto.backend` on child modules.
- PTODSL emits `pto.ptodsl.logical_name` on ABI-specialized helper and
  kernel-module symbols so PTOAS can map specialized names back to the authored
  logical name.
- For EmitC non-entry kernel modules with an authored `kernel_kind` of `vector`
  or `cube`, PTODSL may also attach `pto.kernel_kind` to the child module and
  the primary function so the downstream EmitC path preserves the intended
  physical helper role.

### Kernel-Module Call Lowering

When an `@pto.jit(entry=True)` kernel calls an `@pto.jit(entry=False)` kernel
module, PTODSL does not inline the callee into the caller child. Instead it
builds an explicit caller/callee contract in IR.

The current lowering does all of the following:

1. Emit one public specialized definition in the callee child module.
2. Emit one private import declaration in the caller child module.
3. Emit a `func.call` in the caller body that targets the specialized callee
   symbol.
4. Attach `pto.ptodsl.logical_name` to both sides so PTOAS can reason in terms
   of the authored symbol rather than only the specialized ABI symbol.
5. Record caller-to-callee dependency metadata in the compiled PTODSL artifact.

This is why mixed-backend PTODSL IR already resembles the PTOAS child-assembly
contract: the caller owns declarations and calls, while the callee owns the
public definition.

Today this cross-module ABI is intentionally narrow. PTODSL mixed-backend
kernel-module calls are expected to stay within the currently supported
C-ABI-compatible subset — specifically GM pointers and scalar values. PTODSL
does not use Tile values as the cross-child ABI for mixed-backend kernel-module
calls.

### Subkernel Helper Lowering

Decorated and inline subkernels are also lowered explicitly rather than left as
Python-only structure. This lowering records PTODSL helper structure and call
boundaries; it does not make PTODSL responsible for the final Vector/Cube
section partition.

For `@pto.tileop` and inline `with pto.tileop():` scopes, PTODSL:

- outlines the subkernel body into a helper `func.func` when needed
- marks the helper with canonical `pto.tileop.helper`
- emits a helper call from the caller body

This is the PTODSL-side expression of a logical mixed kernel: the entry or
owning helper remains one logical function, while PTOAS later infers,
normalizes, and splits the physical Vector/Cube sections from the generated IR.

### PTODSL-to-PTOAS Handoff

The current handoff is intentionally split by responsibility:

- PTODSL is responsible for:
  - building the backend-partitioned outer container
  - choosing child-level `pto.backend`
  - emitting explicit launch-entry markers
  - emitting specialized caller/callee symbols plus
    `pto.ptodsl.logical_name`
- PTOAS is responsible for:
  - resolving the effective backend of each child
  - assembling each child compile unit
  - inferring and normalizing Vector/Cube section structure
  - splitting VPTO logical kernels into physical Vector/Cube units
  - compiling child outputs and linking the final fatobj

One practical consequence: PTODSL may leave tile paths outside explicit
sections. PTOAS normalization is expected to infer and materialize the section
structure before backend-specific compilation continues.

## Current User Contract

### Backend Selection

Backend selection is module-level. Function-level backend selection is not part
of the current contract.

The effective backend is resolved in this order:

1. `--pto-backend=emitc|vpto` forces a single backend for the whole input.
2. Without `--pto-backend`, PTOAS reads `pto.backend` on the module or child
   module.
3. A missing `pto.backend` defaults to `emitc`.

If PTOAS can resolve the input to one effective backend, it runs a single
backend job. If the outer module contains child modules with different effective
backends, PTOAS enters mixed-backend fatobj mode.

Mixed-backend fatobj mode has two user-visible restrictions:

- It requires an explicit output path, for example `-o kernel.o`.
- It rejects debug IR output modes such as `--emit-mlir`, `--emit-vpto`,
  `--pto-print-seam-ir`, and `--pto-seam-ir-file`.

### Entry Functions

PTOAS now treats entry functions as explicit. A function is recognized as a
launched entry only when it carries one of the accepted entry attributes:

- `pto.entry`
- `pto.kernel`
- legacy `hacc.entry`
- legacy `pto.aicore`

Entry functions must be definitions and must return `void`. The driver uses the
same predicate to decide whether a VPTO child should emit a host stub.

`pto.internal.non_entry` remains useful as frontend metadata for helper
functions, but current PTOAS entry detection does not infer entries from
heuristics such as "single function" or "only non-helper excluded."

## IR Shape

A mixed-backend input is an outer module with backend-selected children:

```mlir
module attributes {pto.target_arch = "a5"} {
  module attributes {pto.backend = "emitc"} {
    func.func private @vpto_post(%arg0: i64)

    func.func @entry() attributes {pto.kernel} {
      %c0 = arith.constant 0 : i64
      func.call @vpto_post(%c0) : (i64) -> ()
      return
    }
  }

  module attributes {
    pto.backend = "vpto",
    pto.kernel_kind = #pto.kernel_kind<vector>
  } {
    func.func public @vpto_post(%arg0: i64) {
      return
    }
  }
}
```

The outer module carries shared attributes such as `pto.target_arch`. When the
driver constructs a child compile unit, it copies outer attributes first, then
copies child attributes, so a child can override shared defaults.

## Driver Flow

The implemented driver flow is:

```text
input .pto / .ptobc
  -> load and verify MLIR module
  -> resolve target arch
  -> resolve backend mode
  -> run single backend job
     or
     build child jobs, compile child fatobjs, link child fatobjs
```

In single-backend mode:

- `emitc` compiles to CCE C++ text and writes that text to `-o`.
- `vpto` compiles to VPTO object components and emits a host-linkable fatobj at
  `-o`.

In mixed-backend mode:

1. PTOAS builds one backend child compile unit per child module.
2. Each EmitC child compiles to CCE C++ text, then `emitFatobjCCE` compiles that
   text to a temporary child fatobj.
3. Each VPTO child compiles to VPTO object components, then `emitFatobjLLVM`
   compiles the Cube LLVM module, Vector LLVM module, and optional host stub
   source into a temporary child fatobj.
4. `linkFatobjs` invokes BiSheng fatobj linking to produce the final `-o`
   output.

The final mixed-backend output is therefore an object/fatobj product, not an
MLIR-level merged module.

## Child Compile Unit Assembly

Before running a child backend job, PTOAS builds a detached compile unit:

1. Create a new top-level `module`.
2. Copy outer module attributes except symbol name and `pto.backend`.
3. Copy child module attributes except symbol name and `pto.backend`.
4. Clone all operations from the child body.
5. Add the cross-child declarations or helper clones needed by the child.

This gives backend passes a normal top-level module while preserving the child's
backend-specific configuration.

Set `PTOAS_DEBUG_CHILD_UNIT=1` to dump each assembled child compile unit before
backend compilation.

## Cross-Child Symbol Rules

The current implementation supports a narrow, explicit cross-child contract.

### Direct `func.call`

For each direct `func.call` in a child:

- If the callee exists in the child compile unit, no action is needed.
- Otherwise, PTOAS searches sibling child modules for exactly one public
  `func.func` definition with the same symbol name.
- If found, PTOAS clones a private declaration into the caller compile unit.
- If zero or multiple public sibling definitions are found, compilation fails.

This covers source-level calls from an EmitC entry to a VPTO helper and similar
patterns. The source should reference the logical function name, not a generated
backend ABI suffix.

### Exported Logical Wrappers

For each public non-external function in a child compile unit, PTOAS checks its
PTODSL logical name. If the logical name differs from the symbol name, PTOAS
ensures a private implementation exists at the logical name and rewrites the
public export to a small wrapper that calls the logical implementation.

Functions that carry a function-level `pto.kernel_kind` are skipped by this
wrapper rewrite.

### `pto.import_reserved_buffer peer_func`

For each `pto.import_reserved_buffer`, PTOAS resolves `peer_func` against
sibling public functions by symbol name or PTODSL logical name:

- The peer reference must resolve to exactly one sibling public function.
- PTOAS clones the peer function body into the current child as a private
  helper.
- PTOAS rewrites the `peer_func` reference to the actual cloned symbol when
  needed.
- The cloned peer function must currently be leaf-only. If it directly calls
  another function, PTOAS rejects the input instead of recursively cloning a
  transitive function closure.

This leaf-only limit is intentional in the current implementation and is covered
by regression tests.

## Mix-Kernel in the VPTO Backend

Mix-kernel is handled inside the VPTO pipeline after backend selection. The
input may contain `pto.section.vector` and `pto.section.cube` regions inside a
logical kernel. VPTO normalization and split passes turn that logical shape into
physical Vector and Cube modules.

The normalized VPTO container shape is:

```text
module {
  module { ... optional ABI-boundary entry ... }
  module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} { ... }
  module attributes {pto.kernel_kind = #pto.kernel_kind<cube>} { ... }
}
```

Important constraints:

- A physical VPTO child must carry `pto.kernel_kind`.
- At most one ABI-boundary child without `pto.kernel_kind` may exist in a
  normalized VPTO container.
- If an ABI-boundary child exists, it must pair with at least one physical child.
- Section wrappers are a frontend/backend partitioning device; physical modules
  consume the section bodies rather than preserving the wrappers as runtime
  constructs.

## Object Emission and Linking

All mixed-backend children are normalized to fatobj inputs:

```text
EmitC C++ source
  -> emitFatobjCCE
  -> child fatobj

VPTO Cube LLVM + VPTO Vector LLVM + optional stub source
  -> emitFatobjLLVM
  -> child fatobj

child fatobjs
  -> linkFatobjs
  -> final fatobj
```

For VPTO children, host stub emission depends on whether the child compile unit
contains an explicit PTO entry. If the child has no entry, PTOAS still emits a
minimal stub source so the object emission path has valid C++ stub input.

`linkFatobjs` currently delegates the final merge to the CANN/BiSheng toolchain.
Mixed-backend mode therefore always requires the toolchain environment, even if
one child backend would otherwise be able to produce text.

## Debugging Checklist

Use this order when debugging mixed compilation:

| Step | What to check | How |
|------|---------------|-----|
| 1 | Backend mode | Confirm whether `--pto-backend` forced single-backend mode. |
| 2 | Child compile units | Run with `PTOAS_DEBUG_CHILD_UNIT=1`. |
| 3 | Entry markers | Check that launched functions carry `pto.entry` or `pto.kernel` and return `void`. |
| 4 | Cross-child calls | Ensure each external callee resolves to exactly one sibling public function. |
| 5 | Reserved-buffer peers | Ensure each `peer_func` resolves to exactly one sibling public leaf function with a matching `pto.reserve_buffer`. |
| 6 | Toolchain setup | Mixed mode needs CANN/BiSheng discovery because it emits and links fatobjs. |
| 7 | Final symbols | Use `readelf -Ws` on the final object when diagnosing unresolved launch or helper symbols. |

## Known Limits

- Mixed-backend mode is object-only and does not support debug IR output flags.
- Child compile units are isolated. Cross-child helper graphs are not
  recursively imported.
- `pto.import_reserved_buffer peer_func` targets are leaf-only for now.
- Backend-partitioned container detection is structural but simple: a
  multi-child outer module can enter mixed mode when debug IR output is not
  requested and no single backend is forced.
- SIMT helper metadata may appear in frontend IR, but current mixed-kernel
  physical splitting is primarily Vector/Cube-oriented and depends on the
  backend support for the specific ops used.
- Single-backend EmitC still writes C++ text; only mixed EmitC child jobs are
  immediately compiled to child fatobjs by the driver.

## Attribute Quick Reference

| Attribute | Location | Current meaning |
|-----------|----------|-----------------|
| `pto.backend` | module / child module | Backend selector: `"emitc"` or `"vpto"`. |
| `pto.target_arch` | module / child module | Target arch, for example `"a5"`. |
| `pto.kernel_kind` | module / function | Physical VPTO unit kind: Vector or Cube. |
| `pto.entry` | `func.func` | Preferred explicit launched entry marker. |
| `pto.kernel` | `func.func` | VPTO-style explicit launched entry marker. |
| `pto.aicore` | `func.func` | Legacy entry marker accepted for compatibility. |
| `pto.internal.non_entry` | `func.func` | Frontend/helper metadata; not used for current entry inference. |
| `pto.ptodsl.logical_name` | `func.func` | Source-level logical name used when assembling wrappers and peer references. |
| `pto.tileop.helper` | `func.func` | Canonical tileop-style helper marker emitted for `@pto.tileop` helpers. |
| `pto.ptodsl.subkernel_helper` | `func.func` | Legacy helper role marker retained for compatibility with older/manual IR. |
