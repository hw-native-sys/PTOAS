// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

# PTO IR family ownership design

## Status

This document defines the intended ownership boundaries for PTO IR, generated
operation declarations and definitions, handwritten semantics, transforms, and
backends. The first implementation step is now implemented: family ODS files
and generated public declarations exist for PTO Base, PTO Tile, VMI, and VPTO,
while `PTO.h` remains the compatibility aggregate. `PTOCommon.h` owns
declarations shared by those families. Later family migrations follow this
contract without changing IR syntax or operation semantics.

The primary goal is to give each declaration and implementation one natural
owner. Compile-time and memory measurements may validate that the boundaries
are effective, but are not a reason to split otherwise cohesive components.

## Design principles

1. A family represents a stable semantic boundary, not an equal-sized shard.
2. PTO remains one MLIR dialect. Families do not introduce independently
   named dialects or change operation names.
3. Compatibility aggregates remain available to downstream users.
4. Project implementation files include their narrow owner headers instead of
   relying on compatibility umbrellas.
5. Generated declarations, generated definitions, registration, and
   handwritten semantics should agree on the same owner.
6. A pass is split only when it contains independently maintainable
   transactions. File size or peak compiler memory alone is insufficient.
7. Shared code becomes a component only when it has a stable contract and more
   than one real consumer. Generic `Utils` collections are not ownership
   boundaries.

## IR ownership model

```text
PTO Base
PTO Tile
  +-- Memory
  +-- Cube
  +-- Pipeline
  +-- Vector Data
  +-- Vector Elementwise
  `-- Vector Reduction

VMI
  +-- Types and layout attributes
  +-- Public semantic operations
  `-- Internal layout-assigned operations

VPTO
  +-- Control
  +-- SIMT
  +-- MTE
  +-- Cube
  +-- Vector
  `-- UB
```

### PTO Base

PTO Base owns dialect-wide infrastructure and operations that are not tied to
one Tile, VMI, or VPTO execution family. Common attributes, types, and
interfaces belong here only when they are required by multiple families.

PTO Base must not become a home for an operation merely because its final
family has not yet been identified.

### PTO Tile

Tile operations are grouped by their semantic role:

- Memory owns allocation, views, and memory-facing tile operations.
- Cube owns matrix/cube computation operations.
- Pipeline owns synchronization, communication, and pipeline-lifetime
  operations.
- Vector Data owns vector data movement and rearrangement.
- Vector Elementwise owns elementwise computation.
- Vector Reduction owns reduction computation.

Helpers shared by multiple vector families may form a narrowly named vector
verifier or semantic component. They should not be promoted into PTO Base by
default.

### VMI public and internal IR

VMI has two intentional operation surfaces with different stability:

```text
VMI public semantic IR
  -> layout assignment
  -> VMI internal layout-assigned IR
  -> VPTO
```

Public VMI operations describe user-facing vector semantics independently of
one physical layout. They are the stable authoring surface.

VMI internal operations represent the explicit intermediate form used after
layout assignment. Existing legacy operations may serve this role when they
encode the required physical grouping or layout semantics.

The following dependency rules apply:

- layout assignment and layout support may depend on VMI internal IR;
- public VMI APIs must not expose internal operation classes as their stable
  contract;
- user documentation and stable Python APIs describe the public VMI surface;
- backends do not bypass VPTO by directly consuming VMI internal IR;
- verifiers that only apply to internal operations belong to the internal
  operation owner.

### VPTO

VPTO families follow hardware-facing semantic roles. Control owns section,
container, pointer, and execution-control operations; SIMT, MTE, Cube, Vector,
and UB own their corresponding micro-operation families.

Cross-family legality is not forced into one family. A validation component
may intentionally depend on several VPTO family headers when it validates the
contract between them.

## Compatibility and internal include policy

Compatibility entry points remain supported:

```text
PTO/IR/PTO.h          all public PTO operation families
PTO/IR/PTOCommon.h    dialect-wide declarations shared by operation families
PTO/IR/VPTO.h         all VPTO operation families
PTO/IR/PTOOps.td      aggregate TableGen entry point
PTO/IR/VPTOOps.td     aggregate VPTO TableGen entry point
```

These files are public conveniences, not the default internal dependency.

- Tools, bindings, C API entry points, and downstream clients that truly need
  the full dialect may use an aggregate header.
- IR and transform implementation files include the family headers for the
  operation classes they inspect or create.
- Public component headers expose the smallest stable contract possible.
  Concrete operation family headers are included by implementation files when
  the concrete classes are an implementation detail.
- Compatibility forwarding headers may preserve old paths, but new internal
  code uses the canonical owner path.

## Generated and handwritten implementation policy

Each operation family should eventually own four aligned artifacts:

```text
family ODS source
family generated declarations
family generated definitions and registration
family handwritten semantic implementation
```

Generated definitions and registration may use separate translation units.
Registration is compile-time infrastructure and should not force unrelated
handwritten semantics into the same owner.

Custom parsers, printers, verifiers, folders, type inference, and interface
implementations belong to the semantic family that defines their contract.
Dialect-wide type or attribute behavior belongs to a type/attribute owner, not
to an arbitrary operation implementation file.

## Transform ownership

Transforms are organized by the IR contract they primarily consume or
produce:

```text
Transforms/PTO
Transforms/VMI
Transforms/VPTO
Transforms/Sync
Transforms/EmitC
```

Each family owns its pass declarations and registration. A top-level pass
header may remain as a compatibility aggregate for tools and downstream
clients. Pass implementations include only their family pass header.

Validation and analysis APIs are separate components rather than implicit
contents of pass headers. This lets pipelines invoke validation without making
all pass consumers depend on its implementation contract.

## Backend ownership

VPTO LLVM emission follows three conceptual layers:

```text
VPTO preparation
  -> target-independent emitter contract
  -> target-specific lowering implementation
```

Target implementations such as Beta1 and CANN900 own their lowering
registrations and do not include each other's implementation. Common code is
limited to stable pipeline, module-translation, dispatch, and helper contracts.

EmitC is a separate backend owner. EmitC-specific dependencies do not leak
into core PTO transforms or the VPTO LLVM backend.

## Cohesive pass exemptions

Some passes intentionally span multiple operation families because they own a
single rewrite or conversion transaction.

`VPTOExpandWrapperOps` is one such component. DMA, MTE, Cube, atomic, and SIMT
wrapper patterns participate in one wrapper-expansion contract. It remains one
pass and is not split merely to reduce translation-unit size.

Likewise, pointer normalization may jointly convert PTO pointers, function
signatures, calls, returns, and SCF region types. Those changes form one type
conversion transaction and should not be separated into independently invalid
intermediate passes.

Cross-family validation may also remain centralized when its purpose is to
validate relationships between families.

## Build target policy

Source ownership is established before library ownership. Initially, the
historical `PTOIR` and transform targets remain compatible aggregates.

Only after family declarations, registration, handwritten semantics, and
helper ownership are stable should the build introduce narrower libraries such
as `PTOBaseIR`, `PTOTileIR`, `VMIIR`, or `VPTOIR`. Narrow targets must reflect
real acyclic dependencies; they must not be created first and then justified
with artificial interfaces.

## Refactoring sequence

Follow-up changes should proceed in dependency order:

1. split family ODS and generated public declarations; **implemented by this
   PR**. Generated definitions, registration, and handwritten semantics remain
   in their historical aggregate owners for the following steps.
2. align generated definitions and registration with those families;
3. move handwritten type, attribute, and operation semantics to their owners;
4. separate VMI public and internal IR ownership;
5. align VPTO handwritten semantics with VPTO families;
6. establish family pass APIs and explicit analysis/validation components;
7. stabilize backend and driver component contracts;
8. decouple Sync public models from concrete operation families;
9. evaluate narrower CMake targets;
10. clean remaining aggregate includes as a final conformance step.

Each step should preserve compatibility aggregates, be independently
reviewable and reversible, and include focused build and regression evidence.

## Non-goals

This design does not:

- introduce new dialect names or change textual IR syntax;
- require every operation family to have the same size;
- require one source file per operation;
- require all large passes to be split;
- make compiler-memory thresholds part of IR semantics;
- immediately remove compatibility headers or historical CMake targets;
- redesign VMI layout assignment, VPTO lowering, or synchronization algorithms.

## Review checklist for follow-up changes

- Is the new owner based on semantic responsibility rather than file size?
- Do ODS, declarations, definitions, registration, and handwritten semantics
  identify the same owner?
- Does a public header expose only a stable component contract?
- Are concrete operation dependencies located in implementation files where
  possible?
- Is an aggregate header retained only for compatibility or a genuine
  full-dialect consumer?
- If a pass is split, are the resulting transactions independently valid and
  maintainable?
- If a shared helper is introduced, does it have a named model and multiple
  real consumers?
- Can the change build, regress, and be reverted independently?
