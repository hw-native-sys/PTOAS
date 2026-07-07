# 3. Kernel Entries, Kernel Modules, and Sub-Kernels

PTODSL provides one kernel decorator (`@pto.jit`) with two roles
(`entry=True` / `entry=False`), two compilation backends (`vpto` / `emitc`),
custom OP sub-kernel decorators (`@pto.tileop` and `@pto.simt`), plus
matching context managers for inline use. Legacy `@pto.cube` / `@pto.simd`
and `with pto.cube():` / `with pto.simd():` now raise migration diagnostics
that direct users to `@pto.tileop`. This chapter covers the `@pto.jit` entry
and module contracts, the two programming models, the two compilation
backends, sub-kernel reference, parameter contracts, and boundary constraints.


## 3.1 `@pto.jit` — roles, backends, and modes

Decorator overview:

```text
@pto.jit(entry=True)        host-launchable kernel entry
@pto.jit(entry=False)       kernel module, callable from entries and other modules
  backend="vpto"            VPTO backend (default) — mode="auto" or "explicit"
  backend="emitc"           EmitC backend — mode="auto" only
  mode="auto"               tile-first authoring, compiler-managed staging (default)
  mode="explicit"           micro-instruction authoring, user-managed staging

@pto.tileop                 Custom tile-op sub-kernel (vector- or cube-style body)
@pto.cube / @pto.simd       Legacy sub-kernel aliases; use @pto.tileop
@pto.simt                   SIMT-unit scalar sub-kernel
```

### Role

`@pto.jit` marks a function as a PTO kernel. Its **`entry`** parameter
selects the role:

- **`entry=True`** (the default): a host-launchable kernel entry. The public
  ABI is pointer-first — explicit GM pointers, runtime scalars, and
  keyword-only `const_expr` compile-time constants. This is the only form that
  can be compiled with `.compile(...)` and launched with `[grid, stream]`.
- **`entry=False`**: a kernel module — a device-side function with a **C ABI**.
  Parameters are `pto.ptr(...)` and PTO scalars only. `Tile`, `TensorView`, and
  `PartitionTensorView` are constructed locally inside the module body from
  the caller's raw pointers. Modules are called from entries (or other modules)
  and are not host-launchable.

The **`backend`** parameter selects the compilation target:

- `backend="vpto"` (default) compiles through the VPTO backend. Supports both
  `mode="auto"` and `mode="explicit"`.
- `backend="emitc"` compiles through the EmitC (C++ codegen) backend. Only
  supports `mode="auto"`. Using `mode="explicit"` with `backend="emitc"` is
  rejected at decoration time with an actionable diagnostic.

The **`mode`** parameter selects the programming model within the kernel body
(see Section 3.4). For native launch builds, `mode="auto"` keeps PTOAS default
build level and enables sync insertion by default, while `mode="explicit"` uses
`--pto-level=level3` and disables sync insertion by default. This matches the
manual-address, user-managed staging contract of explicit kernels.

`@pto.jit` owns compilation (tracing + lowering), caching, and — for
`entry=True` — runtime launch binding. The supported sub-kernel decorators
(`@pto.tileop` and `@pto.simt`) define helpers that are called from within
`@pto.jit` bodies. Legacy `@pto.cube` / `@pto.simd` are diagnosed at the
front-end boundary and should be migrated to `@pto.tileop`.


## 3.2 `entry=True` — host-launchable kernel entry

### Signature

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel_name","compile":{"CONST_A":128,"CONST_B":64}} -->
```python
@pto.jit(target="a5", entry=True)
def kernel_name(
    x_ptr: pto.ptr(pto.f32, "gm"),  # explicit GM pointer (positional)
    y_ptr: pto.ptr(pto.f32, "gm"),  # explicit GM pointer (positional)
    rows: pto.i32,                  # runtime metadata (positional)
    cols: pto.i32,                  # runtime metadata (positional)
    *,
    CONST_A: pto.const_expr = 128,  # compile-time constant (keyword-only)
    CONST_B: pto.const_expr = 64,   # compile-time constant (keyword-only)
):
    x_view = pto.make_tensor_view(x_ptr, shape=[rows, cols], strides=[cols, 1])
    y_view = pto.make_tensor_view(y_ptr, shape=[rows, cols], strides=[cols, 1])
    # ... tile allocation, view partitioning, and kernel logic ...
    return
```

Since `entry=True` is the default, you can omit it:
`@pto.jit(target="a5")` is equivalent to `@pto.jit(target="a5", entry=True)`.

### How to declare and pass parameters

A host-entry kernel accepts three kinds of parameters. Each has a distinct role,
position in the signature, and way to supply the value:

| Parameter kind | Position | Annotation | Pass the value at |
|---|---|---|---|
| **Device buffer** | positional (before `*`) | `pto.ptr(dtype, "gm")` | launch time |
| **Runtime scalar** | positional (before `*`) | `pto.i32`, `pto.f32`, `pto.i1`, etc. | launch time |
| **Compile-time constant** | keyword-only (after `*`) | `pto.const_expr = <default>` | compile time |

#### 1. Device-buffer parameters

Declare a positional parameter with an explicit GM pointer type such as
`pto.ptr(pto.f32, "gm")`. At launch time, pass a pointer-like value — for
example, a framework tensor with `.data_ptr()` or an integer device address:

```python
@pto.jit(target="a5")
def my_kernel(
    X_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    # Inside the body, reconstruct the GM descriptor explicitly:
    x_view = pto.make_tensor_view(X_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
```

#### 2. Runtime scalar parameters

Declare a positional parameter with a PTO scalar annotation (`pto.i32`,
`pto.f32`, `pto.i1`, etc.). At launch time, pass an ordinary Python
`int`, `float`, or `bool`:

```python
@pto.jit(target="a5")
def my_kernel(
    X_ptr: pto.ptr(pto.f32, "gm"),
    n: pto.i32,          # pass an int at launch
    alpha: pto.f32,      # pass a float at launch
):
    # Scalars arrive as PTO values and can be used directly in
    # index math, loop bounds, comparisons, and sub-kernel calls:
    limit = n // 2
```

#### 3. Compile-time constants

Declare after `*` with `pto.const_expr` and a default value.
Pass the value to `.compile(...)` — **not** at launch time:

```python
@pto.jit(target="a5")
def my_kernel(
    X_ptr: pto.ptr(pto.f32, "gm"),
    *,
    BLOCK: pto.const_expr = 128,
):
    # BLOCK is a Python value at trace time — use it for tile shapes,
    # unrolled loops, or dtype arguments:
    tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
```

The compiler specializes the kernel for each combination of constexpr values.
Once compiled, the values are baked in — they cannot change between launches of
the same compiled instance. To use a different value, call `.compile(...)` again.

### Full example: declare and launch

Bringing all three kinds together:

```python
@pto.jit(target="a5", mode="auto")
def scaled_bias_add(
    X_ptr: pto.ptr(pto.f32, "gm"),                # device buffer
    O_ptr: pto.ptr(pto.f32, "gm"),                # device buffer
    rows: pto.i32,                                 # runtime scalar
    cols: pto.i32,                                 # runtime scalar
    alpha: pto.f32,                               # runtime scalar
    bias: pto.f32,                                # runtime scalar
    *,
    BLOCK: pto.const_expr = 128,                   # compile-time constant
):
    x_view = pto.make_tensor_view(X_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])
    # ... use alpha, bias, BLOCK inside the kernel body ...
    return
```

```python
# Step 1 — compile: const_expr values go to .compile()
compiled = scaled_bias_add.compile(BLOCK=64)

# Step 2 — launch: pointers and runtime scalars go to compiled[grid, stream](...)
import numpy as np
X = np.random.randn(4, 128).astype(np.float32)
O = np.empty_like(X)
compiled[1, None](X.ctypes.data, O.ctypes.data, 4, 128, 2.0, 1.0)
```

### What is NOT accepted at the entry

The following types are intentionally **not** accepted as `entry=True` parameters:

- `pto.tensor_spec(...)` — legacy host-tensor annotations are deprecated and
  rejected everywhere.
- `Tile`, `PartitionTensorView`, `VReg` — these are created inside the kernel
  body or passed across module boundaries, not from the host.

### Compilation and launch

<!-- ptodsl-doc-test: {"mode":"launch_fragment","fixture":"launch.generic_compile_and_launch","symbol":"kernel_name"} -->
```python
import numpy as np


# Compile (traces the body, lowers through PTOAS, caches the result)
compiled = kernel_name.compile(CONST_A=128, CONST_B=64)

# Allocate or obtain concrete buffers that match the declared host ABI.
A = np.random.randn(4, 128).astype(np.float32)
O = np.empty_like(A)

# Launch on NPU
compiled[grid, stream](A.ctypes.data, O.ctypes.data, 4, 128)
```

- `.compile(**constexprs)` — traces the kernel body with the given constexpr
  values, lowers the IR, and returns a compiled handle. Subsequent calls with
  the same specialization key (function identity, entry annotation signature,
  constexpr values) hit the cache.
- `compiled[grid, stream](args...)` — launches the compiled kernel. `grid` is
  the number of SPMD blocks (an integer); `stream` is the NPU stream (`None`
  for default).

**Only `entry=True` kernels support `.compile()` and `[grid, stream]` launch.**
Calling `.compile()` on an `entry=False` module raises an error.

### Loading an existing PTO file

Use `source=` when the kernel implementation already exists as hand-written PTO
IR, but you still want PTODSL's Python compile and launch workflow. The
decorated Python function declares the host-side ABI; `source=` provides the
kernel body either as a PTO file path or as PTO text directly.

<!-- ptodsl-doc-test: {"mode":"launch_fragment","fixture":"launch.source_backed_tadd","symbol":"tadd","files":{"kernels/tadd_entry.pto":"module {\n  func.func @tadd_entry(%arg0: !pto.ptr<f32, gm>, %arg1: !pto.ptr<f32, gm>, %arg2: !pto.ptr<f32, gm>, %arg3: i32) {\n    return\n  }\n}\n"}} -->
```python
@pto.jit(
    name="tadd_entry",
    target="a5",
    backend="vpto",
    source="kernels/tadd_entry.pto",
)
def tadd(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    numel: pto.i32,
):
    # The body is ignored when source= is provided.
    pass


compiled = tadd.compile()
compiled[grid, stream](A, B, O, numel)
```

The Python function body is not traced in this form. Keep it empty, or leave a
short comment for readers. Positional parameters still matter: PTODSL uses them
to build the launch wrapper and marshal Python, NumPy, or torch-npu arguments.

The PTO source must contain one non-declaration `func.func` whose symbol matches
the JIT entry name. By default, the entry name is the Python function name. Use
`name=` when the PTO symbol has a different name, or when the source contains more
than one kernel:

```mlir
module {
  func.func @tadd_entry(
      %a: !pto.ptr<f32, gm>,
      %b: !pto.ptr<f32, gm>,
      %o: !pto.ptr<f32, gm>,
      %numel: i32) {
    // hand-written PTO body
    return
  }
}
```

PTODSL checks the selected PTO function before compiling:

- The number of PTO function arguments must match the Python positional
  parameters.
- Each argument type must match the Python annotation, position by position.
- The PTO entry must return no values.
- If the file or entry cannot be found, the diagnostic names the requested entry
  and source path.

When `source` is a filesystem path, relative paths are resolved from the Python
file that declares the decorated function, so tests can keep the Python wrapper
next to the PTO file:

```text
case.py
kernels/tadd_entry.pto
```

```python
# case.py
@pto.jit(name="tadd_entry", source="kernels/tadd_entry.pto")
def tadd(A_ptr: pto.ptr(pto.f32, "gm"), O_ptr: pto.ptr(pto.f32, "gm")):
    pass
```

For short tests, `source` can also embed the PTO text directly:

```python
tadd_source = """module {
  func.func @tadd_entry(%a: !pto.ptr<f32, gm>, %o: !pto.ptr<f32, gm>) {
    return
  }
}
"""

@pto.jit(name="tadd_entry", source=tadd_source)
def tadd(A_ptr: pto.ptr(pto.f32, "gm"), O_ptr: pto.ptr(pto.f32, "gm")):
    pass
```

Source-backed entries use the same `.compile()` and `compiled[grid, stream](...)`
launch syntax as ordinary traced entries. If the PTO file contents change,
compiling the same declaration again rebuilds the cached artifact.

Limitations:

- `source=` is only supported for launchable `entry=True` kernels.
- Keyword-only `pto.const_expr` parameters are not supported with `source=`.
  Source files are loaded as fixed PTO IR text; PTODSL does not template or
  specialize the source file.
- `.compile(...)` does not accept constexpr bindings for source-backed kernels.
- `backend=`, `mode=`, and `insert_sync=` still matter. For source-backed VPTO
  files, set `backend="vpto"`. When `mode="explicit"`, PTODSL compiles the
  source as explicit PTO; otherwise sync insertion follows the same policy as
  ordinary `@pto.jit` entries.

### SPMD built-ins

Available inside an `entry=True` body:

| Built-in | Returns | Description |
|----------|---------|-------------|
| `pto.get_block_idx()` | `int` | Index of the current block (0-based) |
| `pto.get_block_num()` | `int` | Total number of blocks in the grid |
| `pto.get_subblock_idx()` | `int` | Index of the current sub-block |
| `pto.get_subblock_num()` | `int` | Total number of sub-blocks |

### Typical body (auto mode)

```python
@pto.jit(target="a5", mode="auto")
def my_kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    b_view = pto.make_tensor_view(B_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    for row in range(0, rows, 1):
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
        b_part = pto.partition_view(b_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)
        pto.tile.add(a_tile, b_tile, o_tile)
        pto.tile.store(o_tile, o_part)
```


## 3.3 `entry=False` — kernel modules

A kernel module is an `@pto.jit(entry=False)` function that entry kernels or
other modules call directly from their traced bodies.

### Positioning

Kernel modules are the mechanism for **composing algorithm pieces with
independent compilation settings**. When a kernel grows beyond a single
function, modules let you split it along natural algorithm boundaries while
giving each piece its own `mode` and `backend`:

- An `auto`-mode entry handles the top-level orchestration (tile allocation,
  GM view partitioning, `tile.load` / `tile.store`), then calls an
  `explicit`-mode module for a hand-tuned compute kernel that needs
  micro-instruction control.
- An `emitc` entry integrates with a C++ codegen pipeline, while a
  compute-heavy `vpto` module underneath gets direct hardware access — all
  within the same compilation unit.

Modules are **not** host-launchable — you cannot call `.compile()` directly on
a module, nor invoke it with `[grid, stream]`. Instead, the module is compiled
and linked together with its callers automatically.

### Compilation boundary

Each kernel module is compiled into a **real function boundary** in the
generated MLIR. When a traced body calls a module, PTODSL:

1. Creates a separate child MLIR module for the callee.
2. Compiles it through its own backend pipeline — `vpto` or `emitc`, chosen
   independently from the caller.
3. Emits a `func.func` → `func.call` pair to wire the call site.

This means modules are **not inlined or macro-expanded** — they are genuine
function calls with a defined ABI. Each module carries its own compilation
attributes (`pto.backend`, mode policy) and is linked into the final binary
by the PTOAS linker. The caller never needs to manage this: calling a module
function from a traced body is all it takes to record the dependency and
trigger automatic compilation and linking.

**Backend default**: if a module does not specify `backend=`, it defaults to
`"vpto"` — independently of the caller's backend. An `emitc` entry calling a
bare `@pto.jit(entry=False)` module will compile the entry through EmitC and
the module through VPTO. To keep the same backend, set it explicitly on both.

If you need to combine kernels that were compiled separately (e.g., a
pre-compiled module reused across multiple entries), use
`pto.merge_jit_modules()` — see Section 3.6.

### Module body: control flow and vector operations

Module bodies follow the same AST rewrite rules as `@pto.jit(entry=True)`
(see Chapter 5). In the default `mode="auto"`, Python `for` / `if` are
rewritten to device-side control flow, and the compiler handles hardware
section placement automatically — you can write `vlds` / `vadd` / `vsts`
directly in the module body without an explicit `with pto.tileop():`. In
`mode="explicit"`, you must manage hardware sections yourself with
`with pto.tileop():` or `with pto.simt():`.

### Interface protocol

Modules are compiled to real function boundaries with a **C ABI**. Only
C-compatible types can cross the boundary:

```python
@pto.jit(entry=False)
def my_module(
    gm_in: pto.ptr(pto.f32, "gm"),               # GM pointer
    ub_buf: pto.ptr(pto.f32, pto.MemorySpace.UB), # UB pointer
    rows: pto.i32,                                 # PTO scalar
    cols: pto.i32,                                 # PTO scalar
):
    # Module body: construct tiles locally from raw pointers,
    # then operate on them with tile ops / sub-kernels.
    in_tile = pto.alloc_tile(shape=[1, cols], dtype=pto.f32, addr=ub_buf)
    # ...
    return
```

The module ABI accepts:

| Type | Description |
|------|-------------|
| `pto.ptr(dtype, "gm")` | Typed GM pointer |
| `pto.ptr(dtype, space)` | Typed pointer in any memory space (UB, L1, L0A, L0B, ...) |
| PTO scalar (`pto.i32`, `pto.f32`, ...) | Device-side scalar value |

**`Tile`, `TensorView`, and `PartitionTensorView` cannot cross the module
boundary.** These are complex data structures (carrying shape, strides, and
memory space metadata) that do not map to the C ABI. Modules allocate their
own tiles internally and receive raw pointers from the caller.

### Protocol constraints

The following are intentionally **not** supported across the module boundary:

- `pto.const_expr` — compile-time constants belong at the `entry=True` boundary.
  If a module needs a compile-time-known value, pass it as a PTO scalar from
  the caller (where it may originate from a `const_expr`).
- Return values — module functions must return `None`. Data crosses the module
  boundary only through mutable references (tiles, pointers). VReg and mask
  values cannot escape the module boundary.
- Host launch — calling `.compile()` or `[grid, stream]` on an `entry=False`
  module raises an error. Modules are compiled automatically as dependencies
  when their calling entry kernel is compiled.
- **Complex types**: `Tile`, `TensorView`, and `PartitionTensorView` cannot
  be module parameters — see "Interface protocol" above for the C ABI rule.

**C ABI for all module boundaries.** Because every module is compiled to a
real `func.func` → `func.call` boundary, the parameter passing convention is
always the C ABI — regardless of whether the caller and callee share the
same backend. Only `pto.ptr` and PTO scalars can cross the module boundary.
Passing `Tile`, `TensorView`, or `PartitionTensorView` as module parameters
is not supported.

### Calling modules from entries and other modules

A module is invoked with a normal Python function call inside a traced body.
At compile time, PTODSL records a `func.call` into the module's compiled
function — the call is a real function invocation, not an inline or macro
expansion. Argument types must match the module's declared C ABI (pointers
and scalars — see Interface protocol above). The caller passes raw pointers
(via `tile.as_ptr()`) and shape scalars; the module constructs its local
tiles from them:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"my_kernel","compile":{"BLOCK":128}} -->
```python
@pto.jit(entry=False)
def process_tile(
    a_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),
    b_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),
    o_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),
    rows: pto.i32,
    cols: pto.i32,
):
    VEC = pto.elements_per_vreg(pto.f32)
    for r in range(0, rows, 1):
        remained = cols
        for c in range(0, cols, VEC):
            mask, remained = pto.make_mask(pto.f32, remained)
            a_vec = pto.vlds(a_ptr, c)
            b_vec = pto.vlds(b_ptr, c)
            o_vec = pto.vadd(a_vec, b_vec, mask)
            pto.vsts(o_vec, o_ptr, c, mask)


@pto.jit(target="a5", entry=True)
def my_kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    B_ptr: pto.ptr(pto.f32, "gm"),
    O_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    a_view = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    b_view = pto.make_tensor_view(B_ptr, shape=[rows, cols], strides=[cols, 1])
    o_view = pto.make_tensor_view(O_ptr, shape=[rows, cols], strides=[cols, 1])

    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    b_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    for row in range(0, rows, 1):
        a_part = pto.partition_view(a_view, offsets=[row, 0], sizes=[1, cols])
        b_part = pto.partition_view(b_view, offsets=[row, 0], sizes=[1, cols])
        o_part = pto.partition_view(o_view, offsets=[row, 0], sizes=[1, cols])

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)

        process_tile(a_tile.as_ptr(), b_tile.as_ptr(), o_tile.as_ptr(), 1, cols)

        pto.tile.store(o_tile, o_part)
```

When a traced body calls a module, PTODSL automatically records the
dependency and lowers it to a `func.call` into the module's compiled function.
You don't need to register modules or declare imports — just call the
function. Each module is compiled through its own backend pipeline as a
separate child MLIR module, then linked together with the caller by the PTOAS
linker. Modules can call other modules; the same automatic wiring and
per-function compilation applies transitively.


### Plain helpers vs sub-kernels

PTODSL exposes one public launch boundary: `@pto.jit`. Inside that entry,
there are two kinds of helpers:

- **Plain Python helpers** for code organization, repeated index math,
  partition construction, and orchestration that should stay in the caller's
  context.
- **Sub-kernels** (`@pto.tileop`, `@pto.simt`) when the helper must
  run on a specific hardware unit or use unit-local value categories such as
  `vreg` or cube-local scratch.

Use a plain helper when the code should not introduce a new hardware-unit
boundary. Plain helpers do not define a separate ABI, target, mode, or
backend. They are traced as part of the enclosing `@pto.jit` specialization
and therefore inherit the caller's context.

Use a sub-kernel when the helper's semantics belong to a specific unit:
vector register math or cube instructions via `@pto.tileop`, or scalar-thread
work via `@pto.simt`. Sub-kernels are the only public way to express that
boundary.

Named sub-kernels and plain nested helpers both use the same default AST
rewrite behavior when they are traced from a compiled specialization.

Sub-kernels are the mechanism for custom compute in PTODSL — when Tile Ops
cover your needs, you don't need one; when they don't, a sub-kernel gives you
direct access to the hardware unit. In auto mode, a sub-kernel's parameters
follow the decorator's role-specific ABI — the compiler still owns staging and
sync for tileop-style helpers. `@pto.tileop` accepts `Tile`, `TensorView`,
`PartitionTensorView`, and PTO scalar parameters. `@pto.simt` additionally
accepts typed `pto.ptr(...)` parameters. These richer in-kernel boundary types
do not change the public `@pto.jit` host entry ABI. Section 3.3 covers each
sub-kernel decorator in detail.

### Module vs sub-kernel

**Module or sub-kernel?** A simple rule:
- Logic that **must run as a custom tile op or SIMT helper**
  and operates on tiles → use a sub-kernel (`@pto.tileop` or `@pto.simt`).
- General device-side code organisation — allocating tiles, partitioning GM
  views, calling sub-kernels, mixing backends → use a kernel module
  (`@pto.jit(entry=False)`).

Modules **can** call sub-kernels (they are callable from both entries and
modules). Sub-kernels **cannot** call modules — data crosses the sub-kernel
boundary only through UB tiles, not through nested function calls.

| | `@pto.jit(entry=False)` module | `@pto.tileop` / `@pto.simt` |
|---|---|---|
| Positioning | General device-side function | **Custom tile op** — hardware-bound compute primitive |
| Scope | Orchestration, tile allocation, data movement, sub-kernel dispatch | Single-hardware-unit compute logic |
| ABI | **C ABI: ptr + PTO scalars only**. Tile/TensorView/PartitionTensorView cannot cross the function boundary. Caller passes `tile.as_ptr()`; module constructs local tiles internally | Role-specific PTODSL helper ABI. `@pto.tileop`: `Tile`, `TensorView`, `PartitionTensorView`, PTO scalars. `@pto.simt`: `Tile`, typed `pto.ptr(...)`, PTO scalars |
| Backend | VPTO or EmitC | Always VPTO |
| Compilation | Compiled as a separate child module, linked automatically | Outlined as a helper function inside the owning caller/module |
| Callable from | Entries and other modules | Entries and modules |
| Can call modules | Yes | No (data crosses boundary only through UB tiles) |
| Can call sub-kernels | Yes | No (sub-kernels cannot nest) |


## 3.4 Programming models: auto vs explicit

The `mode` you choose for a kernel and the `mode` you choose for its modules
are independent. An `entry=True, mode="auto"` kernel can call an
`entry=False, mode="explicit"` module — this is a common pattern where the
entry stays at tile level for simplicity, and the module uses micro-instructions
for hand-tuned compute.

### `mode="auto"` — tile-centric

In auto mode you think in tiles. You allocate tiles, partition GM views, move
data with `tile.load` and `tile.store`, compute with Tile Ops like
`tile.add` and `tile.exp`, and call sub-kernels for hardware-specific compute.
The compiler handles the lowering of tiles to micro-instructions: inferring
staging, inserting synchronization between Tile Ops and sub-kernels, and
managing tile-level scheduling.

Use auto mode for the majority of kernels. It gives you the full performance
of the NPU without requiring you to reason about instruction-level ordering.

### `mode="explicit"` — tile + micro-instruction

Explicit mode extends auto mode with direct micro-instruction access. You keep
everything available in auto — tiles, Tile Ops, sub-kernels — and additionally
gain access to MTE ops, explicit synchronization, and pointer manipulation.
When you need precise control over individual instructions and phase ordering,
you can drop below the tile abstraction without leaving the `@pto.jit` entry.

Explicit mode broadens the orchestration code you can write inside `@pto.jit`
and `@pto.jit(entry=False)` bodies. Sub-kernel ABIs themselves remain
role-specific rather than mode-specific: `@pto.tileop` accepts `Tile`,
`TensorView`, `PartitionTensorView`, and PTO scalars, while `@pto.simt`
accepts `Tile`, typed `pto.ptr(...)`, and PTO scalars.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.explicit_signature","symbol":"kernel_entry_explicit_signature_probe","compile":{"BLOCK":16}} -->
```python
@pto.jit(entry=False, mode="explicit")
def my_orchestration_helper(
    gm_ptr: pto.ptr(pto.f32, "gm"),               # GM pointers
    ub_ptr: pto.ptr(pto.f32, pto.MemorySpace.UB), # typed UB pointers
    l0a_ptr: pto.ptr(pto.f16, pto.MemorySpace.LEFT),  # cube-local pointers
    rows: pto.i32,                                 # PTO scalar values
    cols: pto.i32,
):
    return
```

**Typical pattern**: GM↔UB movement uses ptr-based `mte_load`/`mte_store`
rather than `tile.load`/`tile.store`. The user places `pipe_barrier` at phase
boundaries and explicitly sequences sub-kernel calls:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.explicit_body","symbol":"kernel_entry_explicit_body_probe","compile":{"ROWS":8,"COLS":16}} -->
```python
def process_block(q_tile, k_part, v_part, k_tile, v_tile,
                  s_tile, o_tile, o_part, rows: pto.i32, cols: pto.i32):
    in_row_bytes = cols * pto.bytewidth(pto.f16)
    out_row_bytes = cols * pto.bytewidth(pto.f32)
    gm_row_stride = k_part.strides[0] * pto.bytewidth(pto.f16)
    ub_row_stride = k_tile.shape[1] * pto.bytewidth(pto.f16)

    # Stage current block from GM to UB
    pto.mte_load(k_part.as_ptr(), k_tile.as_ptr(), 0, in_row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.mte_load(v_part.as_ptr(), v_tile.as_ptr(), 0, in_row_bytes,
                 nburst=(rows, gm_row_stride, ub_row_stride))
    pto.pipe_barrier(pto.Pipe.ALL)

    # Dispatch sub-kernels
    qk_matmul(q_tile, k_tile, s_tile)
    pto.pipe_barrier(pto.Pipe.ALL)

    online_softmax(s_tile, o_tile, rows, cols)
    pto.pipe_barrier(pto.Pipe.ALL)

    # Write result back
    pto.mte_store(o_tile.as_ptr(), o_part.as_ptr(), out_row_bytes,
                  nburst=(rows, ub_row_stride, gm_row_stride))
```

Sub-kernel calls and inline sub-kernel scopes (`with pto.tileop():`, etc.) work
identically in both modes.

### Choosing between modes

| | `mode="auto"` | `mode="explicit"` |
|---|---|---|
| Abstraction | Tiles | Tiles + micro-instructions |
| Data movement | `tile.load` / `tile.store` | `mte_load` / `mte_store` (ptr-based) |
| Sync | Compiler-managed | User-authored |
| Use case | Most kernels | Hand-tuned instruction scheduling |

Start with auto. Move to explicit when you need to control the exact sequence
of micro-instructions — for example, to overlap DMA and compute with
double-buffering, or to hand-optimize a phase boundary that the compiler
doesn't fuse as aggressively as you need.


## 3.5 `backend`: VPTO vs EmitC

`backend` chooses which compiler pipeline builds your kernel:

- **`backend="vpto"`** (the default) is the native VPTO path. It works with
  both `mode="auto"` and `mode="explicit"`. Use this unless you specifically
  need C++ codegen.
- **`backend="emitc"`** generates C++ through the EmitC pipeline. It only
  works with `mode="auto"` — if you try `mode="explicit"` with it, PTODSL
  raises an error at decoration time.

### Choosing a backend

| | `backend="vpto"` | `backend="emitc"` |
|---|---|---|
| Supported modes | `auto`, `explicit` | `auto` only |
| Typical use | Most kernels (default) | C++ codegen integration |

### Mixing backends with modules

You can use different backends for different functions within the same
compilation. Because each module is compiled as an independent function
boundary with its own backend pipeline, the caller's backend does not
constrain the callee. Common patterns:

- **Pure VPTO**: entry and all modules use `backend="vpto"` (the default).
- **EmitC entry + VPTO modules**: the entry uses `backend="emitc"` for C++
  codegen, while compute-heavy modules use `backend="vpto"` for direct
  hardware access.
- **Same-backend modules**: VPTO→VPTO or EmitC→EmitC — modules are compiled
  as separate units and linked together automatically.

The compiler resolves cross-backend calls automatically — when an `emitc`
entry calls a `vpto` module, the linker wires them together into the final
binary. You just write the Python call; the toolchain handles the rest.

### Mixed-backend: compilation model and current limitations

Under the hood, PTODSL emits a **backend-partitioned outer container**:
a top-level `builtin.module` that holds per-function child modules, each
tagged with a `pto.backend` attribute (`"vpto"` or `"emitc"`). The PTOAS
driver compiles each child module through its respective pipeline, then links
the results into a single fat binary.

The container shape follows these invariants:

- **Outer module**: holds child modules and shared declarations (type
  aliases, external symbols). It does not contain executable code directly.
- **Child modules**: each carries a `pto.backend` attribute and contains
  exactly one backend-specific compilation unit — the traced function body
  lowered through the designated pipeline.
- **Same-backend children**: same-backend multi-child containers (e.g.,
  VPTO→VPTO or EmitC→EmitC) are first-class supported and follow the same
  compilation path as single-child containers. No special flags needed.

Current driver-level limitations to be aware of:

- **Output path required**: mixed-backend compilation requires an explicit
  `-o` output path. The driver writes the linked fat binary there.
- **Debug IR output**: dump/debug IR output flags (e.g., `--emit-pto-ir`,
  `--emit-llvm-ir`) are not yet supported in mixed-backend mode. Use
  `compiled.mlir_text()` in PTODSL to inspect the pre-lowering MLIR, or
  compile each function in isolation for IR debugging.
- **Linker scope**: the fat-object linker currently operates on a single
  compilation unit (one entry + its transitively called modules). Linking
  across separately compiled entry kernels requires `merge_jit_modules()`
  (Section 3.6).


## 3.6 Combining compiled kernels with `merge_jit_modules()`

When you compile an `entry=True` kernel that calls modules, everything is
bundled together automatically. Sometimes you need to combine kernels that
were compiled separately — for example, a pre-compiled module that you want
to reuse across multiple entry kernels without recompiling it each time.

`pto.merge_jit_modules()` merges multiple compiled handles into one module:

```python
from ptodsl import pto

merged = pto.merge_jit_modules(
    entry_kernel,
    module_a,
    module_b,
)
# merged contains all three functions, linked and ready
```

All handles in the merge must share the same target architecture. Backend and
mode can differ per handle — the merge appends the functions together and
preserves each one's compilation settings.


## 3.7 Sub-kernels — custom tile operations

Sub-kernels are the mechanism for authoring **custom tile-level operations**
in PTODSL. While the built-in Tile Ops (`tile.load`, `tile.store`,
`tile.add`, `tile.exp`, etc.) cover common patterns, sub-kernels let you
write operations that map directly to a specific NPU compute unit when the
built-in ops don't cover your needs.

**Sub-kernels are custom tile ops.** Their contract is strict:

- **Inputs**: role-specific PTODSL boundary values. `@pto.tileop` accepts
  `Tile`, `TensorView`, `PartitionTensorView`, and PTO scalars (`pto.i32`,
  `pto.f32`, ...). `@pto.simt` accepts `Tile`, typed `pto.ptr(...)`, and PTO
  scalars. Data still flows through device-side boundary objects; sub-kernels
  do not define a host-visible ABI.
- **Outputs**: written back to UB tiles. Mutable `Tile` parameters remain the
  primary output path. In the current MVP, decorated tileop-style helpers may
  additionally return PTO scalar values through `func.call` results, but
  `Tile`, `TensorView`, `PartitionTensorView`, `vreg`, and mask values still do
  not cross the helper boundary as results.
- **No cross-boundary vreg**: vector registers (`vreg`) and cube-local state
  (LEFT, RIGHT, ACC) are private to the sub-kernel body and never escape.

When to use a sub-kernel vs a kernel module:
- If the logic **must execute on a specific hardware unit** (Cube, SIMD, or
  SIMT) and operates on tiles → use a sub-kernel.
- If you need to orchestrate data movement, allocate tiles, partition GM
  views, or mix backends → use an `@pto.jit(entry=False)` kernel module
  instead. Modules can call sub-kernels, but sub-kernels cannot call modules.

Sub-kernels are decorated with `@pto.tileop` or `@pto.simt`. PTODSL lowers
them to real helper `func.func` bodies instead of flattening them directly
into the surrounding caller. Legacy `@pto.cube` / `@pto.simd` and inline
`with pto.cube():` / `with pto.simd():` are removed public surfaces; PTODSL
diagnoses them and directs users to `@pto.tileop`. Sub-kernels can be authored
in two ways:

1. **As decorated functions** — reusable, named sub-kernels called from
   `@pto.jit` entries and modules.
2. **As context managers** (`with pto.tileop():`, etc.) — inline blocks for
   one-off snippets (see Section 3.8).

Named sub-kernel decorators use the same default AST rewrite model as
`@pto.jit`: supported Python `if` and `for range(...)` statements lower to
device-side control flow.

### 3.7.1 Cube-style `@pto.tileop` — matrix operations

**Role**: `@pto.tileop` is also the custom tile-op surface for matrix
multiplication on the Cube unit. It consumes UB-resident tiles and explicit
cube-local scratch buffers. All parameters are `Tile` references — the caller
owns tile allocation, and the sub-kernel only expresses the compute dataflow.

**Signature**: `@pto.tileop(fn=None, *, name=None, target="a5")`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.cube_signature","symbol":"kernel_entry_cube_signature_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
@pto.tileop
def my_cube_kernel(
    input_tile: pto.Tile,            # UB tile (source data)
    output_tile: pto.Tile,           # UB tile (destination)
    left_scratch: pto.Tile,          # LEFT buffer (cube-local)
    right_scratch: pto.Tile,         # RIGHT buffer (cube-local)
    acc_scratch: pto.Tile,           # ACC buffer (cube-local)
):
    return
```

All parameters are `Tile` references. Tiles marked as cube-local must be
allocated with the appropriate `memory_space` (e.g., `pto.MemorySpace.LEFT`,
`pto.MemorySpace.ACC`).

**Typical body**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.cube_helper","symbol":"data_movement_cube_helper_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
@pto.tileop
def qk_matmul(
    q_tile: pto.Tile,
    k_tile: pto.Tile,
    q_l0a: pto.Tile,
    k_l0b: pto.Tile,
    s_acc: pto.Tile,
    s_tile: pto.Tile,
):
    m = q_tile.valid_shape[0]
    k = q_tile.valid_shape[1]
    n = k_tile.valid_shape[1]

    pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
```

Cube-local state (LEFT, RIGHT, ACC, BIAS) never leaks into UB — it is the
caller's responsibility to allocate scratch buffers and pass them in
explicitly.

**Lowering model**: a decorated cube-style `@pto.tileop` function becomes one
reusable helper function inside the owning PTODSL child module. Each callsite
lowers to `func.call` of that helper. The helper body stays as a naked
`tileop` helper; backend passes infer the primary domain and materialize
`pto.section.cube` later in the PTOAS pipeline.

**Invocation modes**: can be called from `@pto.jit` in either mode, or authored
as an anonymous inline helper with `with pto.tileop():` (Section 3.8).

### 3.7.2 `@pto.tileop` — custom tile op (vector operations)

**Role**: `@pto.tileop` is the recommended custom tile-op surface for
row-wise vector compute. It operates on vector registers (`vreg`) loaded from UB tiles
and stores results back to UB tiles. Parameters may be `Tile`,
`TensorView`, `PartitionTensorView`, and PTO scalars — the sub-kernel reads
device-side descriptors, computes on vector hardware, and writes results back
through mutable tile parameters. Vector registers are local to the function
and never cross its boundary.

**Signature**: `@pto.tileop(fn=None, *, name=None, target="a5")`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.tileop_signature","symbol":"kernel_entry_simd_signature_probe","compile":{"BLOCK":128}} -->
```python
@pto.tileop
def my_tileop_kernel(
    input_tile: pto.Tile,            # UB tile
    output_tile: pto.Tile,           # UB tile
    rows: pto.i32,                   # PTO scalar
    cols: pto.i32,                   # PTO scalar
):
    return
```

Parameters are device-side `Tile` / `TensorView` / `PartitionTensorView`
references and PTO scalar values (`pto.i32`, `pto.f32`, etc.). Scalar
parameters may come from `lds` reads or compile-time constants.

This interface contract is enforced unconditionally. A decorated `@pto.tileop`
function does not accept `pto.ptr(...)`; typed pointers remain SIMT-only. If
you need a broader C-style ABI, use `@pto.jit(entry=False)` instead.

**Typical body**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.tileop_body","symbol":"kernel_entry_simd_body_probe","compile":{"BLOCK":128}} -->
```python
@pto.tileop
def add_rows(a_tile: pto.Tile, b_tile: pto.Tile, o_tile: pto.Tile,
             rows: pto.i32, cols: pto.i32):
    VEC = pto.elements_per_vreg(pto.f32)
    for r in range(0, rows, 1):
        remained = cols
        for c in range(0, cols, VEC):
            mask, remained = pto.make_mask(pto.f32, remained)
            a_vec = pto.vlds(a_tile[r, c:])
            b_vec = pto.vlds(b_tile[r, c:])
            o_vec = pto.vadd(a_vec, b_vec, mask)
            pto.vsts(o_vec, o_tile[r, c:], mask)
```

The boundary contract: `vreg` values (`a_vec`, `b_vec`, `o_vec`) are local to
the function. The only way to persist data across a `@pto.tileop` call is to
write it back to a UB tile via `vsts` (or `psts`, etc.).

**Lowering model**: a decorated `@pto.tileop` function becomes one reusable
helper function inside the owning PTODSL child module. Each callsite lowers to
`func.call` of that helper. The helper body is emitted as a naked `tileop`
helper with `pto.tileop.helper`; backend passes infer the
primary domain and materialize `pto.section.vector` later in the PTOAS
pipeline.

**Invocation modes**: can be called from `@pto.jit` in either mode, or authored
as an anonymous inline helper with `with pto.tileop():` (Section 3.8).

PTODSL no longer exposes a separate `@pto.simd` public surface. Vector-style
helpers use `@pto.tileop`, and the backend infers `pto.section.vector` from
the helper body later in the PTOAS pipeline.

### 3.7.3 `@pto.simt` — SIMT unit (scalar-parallel operations)

**Role**: `@pto.simt` is the SIMT helper surface for per-element scalar-parallel
compute. SIMT (Single Instruction, Multiple Threads) is a
programming model where you write instructions in scalar syntax
(`scalar.load`, `scalar.store`, `a + b`), and the hardware executes them in
parallel across many threads — analogous to how a GPU SM runs a CUDA kernel.
Parameters are `Tile` references, typed UB pointers, and PTO scalars. The
sub-kernel reads and writes individual elements through tile handles; results
flow back to the caller via mutable tile parameters.

**Signature**: `@pto.simt(fn=None, *, name=None, target="a5", max_threads=None, max_regs=None)`

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.simt_signature","symbol":"kernel_entry_simt_signature_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def my_simt_kernel(
    tile: pto.Tile,                  # UB tile
    ptr: pto.ptr(pto.f32, pto.MemorySpace.UB),  # typed UB pointer
    scalar_value: pto.i32,           # PTO scalar
):
    return
```

**Typical body**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_blend","symbol":"flash_attention_simt_blend_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile, pv_tile: pto.Tile,
    alpha_tile: pto.Tile, beta_tile: pto.Tile,
    o_next_tile: pto.Tile,
    row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
):
    for row in range(row_start, row_stop, 1):
        alpha = scalar.load(alpha_tile[row, 0])
        beta = scalar.load(beta_tile[row, 0])
        for col in range(0, valid_dim, 1):
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[row, col])
```

SIMT kernels read and write individual scalar elements from tiles or typed
pointers. The unit executes the same scalar instruction across many work-items
in parallel, making it efficient for per-element operations.

#### SIMT resource attributes

Optional `max_threads` and `max_regs` arguments attach VPTO resource attributes
to the generated `pto.simt_entry` helper.

**Signature**: `@pto.simt(fn=None, *, name=None, target="a5", max_threads=None, max_regs=None)`

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_threads` | positive Python `int` | backend default `1024` | Compile-time launch envelope for this SIMT helper |
| `max_regs` | positive Python `int` | backend default `32` | Scalar register budget per work-item |

`max_threads` is not the launch size. The actual work-item count comes from the
SIMT launch dimensions. Both arguments must be Python integers known at trace
time, greater than zero, and fit in signless `i32`. They are only valid on
decorated SIMT helper functions, not inline `with pto.simt():` scopes.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel_entry_simt_resource_probe","compile":{}} -->
```python
@pto.simt(max_threads=256, max_regs=48)
def write_tid(dst: pto.ptr(pto.i32, "gm")):
    tid = pto.get_tid_x()
    idx = scalar.index_cast(tid)
    pto.stg(tid, dst, idx)


@pto.jit(target="a5")
def kernel_entry_simt_resource_probe(dst: pto.ptr(pto.i32, "gm")):
    write_tid[128, 1, 1](dst)
```

This interface contract is enforced unconditionally. `@pto.simt` may accept
Tiles, typed pointers, and PTO scalars, but not broader module-only boundary
types.

#### Explicit SIMT launch dimensions

Calling a decorated SIMT helper directly uses the default launch descriptor
emitted by the tracer. Use indexed launch syntax when the launch dimensions must
be authored explicitly. `pto.simt_launch(...)` is the equivalent functional
form.

**Signatures**:

```python
body[dim_x, dim_y, dim_z](*args, **static_kwargs)
pto.simt_launch(body, *args, dims=(dim_x, dim_y, dim_z), **static_kwargs)
```

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `body` | `@pto.simt` function | SIMT entry body to launch |
| `*args` | PTO values | Runtime arguments passed to the SIMT body |
| `dim_x`, `dim_y`, `dim_z` | `i32`-compatible values | Launch dimensions in source-level `x, y, z` order |
| `**static_kwargs` | hashable Python values | Trace-time specialization arguments for the SIMT body |

**Returns**: None.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel_entry_simt_launch_probe","compile":{}} -->
```python
@pto.simt
def fill_tid(dst: pto.ptr(pto.i32, "gm")):
    tid = pto.get_tid_x()
    pto.stg(tid, dst, scalar.index_cast(tid))


@pto.jit(target="a5")
def kernel_entry_simt_launch_probe(dst: pto.ptr(pto.i32, "gm")):
    fill_tid[32, 1, 1](dst)
```

Specific SIMT micro-op APIs are documented in Chapter 13.

## 3.8 Inline context manager syntax

In addition to the decorator form, custom tile-op helpers can be written with
`with pto.tileop():`, and SIMT helpers can be written with `with pto.simt():`.
These open one-off anonymous sub-kernel bodies without requiring a separate
named Python function. Inline scopes are supported in top-level `@pto.jit`
bodies. Legacy `with pto.cube():` / `with pto.simd():` are diagnosed and
should be rewritten to `with pto.tileop():`.

### Syntax

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.inline_simd_scope","symbol":"kernel_entry_inline_simd_scope_probe","compile":{"BLOCK":128}} -->
```python
with pto.tileop():
    a_vec = pto.vlds(a_tile[r, c:])
    b_vec = pto.vlds(b_tile[r, c:])
    o_vec = pto.vadd(a_vec, b_vec, mask)
    pto.vsts(o_vec, o_tile[r, c:], mask)
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.inline_simt_scope","symbol":"kernel_entry_inline_simt_scope_probe","compile":{"BLOCK":8}} -->
```python
with pto.simt():
    alpha = scalar.load(alpha_tile[row, 0])
    beta = scalar.load(beta_tile[row, 0])
    o_next = alpha * o_prev + beta * pv_val
    scalar.store(o_next, o_next_tile[row, col])
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"kernel_entry.inline_cube_scope","symbol":"kernel_entry_inline_cube_scope_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
with pto.tileop():
    pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)
```

### Semantics

- Inside the `with` block, instructions execute on the corresponding hardware
  unit.
- On block exit, PTODSL outlines the block into one anonymous helper
  `func.func` and replaces the original region with a `func.call`.
- Named/decorated `@pto.tileop` helpers and inline `with pto.tileop():`
  scopes both lower as naked `pto.tileop.helper` bodies. PTOAS infers the
  primary domain and materializes `pto.section.vector` /
  `pto.section.cube` later in the pipeline.
- `with pto.simt():` preserves its scalar body inside one outlined
  `pto.simt_entry` helper, and the caller emits `pto.simt_launch`.
- Values defined inside the inline sub-kernel cannot escape the block directly.
  Use Tiles, typed pointers, or other mutable references to communicate results
  back to the caller.
- Cube-local scratch (`l0a`, `l0b`, `acc`) must be allocated by the caller
  before entering the block.

### Comparison

| | Decorator form | Context manager form |
|---|---|---|
| Reuse | Named, callable from multiple sites | Anonymous helper, single-use |
| Readability | Good for complex, multi-step logic | Good for short (3-10 line) snippets |
| Lowering | Reusable helper `func.func` | Anonymous helper `func.func` created on block exit |
| Testing | Can be unit-tested independently | Tested only through the enclosing kernel |
| Cube-local args | Explicit parameters | Captured from enclosing scope |

The two forms can be freely mixed in the same `@pto.jit` body.


## 3.9 Boundary contracts

**Sub-kernels are custom tile ops.** Their I/O contract is strict: tileop-style
helpers accept `Tile`, `TensorView`, `PartitionTensorView`, and PTO scalars;
SIMT helpers accept `Tile`, typed `pto.ptr(...)`, and PTO scalars. Results
still exit by writing to mutable references or returning PTO scalar values only.

**Modules use the C ABI.** Module boundaries (`entry=False`) are real function
calls — only `pto.ptr` and PTO scalars can cross. `Tile`, `TensorView`, and
`PartitionTensorView` are allocated locally on each side.

| Boundary | Allowed |
|----------|---------|
| Host → `@pto.jit(entry=True)` | explicit GM pointers + runtime scalars |
| Entry / module → `@pto.jit(entry=False)` module | **`pto.ptr` + PTO scalars only** (C ABI). Caller passes `tile.as_ptr()`; module constructs local tiles internally |
| Entry / module → `@pto.tileop` | `Tile`, `TensorView`, `PartitionTensorView`, PTO scalars |
| Entry / module → `@pto.simt` | `Tile`, typed `pto.ptr(...)`, PTO scalars |
| `@pto.jit` → `with pto.{tileop,simt}:` | Captured `Tile` / ptr / scalar values from enclosing scope |
| Sub-kernel → sub-kernel | Not allowed (go through UB tiles via the caller) |
| Sub-kernel → module | Not allowed (sub-kernels cannot call out) |
| Inline sub-kernel → caller | No direct SSA return path; write through Tile / ptr / mutable references |
| `@pto.tileop` → caller | Write results through mutable Tile / view operands; transient `vreg` and cube-local state cannot escape |
| Cube-local → UB | Only via `mte_l0c_ub`; LEFT/RIGHT/ACC/BIAS are private |
| `entry=False` module → caller | No return values; data crosses only via mutable references |


## 3.10 `pto.const_expr`

`pto.const_expr` marks a `@pto.jit(entry=True)` keyword-only parameter as a
compile-time constant. The compiler specializes the kernel for each combination
of constexpr values, and the compiled artifact is cached by specialization key
together with the kernel's entry annotation contract.

`pto.const_expr` is the public API name. In prose, this manual sometimes uses
"constexpr" as shorthand for "compile-time constant"; it does not refer to a
second PTODSL symbol.

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    *,
    BLOCK: pto.const_expr = 128,
    DTYPE: pto.const_expr = pto.f32,
):
    # ... use BLOCK / DTYPE in tile shapes, loop bounds, or dtype-specialized paths ...
    return
```

- Must appear as a keyword-only argument (after `*`).
- Must have a default value.
- Must be provided at `.compile()` time if the caller needs to override the
  default.
- Cannot change between launches of the same compiled instance — compile a new
  variant for a different value.
- **Only valid in `entry=True` kernels.** `entry=False` modules do not
  support `const_expr` — pass compile-time-known values as PTO scalars from the
  caller instead.

`pto.const_expr` parameters can be used anywhere in the kernel body where a
Python value is expected: tile shapes, loop bounds that are known at compile
time, dtype arguments, etc. They are evaluated at trace time, so `for i in
range(BLOCK)` would unroll `BLOCK` times.

In contrast, values passed as runtime shape/stride scalars (for example,
`rows`, `cols`, or `x_stride0`) are dynamic — they vary per launch and should
be used with `pto.for_` to produce device-side loops.
