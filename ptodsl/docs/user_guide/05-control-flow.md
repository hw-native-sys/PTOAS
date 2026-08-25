# 5. Control Flow

PTODSL uses a **tracing** compilation model. When you call `kernel.compile(...)`, PTODSL executes your Python function body once to record every PTO instruction — this pass is called *tracing*. The recorded program is then lowered and optimized into device code. Once compiled, launching the kernel runs the already-built device code directly on the NPU.

This has one critical implication for how you write loops and branches:

 - **Python native `for`/`if`** is rewritten to device-side control flow by default in `@pto.jit` bodies, `@pto.func` helpers, and named `@pto.tileop` / `@pto.simt` helpers. A `for i in range(rows)` loop records a device loop, and a runtime `if` records both branches.
- **Assign-form Python conditional expressions** such as `x = a if cond else b` are normalized through the same AST rewrite path, so runtime conditions lower to device-side branches before the assignment is merged back into `x`.
- **`pto.const_expr` / `pto.static_range`** keep compile-time Python behavior when you want trace-time specialization or unrolling.
- **`pto.for_` / `pto.if_`** produce device-side control flow. The loop bound or branch condition can be a runtime value, and the hardware will execute the loop or take the branch dynamically.

**Simple rule: Python control flow = device-side by default. Use `pto.const_expr` / `pto.static_range` for compile-time control flow.**

## 5.1 Trace-time unrolling

Use `pto.static_range(...)` when a loop should execute during tracing. Each
iteration records its instructions separately, so the device code gets a linear
sequence with the body repeated:

```python
@pto.jit(target="a5")
def unrolled_kernel(A, O, *, N: pto.const_expr):
    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    # N is constexpr, so static_range(N) is known at trace time.
    # The loop unrolls: the device gets N copies of the body.
    for i in pto.static_range(N):
        a_part = pto.partition_view(a_view, offsets=[i], sizes=[1])
        o_part = pto.partition_view(o_view, offsets=[i], sizes=[1])
        a_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
        o_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
        pto.tile.load(a_part, a_tile)
        pto.tile.add(a_tile, a_tile, o_tile)
        pto.tile.store(o_tile, o_part)
```

This works when the loop bound is a compile-time constant, such as a
`const_expr` parameter. For dynamic bounds, use native `range(...)` in the
default AST rewrite mode or use explicit `pto.for_`.

## 5.2 `pto.for_` — device-side loops

`pto.for_` records a structured loop that executes on the device. Its bound can be any expression involving runtime values (tensor shapes, scalar computations, block indices), and the compiler may optimize it further — unrolling when the bound is known at compile time, or keeping it as a runtime loop otherwise.

### Basic form

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.basic_for","symbol":"control_flow_basic_for_probe","compile":{"BLOCK":8}} -->
```python
with pto.for_(start, stop, step=step) as iv:
    pto.tile.load(pto.partition_view(a_view, offsets=[iv, 0], sizes=[1, cols]), tile)
```

- `start`, `stop`, `step` are PTO scalar expressions. They are evaluated on the device.
- The loop body executes `(stop - start + step - 1) // step` times.
- Use with `step=1` unless you need a strided iteration.

Compare the two approaches:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.compare_loops","symbol":"control_flow_compare_loops_probe","compile":{"BLOCK":8}} -->
```python
# Trace-time unrolling — BLOCK must be constexpr
for i in pto.static_range(BLOCK):
    pto.tile.load(pto.partition_view(a_view, offsets=[0, 0], sizes=[1, cols]), tile)

# Device-side loop — num_blocks can be dynamic
with pto.for_(0, num_blocks, step=1) as i:
    pto.tile.load(pto.partition_view(a_view, offsets=[i, 0], sizes=[1, cols]), tile)
```

### Nested loops

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.nested_loops","symbol":"control_flow_nested_loops_probe","compile":{"BLOCK":8}} -->
```python
with pto.for_(0, rows, step=1) as r:
    with pto.for_(0, cols, step=1) as c:
        val = scalar.load(tile[r, c])
```

Both loops execute on the device. The outer loop bound `rows` and inner loop bound `cols` can be runtime values.

### Loop with carry state

When a loop needs to propagate state from one iteration to the next, use the `.carry(...)` method. This is the PTODSL equivalent of a loop that accumulates or updates variables across iterations. The following self-contained kernel is the smallest compileable carry example used by the docs-as-test harness:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"carry_loop_probe","compile":{"BLOCK":128}} -->
```python
@pto.jit(target="a5")
def carry_loop_probe(*, BLOCK: pto.const_expr = 128):
    m_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    m_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    m_prev.fill(0.0)
    l_prev.fill(0.0)
    o_prev.fill(0.0)

    kv_loop = pto.for_(0, 4, step=1).carry(m=m_prev, l=l_prev, o=o_prev)
    with kv_loop:
        kv_loop.m.fill(1.0)
        kv_loop.l.fill(2.0)
        kv_loop.o.fill(3.0)
        kv_loop.update(m=m_next, l=l_next, o=o_next)

    final_o = kv_loop.final("o")
    final_o.fill(4.0)
```

`.carry(name=initial_value)` declares named state variables that are passed from one iteration to the next. Inside the loop body, access the current value with `loop.name`. At the end of the body, call `loop.update(name=new_value)` to set what the next iteration receives. After the loop exits, `loop.final("name")` retrieves the value from the last iteration.

This pattern is central to algorithms like online softmax, where each KV block updates running statistics (row max, sum, output accumulator). The ping-pong tile pattern — allocating two tiles and swapping them each iteration — is the idiomatic way to manage this state:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.carry_pingpong","symbol":"control_flow_carry_pingpong_probe","compile":{"Br":16,"num_blocks":4}} -->
```python
# Allocate ping-pong state tiles
m_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
m_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
l_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
l_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")

# Initialize prev tiles
m_prev.fill(float("-inf"))
l_prev.fill(0.0)

loop = pto.for_(0, num_blocks, step=1).carry(m=m_prev, l=l_prev)
with loop:
    m_cur = loop.m
    l_cur = loop.l

    m_next.fill(1.0)
    l_next.fill(2.0)

    loop.update(m=m_next, l=l_next)
```

### Chunked inner loop with carry (tail handling)

For SIMD kernels that process data in vector-width chunks, use a carry loop to track the remaining element count across column iterations:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"tail.chunked_inner_loop","symbol":"tail_chunked_inner_loop_probe","compile":{"BLOCK":128}} -->
```python
VEC = pto.elements_per_vreg(pto.f32)
col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
with col_loop:
    c = col_loop.iv
    remained = col_loop.remained
    mask, remained = pto.make_mask(pto.f32, remained)
    vec = pto.vlds(tile[r, c:])
    # ... operate under mask ...
    pto.vsts(vec, out_tile[r, c:], mask)
    col_loop.update(remained=remained)
```

`make_mask(dtype, n)` returns two values: the predicate mask for the current chunk and the updated remaining count. Passing the updated count back via `col_loop.update(remained=...)` feeds it into the next iteration, so each chunk correctly computes how many elements are left. If `n` is an `index`, the updated remaining count stays an `index`; PTODSL hides the hardware `i32` tail-mask bookkeeping internally.

### Loop unroll hints

Device-side loops accept optional unroll hints, `unroll=` or `unroll_factor=` (mutually exclusive), on every loop construction path:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"unroll_hint_probe","compile":{"BLOCK":8}} -->
```python
@pto.jit(target="a5")
def unroll_hint_probe(*, BLOCK: pto.const_expr = 8):
    acc = pto.const(0, dtype=pto.i32)
    # AST-rewrite path: pto.range marks a device loop with a hint.
    for i in pto.range(BLOCK, unroll="full"):
        acc = acc + pto.const(1, dtype=pto.i32)
    # Explicit pto.for_ path (the .carry(...) form takes the same keywords).
    with pto.for_(0, BLOCK, step=1, unroll_factor=2) as i:
        acc = acc + pto.const(2, dtype=pto.i32)
    _ = acc
```

| Hint | Meaning |
|---|---|
| `unroll="full"` | Unroll completely when the trip count is a compile-time constant; otherwise the hint is dropped with a remark. |
| `unroll_factor=N` | Unroll by N when the step is a compile-time constant (dynamic upper bounds are supported and produce an epilogue loop); otherwise the hint is dropped with a remark. |

Rules and limitations:

- `unroll_factor` must be a positive Python `int` no larger than `2**31 - 1` (the hint is encoded as an i32 attribute).
- **A hint never changes program semantics.** When the loop cannot be unrolled natively the hint is dropped with a compiler remark and the loop is compiled unchanged. This happens for: a dynamic trip count with `unroll="full"`; a dynamic step with `unroll_factor`; `unroll_factor=1` (a no-op); a factor above the `pto-unroll-loops` pass's `max-unroll-factor` (default 1024); an empty loop body; a statically empty iteration space (`stop <= start`); and a loop whose induction variable is not `index` (PTODSL loops are `index`-typed, so this only applies to hand-written IR). This is unlike `pto.static_range`, which always unrolls at trace time and requires compile-time-constant bounds.
- Plain `range(...)` / `pto.range(...)` loops require a positive step (they lower to `scf.for`, which only supports ascending iteration); a constant non-positive step is rejected. Loops with `break` / `continue` / `else` lower through `pto._while` and cannot carry hints — using one raises an error.
- Loops without any hint are compiled exactly as before.
- Unroll hints are honored by the VPTO LLVM backends only (the default `pto-backend=vpto` pipelines). The EmitC backend (`PTOToEmitC`) does not consume `pto.unroll` / `pto.unroll_factor` — hints are silently dropped there.

## 5.3 `pto.if_` — device-side conditionals

`pto.if_` records a device-side conditional branch. Unlike a Python `if`, the condition can be a runtime PTO scalar, and both branches are recorded into the program so the hardware can choose at runtime.

The condition must be a PTO scalar value (e.g., the result of a comparison like `a > b` or a value loaded from a tile). Python booleans evaluated at trace time should use a plain `if` instead.

### Recommended block structure

PTODSL should treat one device-side conditional as one explicit branch object.
The recommended surface is:

```python
with pto.if_(cond) as br:
    with br.then_:
        ...
    with br.else_:
        ...
```

This keeps the `if` / `else` pairing explicit. The `else_` branch is optional
for side-effect-only conditionals.

### Automatic named merge across branches

When a value must flow out of both branches, PTODSL should merge by explicit
name. Each branch assigns the same output names with `br.assign(...)`, and the
merged results are read back from the branch handle after the conditional:

```python
@pto.simt
def conditional_scale(
    tile: pto.Tile,
    threshold: pto.f32,
    scale: pto.f32,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        with pto.for_(0, cols, step=1) as c:
            val = scalar.load(tile[r, c])
            big = val > threshold

            with pto.if_(big) as br:
                with br.then_:
                    br.assign(val=val * scale)
                with br.else_:
                    br.assign(val=val)

            val = br.val
            scalar.store(val, tile[r, c])
```

In this example, both branches define the merged value named `val`. After the
conditional closes, `br.val` is the SSA-merged result seen by downstream code.
This surface avoids explicit result-type declarations and explicit
`pto.yield_(...)` in user code while still keeping the merge contract explicit.

When one branch yields `pto.i1` and the other yields an integer-like scalar,
the integer value is normalized to `i1` with nonzero truthiness (`0` is false;
any nonzero value is true). Other incompatible branch result types still
require an explicit conversion.

## 5.4 `pto.const_expr` and tracing

`pto.const_expr` parameters (Section 3.6) are compile-time constants. They are fixed at `.compile()` time and cannot change between launches of the same compiled kernel. Because their values are known during tracing, they interact naturally with Python control flow:

```python
@pto.jit(target="a5")
def kernel(
    A,
    *,
    BLOCK: pto.const_expr = 128,
    NUM_BLOCKS: pto.const_expr = 8,
    UNROLL: pto.const_expr = False,
):
    N = A.shape[0]
    num_blocks = (N + BLOCK - 1) // BLOCK

    # N and num_blocks are runtime values derived from tensor metadata.
    # They can drive native range(...) loops in the default AST rewrite mode.
    with pto.for_(0, num_blocks, step=1) as i:
        ...

    if pto.const_expr(UNROLL):
        # Trace-time: UNROLL and NUM_BLOCKS are both known during tracing.
        # Each iteration records separately, so the loop is fully unrolled.
        for i in pto.static_range(NUM_BLOCKS):
            ...
    else:
        # The non-unrolled path can still use a device-side loop whose bound
        # is a constexpr value captured into the traced program.
        with pto.for_(0, NUM_BLOCKS, step=1) as i:
            ...
```

This lets you write a single kernel that specializes into different strategies based on constexpr knobs, while still using runtime tensor metadata for device-side control flow.

## 5.5 Native Python control-flow rewrite

`@pto.jit`, `@pto.func`, and named `@pto.tileop` / `@pto.simt`
callables rewrite supported native Python control flow before tracing their
bodies. In the default mode, plain Python `if` and `for range(...)`, and `while`
in the rewritten scope become device-side control flow. Use `pto.const_expr(...)`
and `pto.static_range(...)` when you want trace-time behavior.

PTODSL does not recursively rewrite arbitrary undecorated Python callees. If an
external helper should contain runtime native control flow, decorate it with
`@pto.func` or one of the other PTODSL callable decorators. `@pto.func` helpers
must explicitly declare their return type with `returns=...` or a Python return
annotation. A plain Python helper is still executed during tracing; static
`range(...)` loops in such a helper unroll at trace time, and runtime loop
bounds or branch conditions are not converted into `scf.for` / `scf.if`.

### Runtime branches

By default, a native Python `if` becomes a device-side conditional:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_branch_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_branch_kernel():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)

    if lhs > rhs:
        total = lhs + rhs
    else:
        total = rhs + lhs

    _ = total
```

The assigned value `total` is live after the branch, so PTODSL rewrites the
branch into a `pto.if_` with automatic merge.

Static list slots can also be live across a rewritten branch. The subscript
index must be an integer that can be resolved during AST rewriting, including
compile-time constants and `pto.const_expr` values:

```python
@pto.jit(target="a5")
def ast_rewrite_static_slot_kernel(*, SLOT: pto.const_expr = 0):
    values = [pto.const(0, dtype=pto.i32)]

    if pto.const(1, dtype=pto.i1):
        values[SLOT] = pto.const(1, dtype=pto.i32)

    if values[SLOT]:
        pto.pipe_barrier(pto.Pipe.ALL)
```

The rewritten slot is merged as an `scf.if` result, just like a named scalar
value. Dynamic indices and container aliases remain unsupported; use an
explicit `pto.if_`/`pto.for_` state value or a real buffer load/store for those
cases.

If a live-out value is assigned in only one branch, PTODSL keeps the old value
on the missing branch:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_old_value_branch_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_old_value_branch_kernel():
    lhs = pto.const(4, dtype=pto.i32)
    rhs = pto.const(2, dtype=pto.i32)
    total = rhs

    if lhs > rhs:
        total = lhs + rhs

    _ = total
```

Side-effect-only branches are also supported:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_side_effect_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_side_effect_kernel():
    cond = pto.const(1, dtype=pto.i1)
    if cond:
        pto.pipe_barrier(pto.Pipe.ALL)
```

### Short-circuit `and` / `or`

Python `and` / `or` between PTODSL runtime values are not bitwise
operations: they keep Python's short-circuit semantics, so the right-hand
side is only evaluated on the device where Python would actually need it.
The AST rewrite turns `a and b` / `a or b` into a result-bearing `scf.if`
whose guarded region contains exactly the RHS:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_short_circuit_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_short_circuit_kernel():
    value = pto.const(10, dtype=pto.i32)
    divisor = pto.const(2, dtype=pto.i32)

    pred = (divisor != 0) and ((value // divisor) > 0)
    if pred:
        pto.pipe_barrier(pto.Pipe.ALL)
```

`and` returns the left operand when it is falsy and the right operand
otherwise; `or` returns the left operand when it is truthy and the right
operand otherwise. Integer-typed operands are tested with non-zero
truthiness, and when one merged branch is an `i1` the other operand is
reconciled with the same rules as `pto.if_` branch merges (including Python
`bool` literals, which materialize as `i1` constants). Statically-known
`bool` / `int` operands short-circuit at trace time: `False and rhs` and
`True or rhs` never trace the RHS at all.

`and` / `or` compose with every rewritten expression context: assignments,
call arguments, `return`, and `if` / `while` conditions. Floating-point
values are not valid short-circuit controls and raise a clear error.


### Runtime loops

Native `range(...)` loops become device-side loops:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_loop_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_loop_kernel(rows: pto.i32):
    for row in range(0, rows, 1):
        _ = row
        pto.pipe_barrier(pto.Pipe.ALL)
```

The supported range forms are:

```python
range(stop)
range(start, stop)
range(start, stop, step)
```

The loop target must be a simple name.

### Runtime `while`

Native Python `while` is device-side control flow in the default AST rewrite
mode. Its condition is evaluated at the loop header on every iteration, so it
may depend on runtime scalar values and loop-carried state:

```python
value = pto.const(0, dtype=pto.i32)
while value < limit:
    value = value + pto.const(1, dtype=pto.i32)
```

The runtime rewrite supports condition-driven loops with inferred loop-carried
values. `break`, `continue`, and loop `else` are lowered through hidden
control-state values in `scf.while`. A plain loop over `pto.static_range(...)`
remains trace-time Python and keeps ordinary Python execution semantics.

### Loop-carried values

Accumulator-style loops are rewritten through `pto.for_(...).carry(...)`:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_accumulator_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_accumulator_kernel(rows: pto.i32):
    one = pto.const(1, dtype=pto.i32)
    acc = pto.const(0, dtype=pto.i32)

    for _ in range(rows):
        acc = acc + one

    _ = acc
```

This lowers to an `scf.for` with `iter_args`.

The loop-carried state of a native runtime `while` is inferred from values that
are assigned in the loop and used by the condition, by later iterations, or
after the loop. Each such value must have a stable type and a well-defined
initial value before the loop. Loop-local temporaries that are written before
they are read inside every iteration (e.g. `col = base + index`) are not
loop-carried and need no initial value outside the loop.

For a value assigned only on some iterations but used after the loop, the
default path preserves the entering value and therefore requires that value to
be definitely bound before the `while`. Otherwise AST rewrite reports the
value as `last-iteration-only`; initialize it before the loop or use explicit
loop state.

Because loop-local temporaries stay iteration-local, an `else:` clause of a
runtime `while` must not read them: the clause runs after the rewrite's
single traced execution, so it would observe the trace-time value instead of
the value of the final iteration. Read only genuinely loop-carried names
(or values that are live after the loop) from an `else:` clause. If you need
a more complex loop state pattern, use the explicit API:

```python
loop = pto.for_(0, rows, step=1).carry(acc=acc)
with loop:
    acc = loop.acc
    acc = acc + one
    loop.update(acc=acc)
acc = loop.final("acc")
```

### Compile-time control flow

In the default AST rewrite mode, plain `if` and `range(...)` are runtime
control flow. Use explicit compile-time helpers when you want trace-time
behavior.

Use `pto.const_expr(...)` for trace-time branches:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_static_branch_kernel","compile":{"ENABLE":true}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_static_branch_kernel(*, ENABLE: pto.const_expr = True):
    if pto.const_expr(ENABLE):
        pto.pipe_barrier(pto.Pipe.ALL)
```

Use `pto.static_range(...)` for trace-time unrolling:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_static_loop_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_static_loop_kernel():
    for _ in pto.static_range(2):
        pto.pipe_barrier(pto.Pipe.ALL)
```

`pto.static_range(...)` keeps Python `for` semantics. Bindings created or
deleted by the unrolled body and `else` clause remain in effect after tracing.
The loop target remains available after a non-empty loop unless the body
deletes it:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_static_loop_target_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_static_loop_target_kernel():
    for stage in pto.static_range(2):
        pto.pipe_barrier(pto.Pipe.ALL)
    _ = stage
```

### Compatibility and nested helpers

AST rewrite defaults to `True` on `@pto.jit` and named sub-kernel decorators.

Explicit control-flow APIs continue to work alongside rewritten Python control
flow:

```python
with pto.if_(cond) as br:
    ...

with pto.for_(0, rows, step=1) as row:
    ...
```

For `@pto.jit`, the rewrite runs when a specialization is compiled, so Python
closure values are read at compile time rather than when the decorator is
evaluated. Named sub-kernel decorators use the same source rewrite semantics
when their body is traced.

Nested Python helper bodies follow the same rule as the outer kernel body:
their supported Python `if` and `for range(...)` statements are rewritten to
runtime control flow by default.

Functions created by `exec`, REPLs, notebooks, or other dynamic generators may
not expose retrievable Python source through `inspect.getsource(...)`. In that
case, or when the source cannot be matched back to a unique function
definition, PTODSL falls back to the original trace-time Python function
instead of rewriting it. Use source-backed functions for native Python runtime
`if` / `for range(...)`; fallback tracing still only supports ordinary Python
compile-time control flow.

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"ast_rewrite_nested_helper_kernel","compile":{}} -->
```python
@pto.jit(target="a5")
def ast_rewrite_nested_helper_kernel():
    cond = pto.const(1, dtype=pto.i1)

    def helper(enabled):
        if enabled:
            for _ in range(2):
                pto.pipe_barrier(pto.Pipe.ALL)

    helper(cond)
```

Helper-local compile-time control flow still uses `pto.const_expr(...)` and
`pto.static_range(...)`.
Plain Python boolean guards also remain trace-time branches. This keeps
constexpr comparisons such as `if BLOCK == 128:` and boolean flags such as
`if ENABLE:` compatible with the default rewrite mode. Device-side conditions
must be PTO scalar values and lower through rewritten Python `if` or explicit
`pto.if_`.

### Debugging legacy tracing

`ast_rewrite=False` is a developer/debug escape hatch. It restores legacy
trace-time Python control flow for a specific `@pto.jit` kernel or named
sub-kernel so you can isolate rewrite issues while developing PTODSL itself.
It is not the recommended user-facing mode for new examples or kernels.

```python
@pto.jit(target="a5", ast_rewrite=False)
def debug_kernel(*, BLOCK: pto.const_expr = 4):
    for _ in range(BLOCK):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5", entry=False, mode="explicit", ast_rewrite=False)
def debug_simd_helper():
    if pto.const_expr(True):
        pto.pipe_barrier(pto.Pipe.ALL)
```

The same switch is available through `frontend_options`:

```python
@pto.jit(
    target="a5",
    frontend_options={
        "ast_rewrite": False,
    },
)
def debug_options_disable_rewrite_kernel(*, BLOCK: pto.const_expr = 4):
    for _ in range(BLOCK):
        pto.pipe_barrier(pto.Pipe.ALL)
```

Do not pass conflicting values through both spellings. For example,
`@pto.jit(ast_rewrite=False, frontend_options={"ast_rewrite": True})` is
rejected.

When this mode is disabled for a function, native Python `if` / `for` executes
while tracing that function. Runtime device-side control flow should still use
the default rewrite mode, or explicit `pto.if_` / `pto.for_` APIs when you need
manual control.

The structured `frontend_options` argument is reserved for frontend rewrite
debugging and future rewrite passes. Today it accepts the same AST rewrite
switch plus a rewrite-part selector:

```python
@pto.jit(
    target="a5",
    frontend_options={
        "ast_rewrite": True,
        "rewrite_part": {"control_flow"},
        "dump_rewritten_source": False,
    },
)
def debug_frontend_options_kernel():
    pto.pipe_barrier(pto.Pipe.ALL)
```

`rewrite_part` currently only accepts `"control_flow"`. Future frontend
rewrite passes should add their selector names here. `dump_rewritten_source`
is reserved for debugging output and currently must remain `False`.

### Unsupported patterns

The runtime rewrite does not support dynamic `return` from inside a loop, or
runtime iteration over arbitrary Python iterables. Tuple/list loop targets and
trace-time-only values still require the existing static forms. A plain runtime
`for` cannot expose its induction variable after the loop; the controlled
`for` form using `break`, `continue`, or `else` can carry it explicitly. Use an explicit
kernel-level result/carry protocol when a loop needs to communicate a result;
do not use Python `return` as a dynamic loop exit.

## 5.6 Summary

| Construct | When evaluated | Use for |
|-----------|---------------|---------|
| Python `for` / `if` | Device-side | Native syntax for dynamic loops and branches |
| Python `while` | Device-side | Runtime condition-driven loops |
| Runtime `break` / `continue` / loop `else` | Lowered through hidden `scf.while` control state | Dynamic `return` from a runtime loop remains unsupported |
| `pto.const_expr` / `pto.static_range` | Trace time | Compile-time branches and unrolling |
| `pto.for_` | Device-side | Dynamic bounds, runtime loop counts |
| `pto.for_(...).carry(...)` | Device-side | Loops with accumulated state across iterations |
| `pto.if_` | Device-side | Runtime conditions, data-dependent branching |
| Python `and` / `or` | Device-side short-circuit `scf.if` | Runtime predicates with Python short-circuit semantics |
