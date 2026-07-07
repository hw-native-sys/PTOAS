# PTODSL AST Rewrite Control Flow Design

## Background

PTODSL uses Python tracing today. Native Python `if` and `for` execute while
the frontend records IR, so they are compile-time control flow. Device-side
control flow is already available through explicit APIs:

- `with pto.if_(cond) as br:`
- `with pto.for_(start, stop, step=step) as iv:`
- `pto.for_(...).carry(...)`

This is explicit and stable, but it makes dynamic control-flow kernels look
less like ordinary Python. This design adds a source AST rewrite pass: legal
Python `if` / `for range(...)` syntax is rewritten before tracing into
structured IR control flow, while explicit compile-time escape hatches keep
static specialization readable.

## Goals

- Make native Python control-flow syntax usable by default for runtime control
  flow in `@pto.jit(...)` kernels and named `@pto.tileop` / `@pto.simt`
  sub-kernels.
- Use `ast_rewrite` as the public name for the source rewrite feature.
- Rewrite legal Python `if` / `for range(...)` into existing PTODSL
  control-flow surfaces.
- Preserve explicit control flow: existing `pto.if_` / `pto.for_` code remains
  valid whether AST rewrite is enabled or disabled.
- Provide explicit compile-time escape hatches:
  - `pto.const_expr(value)` for native Python `if`.
  - `pto.static_range(...)` for trace-time loop unrolling.
- Reserve a structured frontend-options shape so later rewrite passes, such as
  scalar-expression rewrite, can be enabled or debugged independently.

## User Model

Default mode rewrites supported native Python control flow:

```python
@pto.jit(target="a5")
def kernel(rows: pto.i32):
    for row in range(rows):
        if row > pto.const(0, dtype=pto.i32):
            ...
```

The frontend rewrites the loop to `pto.for_` and the branch to `pto.if_`.

Compile-time control flow stays explicit:

```python
@pto.jit(target="a5")
def kernel(*, BLOCK: pto.const_expr = 128):
    if pto.const_expr(BLOCK == 128):
        for stage in pto.static_range(4):
            ...
```

For future rewrite passes, the frontend should also reserve an options object.
The exact API can stay small in the first implementation, but the design should
leave room for feature-specific toggles and debug output:

```python
@pto.jit(
    target="a5",
    frontend_options={
        "ast_rewrite": True,
        "rewrite_part": {"control_flow"},
        "dump_rewritten_source": False,
    },
)
def kernel(...):
    ...
```

## Rewrite Strategy

The implementation is source-to-source AST rewriting. The rewrite runs when a
kernel specialization is compiled, not when the `@pto.jit` decorator is
evaluated, so closure nonlocals are read at compile time.
Named sub-kernel decorators keep the original Python function at decoration
time and apply the same rewrite when the sub-kernel body is traced.

1. `kernel.compile(...)` obtains the function source with `inspect.getsource`
   when AST rewrite is enabled. If source is not available, PTODSL keeps the
   original trace-time function so default-on rewrite does not break
   dynamically generated kernels. The same fallback applies when the source
   cannot be matched back to a unique function definition.
2. The source is parsed with Python `ast.parse`, so the input must remain valid
   Python.
3. Decorators are removed from the transformed function to avoid recursive
   decoration.
4. The AST rewriter transforms supported `if` and `for` statements into calls
   to existing PTODSL APIs.
5. The transformed function is compiled and passed to the existing tracing
   runtime.

The generated IR is still produced by current `pto.if_`, `pto.for_`, and
`pto.for_(...).carry(...)` implementations.

## If Rewrite

Input:

```python
if cond:
    value = lhs + rhs
else:
    value = rhs + lhs
out = value
```

Rewritten shape:

```python
with pto.if_(cond) as __br:
    with __br.then_:
        value = lhs + rhs
        __br.assign(value=value)
    with __br.else_:
        value = rhs + lhs
        __br.assign(value=value)
value = __br.value
out = value
```

Live-out names assigned inside both branches are merged through
`br.assign(...)`. If a live-out name is assigned in only one branch, the
rewriter captures the pre-branch value and yields it from the missing branch.
Side-effect-only `if` statements do not create results.

## For Rewrite

Input:

```python
for i in range(start, stop, step):
    body(i)
```

Rewritten shape:

```python
with pto.for_(start, stop, step=step) as i:
    body(i)
```

Loop-carried accumulator input:

```python
acc = init
for i in range(n):
    acc = acc + value(i)
use(acc)
```

Rewritten shape:

```python
__loop = pto.for_(0, n, step=1).carry(acc=acc)
with __loop:
    i = __loop.iv
    acc = __loop.acc
    acc = acc + value(i)
    __loop.update(acc=acc)
acc = __loop.final("acc")
use(acc)
```

The first implementation supports accumulator-style loop carry where the
carried value is read before it is reassigned in the loop body. Last-iteration
only values that have no initial value are rejected with a diagnostic.

## Compile-Time Escape Hatches

`pto.const_expr(value)` returns Python truthiness and marks the enclosing
native `if` as compile-time:

```python
if pto.const_expr(USE_FAST_PATH):
    ...
```

Plain Python boolean conditions also stay at trace time after the condition is
evaluated. This preserves existing constexpr guards such as `if BLOCK == 128:`
and `if ENABLE:` while still rewriting PTO runtime scalar conditions to
`pto.if_`.

`pto.static_range(...)` returns Python `range(...)` and marks the loop as
trace-time unrolled. The loop target keeps normal Python live-after-loop
semantics because the loop is not rewritten to a device-side region:

```python
for i in pto.static_range(4):
    ...
_ = i
```

## Limitations

The first version supports:

- `if` / `elif` / `else` through nested rewritten branches.
- `for name in range(...)` with one to three range arguments.
- Branch live-out merges where both branches assign the merged names.
- One-sided branch assignments that merge with the pre-branch value.
- Accumulator-style loop carry.
- Nested supported control flow in the kernel body, such as `if` / `for` inside
  `with` blocks.
- Nested helper rewriting. Nested Python `def` bodies follow the same
  AST-rewrite semantics as the outer kernel body.

The first version rejects:

- non-Python syntax;
- `for-else`;
- non-`range(...)` runtime loops;
- tuple/list loop targets;
- `break` / `continue`;
- using the runtime loop induction variable after the loop;
- last-iteration-only loop values with no initial carried value;
- rewriting nested Python `lambda` / `class` bodies, which keep normal
  trace-time Python semantics;
- unsupported Python effects such as relying on list/dict mutation as IR
  values.

## Compatibility

The compatibility guarantee is for PTODSL's explicit control-flow APIs.
Existing `with pto.if_(...)` / `with pto.for_(...)` kernels remain valid with
AST rewrite enabled, because those constructs already spell the intended
device-side control flow explicitly.

```python
with pto.if_(cond) as br:
    ...

with pto.for_(0, rows, step=1) as row:
    ...
```

Python-native trace-time control flow should migrate to the explicit
compile-time helpers:

```python
if pto.const_expr(USE_FAST_PATH):
    ...

for stage in pto.static_range(4):
    ...
```
