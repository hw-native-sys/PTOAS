# ptodsl — PTO Python IR Builders

A lightweight Python DSL package for building PTO MLIR IR modules
in Python. PTODSL kernels are ordinary Python functions decorated with
`@pto.jit`. Type annotations carry PTO
types as lazy descriptors, and control-flow maps 1-to-1 to MLIR operations.

PTODSL is distributed through the repository-root `ptoas` package. Use the
root install flows described below.

---

## Directory layout

```
ptodsl/
├── ptodsl/              # pip-installable package
│   ├── __init__.py      # exports: pto
│   ├── pto.py           # main PTO DSL namespace
│   ├── _scalar.py       # internal implementation for pto.* scalar helpers
│   ├── _context.py      # MLIR context factory
│   ├── _types.py        # lazy dtype descriptors and type constructors
│   ├── _ops.py          # PTO operation wrappers
│   ├── _control_flow.py # for_, if_, yield_ context managers
│   ├── _jit.py          # @pto.jit decorator
│   ├── _tracing/        # shared tracing runtime building blocks
│   └── _tile_template_tracing.py # internal tile-template tracing implementation
├── examples/
│   ├── tadd_lowlevel.py    # TADD – raw MLIR Python binding calls
│   ├── tadd_dsl.py         # TADD – @pto.jit DSL style
│   ├── softmax_lowlevel.py # Softmax – raw MLIR Python binding calls
│   └── softmax_dsl.py      # Softmax – @pto.jit DSL style
└── README.md
```

---

## Supported install flows

PTODSL is shipped through the repository-root `ptoas` package. Supported ways
to make `import ptodsl` available are:

```bash
# 1) Released or CI-built wheel: installs PTOAS + PTODSL together
pip install /path/to/ptoas*.whl

# 2) Non-editable source install from the repository root
cd $PTOAS_REPO_ROOT
pip install . --no-build-isolation

# 3) Editable install for PTOAS / PTODSL developers
cd $PTOAS_REPO_ROOT
pip install -e . --no-build-isolation
```

Install verification:

```bash
python3 -c "import ptodsl; from ptodsl import pto; print(ptodsl.__file__)"
```

Not supported:

- `cd ptodsl && pip install -e .`
- runtime repository walks or `sys.path` repair to locate MLIR/PTO bindings

Direct CTest/developer-tree execution must provide the matching MLIR and PTO
Python roots explicitly through the test environment or `PYTHONPATH`; PTODSL
does not search nearby build and install directories at import time.

Artifact boundary:

- `ptoas` wheel: PTODSL-capable Python distribution
- `ptoas-bin-*.tar.gz`: compiler-only artifact, does **not** imply `import ptodsl`

For source development, install the checkout into the active Python environment.
The editable install retains an incremental CMake build tree without requiring
manual `PYTHONPATH` or build-tree wrapper setup:

```bash
cd $PTOAS_REPO_ROOT
LLVM_BUILD_DIR=/path/to/llvm/build ./quick_install.sh
```

---

## PTODSL TileLib backend

PTOAS uses its in-process PTODSL TileLib service for VPTO tile-op expansion:

```bash
ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  input.pto -o -
```

Wheel and CMake-tree launchers add their packaged `TileOps` resource directory
to the host interpreter while the native compiler runs. Template discovery,
metadata queries, and MLIR materialization therefore stay in one process and
do not require a Python executable, daemon socket, or package-path CLI option.

`InsertTemplateAttributes` queries legal-candidate metadata before fusion and
stores an ordered `candidates` array containing only `id`, `name`,
`loop_depth`, `postupdate`, and `tail`. Fusion may filter this array.
Candidates are ordered by unique `id`. `ExpandTileOp` renders the first
candidate that remains, providing a deterministic fallback when several
candidates reach expansion.

See the
[PTODSL TileLib migration test checklist](docs/tilelib-migration-testing.md)
for the complete test inventory, commands, and expected outcomes.

---

## JIT examples

`ptodsl/examples/` contains self-contained `@pto.jit` examples that cover
both compile-only and end-to-end launch flows.

All examples below assume you already prepared one of the supported install
flows above. They should not require manual `PYTHONPATH` edits or example-local
`sys.path` bootstrap.

### Prerequisites for launch examples

- `ptoas` with bundled PTODSL installed as above
- CANN 9.0+ with `ASCEND_HOME_PATH` set
- For end-to-end launch: `torch`, `torch_npu`, `numpy`
- `bisheng` on `PATH`

Set up the environment in each new shell:

```bash
cd $PTOAS_REPO_ROOT
pip install -e . --no-build-isolation
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
```

For CPU simulation with `msprof`, the wrapper script below will set the
simulator library path and `ulimit` for you. The normal PTOAS + CANN shell
setup above is still required.

### `tadd_launch.py`

Single script: kernel definition, compile, launch, and accuracy check.
Equivalent IR to the bundled `tadd.pto` regression testcase.

Compile-only:

```bash
python3 ptodsl/examples/tadd_launch.py --emit-mlir
```

Expected: MLIR containing `@TADD_f32_16x64` and `@TADD_f32_32x32`.

Optional PTOAS frontend smoke:

```bash
python3 ptodsl/examples/tadd_launch.py --emit-mlir > /tmp/tadd_dsl.mlir
ptoas --emit-pto-ir /tmp/tadd_dsl.mlir -o - | head
```

End-to-end under the `msprof` CPU simulator:

```bash
scripts/sim_dsl.sh ptodsl/examples/tadd_launch.py
```

Expected output:

```text
PASS f32_16x64  compile=0.024s launch=35.193s
PASS f32_32x32  compile=0.022s launch=35.926s
All cases passed.
```

Direct run on a real NPU:

```bash
python3 ptodsl/examples/tadd_launch.py
```

### `flash_attention_softmax_launch.py`

Launchable row-wise softmax demo. The kernel surface is the ordinary
`scores -> out` contract, while the implementation preloads the score matrix to
UB and then uses a packed online-softmax recurrence so one NPU can stream
64-row packs sequentially from UB.

Compile-only:

```bash
python3 ptodsl/examples/flash_attention_softmax_launch.py --emit-mlir
```

End-to-end under the `msprof` CPU simulator:

```bash
scripts/sim_dsl.sh ptodsl/examples/flash_attention_softmax_launch.py
```

Expected output:

```text
PASS rows64_seq128
PASS rows81_seq96
All cases passed.
```

Direct run on a real NPU:

```bash
python3 ptodsl/examples/flash_attention_softmax_launch.py
```

### `rms_norm/rmsnorm_alloc_buffer_simt.py`

Compile-only RMSNorm example for explicit-mode SIMT kernels. It models the
RMSNorm SIMT-VF token body with a named `@pto.simt` helper launched from the
main kernel through `helper[threads, 1, 1](...)`.

The example exercises the PTODSL surfaces needed by this style of kernel:

- SIMT-local `pto.alloc_buffer(...)` for per-thread fragment storage
- hand-authored dynamic UB scratch layout with `pto.castptr` / `pto.addptr`
- contiguous vector loads through `pto.load(..., contiguous=N)` and vector
  stores through `pto.store(vector, ...)`
- `pto.Vec(..., init=scalar)` for scalar-to-vector broadcast
- `pto.simt_allreduce_sum(...)` lowered inline through the type-specific
  `pto.redux_addi` / `pto.redux_addf` operation and
  `pto.syncthreads`
- explicit pipe `set_flag` / `wait_flag` synchronization, including dynamic
  ping-pong event ids inside the token loop
- a runtime token loop that lowers to `scf.for`

```bash
python3 ptodsl/examples/rms_norm/rmsnorm_alloc_buffer_simt.py --variant x128 > /tmp/rmsnorm_x128.mlir
python3 ptodsl/examples/rms_norm/rmsnorm_alloc_buffer_simt.py --variant x64 > /tmp/rmsnorm_x64.mlir
```

Expected: MLIR containing `@rmsnorm_4096_alloc_buffer_simt_context_kernel`,
the named SIMT helper `@rmsnorm_simt_token_body__simt_...`, explicit
`pto.simt_launch`, `scf.for`, `vector<4xf32>`, `llvm.alloca` for local
fragments, inline `pto.redux_addi` / `pto.redux_addf` and `pto.syncthreads`
allreduce operations, and
dynamic `pto.set_flag_dyn` / `pto.wait_flag_dyn` operations for the ping-pong
events.

### Launch artifacts

- `~/.cache/ptodsl/` — JIT-compiled kernel `.so` cache
- `build/msprof_res/` — `msprof` simulator trace output

---

## Running regression checks

Run these checks from an environment where `import ptodsl` already works via a
supported `ptoas` install flow.

```bash
cd $PTOAS_REPO_ROOT
python3 ptodsl/tests/test_jit_compile.py
python3 ptodsl/tests/test_jit_diagnostics.py
python3 ptodsl/tests/test_subkernel_diagnostics.py
python3 ptodsl/tests/test_flash_attention_demo_compile.py
python3 ptodsl/tests/test_ptoas_frontend_verify.py
python3 ptodsl/tests/test_docs_as_test.py
```

Expected output:

```
ptodsl_jit_compile: PASS
ptodsl_jit_diagnostics: PASS
ptodsl_subkernel_diagnostics: PASS
ptodsl_flash_attention_demo_compile: PASS
ptodsl_ptoas_frontend_verify: PASS
ptodsl_docs_as_test: PASS
```

`ptodsl/tests/` is the canonical home for PTODSL-specific regression scripts.
The launchable sources under `ptodsl/examples/` remain examples; the
regressions that protect compile-only behavior live here alongside docs checks.
Runtime correctness coverage for PTODSL examples belongs under `test/dsl-st/`;
for example, the RMSNorm alloc-buffer/SIMT example is covered by
`test/dsl-st/rmsnorm_alloc_buffer_simt.py`.

`test_docs_as_test.py` is the docs-as-test regression for the PTODSL user
guide under `ptodsl/docs/user_guide/`. It scans every Python fenced code block
and requires each one to be explicitly classified with either
`ptodsl-doc-test` or `ptodsl-doc-pending` metadata.

- `mode="compile"` blocks are executed as-authored and must pass the PTODSL
  compile-only path, MLIR verify, and shared PTOAS frontend validation.
- `mode="compile_fragment"` blocks are embedded into explicit test fixtures so
  representative partial snippets can be compiled under a declared outer
  kernel context instead of relying on hidden heuristic context synthesis.
- `ptodsl-doc-pending` marks snippets the manual intends to treat as contract
  later, but which are still blocked on missing implementation or missing test
  harness support.

Run it directly while editing the manual:

```bash
cd $PTOAS_REPO_ROOT
python3 ptodsl/tests/test_docs_as_test.py
```

When it fails, the diagnostic includes the Markdown path, starting line number,
and target symbol so the drift can be fixed in the manual instead of searching
through generated IR logs.

These PTODSL regressions are intentionally complementary:

- `test_jit_compile.py` protects canonical authored compile probes and
  lowering contracts for the public PTODSL surface.
- `test_flash_attention_demo_compile.py` protects the bundled
  `ptodsl/examples/flash_attention_sketch.py` authored demo as a stable end-to-end
  contract.
- `test_ptoas_frontend_verify.py` protects the handoff from PTODSL-emitted
  MLIR into standalone `ptoas` frontend verification.
- `test_docs_as_test.py` protects the user manual itself: documented
  self-contained examples must still compile, fixture-backed partial fragments
  must still compile inside their declared context, and explicitly marked
  pending snippets remain visible as docs/test debt.

`test_docs_as_test.py` is not a replacement for the authored compile/demo
regressions above. It reuses the same compile-only and frontend-validation
boundaries, but its job is to keep `ptodsl/docs/user_guide/` honest rather than
to redefine the canonical demo contracts.

The legacy `ptodsl/check_ir.py` script has been retired. PTODSL validation now
lives under `ptodsl/tests/` so every regression shares the same bootstrap,
public surface, and canonical authored targets as the tracing/JIT
implementation.

---

## Public API map

The user guide under `ptodsl/docs/user_guide/` is the canonical PTODSL API
reference. This README keeps only a compact map of the public surface:

- `@pto.jit`: the only host-visible kernel entry
- `@pto.tileop`, `@pto.simt`: reusable single-core and SIMT helpers
- `pto.ptr(...)` + runtime PTO scalar annotations: public entry ABI
- `pto.make_tensor_view(...)`, `pto.partition_view(...)`, `pto.alloc_tile(...)`:
  core data-model builders
- `pto.tile.*`, `pto.mte_*`, `pto.v*`, and `pto.*` scalar helpers: operational surfaces
- default AST rewrite for Python `for` / `if`, plus explicit `pto.for_` /
  `pto.if_`: control-flow surface

Start here for the full reference:

- `ptodsl/docs/user_guide/01-introduction.md`
- `ptodsl/docs/user_guide/03-kernel-entry-and-subkernels.md`
- `ptodsl/docs/user_guide/04-type-system-and-buffer.md`
- `ptodsl/docs/user_guide/05-control-flow.md`
- `ptodsl/docs/user_guide/06-scalar-and-pointer-ops.md`
- `ptodsl/docs/user_guide/07-data-movement-ops.md`
- `ptodsl/docs/user_guide/08-compute-operations.md`
- `ptodsl/docs/user_guide/09-predicate-and-mask-ops.md`
- `ptodsl/docs/user_guide/10-sync-ops.md`
- `ptodsl/docs/user_guide/13-simt-micro-ops.md`
- `ptodsl/docs/user_guide/15-special-scalar.md`

## How the IR check works

```
generated IR  ──┐
                ├── Module.parse() → canonical string ──── == ──── PASS/FAIL
reference .pto ──┘  (strips comments, normalises SSA names and attr order)
```

Constant declaration order is preserved after the round-trip; builders must
emit constants in the same order as the reference.  The diff output makes any
mismatch immediately visible.
