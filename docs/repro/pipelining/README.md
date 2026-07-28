# pipelining — VMI Persistent stages vs CCE ping-pong repro

Self-contained compile (+ optional on-device CCE) repro for long-K Persistent
quant / dequant and residual-mix-style scale: production wants stages=2 and
`block_k > 512` so MTE and vector stay overlapped. Ascend CCE can express and
**run** stages=2 ping-pong; VMI on tag **vmi-v0.1.3** cannot reach the same
wide-tile schedule — layout assignment rejects 1024-wide tiles, and the
hand-written MI path crashes bisheng even after ptoas emits LLVM IR.

## Pin

| Component | Ref |
|-----------|-----|
| PTOAS | tag **vmi-v0.1.3** (`ptoas` from that build; not PyPI-only) |
| Branch | `zjw/pipelining_issue` |
| CANN | 9.0 / `bisheng` |
| `PTOAS_ROOT` | repo root (`../../..` from this directory) |

If `install/bin/ptoas` crashes on missing MLIR libs, point tools at a build tree:

```bash
export PTOAS_TOOLS_ROOT=/path/to/PTOAS-built-at-vmi-v0.1.3
```

## What breaks

| Path | stages | block_k | Fault mode |
|------|-------:|--------:|------------|
| CCE (`fixtures/reference_asc_cce*.asc`, `device/scale_stages2.asc`) | 2 | 512–1024 | **OK** — compile; device host numerical OK |
| VMI stages=1 / 512 (`current_slow_vmi.*`) | 1 | 512 | **emit-vpto OK** |
| VMI stages=2 / 512 (`isolate_stages2_blockk512_vmi.*`) | 2 | 512 | **emit-vpto OK** (dual-buffer alone is fine) |
| VMI stages=1 / 1024 (`isolate_stages1_blockk1024_vmi.*`) | 1 | 1024 | **Layout reject** on 1024-wide vmul |
| VMI stages=2 / 1024 (`desired_vmi.*`) | 2 | 1024 | **Layout reject** (stages+wide) |
| Cast-back VMI (`cast_back_stages*_*.*`) | 1–2 | 512–1024 | **Layout reject** on f8↔bf16 Persistent |
| VMI target MI (`target_mi.pto`) | 2 | 1024 | **Bisheng crash** after LLVM IR emit |

So the blocker for the desired quant schedule is **wide `block_k=1024` layout**,
not dual-buffer stages alone. Cast-back Persistent hits the same layout wall.

## Check results (vmi-v0.1.3 tools, 2026-07-28)

```bash
source scripts/env.sh
# optional: export PTOAS_TOOLS_ROOT=...
./scripts/check_stages.sh
```

| Step | Result |
|------|--------|
| `reference_asc_cce.asc` / `_f8` / `_cast_back` | **PASS** bisheng `--cce-aicore-only` |
| `device/scale_stages2.asc` aicore object | **PASS** |
| `current_slow_vmi.pto` `--emit-vpto` | **PASS** |
| `isolate_stages2_blockk512_vmi.pto` | **PASS** |
| `isolate_stages1_blockk1024_vmi.pto` | **FAIL** layout |
| `desired_vmi.pto` | **FAIL** layout |
| `cast_back_stages1_k512_vmi.pto` | **FAIL** layout |
| `cast_back_stages2_k1024_vmi.pto` | **FAIL** layout |
| `lowered_vpto.pto` → LLVM IR | **PASS** |
| `target_mi.pto` → emit + LLVM IR | **PASS**; bisheng `.o` **FAIL** |

Full log: `sim_outputs/check_stages/compile_results.txt`

## On-device CCE

```bash
cd device
./run.sh                 # build + launch stages=2 f32 scale, check max_err
MSOPPROF=1 ./run.sh      # optional msopprof wrap if installed
```

Measured 2026-07-28:

```
scale_stages2: N=2048 mism=0 max_err=0 OK
```

CCE stages=2 runs with dual-buffer HardEvent overlap. VMI stages=2 / `block_k=1024`
does not compile through layout — that is the PTOAS gap this package isolates.

## Layout

```
pipelining/
  fixtures/     CCE twins, VMI isolate + cast-back cells, target_mi
  device/       AscendC host for stages=2 CCE numerical check
  scripts/      env.sh, check_stages.sh
```

## Done when

VMI can compile and run stages=2 / `block_k=1024` Persistent tiles with
dual-buffer ping-pong — same overlap pattern as CCE and `target_mi.pto`.
