# pipelining — SoftPipeline stages with ≤256-lane VF

Minimal, **kernel-agnostic** compile (+ optional on-device CCE) repro for
Persistent SoftPipeline. ASC/CCE uses **chunked SIMD (≤256)** + `stages≥2` +
real MTE; TileKernels VMI already authors that way. The remaining asks are
healthy multi-stage launch / fatobj, not “must have single-op `size=1024` VMI”.

Product TK measurements (2026-07-28 stages sweep):

| Kernel | After sweep | Remaining gap |
|--------|-------------|---------------|
| `per_block_cast` | landed **stages=2 / `block_k=1024`** (chunked strip 128) | still ~0.85× historical ASC GB/s @ 8192×2048 |
| `cast_back` | keep **stages=1 / `tile_k=512`** (HALF_TILE=256); stages=2 regresses BW | ~0.82× ASC |
| MHC `pre_fwd` / `post_fwd` | keep **stages=2**; stages=3 regresses N=8192 | ~0.84–0.87× ASC (gap **07** UNPK) |
| MHC `post_bwd` | SoftPipeline out of scope | gap **03** reduce pad |

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

## Structure (report shape)

1. **Working ASC/CCE baseline** — `fixtures/reference_asc_cce.asc`,
   `device/scale_stages2.asc`: stages=2, HardEvent ping-pong, device numerical OK.
2. **Desired VMI** — same outer schedule (`stages=2`, DMA `block_k=512`) with
   **chunked ≤256** VF body (`isolate_stages2_chunked256_vmi.py`). Primary ask.
3. **Current slow / isolate** — stages=1 chunked (`isolate_stages1_chunked256_vmi.py`);
   stages=2 with size=512 single-op still emit-OK (`isolate_stages2_blockk512_vmi.*`).
4. **Optional wide VF** — `desired_vmi.*` / `isolate_stages1_blockk1024_*` single-op
   `size=1024` layout reject. Demoted: ASC SoftPipeline parity does **not** require this.
5. **Remaining PTOAS asks after TK sweep** — healthy fatobj/HIVM for multi-stage
   product kernels on some pins (`bisheng` exit 139); gap **03** for `post_bwd`;
   gap **07** for MHC fwd UNPK. Not “need 512/1024-wide logical VF”.

## What breaks / what works

| Path | stages | VF body | Fault mode |
|------|-------:|---------|------------|
| CCE (`reference_asc_cce.asc`, `device/scale_stages2.asc`) | 2 | chunked | **OK** device |
| VMI chunked-256 stages=1/2 (`.py` fixtures) | 1–2 | 256×2 over 512 | frontend emit (primary cells) |
| VMI stages=2 / single-op 512 | 2 | size=512 | **emit-vpto OK** |
| VMI stages=1 / single-op 1024 | 1 | size=1024 | **Layout reject** (optional feature) |
| VMI `desired_vmi` stages=2 / 1024 | 2 | size=1024 | **Layout reject** (optional) |
| VMI `target_mi` | 2 | wide MI | emit OK; **bisheng** often exit 139 |

## Check

```bash
source scripts/env.sh
./scripts/check_stages.sh
```

Full log: `sim_outputs/check_stages/compile_results.txt`

## On-device CCE

```bash
cd device
./run.sh
```

```
scale_stages2: N=2048 mism=0 max_err=0 OK
```

TileLang SoftPipeline isolate (ASC stages 1→2 BW gain with max_err=0; PTO fatobj
bisheng 139): `tilelang-vmi-pipelining_issue/examples/vmi_pipelining_repro`
(`micro_stages256.py` / `run_micro_stages.py`).

E2E product patterns: `vmi-demo-pipeline/pipeline_study`.

## Done when

VMI SoftPipeline with **chunked ≤256** VF + `stages≥2` + real MTE is healthy
end-to-end (including fatobj) for the TK schedules above. Wide single-op
`size=1024` is optional sugar, not the SoftPipeline blocker.
