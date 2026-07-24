# VMI feature gaps (docs fixtures)

Minimal reproducers for PTOAS capabilities that still block VMI from reaching
ASC bandwidth on quantization and residual-mix style vector kernels. These are
**documentation fixtures**, not lit tests (no `// RUN:` lines).

Each subdirectory has:

| File | Role |
|---|---|
| `README.md` | Algorithm need, current slow path + reproduce, desired failure, target MI / bisheng status |
| `current_slow_vmi.pto` | Compiles and runs today, but slow / non-optimal; **must** lower with `pto-test-opt` + `$PASS` |
| `lowered_vpto.pto` | Checked-in dump from that lower |
| `desired_vmi.pto` | Idiomatic ask (may fail — README pastes the error) |
| `target_mi.pto` | Desired `pto.mi` / VPTO shape; exercised with `ptoas` + `bisheng` |

## Gaps

| # | Gap | Why ASC-level BW cares |
|---|---|---|
| 01 | [Scalar GM dcache bypass](01_scalar_gm_dcache_bypass/) | Tiny per-block SF writes still force bulk MTE |
| 02 | [Compact inverse `vbrc`](02_compact_inv_vbrc/) | Padded recip + BRC reload tax on quant |
| 03 | [Direct scalar / 1PT reduce store](03_direct_scalar_reduce_store/) | Residual-mix post-bwd still pads to 8; ASC uses ONEPT/1PT |
| 04 | [Packed UE8M0 SF](04_packed_ue8m0_sf/) | Unpacked SF doubles traffic |
| 05 | [Persistent stages / `block_k`](05_persistent_stages_block_k/) | Per-block + cast-back capped at stages=1 / `block_k≤512`; stages=2 faults in practice |
| 06 | [Small-L ui8→ui16 widen](06_small_l_ui8_widen/) | L=8 `vcvt` illegalizes to residual `extui` (blocks fused quant→dequant) |
| 07 | [bf16 UNPK / PK fuse](07_bf16_unpk_pk_fuse/) | Residual-mix **fwd** strips: continuous load only feeds EVEN after UNPK; `dist_mode=unpack` illegal |

Tip findings (high level, no external repo names): residual-mix **bwd** still
pays a reduce pad-to-8 store tax (**03**); residual-mix **fwd** needs full
UNPK/PK strip fuse (**07**); per-block / cast-back Persistent stay on stages=1 /
`block_k≤512` (**05**); fused small-L UE8M0 widen hits L=8 `extui` (**06**).

## How to reproduce

Assumes a local `build/` of this tree (or any install with `pto-test-opt` /
`ptoas` on `PATH`) and Ascend `bisheng` available:

```bash
# From repo root after a normal build:
PTO_TEST_OPT=${PTO_TEST_OPT:-$PWD/build/tools/pto-test-opt/pto-test-opt}
PTOAS=${PTOAS:-$PWD/build/tools/ptoas/ptoas}
# Example Ascend install; override if yours differs:
BISHENG=${BISHENG:-/usr/local/Ascend/cann-9.0.0/tools/bisheng_compiler/bin/bisheng}
export PATH="$(dirname "$BISHENG"):$PATH"

PASS='-vmi-lower-unified-to-legacy -vmi-mask-granularity-assignment -vmi-layout-assignment -vmi-to-vpto'
cd docs/feature-gaps/vmi/<gap_dir>

# Current slow path (must succeed; refresh lowered_vpto.pto)
"$PTO_TEST_OPT" current_slow_vmi.pto $PASS -o lowered_vpto.pto

# Desired (often fails; paste error into README)
"$PTO_TEST_OPT" desired_vmi.pto $PASS -o /tmp/desired.pto

# Target MI → VPTO IR → HIVM LLVM → device object
"$PTOAS" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --emit-vpto -o /tmp/t.emit.pto target_mi.pto
"$PTOAS" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --emit-vpto-llvm-ir -o /tmp/t.ll target_mi.pto
"$BISHENG" --target=hiipu64-hisilicon-cce -march=dav-c310-vec \
  --cce-aicore-arch=dav-c310-vec --cce-aicore-only -c -x ir /tmp/t.ll -o /tmp/t.o
```

Use `--emit-vpto-llvm-ir` for LLVM IR (not the older `--vpto-emit-hivm-llvm` name).
