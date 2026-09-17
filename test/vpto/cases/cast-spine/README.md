# cast-spine: end-to-end cases for `--vmi-prefer-cast-spine-round-trip`

Five hand-authored VPTO kernels that exercise the direction-spine peephole in
VMI layout assignment.  Each case is a complete `test/vpto/cases` entry
(`kernel.pto`, `launch.cpp`, `main.cpp`, `golden.py`, `compare.py`,
`ptoas.flags`) and runs through
`test/vpto/scripts/run_host_vpto_validation.sh` unchanged.

The predicate under test recognises a *closed nested round trip* in a cast
spine: a direction sequence `<up, down, up, down>` whose first leg's source
element type equals the last leg's result element type.  When it matches, the
peephole seeds the deinterleaved layout family instead of the lane-stride
family.

| case | chain | legs | recognised |
| --- | --- | --- | --- |
| `roundtrip-bf16-f8-bf16` | `bf16 -vcvt-> f32 -vcvt-> f8E4M3FN -vcvt-> f32 -vmul-> bf16` | `<up, down, up, down>`, closes on bf16 (32<->8 inner leg) | **yes** |
| `roundtrip-bf16-f16-bf16` | `bf16 -vcvt-> f32 -vcvt-> f16 -vcvt-> f32 -vmul-> bf16` | `<up, down, up, down>`, closes on bf16 (32<->16 inner leg) | **yes** |
| `roundtrip-f32-f16-f32` | `f32 -vcvt-> f16 -vcvt-> f32 -vmul-> f32` | `<down, up>` | no |
| `roundtrip-si32-si16-si32` | `si32 -vcvt-> si16 -vcvt-> si32 -vadd-> si32` | `<down, up>` | no |
| `roundtrip-ui32-ui8-ui32` | `ui32 -vcvt-> ui8 -vcvt-> ui32 -vadd-> ui32` | `<down, up>` | no |

The three counterexamples are the counterexamples of
`test/lit/vmi_new/vmi_layout_assignment_cast_roundtrip.pto`; the two positive
cases are that file's two nested round trips made runnable (the 8-bit chain is
also the `fused_quant_dequant_vmi_opt.pto` shape).  Both take a spine-scoped
composite narrow form (one conversion per wide part, no `pto.vor`): the 8-bit
chain converts in sub-lane 0 of every 32-bit container (`part = "P0"`), the
16-bit chain in the even lane of every 16-bit container (`part = "EVEN"`).

## Kernel structure

Every kernel has the same shape:

```
mte_gm_ub -> vecscope { scf.for REPEAT { scf.for TILE { chain } } } -> mte_ub_gm
```

* the inner `TILE` loop walks the whole UB buffer once (64 tiles);
* the outer `REPEAT` loop (16) replays it, so one launch performs
  `16 * 64` chain iterations.  This keeps the per-iteration VPTO shape stable -
  so the op census of the two compiler arms is directly comparable - while
  making a single launch long enough to time above the launch-overhead floor;
* input and output are read from / written to GM through the standard
  `pto.mte_gm_ub` / `pto.mte_ub_gm` pair, so the case is a real kernel, not a
  fragment.

Input values are chosen to be exactly representable in every intermediate type,
so each case is *lossless* and the oracle in `golden.py` is bit-exact
independent of rounding mode; a wrong layout still shows up because the value
pattern differs per lane.

## Running (default arm = whatever the compiler default is)

```bash
cd /home/mouliangyu/projects/github.com/mouliangyu/PTOAS-0
source .work/issue1337-env/env.sh
ACL_DEVICE_ID=1 DEVICE=BOARD \
  CASE_NAME=cast-spine/roundtrip-bf16-f8-bf16 \
  WORK_SPACE=/tmp/cast-spine-work \
  bash test/vpto/scripts/run_host_vpto_validation.sh
```

`aclrtSetDevice` needs a root context on this host, so submit the run through
`task-submit --device 1 --run ...` (see the experiment report).

## Running both compiler arms

`ptoas.flags` in these directories intentionally does *not* pin the switch, so
the case follows the compiler default.  To pin the arms explicitly - which is
what the A/B experiment does - pass the flag on the command line or copy the
case and override `ptoas.flags`:

```bash
# OFF arm (pre-recognition layout)
ptoas --pto-arch a5 --pto-backend=vpto --emit-vpto \
  --vmi-prefer-cast-spine-round-trip=false \
  test/vpto/cases/cast-spine/roundtrip-bf16-f8-bf16/kernel.pto -o /tmp/off.pto

# ON arm (deinterleaved family)
ptoas --pto-arch a5 --pto-backend=vpto --emit-vpto \
  --vmi-prefer-cast-spine-round-trip=true \
  test/vpto/cases/cast-spine/roundtrip-bf16-f8-bf16/kernel.pto -o /tmp/on.pto
```

`.work/issue1337-env/experiments/issue1337-cast-spine-e2e/tools/` contains the
harness that automates this (`run_cast_spine_arm.sh`, `measure_all.sh`,
`vpto_census.py`).

## Timing knobs

`main.cpp` reads two environment variables:

| variable | default | meaning |
| --- | --- | --- |
| `CAST_SPINE_ITERS` | 1 | kernel launches inside the ACL-event timed window |
| `CAST_SPINE_WARMUP` | 3 | untimed launches issued before the window |

The timed window is bracketed by `aclrtRecordEvent` on the same stream, so
device time only is measured; H2D/D2H copies and allocation are outside it.
Running the same binary at several `CAST_SPINE_ITERS` values and fitting a line
removes the residual fixed launch cost.
