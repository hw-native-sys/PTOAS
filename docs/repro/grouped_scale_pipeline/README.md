# Grouped reduce/scale/convert pipeline is slower through VMI

> **Scope:** the checked-in PTO and CCE retain the extracted production-shaped
> `8001x16384` packed per-token schedule (8032 padded rows, 72 workers,
> double-buffered UB). Both compile and launch on the pinned toolchain.

This is a framework-free A5 reproducer for a vector pattern that reduces 256
BF16 lanes into eight group maxima, expands the eight values, scales the input,
and converts it to FP8.  Both sides are complete GM-to-UB-to-GM kernels:

- `fixtures/production_group_vmi.pto` is compiled by stock PTOAS/VMI into the
  complete 72-core VMI path.
- `fixtures/reference_device.asc` plus `fixtures/reference_cce.cpp` is the
  direct CCE peer: an A5 vector body and a tiny stream-first host launcher.

Run `bash check.sh compile` to compile both sides and inspect the emitted VPTO.
`ACL_DEVICE_ID=0 bash check.sh benchmark` builds, launches, checks, and times
both libraries through `torch_npu` and ctypes. Generated files go under
`outputs/`. It reports both identical cold-L2 event timing and device-only
FFTS timing; the latter is the primary comparison.

| Verified device-0 measurement | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| Cold-L2 event median | 183.877 | 494.887 | 0.3716 |
| Device-only FFTS | 134.580 | 491.501 | 0.2738 |

Absolute values depend on device load; the live harness is the source of truth.
The full VMI body retains reduction, broadcast, conversion, persistent tiling,
and pipeline events. For the ragged production shape, the VMI adapter's padded
input/output path is included in the device-only total; this reproduces the
observed slowdown. The harness uses identical 256-MB L2 flushes and device
events for both paths, while FFTS excludes host launch overhead.

Requested behavior: retain group results and predicates across broadcast and
conversion, avoid layout materialization and UB spill/reload, and enable a
single tiled VMI launch to approach the demonstrated CCE throughput.
