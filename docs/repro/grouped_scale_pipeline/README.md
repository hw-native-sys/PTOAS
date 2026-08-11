# Grouped reduce/scale/convert pipeline is slower through VMI

> **Scope:** the checked-in PTO and CCE retain the extracted production-shaped
> `8001x16384` packed per-token schedule (8032 padded rows, 72 workers,
> double-buffered UB). Both compile and launch on the pinned toolchain.

This is a framework-free A5 reproducer for a vector pattern that reduces 256
BF16 lanes into eight group maxima, expands the eight values, scales the input,
and converts it to FP8.  Both sides are complete GM-to-UB-to-GM kernels:

- `fixtures/grouped_scale_vmi.pto` is compiled by stock PTOAS/VMI and currently
  reproduces the device-LLVM failure on the production shape.
- `fixtures/reference_device.asc` plus `fixtures/reference_cce.cpp` is the
  direct CCE peer: an A5 vector body and a tiny stream-first host launcher.

Run `bash check.sh compile` to compile both sides and inspect the emitted VPTO.
`ACL_DEVICE_ID=0 bash check.sh benchmark` builds, launches, checks, and times
both libraries through `torch_npu` and ctypes. Generated files go under
`outputs/`. The live control enqueues 64 identical launches between each event pair and reports synchronized per-launch medians; the batch is identical on both sides.

| Verified device-0 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| production ragged packed conversion (`8001x16384`, padded VMI path) | 134.999 | 491.924 | 0.2744 |

Absolute values depend on device load; the live harness is the source of truth.
The full VMI body retains reduction, broadcast, conversion, persistent tiling,
and pipeline events. For the ragged production shape, the VMI adapter's padded
input/output path is included in the device-only total; this reproduces the
private study's slowdown. The harness uses identical 256-MB L2 flushes and
device events for both paths.

Requested behavior: retain group results and predicates across broadcast and
conversion, avoid layout materialization and UB spill/reload, and enable a
single tiled VMI launch to approach the demonstrated CCE throughput.
