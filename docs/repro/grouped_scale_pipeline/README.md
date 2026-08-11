# Grouped reduce/scale/convert pipeline is slower through VMI

This is a framework-free A5 reproducer for a vector pattern that reduces 256
BF16 lanes into eight group maxima, expands the eight values, scales the input,
and converts it to FP8.  Both sides are complete GM-to-UB-to-GM kernels:

- `fixtures/grouped_scale_vmi.pto` is compiled by stock PTOAS/VMI.
- `fixtures/reference_device.asc` plus `fixtures/reference_cce.cpp` is the
  direct CCE peer: an A5 vector body and a tiny stream-first host launcher.

Run `bash check.sh compile` to compile both sides and inspect the emitted VPTO.
`ACL_DEVICE_ID=1 bash check.sh benchmark` builds, launches, checks, and times
both libraries through `torch_npu` and ctypes. Generated files go under
`outputs/`. The command uses a common 128x7168 output extent: the compact
VMI work item is invoked once per contiguous 256 values, while CCE uses its
single 72-core tiled launch.

| Verified device-1 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| 128x7168 BF16 grouped conversion | 32.543 | 17557.001 | 0.0019 |

Absolute values depend on device load; the live harness is the source of truth.
The compact VMI work item is intentionally small enough for compiler review
while retaining reduction, broadcast, and conversion. Its tiled launch count
is reported explicitly so it is not mistaken for a single-kernel parity claim.

Requested behavior: retain group results and predicates across broadcast and
conversion, avoid layout materialization and UB spill/reload, and enable a
single tiled VMI launch to approach the demonstrated CCE throughput.
