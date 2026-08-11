# Grouped reduce/scale/convert pipeline is slower through VMI

This is a framework-free A5 reproducer for a vector pattern that reduces 256
BF16 lanes into eight group maxima, expands the eight values, scales the input,
and converts it to FP8.  Both sides are complete GM-to-UB-to-GM kernels:

- `fixtures/grouped_scale_vmi.pto` is compiled by stock PTOAS/VMI.
- `fixtures/reference_device.asc` plus `fixtures/reference_cce.cpp` is the
  direct CCE peer: an A5 vector body and a tiny stream-first host launcher.

Run `bash check.sh compile` to compile both sides and inspect the emitted VPTO.
`ACL_DEVICE_ID=0 bash check.sh benchmark` builds, launches, checks, and times
both libraries through `torch_npu` and ctypes. Generated files go under
`outputs/`. The live control uses one 256-value work item and one device launch per side; events are synchronized before collection. A Python tile loop is intentionally excluded.

| Verified device-0 event median | CCE us | VMI us | CCE/VMI |
|---|---:|---:|---:|
| one 256-value BF16 grouped conversion | 32.922 | 32.680 | 1.0074 |

Absolute values depend on device load; the live harness is the source of truth.
The compact VMI work item is intentionally small enough for compiler review while retaining reduction, broadcast, and conversion. Full-shape throughput requires partitioning inside one launch.

Requested behavior: retain group results and predicates across broadcast and
conversion, avoid layout materialization and UB spill/reload, and enable a
single tiled VMI launch to approach the demonstrated CCE throughput.
