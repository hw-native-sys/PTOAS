# Feature request

PTODSL cannot express a typed GM pointer table. The checked negative fixture
fails before lowering with `pto.ptr(...) does not support element type
!pto.ptr<f32, gm>`, while the direct CCE pointer-table kernel builds and
launches through the same ctypes stream ABI.

The fixed-argument VMI program is a compile/run control only. It does not
implement the indirect pointer-table workload, so no CCE/VMI throughput ratio
is claimed. Add the bounds-checkable pointer-table load and launcher support
specified in `README.md` before a fair performance comparison is possible.
