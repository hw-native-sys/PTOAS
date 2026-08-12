# Pointer-table ABI and correctness report

PTODSL cannot express a typed GM pointer table. The checked negative fixture
fails before lowering with `pto.ptr(...) does not support element type
!pto.ptr<f32, gm>`, while the direct CCE pointer-table kernel builds and
launches through the same ctypes stream ABI.

The direct CCE path loads six runtime pointer tables and performs the full
`8192 x 4096`, 10-layer recurrence. The equivalent VMI body uses dense
layer-major buffers and the host stack/unstack copies required by the current
ABI. This is the production-shaped performance comparison; the nested-pointer
negative test separately shows why VMI cannot consume the CCE table directly.

On device 0, CCE takes `1460.010 us` and the complete stacked VMI callable
takes `4594.875 us` by FFTS device time (`CCE/VMI = 0.3177`). Event medians
with an L2 flush independently report `1485.899 us` and `4630.336 us`
(`0.3209`). The nonzero correctness probe fails fast with
`layer_input_maxabs=0.1875` and `residual_maxabs=0.06244755`; the VMI body
therefore needs correctness repair as well as a pointer-table ABI that avoids
the staging copies. The checked-in timings are diagnostic evidence for the ABI
cost, not an accepted throughput result while correctness fails.
