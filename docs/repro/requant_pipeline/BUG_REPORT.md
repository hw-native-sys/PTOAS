# Performance issue

The former 128-row control did not retain the production operation and cannot
establish a performance comparison.  This report now uses the full `8064 x
7168` CCE kernel and the corresponding eight-strip-plus-requant VMI
composition.  Both are launched through the same stream-first ctypes ABI and
timed with the same event and device-only profiler policy.

The retained static final strip also produces a VMI-only mismatch on its valid
tail rows.  The harness reports the mismatch count and continues with a full
device-time measurement, so the report preserves both the correctness blocker
and the gap between the one-pass CCE schedule and the composed VMI lowering.

Verified device-0 timings are CCE `46.808 us` versus VMI `176.554 us` for the
cold-L2 event median (`0.2651` CCE/VMI), and CCE `45.300 us` versus VMI
`126.602 us` for device-only FFTS (`0.3578`). The matching head/tail split is
`0` mismatches through row `7167`, then VMI-only mismatches in the padded tail.
