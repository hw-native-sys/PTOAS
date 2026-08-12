# Performance issue

The former 128-row control did not retain the production operation and cannot
establish a performance comparison.  This report now uses the full `8064 x
7168` CCE kernel and the corresponding eight-strip-plus-requant VMI
composition.  Both are launched through the same stream-first ctypes ABI and
timed with the same event and device-only profiler policy.

The final static VMI strip has 128 inactive rows. The standalone wrapper
zero-pads those rows because the static unpack kernel reads them. They are not
part of the public `8064`-row result; every defined output and scale element is
numerically checked against CCE before timing.

Verified device-0 timings are CCE `47.041 us` versus VMI `302.386 us` for the
cold-L2 event median (`0.1556` CCE/VMI), and CCE `45.430 us` versus VMI
`145.072 us` for device-only FFTS (`0.3132`).
