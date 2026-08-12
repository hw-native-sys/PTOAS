# Performance issue

The cold-L2 event interval is launch-floor dominated for this one-kernel
operation and therefore does not expose the device gap. The standalone FFTS
measurement instead records CCE `21.555 us` versus VMI `26.990 us` on device 0
over 30 repetitions, CCE/VMI `0.7986`. Both paths use one 72-core persistent
kernel and produce matching output.
