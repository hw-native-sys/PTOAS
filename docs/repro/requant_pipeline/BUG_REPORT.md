# Performance issue

The corrected 64-launch synchronized batch measures CCE 14.682 us and VMI 14.502 us per launch (ratio 1.0124) on device 0. The earlier 50 ms number timed 28,672 host-side VMI launches. Some runs also expose a PTOAS device-LLVM frontend crash. See `README.md`.
