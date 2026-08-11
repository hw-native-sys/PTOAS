# Performance issue

The standalone CCE reference launches and measures 29.542 us for 128x7168 on
device 1, with an exact host golden. The corresponding VMI fixture reaches the
PTOAS device-LLVM stage and crashes the CANN frontend (exit 139). This is a
compiler blocker before performance parity can be measured. See `README.md`.
