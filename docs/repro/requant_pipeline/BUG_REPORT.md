# Performance issue

The standalone device-0 run launches both paths over 128x7168: CCE is 32.960
us and VMI is 50553.814 us (ratio 0.0007), with both host goldens checked.
Some runs also expose a PTOAS device-LLVM frontend crash. The direct CCE
baseline proves the algorithm is fast and launchable independently. See
`README.md`.
