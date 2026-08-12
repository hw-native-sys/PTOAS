# Performance issue

The retained production-shaped ragged conversion executes correctly on device
0 but VMI remains much slower. Device-only FFTS measures CCE `134.871 us` and
VMI `492.160 us`, CCE/VMI `0.2740`. The VMI path includes its required padded
input, output, and scale copies; the CCE control processes the requested
`8001 x 16384` extent directly.
