# PTOAS bug report: BF16 widen and multiply

The VMI lowering of contiguous BF16-to-FP32 conversion followed by FP32 multiply is 4.8-8.5% slower than an equivalent direct CCE implementation on the larger measured workloads. The included CCE and PTODSL sources are self-contained and use the same grid, tiled data movement, launch ABI, and correctness oracle.

Please improve VMI lowering or scheduling so this operation reaches `CCE/VMI >= 0.98`.
