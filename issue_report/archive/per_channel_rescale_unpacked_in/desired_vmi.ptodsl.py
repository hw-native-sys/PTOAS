"""Desired legal 1xT VMI for per_channel e4m3 rescale unpacked row-major input SF.

On-device dequant of unpacked float input SF then per-channel requant (ASC L1). Packed TMA-in fused path exists but fails on large tiles (see per_channel_tma_in_large_shape).

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
