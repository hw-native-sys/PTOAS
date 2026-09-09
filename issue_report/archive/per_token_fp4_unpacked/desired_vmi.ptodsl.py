"""Desired legal 1xT VMI for per_token FP4 unpacked SF.

On-device e2m1 quant with unpacked float SF (ASC L1). Packed UE8M0 e2m1 cast is already ported.

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
