"""Desired legal 1xT VMI for per_token TMA-col unpacked SF.

On-device TMA-col store of unpacked float SF. Host permute of row-major SF is not a port.

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
