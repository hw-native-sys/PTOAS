"""Desired legal 1xT VMI for per_block sf_only+packed fp32.

fp32 sf_only+packed bitwise vs ASC (bf16 sf_only+packed already matches, including TMA).

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
