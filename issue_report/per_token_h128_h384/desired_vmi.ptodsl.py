"""Desired legal 1xT VMI for per_token H=128/384 (ASC hidden % 128; VMI VL=256).

Legal 1xT 128-lane strip (create_mask in {1,64,128,256}) so Persistent tiles cover hidden=128 (one tile) and hidden=384 (three tiles), matching ASC hidden % 128. Host-pad is not a port.

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
