"""Desired legal 1xT VMI for per_token rescale non-a5 layout (row-major / compose).

Fused 1xT rescale for row-major packed in / (1,32) packed out, or a compose that matches ASC fused SF bitwise.

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
