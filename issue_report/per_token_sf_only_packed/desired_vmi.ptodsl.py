"""Desired legal 1xT VMI for per_token sf_only+packed.

On-device packed UE8M0 SF-only path (same VF amax/UE8M0 as full cast, no payload store). ASC tests this for every packed combo.

Production VMI does not yet emit this IR (host assert or leftover).
Do not substitute host-pad or host TMA permute.
"""
