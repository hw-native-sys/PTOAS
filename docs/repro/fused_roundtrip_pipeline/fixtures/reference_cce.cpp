// Direct reference keeps amax, reciprocal scales, packed FP8 values, and
// dequant scales in vector registers across the round trip.
__simd_vf__ void fused_roundtrip_reference();
