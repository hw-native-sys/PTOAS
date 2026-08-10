// Pseudocode-level CCE reference for one 256-lane group-of-eight body.
// It intentionally contains no application or framework wrapper.
__simd_vf__ void grouped_scale_reference(__ubuf__ bfloat16_t *src,
                                         __ubuf__ bfloat16_t *scale,
                                         __ubuf__ int8_t *dst) {
  // Direct implementations use deinterleaved loads, vcgmax, E2B/BRC scale
  // expansion, four packed FP8 conversions, and four PK4 stores. The desired
  // VMI lowering should not add UB spills or layout shuffles around this chain.
}
