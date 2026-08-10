// Direct CCE reference outline: E2B/BRC input-scale load, UNPK4 FP8 widen,
// vmul, grouped vmax/vcgmax, reciprocal scale construction, packed FP8 vcvt,
// and PK4 stores. All intermediates remain vector registers.
__simd_vf__ void requant_reference();
