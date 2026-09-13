#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void per_channel_cast_kernel_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t x_ub_version_counter_1, int32_t w) {
  vector_f32 amax[4];
  vector_f32 sf_inv[4];
  vector_bf16 amax_bf16[2];
  vector_bool v = asc_create_mask_b16(PAT_ALL);
  auto zero_bf16 = simd_inst::vdup<bfloat16_t>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
    vector_u16 v_ = simd_inst::vdup<uint16_t>(32767, v, MODE_ZEROING);
  auto bf16_abs_mask = (*(vector_bf16 *)(&(v_)));
    vector_s16 v__1 = simd_inst::vshrs(simd_inst::vci<int16_t>(0, INC_ORDER), 5, v, MODE_ZEROING);
  auto sf_lane = (*(vector_u16 *)(&(v__1)));
  auto sf_offsets = simd_inst::vadd(simd_inst::vmuls(simd_inst::vshrs(sf_lane, 1, v, MODE_ZEROING), 64, v, MODE_ZEROING), simd_inst::vand(sf_lane, simd_inst::vdup<uint16_t>(1, v, MODE_ZEROING), v, MODE_ZEROING), v, MODE_ZEROING);
  vector_bool sf_mask = asc_create_mask_b8(PAT_ALL);
  for (int32_t chunk = 0; chunk < 4; ++chunk) {
    amax_bf16[0] = simd_inst::vdup<bfloat16_t>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
    for (int32_t row = 0; row < 32; ++row) {
      vector_bool v_1 = asc_create_mask_b8(PAT_ALL);
      auto raw_f32_0 = simd_inst::vcvt<float>(simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((((x_ub_version_counter_1 & 1) * 32768) + (row * 1024)) + (chunk * 256))])), 0), v_1, PART_P0, MODE_ZEROING);
      auto raw_f32_1 = simd_inst::vcvt<float>(simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (row * 1024)) + (chunk * 256)) + 64)])), 0), v_1, PART_P0, MODE_ZEROING);
      auto v_2 = simd_inst::vdintlv((*(vector_bf16 *)(&(raw_f32_0))), (*(vector_bf16 *)(&(raw_f32_1))));
      auto sf_exponents = simd_inst::vgather2((&(buf_dyn_shmem[(((((x_ub_version_counter_1 & 1) * 1024) + (chunk * 256)) + (row * 2)) + 131072)])), sf_offsets, sf_mask);
      auto sf_bits = simd_inst::vshls(sf_exponents, 7, v, MODE_ZEROING);
      auto values_bf16 = simd_inst::vmul(v_2.v1, (*(vector_bf16 *)(&(sf_bits))), v, MODE_ZEROING);
      simd_inst::vsts_norm(values_bf16, (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((row * 256) + 66560)])), 0, v);
      amax_bf16[0] = simd_inst::vmax(amax_bf16[0], simd_inst::vand(values_bf16, bf16_abs_mask, v, MODE_ZEROING), v, MODE_ZEROING);
    }
    auto v_3 = simd_inst::vintlv(zero_bf16, amax_bf16[0]);
      vector_bf16 v__2 = v_3.v0;
    amax[0] = (*(vector_f32 *)(&(v__2)));
      vector_bf16 v__3 = v_3.v1;
    amax[1] = (*(vector_f32 *)(&(v__3)));
    amax_bf16[1] = simd_inst::vdup<bfloat16_t>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
    for (int32_t row_1 = 0; row_1 < 32; ++row_1) {
      vector_bool v_4 = asc_create_mask_b8(PAT_ALL);
      auto raw_f32_0_1 = simd_inst::vcvt<float>(simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (row_1 * 1024)) + (chunk * 256)) + 128)])), 0), v_4, PART_P0, MODE_ZEROING);
      auto raw_f32_1_1 = simd_inst::vcvt<float>(simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (row_1 * 1024)) + (chunk * 256)) + 192)])), 0), v_4, PART_P0, MODE_ZEROING);
      auto v_5 = simd_inst::vdintlv((*(vector_bf16 *)(&(raw_f32_0_1))), (*(vector_bf16 *)(&(raw_f32_1_1))));
      auto sf_exponents_1 = simd_inst::vgather2((&(buf_dyn_shmem[(((((x_ub_version_counter_1 & 1) * 1024) + (chunk * 256)) + (row_1 * 2)) + 131200)])), sf_offsets, sf_mask);
      auto sf_bits_1 = simd_inst::vshls(sf_exponents_1, 7, v, MODE_ZEROING);
      auto values_bf16_1 = simd_inst::vmul(v_5.v1, (*(vector_bf16 *)(&(sf_bits_1))), v, MODE_ZEROING);
      simd_inst::vsts_norm(values_bf16_1, (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((row_1 * 256) + 66688)])), 0, v);
      amax_bf16[1] = simd_inst::vmax(amax_bf16[1], simd_inst::vand(values_bf16_1, bf16_abs_mask, v, MODE_ZEROING), v, MODE_ZEROING);
    }
    auto v_6 = simd_inst::vintlv(zero_bf16, amax_bf16[1]);
      vector_bf16 v__4 = v_6.v0;
    amax[2] = (*(vector_f32 *)(&(v__4)));
      vector_bf16 v__5 = v_6.v1;
    amax[3] = (*(vector_f32 *)(&(v__5)));
    simd_inst::mem_bar(VST_VLD);
    auto amax_1 = amax[0];
    vector_bool v_7 = asc_create_mask_b32(PAT_ALL);
    auto clamped_amax = simd_inst::vmax(amax_1, simd_inst::vdup<float>(float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v_7, MODE_ZEROING), v_7, MODE_ZEROING);
    auto raw_sf = simd_inst::vmul(clamped_amax, simd_inst::vdup<float>(float(0x1.2492492492492p-9f/*2.232143e-03*/), v_7, MODE_ZEROING), v_7, MODE_ZEROING);
    auto raw_sf_bits = (*(vector_u32 *)(&(raw_sf)));
    auto sf = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(raw_sf))), simd_inst::vdup<uint32_t>(1, v_7, MODE_ZEROING), v_7, MODE_ZEROING), 23, v_7, MODE_ZEROING), 1, v_7, MODE_ZEROING);
    auto sf_inv_exponent = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v_7, MODE_ZEROING), sf, v_7, MODE_ZEROING);
      vector_u32 v__6 = simd_inst::vshls(sf_inv_exponent, 23, v_7, MODE_ZEROING);
    auto __2 = (*(vector_f32 *)(&(v__6)));
    sf_inv[0] = __2;
    simd_inst::vsts_pack_quarter(sf, (__ubuf__ uint32_t*)(&(buf_dyn_shmem[((((w & 1) * 1024) + (chunk * 256)) + 149504)])), 0, v_7);
    auto amax_2 = amax[1];
    vector_bool v_8 = asc_create_mask_b32(PAT_ALL);
    auto clamped_amax_1 = simd_inst::vmax(amax_2, simd_inst::vdup<float>(float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v_8, MODE_ZEROING), v_8, MODE_ZEROING);
    auto raw_sf_1 = simd_inst::vmul(clamped_amax_1, simd_inst::vdup<float>(float(0x1.2492492492492p-9f/*2.232143e-03*/), v_8, MODE_ZEROING), v_8, MODE_ZEROING);
    auto raw_sf_bits_1 = (*(vector_u32 *)(&(raw_sf_1)));
    auto sf_1 = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(raw_sf_1))), simd_inst::vdup<uint32_t>(1, v_8, MODE_ZEROING), v_8, MODE_ZEROING), 23, v_8, MODE_ZEROING), 1, v_8, MODE_ZEROING);
    auto sf_inv_exponent_1 = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v_8, MODE_ZEROING), sf_1, v_8, MODE_ZEROING);
      vector_u32 v__7 = simd_inst::vshls(sf_inv_exponent_1, 23, v_8, MODE_ZEROING);
    auto __3 = (*(vector_f32 *)(&(v__7)));
    sf_inv[1] = __3;
    simd_inst::vsts_pack_quarter(sf_1, (__ubuf__ uint32_t*)(&(buf_dyn_shmem[((((w & 1) * 1024) + (chunk * 256)) + 149568)])), 0, v_8);
    for (int32_t value_row = 0; value_row < 32; ++value_row) {
      auto dequant_f32 = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((value_row * 256) + 66560)])), 0), v, PART_EVEN, MODE_ZEROING);
      vector_bool v_9 = asc_create_mask_b32(PAT_ALL);
      auto scaled = simd_inst::vmul(dequant_f32, sf_inv[0], v_9, MODE_ZEROING);
      vector_bool v_10 = asc_create_mask_b8(PAT_ALL);
      simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>(scaled, v_10, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (value_row * 1024)) + (chunk * 256)) + 65536)])), 0, v_10);
    }
    for (int32_t value_row_1 = 0; value_row_1 < 32; ++value_row_1) {
      auto dequant_f32_1 = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((value_row_1 * 256) + 66624)])), 0), v, PART_EVEN, MODE_ZEROING);
      vector_bool v_11 = asc_create_mask_b32(PAT_ALL);
      auto scaled_1 = simd_inst::vmul(dequant_f32_1, sf_inv[1], v_11, MODE_ZEROING);
      vector_bool v_12 = asc_create_mask_b8(PAT_ALL);
      simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>(scaled_1, v_12, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (value_row_1 * 1024)) + (chunk * 256)) + 65600)])), 0, v_12);
    }
    auto amax_3 = amax[2];
    vector_bool v_13 = asc_create_mask_b32(PAT_ALL);
    auto clamped_amax_2 = simd_inst::vmax(amax_3, simd_inst::vdup<float>(float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v_13, MODE_ZEROING), v_13, MODE_ZEROING);
    auto raw_sf_2 = simd_inst::vmul(clamped_amax_2, simd_inst::vdup<float>(float(0x1.2492492492492p-9f/*2.232143e-03*/), v_13, MODE_ZEROING), v_13, MODE_ZEROING);
    auto raw_sf_bits_2 = (*(vector_u32 *)(&(raw_sf_2)));
    auto sf_2 = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(raw_sf_2))), simd_inst::vdup<uint32_t>(1, v_13, MODE_ZEROING), v_13, MODE_ZEROING), 23, v_13, MODE_ZEROING), 1, v_13, MODE_ZEROING);
    auto sf_inv_exponent_2 = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v_13, MODE_ZEROING), sf_2, v_13, MODE_ZEROING);
      vector_u32 v__8 = simd_inst::vshls(sf_inv_exponent_2, 23, v_13, MODE_ZEROING);
    auto __4 = (*(vector_f32 *)(&(v__8)));
    sf_inv[2] = __4;
    simd_inst::vsts_pack_quarter(sf_2, (__ubuf__ uint32_t*)(&(buf_dyn_shmem[((((w & 1) * 1024) + (chunk * 256)) + 149632)])), 0, v_13);
    auto amax_4 = amax[3];
    vector_bool v_14 = asc_create_mask_b32(PAT_ALL);
    auto clamped_amax_3 = simd_inst::vmax(amax_4, simd_inst::vdup<float>(float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v_14, MODE_ZEROING), v_14, MODE_ZEROING);
    auto raw_sf_3 = simd_inst::vmul(clamped_amax_3, simd_inst::vdup<float>(float(0x1.2492492492492p-9f/*2.232143e-03*/), v_14, MODE_ZEROING), v_14, MODE_ZEROING);
    auto raw_sf_bits_3 = (*(vector_u32 *)(&(raw_sf_3)));
    auto sf_3 = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(raw_sf_3))), simd_inst::vdup<uint32_t>(1, v_14, MODE_ZEROING), v_14, MODE_ZEROING), 23, v_14, MODE_ZEROING), 1, v_14, MODE_ZEROING);
    auto sf_inv_exponent_3 = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v_14, MODE_ZEROING), sf_3, v_14, MODE_ZEROING);
      vector_u32 v__9 = simd_inst::vshls(sf_inv_exponent_3, 23, v_14, MODE_ZEROING);
    auto __5 = (*(vector_f32 *)(&(v__9)));
    sf_inv[3] = __5;
    simd_inst::vsts_pack_quarter(sf_3, (__ubuf__ uint32_t*)(&(buf_dyn_shmem[((((w & 1) * 1024) + (chunk * 256)) + 149696)])), 0, v_14);
    for (int32_t value_row_2 = 0; value_row_2 < 32; ++value_row_2) {
      auto dequant_f32_2 = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((value_row_2 * 256) + 66688)])), 0), v, PART_EVEN, MODE_ZEROING);
      vector_bool v_15 = asc_create_mask_b32(PAT_ALL);
      auto scaled_2 = simd_inst::vmul(dequant_f32_2, sf_inv[2], v_15, MODE_ZEROING);
      vector_bool v_16 = asc_create_mask_b8(PAT_ALL);
      simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>(scaled_2, v_16, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (value_row_2 * 1024)) + (chunk * 256)) + 65664)])), 0, v_16);
    }
    for (int32_t value_row_3 = 0; value_row_3 < 32; ++value_row_3) {
      auto dequant_f32_3 = simd_inst::vcvt<float>(simd_inst::vlds_unpack<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((value_row_3 * 256) + 66752)])), 0), v, PART_EVEN, MODE_ZEROING);
      vector_bool v_17 = asc_create_mask_b32(PAT_ALL);
      auto scaled_3 = simd_inst::vmul(dequant_f32_3, sf_inv[3], v_17, MODE_ZEROING);
      vector_bool v_18 = asc_create_mask_b8(PAT_ALL);
      simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>(scaled_3, v_18, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 32768) + (value_row_3 * 1024)) + (chunk * 256)) + 65728)])), 0, v_18);
    }
  }
}

__simd_vf__ inline void per_channel_cast_kernel_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int32_t packed_sf_ub_version_counter_2) {
  for (int32_t _tmp = 0; _tmp < 4; ++_tmp) {
    auto v = simd_inst::vintlv(simd_inst::vlds_norm<uint8_t>((__ubuf__ uint8_t*)(&(buf_dyn_shmem[((_tmp * 256) + 149504)])), 0), simd_inst::vlds_norm<uint8_t>((__ubuf__ uint8_t*)(&(buf_dyn_shmem[((_tmp * 256) + 150528)])), 0));
    vector_bool v_1 = asc_create_mask_b8(PAT_ALL);
    simd_inst::vsts_norm(v.v0, (__ubuf__ uint8_t*)(&(buf_dyn_shmem[((((packed_sf_ub_version_counter_2 & 1) * 2048) + (_tmp * 512)) + 151552)])), 0, v_1);
    simd_inst::vsts_norm(v.v1, (__ubuf__ uint8_t*)(&(buf_dyn_shmem[((((packed_sf_ub_version_counter_2 & 1) * 2048) + (_tmp * 512)) + 151808)])), 0, v_1);
  }
}

extern "C" __global__ __vector__ void per_channel_cast_kernel_kernel(__gm__ fp8_e4_t* out, __gm__ uint8_t* out_sf, __gm__ fp8_e4_t* x, __gm__ int16_t* x_sf_invs, int32_t num_tokens, int32_t sf_stride) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t valid_rows = 0;
  int32_t x_ub_version_counter_1 = 0;
  int32_t packed_sf_ub_version_counter_2 = 0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(3));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  x_ub_version_counter_1 = 0;
  packed_sf_ub_version_counter_2 = 0;
  for (int32_t w = 0; w < (((((((num_tokens + 63) >> 6) * 7) + 71) / 72) * 2) + (((((((num_tokens + 63) >> 6) * 7) + 71) % 72) >> 31) * 2)); ++w) {
    bool __cond_2 = ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7));
    int8_t __cond_0 = ((int8_t)(bool)0);
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      __cond_0 = ((int8_t)(((((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64) + ((w & 1) * 32)) < num_tokens));
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
    }
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      if ((bool)__cond_0) {
        if (0 < ((num_tokens - ((w & 1) * 32)) - (((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64))) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((x_ub_version_counter_1 & 1) * 32768)]))), (__gm__ uint8_t*)((&(x[(((((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) / (int64_t)7) * (int64_t)458752) + ((((int64_t)w) & (int64_t)1) * (int64_t)229376)) + (((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) % (int64_t)7) * (int64_t)1024))]))), min(32, ((num_tokens - ((w & 1) * 32)) - (((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64))), 1024, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(4), 7168, 1024);
        }
      }
    }
    int8_t __cond_1 = ((int8_t)(bool)0);
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      __cond_1 = ((int8_t)((w & 1) == 1));
      if ((bool)__cond_1) {
        valid_rows = min(1, (((num_tokens + 63) >> 6) - ((((w >> 1) * 72) + ((int32_t)block_idx)) / 7)));
      }
      if ((bool)__cond_0) {
        if (0 < ((num_tokens - ((w & 1) * 32)) - (((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64))) {
          asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ int16_t*)buf_dyn_shmem)[(((x_ub_version_counter_1 & 1) * 512) + 65536)]))), (__gm__ uint8_t*)((&(x_sf_invs[(((((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) / (int64_t)7) * (int64_t)64) + ((((int64_t)w) & (int64_t)1) * (int64_t)32)) + ((((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) % (int64_t)7) * ((int64_t)num_tokens)) * (int64_t)16))]))), 16, (min(32, ((num_tokens - ((w & 1) * 32)) - (((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64))) * 2), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (num_tokens * 2), 64);
        }
      }
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
    }
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      if ((bool)__cond_0) {
        per_channel_cast_kernel_kernel_simd_vf_0(buf_dyn_shmem, x_ub_version_counter_1, w);
      }
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(((x_ub_version_counter_1 & 1) + 2)));
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
    }
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      if ((bool)__cond_1) {
        asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(((packed_sf_ub_version_counter_2 & 1) + 2)));
        per_channel_cast_kernel_kernel_simd_vf_1(buf_dyn_shmem, packed_sf_ub_version_counter_2);
        asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((packed_sf_ub_version_counter_2 & 1)));
      }
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_1)) {
      asc_sync_pipe(PIPE_MTE3);
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_1)) {
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((packed_sf_ub_version_counter_2 & 1)));
    }
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      if ((bool)__cond_1) {
        if ((0 < (((num_tokens + 63) >> 6) - ((((w >> 1) * 72) + ((int32_t)block_idx)) / 7))) && (0 < valid_rows)) {
          asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out_sf[((((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) % (int64_t)7) * (int64_t)2048) + (((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) / (int64_t)7) * ((int64_t)sf_stride)))]))), (__ubuf__ uint8_t*)((&(buf_dyn_shmem[(((packed_sf_ub_version_counter_2 & 1) * 2048) + 151552)]))), 1, 2048, static_cast<asc_store_l2_cache_mode>(4), 2048, 2048);
        }
      }
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_1)) {
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(((packed_sf_ub_version_counter_2 & 1) + 2)));
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_1)) {
      packed_sf_ub_version_counter_2 = (packed_sf_ub_version_counter_2 + 1);
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(((x_ub_version_counter_1 & 1) + 2)));
    }
    if ((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) {
      if ((bool)__cond_0) {
        if (0 < ((num_tokens - ((w & 1) * 32)) - (((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64))) {
          asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out[(((((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) / (int64_t)7) * (int64_t)458752) + ((((int64_t)w) & (int64_t)1) * (int64_t)229376)) + (((((((int64_t)w) >> (int64_t)1) * (int64_t)72) + ((int64_t)((int32_t)block_idx))) % (int64_t)7) * (int64_t)1024))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((x_ub_version_counter_1 & 1) * 32768) + 65536)]))), min(32, ((num_tokens - ((w & 1) * 32)) - (((((w >> 1) * 72) + ((int32_t)block_idx)) / 7) * 64))), 1024, static_cast<asc_store_l2_cache_mode>(4), 7168, 1024);
        }
      }
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
    }
    if (((((w >> 1) * 72) + ((int32_t)block_idx)) < (((num_tokens + 63) >> 6) * 7)) && ((bool)__cond_0)) {
      x_ub_version_counter_1 = (x_ub_version_counter_1 + 1);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(3));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
}

