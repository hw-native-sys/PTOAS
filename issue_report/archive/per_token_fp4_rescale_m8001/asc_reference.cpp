#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void per_token_cast_kernel_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem) {
  auto lane = simd_inst::vci<int16_t>(0, INC_ORDER);
  vector_bool v = asc_create_mask_b16(PAT_ALL);
  auto output_row = simd_inst::vand((*(vector_u16 *)(&(lane))), simd_inst::vdup<uint16_t>(15, v, MODE_ZEROING), v, MODE_ZEROING);
    vector_s16 v_ = simd_inst::vshrs(lane, 4, v, MODE_ZEROING);
  auto group_in_vector = (*(vector_u16 *)(&(v_)));
  auto source_indices = simd_inst::vadd(simd_inst::vmuls(output_row, 32, v, MODE_ZEROING), group_in_vector, v, MODE_ZEROING);
  simd_inst::vsts_norm(source_indices, (__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[49952])), 0, v);
}

__simd_vf__ inline void per_token_cast_kernel_kernel_simd_vf_1(__ubuf__ uint8_t* buf_dyn_shmem, int32_t sf_in_matrix_version_counter_1) {
  vector_bool mask_low = asc_create_mask_b32(PAT_VL32);
  for (int32_t pair = 0; pair < 8; ++pair) {
    auto packed = simd_inst::vlds_brc_elem<uint16_t>((__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[((((sf_in_matrix_version_counter_1 & 1) * 16) + (pair * 2)) + 49152)])), 0);
    vector_bool v = asc_create_mask_b16(PAT_ALL);
    auto exponent = simd_inst::vand(simd_inst::vshrs(packed, 0, v, MODE_ZEROING), simd_inst::vdup<uint16_t>(255, v, MODE_ZEROING), v, MODE_ZEROING);
      vector_u16 v_ = simd_inst::vshls(exponent, 7, v, MODE_ZEROING);
    auto scale = (*(vector_bf16 *)(&(v_)));
    auto packed_1 = simd_inst::vlds_brc_elem<uint16_t>((__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[((((sf_in_matrix_version_counter_1 & 1) * 16) + (pair * 2)) + 49152)])), 0);
    auto exponent_1 = simd_inst::vand(simd_inst::vshrs(packed_1, 8, v, MODE_ZEROING), simd_inst::vdup<uint16_t>(255, v, MODE_ZEROING), v, MODE_ZEROING);
      vector_u16 v__1 = simd_inst::vshls(exponent_1, 7, v, MODE_ZEROING);
    auto scale_1 = (*(vector_bf16 *)(&(v__1)));
    auto scale0 = simd_inst::vsel(simd_inst::vcvt<float>(scale, v, PART_EVEN, MODE_ZEROING), simd_inst::vcvt<float>(scale_1, v, PART_EVEN, MODE_ZEROING), mask_low);
    auto packed_2 = simd_inst::vlds_brc_elem<uint16_t>((__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[((((sf_in_matrix_version_counter_1 & 1) * 16) + (pair * 2)) + 49153)])), 0);
    auto exponent_2 = simd_inst::vand(simd_inst::vshrs(packed_2, 0, v, MODE_ZEROING), simd_inst::vdup<uint16_t>(255, v, MODE_ZEROING), v, MODE_ZEROING);
      vector_u16 v__2 = simd_inst::vshls(exponent_2, 7, v, MODE_ZEROING);
    auto scale_2 = (*(vector_bf16 *)(&(v__2)));
    auto packed_3 = simd_inst::vlds_brc_elem<uint16_t>((__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[((((sf_in_matrix_version_counter_1 & 1) * 16) + (pair * 2)) + 49153)])), 0);
    auto exponent_3 = simd_inst::vand(simd_inst::vshrs(packed_3, 8, v, MODE_ZEROING), simd_inst::vdup<uint16_t>(255, v, MODE_ZEROING), v, MODE_ZEROING);
      vector_u16 v__3 = simd_inst::vshls(exponent_3, 7, v, MODE_ZEROING);
    auto scale_3 = (*(vector_bf16 *)(&(v__3)));
    auto scale1 = simd_inst::vsel(simd_inst::vcvt<float>(scale_2, v, PART_EVEN, MODE_ZEROING), simd_inst::vcvt<float>(scale_3, v, PART_EVEN, MODE_ZEROING), mask_low);
    auto v_1 = simd_inst::vdintlv((*(vector_u16 *)(&(scale0))), (*(vector_u16 *)(&(scale1))));
      vector_u16 v__4 = v_1.v1;
    auto scale_bf16_1 = (*(vector_bf16 *)(&(v__4)));
    for (int32_t row = 0; row < 16; ++row) {
      vector_bool v_2 = asc_create_mask_b8(PAT_ALL);
      auto x0 = simd_inst::vcvt<float>(simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((((sf_in_matrix_version_counter_1 & 1) * 16384) + (row * 1024)) + (pair * 128))])), 0), v_2, PART_P0, MODE_ZEROING);
      auto x1 = simd_inst::vcvt<float>(simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((sf_in_matrix_version_counter_1 & 1) * 16384) + (row * 1024)) + (pair * 128)) + 64)])), 0), v_2, PART_P0, MODE_ZEROING);
      auto v_3 = simd_inst::vdintlv((*(vector_u16 *)(&(x0))), (*(vector_u16 *)(&(x1))));
        vector_u16 v__5 = v_3.v1;
      auto x_bf16_1 = (*(vector_bf16 *)(&(v__5)));
        vector_u16 v__6 = v_3.v1;
        vector_u16 v__7 = v_1.v1;
      simd_inst::vsts_norm(simd_inst::vmul((*(vector_bf16 *)(&(v__6))), (*(vector_bf16 *)(&(v__7))), v, MODE_ZEROING), (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((row * 1024) + (pair * 128)) + 32768)])), 0, v);
    }
  }
}

__simd_vf__ inline void per_token_cast_kernel_kernel_simd_vf_2(__ubuf__ uint8_t* buf_dyn_shmem, int32_t sf_in_matrix_version_counter_1) {
  vector_bool mask_vl8 = asc_create_mask_b32(PAT_VL8);
  vector_bool v = asc_create_mask_b16(PAT_ALL);
  auto abs_mask = simd_inst::vdup<uint16_t>(32767, v, MODE_ZEROING);
  auto zero = simd_inst::vdup<bfloat16_t>(float(0x0p+0f/*0.000000e+00*/), v, MODE_ZEROING);
  for (int32_t row = 0; row < 16; ++row) {
    for (int32_t tile = 0; tile < 4; ++tile) {
      auto v_1 = simd_inst::vld_x2<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((row * 1024) + (tile * 256)) + 32768)])), DINTLV_B16);
        vector_bf16 v_ = v_1.v0;
      auto abs_x0 = simd_inst::vand((*(vector_u16 *)(&(v_))), abs_mask, v, MODE_ZEROING);
        vector_bf16 v__1 = v_1.v1;
      auto abs_x1 = simd_inst::vand((*(vector_u16 *)(&(v__1))), abs_mask, v, MODE_ZEROING);
      auto maxima = simd_inst::vcgmax(simd_inst::vmax(abs_x0, abs_x1, v, MODE_ZEROING), v, MODE_ZEROING);
      auto v_2 = simd_inst::vintlv(zero, (*(vector_bf16 *)(&(maxima))));
        vector_bf16 v__2 = v_2.v0;
      simd_inst::vsts_norm((*(vector_f32 *)(&(v__2))), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((row * 64) + (tile * 8)) + 25040)])), 0, mask_vl8);
    }
  }
  simd_inst::mem_bar(VST_VLD);
  for (int32_t row_1 = 0; row_1 < 16; ++row_1) {
    auto amax = simd_inst::vlds_norm<float>((__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((row_1 * 64) + 25040)])), 0);
    vector_bool v_3 = asc_create_mask_b32(PAT_ALL);
    auto clamped = simd_inst::vmaxs(amax, float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v_3, MODE_ZEROING);
      vector_f32 v__3 = simd_inst::vmuls(clamped, float(0x1.2492492492492p-9f/*2.232143e-03*/), v_3, MODE_ZEROING);
    auto bits = (*(vector_u32 *)(&(v__3)));
    auto scale = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub(bits, simd_inst::vdup<uint32_t>(1, v_3, MODE_ZEROING), v_3, MODE_ZEROING), 23, v_3, MODE_ZEROING), 1, v_3, MODE_ZEROING);
      vector_u32 v__4 = simd_inst::vshls(simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v_3, MODE_ZEROING), scale, v_3, MODE_ZEROING), 23, v_3, MODE_ZEROING);
    auto inverse = (*(vector_f32 *)(&(v__4)));
    vector_bool v_4 = asc_create_mask_b8(PAT_ALL);
    simd_inst::vsts_pack_quarter((*(vector_u8 *)(&(scale))), (__ubuf__ uint8_t*)(&(buf_dyn_shmem[((row_1 * 64) + 98368)])), 0, v_4);
    simd_inst::vsts_pack_b32(simd_inst::vcvt<bfloat16_t>(inverse, v, ROUND_R, RS_ENABLE, PART_EVEN, MODE_ZEROING), (__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((row_1 * 64) + 52128)])), 0, v);
  }
  simd_inst::mem_bar(VST_VLD);
  for (int32_t row_2 = 0; row_2 < 16; ++row_2) {
    for (int32_t tile_1 = 0; tile_1 < 4; ++tile_1) {
      auto v_5 = simd_inst::vld_x2<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((row_2 * 1024) + (tile_1 * 256)) + 32768)])), DINTLV_B16);
      auto inverse_1 = simd_inst::vlds_brc_elem2datablock<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((row_2 * 64) + (tile_1 * 8)) + 52128)])), 0);
      auto quantized0 = simd_inst::vmul(v_5.v0, inverse_1, v, MODE_ZEROING);
      auto quantized1 = simd_inst::vmul(v_5.v1, inverse_1, v, MODE_ZEROING);
      vector_bool v_6 = asc_create_mask_b8(PAT_ALL);
      auto quantized_fp8_0 = simd_inst::vcvt<fp8_e4_t>(simd_inst::vcvt<float>(quantized0, v, PART_EVEN, MODE_ZEROING), v_6, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING);
      auto quantized_fp8_2 = simd_inst::vcvt<fp8_e4_t>(simd_inst::vcvt<float>(quantized0, v, PART_ODD, MODE_ZEROING), v_6, ROUND_R, RS_ENABLE, PART_P2, MODE_ZEROING);
      auto quantized_fp8_1 = simd_inst::vcvt<fp8_e4_t>(simd_inst::vcvt<float>(quantized1, v, PART_EVEN, MODE_ZEROING), v_6, ROUND_R, RS_ENABLE, PART_P1, MODE_ZEROING);
      auto quantized_fp8_3 = simd_inst::vcvt<fp8_e4_t>(simd_inst::vcvt<float>(quantized1, v, PART_ODD, MODE_ZEROING), v_6, ROUND_R, RS_ENABLE, PART_P3, MODE_ZEROING);
      auto merged = simd_inst::vor(simd_inst::vor((*(vector_u8 *)(&(quantized_fp8_0))), (*(vector_u8 *)(&(quantized_fp8_2))), v_6, MODE_ZEROING), simd_inst::vor((*(vector_u8 *)(&(quantized_fp8_1))), (*(vector_u8 *)(&(quantized_fp8_3))), v_6, MODE_ZEROING), v_6, MODE_ZEROING);
      simd_inst::vsts_norm((*(vector_f8e4m3 *)(&(merged))), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((sf_in_matrix_version_counter_1 & 1) * 16384) + (row_2 * 1024)) + (tile_1 * 256)) + 32768)])), 0, v_6);
    }
  }
}

__simd_vf__ inline void per_token_cast_kernel_kernel_simd_vf_3(__ubuf__ uint8_t* buf_dyn_shmem) {
  auto source_indices = simd_inst::vlds_norm<uint16_t>((__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[49952])), 0);
  vector_bool v = asc_create_mask_b16(PAT_ALL);
  auto values = simd_inst::vgather2((&(((__ubuf__ uint16_t*)buf_dyn_shmem)[49184])), source_indices, v);
  simd_inst::vsts_norm(values, (__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[49696])), 0, v);
  vector_bool v_1 = asc_create_mask_b16(PAT_ALL);
  auto values_1 = simd_inst::vgather2((&(((__ubuf__ uint16_t*)buf_dyn_shmem)[49192])), source_indices, v_1);
  simd_inst::vsts_norm(values_1, (__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[49824])), 0, v_1);
}

extern "C" __global__ __vector__ void per_token_cast_kernel_kernel(__gm__ fp8_e4_t* out, __gm__ uint8_t* out_sf, __gm__ fp8_e4_t* x, __gm__ uint8_t* x_sf, int32_t num_tokens, int32_t out_sf_stride) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t sf_in_matrix_version_counter_1 = 0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
  sf_in_matrix_version_counter_1 = 0;
  per_token_cast_kernel_kernel_simd_vf_0(buf_dyn_shmem);
  for (int32_t w = 0; w < ((num_tokens + 383) / 384); ++w) {
    bool __cond_0 = (((w * 72) + ((int32_t)block_idx)) < (((num_tokens + 15) >> 4) * 3));
    if (((w * 72) + ((int32_t)block_idx)) < (((num_tokens + 15) >> 4) * 3)) {
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((sf_in_matrix_version_counter_1 & 1)));
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((sf_in_matrix_version_counter_1 & 1) * 16384)]))), (__gm__ uint8_t*)((&(x[(((((int64_t)w) * (int64_t)1179648) + ((((int64_t)((int32_t)block_idx)) / (int64_t)3) * (int64_t)49152)) + ((((int64_t)((int32_t)block_idx)) % (int64_t)3) * (int64_t)1024))]))), min(16, max(((num_tokens - ((((int32_t)block_idx) / 3) * 16)) - (w * 384)), 0)), 1024, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 3072, 1024);
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(buf_dyn_shmem[(((sf_in_matrix_version_counter_1 & 1) * 32) + 98304)]))), (__gm__ uint8_t*)((&(x_sf[(((((((int64_t)((int32_t)block_idx)) % (int64_t)3) * ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)) * (int64_t)32) + (((int64_t)w) * (int64_t)24)) + (((((int64_t)((int32_t)block_idx)) / (int64_t)3) >> (int64_t)1) * (int64_t)2))]))), 16, 2, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (((num_tokens + 31) >> 5) * 2), 2);
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((sf_in_matrix_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((sf_in_matrix_version_counter_1 & 1)));
      per_token_cast_kernel_kernel_simd_vf_1(buf_dyn_shmem, sf_in_matrix_version_counter_1);
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((sf_in_matrix_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((sf_in_matrix_version_counter_1 & 1)));
      per_token_cast_kernel_kernel_simd_vf_2(buf_dyn_shmem, sf_in_matrix_version_counter_1);
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
      per_token_cast_kernel_kernel_simd_vf_3(buf_dyn_shmem);
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>(0));
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out_sf[(((((int64_t)w) * (int64_t)768) + ((((int64_t)((int32_t)block_idx)) / (int64_t)3) * (int64_t)32)) + (((((int64_t)((int32_t)block_idx)) % (int64_t)3) * ((int64_t)out_sf_stride)) * (int64_t)16))]))), (__ubuf__ uint8_t*)((&(buf_dyn_shmem[99392]))), 16, min(32, max((((num_tokens * 2) - ((((int32_t)block_idx) / 3) * 32)) - (w * 768)), 0)), static_cast<asc_store_l2_cache_mode>(4), out_sf_stride, 32);
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out[(((((int64_t)w) * (int64_t)1179648) + ((((int64_t)((int32_t)block_idx)) / (int64_t)3) * (int64_t)49152)) + ((((int64_t)((int32_t)block_idx)) % (int64_t)3) * (int64_t)1024))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((sf_in_matrix_version_counter_1 & 1) * 16384) + 32768)]))), min(16, max(((num_tokens - ((((int32_t)block_idx) / 3) * 16)) - (w * 384)), 0)), 1024, static_cast<asc_store_l2_cache_mode>(4), 3072, 1024);
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((sf_in_matrix_version_counter_1 & 1)));
      sf_in_matrix_version_counter_1 = (sf_in_matrix_version_counter_1 + 1);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
}

