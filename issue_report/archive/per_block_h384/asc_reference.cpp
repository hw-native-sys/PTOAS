#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void per_block_cast_kernel_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t x_ub_version_counter_1, int32_t sf_ub_version_counter_2) {
  vector_f32 inverse[2];
  vector_f32 amax[2];
  vector_bf16 amax_bf16;
  vector_bool m_low = asc_create_mask_b32(PAT_VL32);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto m_high = simd_inst::vcmps_gt(simd_inst::vci<int32_t>(0, INC_ORDER), 31, v);
  auto one_reg = simd_inst::vdup<float>(float(0x1p+0f/*1.000000e+00*/), v, MODE_ZEROING);
  auto eps = simd_inst::vdup<float>(float(0x1.a36e2eb1c432dp-14f/*1.000000e-04*/), v, MODE_ZEROING);
  auto fp8_max = simd_inst::vdup<float>(float(0x1.cp+8f/*4.480000e+02*/), v, MODE_ZEROING);
  auto inv_fp8_max = simd_inst::vdup<float>(float(0x1.2492492492492p-9f/*2.232143e-03*/), v, MODE_ZEROING);
  vector_bool v_1 = asc_create_mask_b16(PAT_ALL);
  auto zero_bf16 = simd_inst::vdup<bfloat16_t>(float(0x0p+0f/*0.000000e+00*/), v_1, MODE_ZEROING);
    vector_u16 v_ = simd_inst::vdup<uint16_t>(32767, v_1, MODE_ZEROING);
  auto abs_mask = (*(vector_bf16 *)(&(v_)));
  amax_bf16 = simd_inst::vdup<bfloat16_t>(float(0x0p+0f/*0.000000e+00*/), v_1, MODE_ZEROING);
  for (int32_t row = 0; row < 32; ++row) {
    amax_bf16 = simd_inst::vmax(amax_bf16, simd_inst::vand(simd_inst::vlds_norm<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((x_ub_version_counter_1 & 1) * 4096) + (row * 128))])), 0), abs_mask, v_1, MODE_ZEROING), v_1, MODE_ZEROING);
  }
  auto v_2 = simd_inst::vintlv(zero_bf16, amax_bf16);
    vector_bf16 v__1 = v_2.v0;
  amax[0] = (*(vector_f32 *)(&(v__1)));
    vector_bf16 v__2 = v_2.v1;
  amax[1] = (*(vector_f32 *)(&(v__2)));
  amax[0] = simd_inst::vmax(amax[0], eps, v, MODE_ZEROING);
  auto amax_vec = amax[0];
  auto a0 = simd_inst::vcmax(amax_vec, m_low, MODE_ZEROING);
  auto a1 = simd_inst::vcmax(amax_vec, m_high, MODE_ZEROING);
  auto amax_1 = simd_inst::vsel(simd_inst::vdupv(a0, v, POS_LOWEST, MODE_ZEROING), simd_inst::vdupv(a1, v, POS_LOWEST, MODE_ZEROING), m_low);
  auto scale_raw = simd_inst::vmul(amax_1, inv_fp8_max, v, MODE_ZEROING);
  auto bits = (*(vector_u32 *)(&(scale_raw)));
  auto exp = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(scale_raw))), simd_inst::vdup<uint32_t>(1, v, MODE_ZEROING), v, MODE_ZEROING), 23, v, MODE_ZEROING), 1, v, MODE_ZEROING);
  auto scale_bits = simd_inst::vshls(exp, 23, v, MODE_ZEROING);
  auto scale01 = (*(vector_f32 *)(&(scale_bits)));
  auto inv_exp = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v, MODE_ZEROING), exp, v, MODE_ZEROING);
  auto inv_bits = simd_inst::vshls(inv_exp, 23, v, MODE_ZEROING);
  auto _tmp = (*(vector_f32 *)(&(inv_bits)));
  inverse[0] = (*(vector_f32 *)(&(inv_bits)));
  simd_inst::vsts_1st((*(vector_f32 *)(&(scale_bits))), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((sf_ub_version_counter_2 & 1) * 8) + 4096)])), 0, v);
  simd_inst::vsts_1st(simd_inst::vdupv((*(vector_f32 *)(&(scale_bits))), v, POS_HIGHEST, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((sf_ub_version_counter_2 & 1) * 8) + 4097)])), 0, v);
  amax[1] = simd_inst::vmax(amax[1], eps, v, MODE_ZEROING);
  auto amax_vec_1 = amax[1];
  auto a0_1 = simd_inst::vcmax(amax_vec_1, m_low, MODE_ZEROING);
  auto a1_1 = simd_inst::vcmax(amax_vec_1, m_high, MODE_ZEROING);
  auto amax_2 = simd_inst::vsel(simd_inst::vdupv(a0_1, v, POS_LOWEST, MODE_ZEROING), simd_inst::vdupv(a1_1, v, POS_LOWEST, MODE_ZEROING), m_low);
  auto scale_raw_1 = simd_inst::vmul(amax_2, inv_fp8_max, v, MODE_ZEROING);
  auto bits_1 = (*(vector_u32 *)(&(scale_raw_1)));
  auto exp_1 = simd_inst::vadds(simd_inst::vshrs(simd_inst::vsub((*(vector_u32 *)(&(scale_raw_1))), simd_inst::vdup<uint32_t>(1, v, MODE_ZEROING), v, MODE_ZEROING), 23, v, MODE_ZEROING), 1, v, MODE_ZEROING);
  auto scale_bits_1 = simd_inst::vshls(exp_1, 23, v, MODE_ZEROING);
  auto scale01_1 = (*(vector_f32 *)(&(scale_bits_1)));
  auto inv_exp_1 = simd_inst::vsub(simd_inst::vdup<uint32_t>(254, v, MODE_ZEROING), exp_1, v, MODE_ZEROING);
  auto inv_bits_1 = simd_inst::vshls(inv_exp_1, 23, v, MODE_ZEROING);
  auto _tmp_1 = (*(vector_f32 *)(&(inv_bits_1)));
  inverse[1] = (*(vector_f32 *)(&(inv_bits_1)));
  simd_inst::vsts_1st((*(vector_f32 *)(&(scale_bits_1))), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((sf_ub_version_counter_2 & 1) * 8) + 4098)])), 0, v);
  simd_inst::vsts_1st(simd_inst::vdupv((*(vector_f32 *)(&(scale_bits_1))), v, POS_HIGHEST, MODE_ZEROING), (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[(((sf_ub_version_counter_2 & 1) * 8) + 4099)])), 0, v);
  auto x0 = inverse[0];
  auto x1 = inverse[1];
  auto v_3 = simd_inst::vdintlv((*(vector_u16 *)(&(x0))), (*(vector_u16 *)(&(x1))));
    vector_u16 v__3 = v_3.v1;
  auto sf_inv = (*(vector_bf16 *)(&(v__3)));
  for (int32_t row_1 = 0; row_1 < 32; ++row_1) {
      vector_u16 v__4 = v_3.v1;
    auto quantized_bf16 = simd_inst::vmul(simd_inst::vlds_norm<bfloat16_t>((__ubuf__ bfloat16_t*)(&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[(((x_ub_version_counter_1 & 1) * 4096) + (row_1 * 128))])), 0), (*(vector_bf16 *)(&(v__4))), v_1, MODE_ZEROING);
    auto v_4 = simd_inst::vintlv(zero_bf16, quantized_bf16);
      vector_bf16 v__5 = v_4.v0;
    auto quantized_low = (*(vector_f32 *)(&(v__5)));
      vector_bf16 v__6 = v_4.v1;
    auto quantized_high = (*(vector_f32 *)(&(v__6)));
    vector_bool v_5 = asc_create_mask_b8(PAT_ALL);
      vector_bf16 v__7 = v_4.v0;
    simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>((*(vector_f32 *)(&(v__7))), v_5, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((((x_ub_version_counter_1 & 1) * 4096) + (row_1 * 128)) + 16448)])), 0, v_5);
      vector_bf16 v__8 = v_4.v1;
    simd_inst::vsts_pack_quarter(simd_inst::vcvt<fp8_e4_t>((*(vector_f32 *)(&(v__8))), v_5, ROUND_R, RS_ENABLE, PART_P0, MODE_ZEROING), (__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((((x_ub_version_counter_1 & 1) * 4096) + (row_1 * 128)) + 16512)])), 0, v_5);
  }
}

extern "C" __global__ __vector__ void per_block_cast_kernel_kernel(__gm__ fp8_e4_t* out, __gm__ float* output_sf, __gm__ bfloat16_t* x, int32_t num_tokens, int32_t sf_stride) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t sf_m_base = 0;
  int32_t sf_k_base = 0;
  int32_t stored_sf_k_base = 0;
  int32_t x_ub_version_counter_1 = 0;
  int32_t sf_ub_version_counter_2 = 0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(3));
  x_ub_version_counter_1 = 0;
  sf_ub_version_counter_2 = 0;
  for (int32_t w = 0; w < ((num_tokens + 767) / 768); ++w) {
    bool __cond_0 = (((w * 72) + ((int32_t)block_idx)) < (((num_tokens + 31) >> 5) * 3));
    if (((w * 72) + ((int32_t)block_idx)) < (((num_tokens + 31) >> 5) * 3)) {
      sf_k_base = ((((int32_t)block_idx) % 3) * 4);
      sf_m_base = ((w * 24) + (((int32_t)block_idx) / 3));
      stored_sf_k_base = ((((int32_t)block_idx) % 3) * 4);
      int32_t token_block = sf_m_base;
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      if (0 < (num_tokens - (token_block * 32))) {
        asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ bfloat16_t*)buf_dyn_shmem)[((x_ub_version_counter_1 & 1) * 4096)]))), (__gm__ uint8_t*)((&(x[((((int64_t)token_block) * (int64_t)12288) + ((((int64_t)((int32_t)block_idx)) % (int64_t)3) * (int64_t)128))]))), min(32, (num_tokens - (token_block * 32))), 256, 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), 768, 256);
      }
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(((sf_ub_version_counter_2 & 1) + 2)));
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      per_block_cast_kernel_kernel_simd_vf_0(buf_dyn_shmem, x_ub_version_counter_1, sf_ub_version_counter_2);
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_pipe(PIPE_MTE3);
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      if (0 < (num_tokens - (token_block * 32))) {
        asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out[((((int64_t)token_block) * (int64_t)12288) + ((((int64_t)((int32_t)block_idx)) % (int64_t)3) * (int64_t)128))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((x_ub_version_counter_1 & 1) * 4096) + 16448)]))), min(32, (num_tokens - (token_block * 32))), 128, static_cast<asc_store_l2_cache_mode>(4), 384, 128);
      }
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      x_ub_version_counter_1 = (x_ub_version_counter_1 + 1);
      asc_sync_pipe(PIPE_MTE3);
      if ((stored_sf_k_base < 12) && (0 < (((num_tokens + 31) >> 5) - sf_m_base))) {
        asc_copy_ub2gm_align((__gm__ uint8_t*)((&(output_sf[((((int64_t)sf_m_base) * ((int64_t)sf_stride)) + ((int64_t)stored_sf_k_base))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((sf_ub_version_counter_2 & 1) * 8) + 4096)]))), 1, (min(4, max((12 - stored_sf_k_base), 0)) * 4), static_cast<asc_store_l2_cache_mode>(4), (min(4, max((12 - stored_sf_k_base), 0)) * 4), (min(4, max((12 - stored_sf_k_base), 0)) * 4));
      }
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(((sf_ub_version_counter_2 & 1) + 2)));
      sf_ub_version_counter_2 = (sf_ub_version_counter_2 + 1);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(2));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(3));
}

