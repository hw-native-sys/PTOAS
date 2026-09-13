#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
#include <tl_templates/ascend/simd_inst.h>
__simd_vf__ inline void cast_back_kernel_ascend_kernel_simd_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t x_ub_version_counter_1) {
  vector_bool mask_low_f32 = asc_create_mask_b32(PAT_VL32);
  vector_bool v = asc_create_mask_b32(PAT_ALL);
  auto sf_shift = simd_inst::vsel(simd_inst::vdup<int32_t>(23, v, MODE_ZEROING), simd_inst::vdup<int32_t>(15, v, MODE_ZEROING), mask_low_f32);
  auto sf_exp_mask = simd_inst::vdup<uint32_t>(2139095040, v, MODE_ZEROING);
  for (int32_t group = 0; group < 5; ++group) {
    for (int32_t chunk = 0; chunk < 2; ++chunk) {
      auto packed = simd_inst::vlds_brc_elem<uint16_t>((__ubuf__ uint16_t*)(&(((__ubuf__ uint16_t*)buf_dyn_shmem)[(((chunk * 16) + group) + 20480)])), 0);
      auto packed_u32 = (*(vector_u32 *)(&(packed)));
        vector_u32 v_ = simd_inst::vand(simd_inst::vshl((*(vector_u32 *)(&(packed))), sf_shift, v, MODE_ZEROING), sf_exp_mask, v, MODE_ZEROING);
      auto scale = (*(vector_f32 *)(&(v_)));
      for (int32_t row = 0; row < 32; ++row) {
        auto x_raw = simd_inst::vlds_unpack4<float8_e4m3_t>((__ubuf__ float8_e4m3_t*)(&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[(((((x_ub_version_counter_1 & 1) * 20480) + (group * 4096)) + (row * 128)) + (chunk * 64))])), 0);
        vector_bool v_1 = asc_create_mask_b8(PAT_ALL);
        auto out_f32 = simd_inst::vmul(simd_inst::vcvt<float>(x_raw, v_1, PART_P0, MODE_ZEROING), scale, v, MODE_ZEROING);
        simd_inst::vsts_norm(out_f32, (__ubuf__ float*)(&(((__ubuf__ float*)buf_dyn_shmem)[((((((x_ub_version_counter_1 & 1) * 20480) + (group * 4096)) + (row * 128)) + (chunk * 64)) + 10256)])), 0, v);
      }
    }
  }
}

extern "C" __global__ __vector__ void cast_back_kernel_ascend_kernel(__gm__ float* out, __gm__ fp8_e4_t* x, __gm__ uint16_t* x_sf, int32_t num_tokens, int32_t sf_stride) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  int32_t x_ub_version_counter_1 = 0;
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(2));
  x_ub_version_counter_1 = 0;
  for (int32_t w = 0; w < ((num_tokens + 11519) / 11520); ++w) {
    bool __cond_0 = (((w * 72) + ((int32_t)block_idx)) < (((num_tokens + 159) / 160) + (((num_tokens + 159) % 160) >> 31)));
    uint16_t source = (uint16_t)0;
    if (((w * 72) + ((int32_t)block_idx)) < (((num_tokens + 159) / 160) + (((num_tokens + 159) % 160) >> 31))) {
      source = x_sf[((((int64_t)w) * (int64_t)360) + (((int64_t)((int32_t)block_idx)) * (int64_t)5))];
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ fp8_e4_t*)buf_dyn_shmem)[((x_ub_version_counter_1 & 1) * 20480)]))), (__gm__ uint8_t*)((&(x[((((int64_t)w) * (int64_t)1474560) + (((int64_t)((int32_t)block_idx)) * (int64_t)20480))]))), 1, (min(160, max(((num_tokens - (((int32_t)block_idx) * 160)) - (w * 11520)), 0)) * 128), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(160, max(((num_tokens - (((int32_t)block_idx) * 160)) - (w * 11520)), 0)) * 128), (min(160, max(((num_tokens - (((int32_t)block_idx) * 160)) - (w * 11520)), 0)) * 128));
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(2));
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(((__ubuf__ uint16_t*)buf_dyn_shmem)[20480]))), (__gm__ uint8_t*)((&(x_sf[((((int64_t)w) * (int64_t)360) + (((int64_t)((int32_t)block_idx)) * (int64_t)5))]))), 2, (min(16, max(((((num_tokens + 31) >> 5) - (((int32_t)block_idx) * 5)) - (w * 360)), 0)) * 2), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (sf_stride * 2), 32);
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>(2));
      asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>(2));
      cast_back_kernel_ascend_kernel_simd_vf_0(buf_dyn_shmem, x_ub_version_counter_1);
      asc_sync_notify(PIPE_V, PIPE_MTE3, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(2));
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_V, PIPE_MTE3, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      asc_copy_ub2gm_align((__gm__ uint8_t*)((&(out[((((int64_t)w) * (int64_t)1474560) + (((int64_t)((int32_t)block_idx)) * (int64_t)20480))]))), (__ubuf__ uint8_t*)((&(((__ubuf__ float*)buf_dyn_shmem)[(((x_ub_version_counter_1 & 1) * 20480) + 10256)]))), 1, (min(160, max(((num_tokens - (((int32_t)block_idx) * 160)) - (w * 11520)), 0)) * 512), static_cast<asc_store_l2_cache_mode>(4), (min(160, max(((num_tokens - (((int32_t)block_idx) * 160)) - (w * 11520)), 0)) * 512), (min(160, max(((num_tokens - (((int32_t)block_idx) * 160)) - (w * 11520)), 0)) * 512));
      asc_sync_notify(PIPE_MTE3, PIPE_V, static_cast<event_t>((x_ub_version_counter_1 & 1)));
      x_ub_version_counter_1 = (x_ub_version_counter_1 + 1);
    }
  }
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(0));
  asc_sync_wait(PIPE_MTE3, PIPE_V, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(2));
}

