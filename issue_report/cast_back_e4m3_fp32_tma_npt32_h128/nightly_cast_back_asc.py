import math

import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_max_ub_per_vector_core
from tile_kernels.quant.common import CastInputConfig, get_packed_ue8m0_pack_factor, get_sf_shape
from tile_kernels.utils import align, ceil_div


@tilelang.jit()
def get_cast_back_kernel_ascend(
    hidden: int,
    in_config: CastInputConfig,
    out_dtype: T.dtype,
    num_vec_cores: int,
):
    num_per_tokens, num_per_channels = in_config.sf_block
    pack_factor = get_packed_ue8m0_pack_factor() if in_config.use_packed_ue8m0 else 1
    assert (num_per_tokens, num_per_channels) in ((1, 32), (32, 1), (32, 32))
    assert not ((num_per_tokens, num_per_channels) == (32, 1) and in_config.use_tma_aligned_col_major_sf)

    min_k = 128 if in_config.dtype == T.float4_e2m1fn else 64
    block_k_limit = (128 if out_dtype == T.float32 else 512) if num_per_tokens == 1 else 256
    block_k = min(block_k_limit, max(min_k, ceil_div(hidden, min_k) * min_k))

    num_stages = 3 if num_per_tokens == 1 and out_dtype != T.float32 else 2
    sf_stages = num_stages if num_per_tokens == 1 else 1
    use_e2b_scale = (
        num_per_tokens == 1 and in_config.use_packed_ue8m0 and in_config.dtype == T.float4_e2m1fn and out_dtype != T.float32 and block_k % 256 == 0
    )
    ub_bytes_per_token = num_stages * block_k * (in_config.dtype.bits + out_dtype.bits) // 8 + sf_stages * 64  # 64 for worst case
    if num_per_tokens > 1:
        block_m_alignment = num_per_tokens
    elif use_e2b_scale:
        block_m_alignment = 128 // math.gcd(128, ceil_div(block_k, num_per_channels))
    else:
        block_m_alignment = 8
    block_m = get_max_ub_per_vector_core(use_simt=False) // ub_bytes_per_token // block_m_alignment * block_m_alignment

    num_sf_rows = ceil_div(block_m, num_per_tokens)
    num_sf_cols = ceil_div(block_k, num_per_channels)

    sf_raw_rows, sf_raw_cols = get_sf_shape((block_m, block_k), in_config)
    sf_raw_cols //= pack_factor
    sf_raw_dtype = T.uint16 if in_config.use_packed_ue8m0 else in_config.sf_dtype
    sf_raw_cols = align(sf_raw_cols, 32 // sf_raw_dtype.bytes)

    num_tokens = T.dynamic('num_tokens')
    sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_input_shape = (sf_shape[0], sf_shape[1] // pack_factor) if in_config.use_packed_ue8m0 else sf_shape
    sf_stride = T.dynamic('sf_stride')
    num_token_tiles = T.ceildiv(num_tokens, block_m)
    num_hidden_tiles = T.ceildiv(hidden, block_k)

    use_lazy_packed_scale = in_config.use_packed_ue8m0 and not use_e2b_scale and num_per_channels == 32
    keep_unpacked_col_major = in_config.use_tma_aligned_col_major_sf and not in_config.use_packed_ue8m0
    sf_ub_shape = (num_sf_cols, num_sf_rows) if keep_unpacked_col_major else (num_sf_rows, num_sf_cols)
    if not in_config.use_packed_ue8m0:
        sf_ub_shape = (sf_ub_shape[0], align(sf_ub_shape[1], 32 // in_config.sf_dtype.bytes))

    @T.macro
    def load_scale_tile(x_sf_raw, sf_raw_ub, sf_ub, token_base, channel_base):
        if num_per_channels == 1:
            source = x_sf_raw[token_base // pack_factor, channel_base]
        elif in_config.use_tma_aligned_col_major_sf:
            source = x_sf_raw[channel_base // pack_factor, token_base]
        else:
            source_channel = channel_base // pack_factor if in_config.use_packed_ue8m0 else channel_base
            source = x_sf_raw[token_base, source_channel]
        if in_config.use_packed_ue8m0:
            T.copy(source, sf_raw_ub)
            if use_e2b_scale:
                assert num_sf_rows * num_sf_cols % 128 == 0
                with T.SimdVF():
                    lane_indices = T.reinterpret(S.vci(0, T.int16), 'uint16x128')
                    byte_mask = S.vdup(0x00FF, T.uint16)
                    for output_vector in T.serial(num_sf_rows * num_sf_cols // 128):
                        output_base = output_vector * 128
                        output_indices = S.vadds(lane_indices, output_base)
                        output_rows = T.reinterpret(S.vshrs(output_indices, int(math.log2(num_sf_cols))), 'uint16x128')
                        output_cols = S.vand(output_indices, S.vdup(num_sf_cols - 1, T.uint16))
                        packed_channel = S.vshrs(output_cols, 1)
                        byte_indices = S.vand(output_cols, S.vdup(1, T.uint16))
                        if in_config.use_tma_aligned_col_major_sf:
                            source_rows, source_cols = packed_channel, output_rows
                        else:
                            source_rows, source_cols = output_rows, packed_channel
                        source_indices = S.vadd(S.vmuls(source_rows, sf_raw_cols), source_cols)
                        packed = S.vgather2(sf_raw_ub[0, 0], source_indices)
                        shifts = T.reinterpret(S.vmuls(byte_indices, 8), 'int16x128')
                        exponents = S.vand(S.vshr(packed, shifts), byte_mask)
                        scales = T.reinterpret(S.vshls(exponents, 7), 'bfloat16x128')
                        S.vsts(sf_ub[output_base // num_sf_cols, output_base % num_sf_cols], scales, dist='NORM_B16')
            elif num_per_channels == 1:
                with T.SimdVF():
                    sf_exp_mask = S.vdup(0x7F800000, T.uint32)
                    for group in T.serial(num_sf_rows):
                        packed_index = group + token_base % pack_factor
                        shift = 23 - packed_index % pack_factor * 8
                        for chunk in T.serial(num_sf_cols // 64):
                            col = chunk * 64
                            packed = S.vld(sf_raw_ub[packed_index // pack_factor, col], dist='US_B16')
                            packed_u32 = T.reinterpret(packed, 'uint32x64')
                            scales = T.reinterpret(S.vand(S.vshls(packed_u32, shift), sf_exp_mask), 'float32x64')
                            S.vsts(sf_ub[group, col], scales)
        else:
            T.copy(source, sf_ub)

    @T.macro
    def load_scale(sf_ub, group, col, mask_low_f32, sf_shift, sf_exp_mask):
        if use_lazy_packed_scale:
            packed = (
                S.vld(sf_ub[col // 64, group], dist='BRC_B16')
                if in_config.use_tma_aligned_col_major_sf
                else S.vld(sf_ub[group, col // 64], dist='BRC_B16')
            )
            packed_u32 = T.reinterpret(packed, 'uint32x64')
            return T.reinterpret(S.vand(S.vshl(packed_u32, sf_shift), sf_exp_mask), 'float32x64')
        if num_per_channels == 1:
            return S.vld(sf_ub[group, col])
        if keep_unpacked_col_major:
            scale_low = S.vld(sf_ub[col // 32, group], dist='BRC_B32')
            scale_high = S.vld(sf_ub[col // 32 + 1, group], dist='BRC_B32')
        else:
            scale_low = S.vld(sf_ub[group, col // 32], dist='BRC_B32')
            scale_high = S.vld(sf_ub[group, col // 32 + 1], dist='BRC_B32')
        return S.vsel(scale_low, scale_high, mask_low_f32)

    @T.macro
    def store_out(out_ub, row, col, out_f32):
        if out_dtype == T.float32:
            S.vsts(out_ub[row, col], out_f32)
        else:
            out_bf16 = S.vcvt(out_f32, T.bfloat16)
            S.vsts(out_ub[row, col], out_bf16, dist='PK_B32')

    @T.macro
    def transform_tile(x_ub, sf_ub, out_ub):
        with T.SimdVF():
            mask_low_f32 = S.pset(32, 'PAT_VL32') if num_per_channels == 32 else None
            sf_shift = S.vsel(S.vdup(23, T.int32), S.vdup(15, T.int32), mask_low_f32) if use_lazy_packed_scale and num_per_channels == 32 else None
            sf_exp_mask = S.vdup(0x7F800000, T.uint32) if use_lazy_packed_scale else None
            if use_e2b_scale:
                for row in T.serial(block_m):
                    for chunk in T.serial(block_k // 256):
                        col = chunk * 256
                        scale_e2b = S.vld(sf_ub[row, chunk * 8], dist='E2B_B16')
                        scale_low, scale_high = S.vintlv(scale_e2b, scale_e2b)
                        x_low = S.vcvt(S.vld(x_ub[row, col], dist='UNPK4_B8'), T.bfloat16)
                        x_high = S.vcvt(S.vld(x_ub[row, col + 128], dist='UNPK4_B8'), T.bfloat16)
                        S.vsts(out_ub[row, col], S.vmul(x_low, scale_low), dist='NORM_B16')
                        S.vsts(out_ub[row, col + 128], S.vmul(x_high, scale_high), dist='NORM_B16')
            elif in_config.dtype == T.float4_e2m1fn:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
                for group in T.serial(num_sf_rows):
                    row_base = group * num_per_tokens
                    for chunk in T.serial(block_k // 128):
                        col = chunk * 128
                        scale_low = load_scale(sf_ub, group, col, mask_low_f32, sf_shift, sf_exp_mask)
                        scale_high = load_scale(sf_ub, group, col + 64, mask_low_f32, sf_shift, sf_exp_mask)
                        for row in T.serial(num_per_tokens):
                            x_raw = S.vld(x_ub[row_base + row, col], dist='UNPK4_B8')
                            x_bf16 = S.vcvt(x_raw, T.bfloat16)
                            x_uint_low, x_uint_high = S.vintlv(zero_bf16, x_bf16)
                            store_out(out_ub, row_base + row, col, S.vmul(T.reinterpret(x_uint_low, 'float32x64'), scale_low))
                            store_out(out_ub, row_base + row, col + 64, S.vmul(T.reinterpret(x_uint_high, 'float32x64'), scale_high))
            else:
                for group in T.serial(num_sf_rows):
                    row_base = group * num_per_tokens
                    for chunk in T.serial(block_k // 64):
                        col = chunk * 64
                        scale = load_scale(sf_ub, group, col, mask_low_f32, sf_shift, sf_exp_mask)
                        for row in T.serial(num_per_tokens):
                            x_raw = S.vld(x_ub[row_base + row, col], dist='UNPK4_B8')
                            store_out(out_ub, row_base + row, col, S.vmul(S.vcvt(x_raw, T.float32), scale))

    @T.prim_func
    def cast_back_kernel_ascend(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[sf_input_shape, (sf_stride, 1), sf_raw_dtype],
        out: T.Tensor[(num_tokens, hidden), out_dtype],
    ):
        with T.Kernel(num_vec_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), in_config.dtype)
            sf_ub = T.alloc_shared(
                (sf_raw_rows, sf_raw_cols) if use_lazy_packed_scale else sf_ub_shape,
                T.uint16 if use_lazy_packed_scale else (T.bfloat16 if use_e2b_scale else T.float32),
            )
            sf_raw_ub = (
                T.alloc_shared((sf_raw_rows, sf_raw_cols), sf_raw_dtype) if in_config.use_packed_ue8m0 and not use_lazy_packed_scale else sf_ub
            )
            out_ub = T.alloc_shared((block_m, block_k), out_dtype)
            T.annotate_buffer_versions({x_ub: num_stages, sf_raw_ub: sf_stages, sf_ub: sf_stages, out_ub: num_stages})
            for pid_token, pid_hidden in T.Persistent(
                [num_token_tiles, num_hidden_tiles], num_vec_cores, core_id, group_size=1, num_stages=num_stages
            ):
                token_base = pid_token * block_m // num_per_tokens
                channel_base = pid_hidden * block_k // num_per_channels
                T.copy(x[pid_token * block_m, pid_hidden * block_k], x_ub)
                load_scale_tile(x_sf, sf_raw_ub, sf_ub, token_base, channel_base)
                transform_tile(x_ub, sf_ub, out_ub)
                T.copy(out_ub, out[pid_token * block_m, pid_hidden * block_k])

    return cast_back_kernel_ascend
