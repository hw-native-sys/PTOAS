from ptodsl import pto, scalar
from tilelang.contrib.ptodsl.dcache_bypass import (
  pto_read_gm_bypass_dcache as _tl_pto_read_gm_bypass_dcache,
  pto_write_gm_bypass_dcache as _tl_pto_write_gm_bypass_dcache,
)
from tilelang.contrib.ptodsl.simt import (
  scalar_div as _tl_scalar_div,
  scalar_rsqrt as _tl_scalar_rsqrt,
  simt_allreduce_max as _tl_simt_allreduce_max,
  simt_allreduce_min as _tl_simt_allreduce_min,
  simt_allreduce_sum as _tl_simt_allreduce_sum,
  vectorize_binary_f32x2 as _tl_vectorize_binary_f32x2,
  vectorize_unary_f32x2 as _tl_vectorize_unary_f32x2,
)
from ptodsl._ops import _coerce_i64 as _tl_coerce_i64
from ptodsl._surface_values import wrap_surface_value as _tl_wrap_surface_value

@pto.jit(name="main_kernel", kernel_kind="vector", target="a5", mode="explicit")
def main_kernel(out: pto.ptr(pto.f32, "gm"), x: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.ui8, "gm"), num_tokens: pto.si32, sf_extent: pto.si32, sf_stride: pto.si32):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  for w in range(0, ((num_tokens + 2303) // 2304), 1):
    __cond_0 = ((w * 72) + pto.get_block_idx()) < ((num_tokens + 31) // 32)
    if ((w * 72) + pto.get_block_idx()) < ((num_tokens + 31) // 32):
      pto.wait_flag("V", "MTE2", event_id=1)
      if 0 < ((sf_extent - (pto.get_block_idx() * 2)) - (w * 144)):
        pto.mte_gm_ub(pto.addptr(x_sf, (scalar.cast(w, pto.si64) * 144) + (scalar.cast(pto.get_block_idx(), pto.si64) * 2)), pto.addptr(buf_dyn_shmem, 4096), 0, scalar.min(2, ((sf_extent - (pto.get_block_idx() * 2)) - (w * 144))), nburst=(2, sf_stride, 64))
      pto.set_flag("MTE2", "V", event_id=1)
      pto.wait_flag("MTE2", "V", event_id=1)
      with pto.vecscope():
        one = pto.vmi.create_mask(1, size=1)
        nan1 = pto.vmi.vbrc(pto.ui16(32704), size=1)
        for row in range(0, 32, 1):
          for scale_i in range(0, 4, 1):
            loaded = pto.vmi.vload(pto.addptr(buf_dyn_shmem, 4096), (((scale_i // 2) * 64) + (scale_i & 1)), size=1)
            u16 = pto.vmi.vcvt(loaded, to_dtype=pto.ui16)
            shifted = pto.vmi.vshls(u16, 7, one)
            is_inf = pto.vmi.vcmps(shifted, 32640, one, "eq")
            pto.vmi.vstore(pto.vmi.vinterpret_cast(pto.vmi.vsel(is_inf, nan1, shifted), to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 2112), ((row * 64) + scale_i), one)
      pto.set_flag("V", "MTE2", event_id=1)
      pto.wait_flag("V", "MTE2", event_id=0)
      pto.mte_gm_ub(pto.addptr(x, (scalar.cast(w, pto.si64) * 294912) + (scalar.cast(pto.get_block_idx(), pto.si64) * 4096)), pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), 0, scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)) * 128, nburst=(1, scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)) * 128, scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)) * 128))
      pto.set_flag("MTE2", "V", event_id=0)
      pto.wait_flag("MTE3", "V", event_id=0)
      pto.wait_flag("MTE2", "V", event_id=0)
      with pto.vecscope():
        mask = pto.vmi.create_mask(64, size=64)
        for row_1 in range(0, 32, 1):
          for half in range(0, 2, 1):
            scale = pto.vmi.vbrc(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 2112), ((row_1 * 64) + (half * 2)), group=2, size=2, stride=1), group=2, size=64)
            source = pto.vmi.vload(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((row_1 * 128) + (half * 64)), size=64)
            value = pto.vmi.vcvt(source, to_dtype=pto.f32)
            scale_f32 = pto.vmi.vcvt(scale, to_dtype=pto.f32)
            pto.vmi.vstore(pto.vmi.vmul(value, scale_f32, mask), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 2080), ((row_1 * 128) + (half * 64)), mask)
      pto.set_flag("V", "MTE3", event_id=0)
      pto.set_flag("V", "MTE2", event_id=0)
      pto.wait_flag("V", "MTE3", event_id=0)
      pto.mte_ub_gm(pto.castptr(pto.const(8320, dtype=pto.int64), pto.ptr(pto.f32, "ub")), pto.addptr(out, (scalar.cast(w, pto.si64) * 294912) + (scalar.cast(pto.get_block_idx(), pto.si64) * 4096)), scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)) * 512, nburst=(1, scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)) * 512, scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)) * 512), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)