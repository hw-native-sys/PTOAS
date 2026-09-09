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
def main_kernel(out: pto.ptr(pto.f32, "gm"), x: pto.ptr(pto.f4e2m1x2, "gm"), x_sf: pto.ptr(pto.ui8, "gm"), num_tokens: pto.si32, sf_extent: pto.si32, sf_stride: pto.si32):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  x_ub_version_counter_1 = pto.const(0, dtype=pto.int64)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("V", "MTE2", event_id=2)
  x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(0, context="PTO local.var store"))
  for w in range(0, ((num_tokens + 2303) // 2304), 1):
    __cond_0 = ((w * 72) + pto.get_block_idx()) < ((num_tokens + 31) // 32)
    if ((w * 72) + pto.get_block_idx()) < ((num_tokens + 31) // 32):
      for pid_k in range(0, 4, 1):
        pto.wait_flag("V", "MTE2", event_id=2)
        if 0 < ((sf_extent - (pto.get_block_idx() * 2)) - (w * 144)):
          pto.mte_gm_ub(pto.addptr(x_sf, ((scalar.cast(w, pto.si64) * 144) + ((scalar.cast(pid_k, pto.si64) * scalar.cast(sf_stride, pto.si64)) * 8)) + (scalar.cast(pto.get_block_idx(), pto.si64) * 2)), pto.addptr(buf_dyn_shmem, 16384), 0, scalar.min(2, ((sf_extent - (pto.get_block_idx() * 2)) - (w * 144))), nburst=(8, sf_stride, 64))
        pto.set_flag("MTE2", "V", event_id=2)
        pto.wait_flag("V", "MTE2", event_id=x_ub_version_counter_1 & 1)
        pto.mte_gm_ub(pto.addptr(x, ((scalar.cast(w, pto.si64) * 2359296) + (scalar.cast(pto.get_block_idx(), pto.si64) * 32768)) + (scalar.cast(pid_k, pto.si64) * 256)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f4e2m1x2, "ub")), (x_ub_version_counter_1 & 1) * 8192), 0, 256, nburst=(scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)), 1024, 256))
        pto.set_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
        pto.wait_flag("MTE2", "V", event_id=2)
        with pto.vecscope():
          one = pto.vmi.create_mask(1, size=1)
          nan1 = pto.vmi.vbrc(pto.ui16(32704), size=1)
          for row in range(0, 32, 1):
            for scale_i in range(0, 16, 1):
              loaded = pto.vmi.vload(pto.addptr(buf_dyn_shmem, 16384), (((scale_i // 2) * 64) + (scale_i & 1)), size=1)
              u16 = pto.vmi.vcvt(loaded, to_dtype=pto.ui16)
              shifted = pto.vmi.vshls(u16, 7, one)
              is_inf = pto.vmi.vcmps(shifted, 32640, one, "eq")
              pto.vmi.vstore(pto.vmi.vinterpret_cast(pto.vmi.vsel(is_inf, nan1, shifted), to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 8448), ((row * 64) + scale_i), one)
        pto.set_flag("V", "MTE2", event_id=2)
        pto.wait_flag("MTE3", "V", event_id=x_ub_version_counter_1 & 1)
        pto.wait_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
        with pto.vecscope():
          mask = pto.vmi.create_mask(128, size=128)
          for row_1 in range(0, 32, 1):
            for half in range(0, 4, 1):
              scale = pto.vmi.vbrc(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 8448), ((row_1 * 64) + (half * 4)), group=4, size=4, stride=1), group=4, size=128)
              source = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f4e2m1x2, "ub")), (x_ub_version_counter_1 & 1) * 8192), ((row_1 * 256) + (half * 64)), size=(128 // 2))
              value_bf16 = pto.vmi.vinterpret_cast(pto.vmi.vcvt(source, to_dtype=pto.vmi.bf16x2), to_dtype=pto.bf16)
              result = pto.vmi.vmul(pto.vmi.vcvt(value_bf16, to_dtype=pto.f32), pto.vmi.vcvt(scale, to_dtype=pto.f32), mask)
              pto.vmi.vstore(result, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 16384) + 5248), ((row_1 * 512) + (half * 128)), mask)
        pto.set_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
        pto.set_flag("V", "MTE2", event_id=x_ub_version_counter_1 & 1)
        pto.wait_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
        pto.mte_ub_gm(pto.castptr(scalar.muli(_tl_coerce_i64(((x_ub_version_counter_1 & 1) * 16384) + 5248, context="PTO local pointer offset"), pto.const(4, dtype=pto.int64)), pto.ptr(pto.f32, "ub")), pto.addptr(out, ((scalar.cast(w, pto.si64) * 4718592) + (scalar.cast(pto.get_block_idx(), pto.si64) * 65536)) + (scalar.cast(pid_k, pto.si64) * 512)), 2048, nburst=(scalar.min(32, scalar.max(((num_tokens - (pto.get_block_idx() * 32)) - (w * 2304)), 0)), 2048, 8192), l2_cache="naci")
        pto.set_flag("MTE3", "V", event_id=x_ub_version_counter_1 & 1)
        x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(x_ub_version_counter_1 + 1, context="PTO local.var store"))
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=2)