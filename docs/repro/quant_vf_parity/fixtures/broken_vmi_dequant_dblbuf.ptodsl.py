# Full double-buffer dequant dump (wide K-chunk). Provenance for Ask 2 device mismatch.
# Renamed entry for the public package.
from ptodsl import pto, scalar
from ptodsl._ops import _coerce_i64 as _tl_coerce_i64
from ptodsl._surface_values import wrap_surface_value as _tl_wrap_surface_value

@pto.jit(name="dequant_dblbuf_wide", kernel_kind="vector", target="a5", mode="explicit")
def dequant_dblbuf_wide(out: pto.ptr(pto.f32, "gm"), x: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.ui8, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("V", "MTE2", event_id=2)
  pto.set_flag("V", "MTE2", event_id=3)
  for w in range(0, 8):
    __cond_0 = ((w * 9) + (pto.get_block_idx() // 8)) < 64
    pto.wait_flag("V", "MTE2", event_id=(w & 1) + 2)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 64:
      pto.mte_gm_ub(pto.addptr(x_sf, ((w * 73728) + ((pto.get_block_idx() // 4) * 4096)) + ((pto.get_block_idx() % 4) * 16)), pto.addptr(buf_dyn_shmem, ((w & 1) * 1024) + 65536), 0, 16, nburst=(64, 64, 16))
    pto.set_flag("MTE2", "V", event_id=(w & 1) + 2)
    pto.wait_flag("V", "MTE2", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 64:
      pto.mte_gm_ub(pto.addptr(x, ((w * 2359296) + ((pto.get_block_idx() // 4) * 131072)) + ((pto.get_block_idx() % 4) * 512)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (w & 1) * 32768), 0, 512, nburst=(64, 2048, 512))
    pto.set_flag("MTE2", "V", event_id=w & 1)
    pto.wait_flag("MTE2", "V", event_id=(w & 1) + 2)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 64:
      nan_u16 = pto.vmi.vbrc(pto.ui16(32704), size=256)
      for chunk in range(0, 4):
        mask256 = pto.vmi.create_mask(256, size=256)
        sf_u16 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(buf_dyn_shmem, (((w & 1) * 1024) + (chunk * 256)) + 65536), 0, size=256), to_dtype=pto.ui16)
        sf_u16_1 = pto.vmi.vshls(sf_u16, 7, mask256)
        is_inf = pto.vmi.vcmps(sf_u16_1, 32640, mask256, "eq")
        pto.vmi.vstore(pto.vmi.vinterpret_cast(pto.vmi.vsel(is_inf, nan_u16, sf_u16_1), to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w & 1) * 1024) + (chunk * 256)) + 33792), 0, mask256)
    pto.set_flag("V", "MTE2", event_id=(w & 1) + 2)
    pto.wait_flag("MTE3", "V", event_id=w & 1)
    pto.wait_flag("MTE2", "V", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 64:
      for row in range(0, 64):
        for half in range(0, 2):
          mask256_1 = pto.vmi.create_mask(256, size=256)
          scale = pto.vmi.vbrc(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((w & 1) * 1024) + (row * 16)) + (half * 8)) + 33792), 0, group=8, size=8, stride=1), group=8, size=256)
          result = pto.vmi.vmul(pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (((w & 1) * 32768) + (row * 512)) + (half * 256)), 0, size=256), to_dtype=pto.f32), pto.vmi.vcvt(scale, to_dtype=pto.f32), mask256_1)
          pto.vmi.vstore(result, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((((w & 1) * 32768) + (row * 512)) + (half * 256)) + 17920), 0, mask256_1)
    pto.set_flag("V", "MTE3", event_id=w & 1)
    pto.set_flag("V", "MTE2", event_id=w & 1)
    pto.wait_flag("V", "MTE3", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 64:
      pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((w & 1) * 32768) + 17920), pto.addptr(out, ((w * 2359296) + ((pto.get_block_idx() // 4) * 131072)) + ((pto.get_block_idx() % 4) * 512)), 2048, nburst=(64, 2048, 8192), l2_cache="naci")
    pto.set_flag("MTE3", "V", event_id=w & 1)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=2)
  pto.wait_flag("V", "MTE2", event_id=3)