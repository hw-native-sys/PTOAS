from ptodsl import pto, scalar
from ptodsl._ops import _coerce_i64 as _tl_coerce_i64
from ptodsl._surface_values import wrap_surface_value as _tl_wrap_surface_value

@pto.jit(name="packed_group_convert_vmi", kernel_kind="vector", target="a5", mode="explicit")
def packed_group_convert_vmi(x: pto.ptr(pto.bf16, "gm"), x_q: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.ui8, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("MTE3", "V", event_id=2)
  pto.set_flag("MTE3", "V", event_id=3)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  for w in range(0, 56):
    __cond_0 = ((w * 9) + (pto.get_block_idx() // 8)) < 502
    pto.wait_flag("V", "MTE2", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 502:
      pto.mte_gm_ub(pto.addptr(x, ((w * 2359296) + ((pto.get_block_idx() // 8) * 262144)) + ((pto.get_block_idx() % 8) * 2048)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (w & 1) * 32768), 0, 4096, nburst=(16, 32768, 4096))
    pto.set_flag("MTE2", "V", event_id=w & 1)
    pto.wait_flag("MTE3", "V", event_id=(w & 1) + 2)
    pto.wait_flag("MTE2", "V", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 502:
      mask256 = pto.vmi.create_mask(256, size=256)
      mask8 = pto.vmi.create_mask(8, size=8)
      abs_mask = pto.vmi.vbrc(pto.ui16(32767), size=256)
      clamp_u = pto.vmi.vbrc(pto.ui16(14546), size=256)
      exp_mask = pto.vmi.vbrc(pto.ui16(32640), size=8)
      emax_u = pto.vmi.vbrc(pto.ui16(1024), size=8)
      bias_u = pto.vmi.vbrc(pto.ui16(32512), size=8)
      for i in range(0, 16):
        for tile in range(0, 8):
          x_u = pto.vmi.vinterpret_cast(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w & 1) * 32768) + (i * 2048)) + (tile * 256)), 0, size=256), to_dtype=pto.ui16)
          abs_u = pto.vmi.vand(x_u, abs_mask, mask256)
          abs_u_1 = pto.vmi.vsel(pto.vmi.vcmp(abs_u, clamp_u, mask256, "lt"), clamp_u, abs_u)
          max_u = pto.vmi.vcmax(abs_u_1, mask256, group=8)
          max_exp = pto.vmi.vand(max_u, exp_mask, mask8)
          max_exp_1 = pto.vmi.vsel(pto.vmi.vcmps(max_exp, 1024, mask8, "le"), emax_u, max_exp)
          shared = pto.vmi.vsub(max_exp_1, emax_u, mask8)
          ue = pto.vmi.vcvt(pto.vmi.vshrs(shared, 7, mask8), to_dtype=pto.ui8)
          recip = pto.vmi.vinterpret_cast(pto.vmi.vsub(bias_u, shared, mask8), to_dtype=pto.bf16)
          pto.vmi.vstore(ue, pto.addptr(buf_dyn_shmem, ((((w & 1) * 1024) + (i * 64)) + (tile * 8)) + 196608), 0, group=8, stride=1)
          pto.vmi.vstore(recip, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((i * 256) + (tile * 32)) + 99328), 0, group=8, stride=1)
    pto.set_flag("V", "MTE3", event_id=(w & 1) + 2)
    pto.wait_flag("MTE3", "V", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 502:
      mask256_q = pto.vmi.create_mask(256, size=256)
      for i_1 in range(0, 16):
        for tile_1 in range(0, 8):
          recip_v = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((i_1 * 256) + (tile_1 * 32)) + 99328), 0, dist_mode="brc", group=8, size=256, stride=1)
          x_bf = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w & 1) * 32768) + (i_1 * 2048)) + (tile_1 * 256)), 0, size=256)
          y_bf = pto.vmi.vmul(x_bf, recip_v, mask256_q)
          pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vcvt(y_bf, to_dtype=pto.f32), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((w & 1) * 32768) + (i_1 * 2048)) + (tile_1 * 256)) + 131072), 0, mask256_q)
    pto.set_flag("V", "MTE3", event_id=w & 1)
    pto.set_flag("V", "MTE2", event_id=w & 1)
    pto.wait_flag("V", "MTE3", event_id=(w & 1) + 2)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 502:
      pto.mte_ub_gm(pto.addptr(buf_dyn_shmem, ((w & 1) * 1024) + 196608), pto.addptr(x_sf, ((w * 73728) + ((pto.get_block_idx() // 8) * 8192)) + ((pto.get_block_idx() % 8) * 64)), 64, nburst=(16, 64, 512), l2_cache="naci")
    pto.set_flag("MTE3", "V", event_id=(w & 1) + 2)
    pto.wait_flag("V", "MTE3", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 502:
      pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((w & 1) * 32768) + 131072), pto.addptr(x_q, ((w * 2359296) + ((pto.get_block_idx() // 8) * 262144)) + ((pto.get_block_idx() % 8) * 2048)), 2048, nburst=(16, 2048, 16384), l2_cache="naci")
    pto.set_flag("MTE3", "V", event_id=w & 1)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=2)
  pto.wait_flag("MTE3", "V", event_id=3)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
