from ptodsl import pto

@pto.jit(name="full_roundtrip_vmi", kernel_kind="vector", target="a5", mode="explicit")
def full_roundtrip_vmi(x: pto.ptr(pto.bf16, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("MTE3", "V", event_id=2)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("V", "MTE2", event_id=2)
  for w in range(0, 29):
    __cond_0 = ((w * 9) + (pto.get_block_idx() // 8)) < 256
    pto.wait_flag("V", "MTE2", event_id=w % 3)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 256:
      pto.mte_gm_ub(pto.addptr(x, ((w * 589824) + ((pto.get_block_idx() // 2) * 16384)) + ((pto.get_block_idx() % 2) * 1024)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (w % 3) * 8192), 0, 2048, nburst=(8, 4096, 2048))
    pto.set_flag("MTE2", "V", event_id=w % 3)
    pto.wait_flag("MTE2", "V", event_id=w % 3)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 256:
      mask256 = pto.vmi.create_mask(256, size=256)
      mask8 = pto.vmi.create_mask(8, size=8)
      abs_mask = pto.vmi.vbrc(pto.ui16(32767), size=256)
      clamp_u = pto.vmi.vbrc(pto.ui16(14546), size=256)
      exp_mask = pto.vmi.vbrc(pto.ui16(32640), size=8)
      emax_u = pto.vmi.vbrc(pto.ui16(1024), size=8)
      bias_u = pto.vmi.vbrc(pto.ui16(32512), size=8)
      nan_u16 = pto.vmi.vbrc(pto.ui16(32704), size=8)
      for i in range(0, 8):
        for tile in range(0, 4):
          x_u = pto.vmi.vinterpret_cast(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w % 3) * 8192) + (i * 1024)) + (tile * 256)), 0, size=256), to_dtype=pto.ui16)
          abs_u = pto.vmi.vand(x_u, abs_mask, mask256)
          abs_u_1 = pto.vmi.vsel(pto.vmi.vcmp(abs_u, clamp_u, mask256, "lt"), clamp_u, abs_u)
          max_u = pto.vmi.vcmax(abs_u_1, mask256, group=8)
          max_exp = pto.vmi.vand(max_u, exp_mask, mask8)
          max_exp_1 = pto.vmi.vsel(pto.vmi.vcmps(max_exp, 1024, mask8, "le"), emax_u, max_exp)
          shared = pto.vmi.vsub(max_exp_1, emax_u, mask8)
          ue_u16 = pto.vmi.vshrs(shared, 7, mask8)
          ue = pto.vmi.vcvt(ue_u16, to_dtype=pto.ui8)
          recip = pto.vmi.vinterpret_cast(pto.vmi.vsub(bias_u, shared, mask8), to_dtype=pto.bf16)
          sf_bits = pto.vmi.vshls(ue_u16, 7, mask8)
          is_inf = pto.vmi.vcmps(sf_bits, 32640, mask8, "eq")
          scale_bf = pto.vmi.vinterpret_cast(pto.vmi.vsel(is_inf, nan_u16, sf_bits), to_dtype=pto.bf16)
          pto.vmi.vstore(ue, pto.addptr(buf_dyn_shmem, ((((w % 3) * 256) + (i * 32)) + (tile * 8)) + 122880), 0, group=8, stride=1)
          pto.vmi.vstore(scale_bf, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((i * 32) + (tile * 8)) + 61824), 0, group=8, stride=1)
          pto.vmi.vstore(recip, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((i * 128) + (tile * 32)) + 62080), 0, group=8, stride=1)
    pto.wait_flag("MTE3", "V", event_id=w % 3)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 256:
      mask256_q = pto.vmi.create_mask(256, size=256)
      for i_1 in range(0, 8):
        for tile_1 in range(0, 4):
          recip_v = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((i_1 * 128) + (tile_1 * 32)) + 62080), 0, dist_mode="brc", group=8, size=256, stride=1)
          x_bf = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w % 3) * 8192) + (i_1 * 1024)) + (tile_1 * 256)), 0, size=256)
          y_bf = pto.vmi.vmul(x_bf, recip_v, mask256_q)
          q = pto.vmi.vcvt(pto.vmi.vcvt(y_bf, to_dtype=pto.f32), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
          pto.vmi.vstore(q, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((w % 3) * 8192) + (i_1 * 1024)) + (tile_1 * 256)) + 49152), 0, mask256_q)
          scale8 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((i_1 * 32) + (tile_1 * 8)) + 61824), 0, group=8, size=8, stride=1)
          scale = pto.vmi.vbrc(scale8, group=8, size=256)
          q_f = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((((w % 3) * 8192) + (i_1 * 1024)) + (tile_1 * 256)) + 49152), 0, size=256), to_dtype=pto.f32)
          out_f = pto.vmi.vmul(q_f, pto.vmi.vcvt(scale, to_dtype=pto.f32), mask256_q)
          pto.vmi.vstore(pto.vmi.vcvt(out_f, to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((((w % 3) * 8192) + (i_1 * 1024)) + (tile_1 * 256)) + 36864), 0, mask256_q)
    pto.set_flag("V", "MTE3", event_id=w % 3)
    pto.set_flag("V", "MTE2", event_id=w % 3)
    pto.wait_flag("V", "MTE3", event_id=w % 3)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 256:
      pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((w % 3) * 8192) + 36864), pto.addptr(x, ((w * 589824) + ((pto.get_block_idx() // 2) * 16384)) + ((pto.get_block_idx() % 2) * 1024)), 2048, nburst=(8, 2048, 4096), l2_cache="naci")
    pto.set_flag("MTE3", "V", event_id=w % 3)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=2)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=2)

