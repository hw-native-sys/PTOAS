from ptodsl import pto

@pto.jit(name="unpack_stage", kernel_kind="vector", target="a5", mode="explicit")
def unpack_stage(out: pto.ptr(pto.bf16, "gm"), x: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.ui8, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("V", "MTE2", event_id=2)
  pto.set_flag("V", "MTE2", event_id=3)
  for w in range(0, 7):
    __cond_0 = ((w * 9) + (pto.get_block_idx() // 8)) < 56
    pto.wait_flag("V", "MTE2", event_id=(w & 1) + 2)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 56:
      pto.mte_gm_ub(pto.addptr(x_sf, ((((w * 72) + pto.get_block_idx()) // 28) * 14336) + ((((w * 72) + pto.get_block_idx()) % 28) * 8)), pto.addptr(buf_dyn_shmem, ((w & 1) * 512) + 32768), 0, 8, nburst=(64, 224, 8))
    pto.set_flag("MTE2", "V", event_id=(w & 1) + 2)
    pto.wait_flag("V", "MTE2", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 56:
      pto.mte_gm_ub(pto.addptr(x, ((((w * 72) + pto.get_block_idx()) // 28) * 458752) + ((((w * 72) + pto.get_block_idx()) % 28) * 256)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (w & 1) * 16384), 0, 256, nburst=(64, 7168, 256))
    pto.set_flag("MTE2", "V", event_id=w & 1)
    pto.wait_flag("MTE2", "V", event_id=(w & 1) + 2)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 56:
      nan_u16 = pto.vmi.vbrc(pto.ui16(32704), size=256)
      for chunk in range(0, 2):
        mask256 = pto.vmi.create_mask(256, size=256)
        sf_u16 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(buf_dyn_shmem, (((w & 1) * 512) + (chunk * 256)) + 32768), 0, size=256), to_dtype=pto.ui16)
        sf_u16_1 = pto.vmi.vshls(sf_u16, 7, mask256)
        is_inf = pto.vmi.vcmps(sf_u16_1, 32640, mask256, "eq")
        pto.vmi.vstore(pto.vmi.vinterpret_cast(pto.vmi.vsel(is_inf, nan_u16, sf_u16_1), to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w & 1) * 512) + (chunk * 256)) + 16896), 0, mask256)
    pto.set_flag("V", "MTE2", event_id=(w & 1) + 2)
    pto.wait_flag("MTE3", "V", event_id=w & 1)
    pto.wait_flag("MTE2", "V", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 56:
      for row in range(0, 64):
        mask256_1 = pto.vmi.create_mask(256, size=256)
        scale = pto.vmi.vbrc(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w & 1) * 512) + (row * 8)) + 16896), 0, group=8, size=8, stride=1), group=8, size=256)
        result = pto.vmi.vmul(pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((w & 1) * 16384) + (row * 256)), 0, size=256), to_dtype=pto.f32), pto.vmi.vcvt(scale, to_dtype=pto.f32), mask256_1)
        pto.vmi.vstore(pto.vmi.vcvt(result, to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (((w & 1) * 16384) + (row * 256)) + 17920), 0, mask256_1)
    pto.set_flag("V", "MTE3", event_id=w & 1)
    pto.set_flag("V", "MTE2", event_id=w & 1)
    pto.wait_flag("V", "MTE3", event_id=w & 1)
    if ((w * 9) + (pto.get_block_idx() // 8)) < 56:
      pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((w & 1) * 16384) + 17920), pto.addptr(out, ((((w * 72) + pto.get_block_idx()) // 28) * 458752) + ((((w * 72) + pto.get_block_idx()) % 28) * 256)), 512, nburst=(64, 512, 14336), l2_cache="naci")
    pto.set_flag("MTE3", "V", event_id=w & 1)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=2)
  pto.wait_flag("V", "MTE2", event_id=3)

