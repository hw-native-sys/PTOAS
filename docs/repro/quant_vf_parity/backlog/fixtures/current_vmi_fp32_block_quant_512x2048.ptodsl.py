# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# VMI: FP32 → e4m3 block quantize with two on-chip faces (same flow as AscendC reference).
# Shape 512×2048; tile 32×512; stages=1; scale block 32×32; UE8M0-style round.
# Vector body: 64-lane abs-max, dual group=1 reduce, in-register recip, quantize.
# Same known-best loop style as the 8192×2048 twin (range() → scf.for for pair,
# abs-max rows, and quantize).

from ptodsl import pto

@pto.jit(name="fp32_block_quant_512x2048", kernel_kind="vector", target="a5", mode="explicit")
def fp32_block_quant_512x2048(out: pto.ptr(pto.f8e4m3, "gm"), out_sf: pto.ptr(pto.f32, "gm"), x: pto.ptr(pto.f32, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  __cond_0 = pto.get_block_idx() < 64
  if pto.get_block_idx() < 64:
    pto.mte_gm_ub(pto.addptr(x, ((pto.get_block_idx() // 4) * 65536) + ((pto.get_block_idx() % 4) * 512)), pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 0, 2048, nburst=(32, 8192, 2048))
  pto.set_flag("MTE2", "V", event_id=0)
  pto.wait_flag("MTE2", "V", event_id=0)
  if pto.get_block_idx() < 64:
    acc = pto.vmi.vreg(64, pto.f32)
    mask64 = pto.vmi.create_mask(64, size=64)
    mask32 = pto.vmi.create_mask(32, size=64)
    mask1 = pto.vmi.create_mask(1, size=1)
    m_high = pto.vmi.vcmps(pto.vmi.vci(pto.i32(0), size=64), 31, mask64, "gt")
    fp8_max = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.cp+8')), size=1)
    clamp_min = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=1)
    u1 = pto.vmi.vbrc(pto.ui32(1), size=1)
    u254 = pto.vmi.vbrc(pto.ui32(254), size=1)
    for pair in range(0, 8):
      acc = pto.vmi.vbrc(pto.f32(float.fromhex('0x0p+0')), size=64)
      for row in range(0, 32):
        values = pto.vmi.vload(
            pto.addptr(
                pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")),
                (row * 512) + (pair * 64)),
            0, size=64)
        acc = pto.vmi.vmax(acc, pto.vmi.vabs(values, mask64), mask64)
      amax0 = pto.vmi.vmax(pto.vmi.vcmax(acc, mask32, group=1), clamp_min, mask1)
      amax1 = pto.vmi.vmax(pto.vmi.vcmax(acc, m_high, group=1), clamp_min, mask1)
      raw0 = pto.vmi.vdiv(amax0, fp8_max, mask1)
      raw1 = pto.vmi.vdiv(amax1, fp8_max, mask1)
      bits0 = pto.vmi.vinterpret_cast(raw0, to_dtype=pto.ui32)
      bits1 = pto.vmi.vinterpret_cast(raw1, to_dtype=pto.ui32)
      exp0 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits0, u1, mask1), 23, mask1), 1, mask1)
      exp1 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits1, u1, mask1), 23, mask1), 1, mask1)
      scale0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp0, 23, mask1), to_dtype=pto.f32)
      scale1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp1, 23, mask1), to_dtype=pto.f32)
      recip0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp0, mask1), 23, mask1), to_dtype=pto.f32)
      recip1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp1, mask1), 23, mask1), to_dtype=pto.f32)
      pto.vmi.vstore(scale0, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (pair * 2) + 16384), 0, group=1, stride=1)
      pto.vmi.vstore(scale1, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (pair * 2) + 16385), 0, group=1, stride=1)
      recip_v = pto.vmi.vsel(mask32, pto.vmi.vbrc(recip0, group=1, size=64), pto.vmi.vbrc(recip1, group=1, size=64))
      for row in range(0, 32):
        values_q = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), (row * 512) + (pair * 64)), 0, size=64)
        quantized = pto.vmi.vcvt(pto.vmi.vmul(values_q, recip_v, mask64), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
        pto.vmi.vstore(quantized, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((row * 512) + (pair * 64)) + 65600), 0, mask64)
  pto.set_flag("V", "MTE3", event_id=0)
  pto.wait_flag("V", "MTE3", event_id=0)
  if pto.get_block_idx() < 64:
    pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), 65600), pto.addptr(out, ((pto.get_block_idx() // 4) * 65536) + ((pto.get_block_idx() % 4) * 512)), 512, nburst=(32, 512, 2048), l2_cache="naci")
    pto.mte_ub_gm(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), 16384), pto.addptr(out_sf, pto.get_block_idx() * 16), 64, nburst=(1, 64, 64), l2_cache="naci")
