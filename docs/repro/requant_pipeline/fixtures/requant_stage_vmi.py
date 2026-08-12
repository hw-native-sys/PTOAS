from ptodsl import pto


@pto.jit(name="requant_stage", kernel_kind="vector", target="a5", mode="explicit")
def requant_stage(
    out: pto.ptr(pto.f8e4m3, "gm"),
    out_sf: pto.ptr(pto.f32, "gm"),
    x: pto.ptr(pto.bf16, "gm"),
):
  ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  for event in pto.static_range(2):
    pto.set_flag("MTE3", "V", event_id=event)
  for event in pto.static_range(2):
    pto.set_flag("V", "MTE2", event_id=event)
  for event in pto.static_range(2):
    pto.set_flag("MTE3", "V", event_id=event + 2)

  for w in pto.static_range(25):
    active = ((w * 2) + (pto.get_block_idx() // 36)) < 49
    ping = w & 1
    pto.wait_flag("MTE3", "V", event_id=ping + 2)
    if active:
      pto.wait_flag("V", "MTE2", event_id=ping)
      gm_offset = ((((w * 72) + pto.get_block_idx()) // 28) * 917504) + (((w * 72) + pto.get_block_idx()) % 28) * 256
      pto.mte_gm_ub(
          pto.addptr(x, gm_offset),
          pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")), ping * 32768),
          0, 512, nburst=(128, 14336, 512),
      )
      pto.set_flag("MTE2", "V", event_id=ping)
      pto.wait_flag("MTE3", "V", event_id=ping)
      pto.wait_flag("MTE2", "V", event_id=ping)

      mask64 = pto.vmi.create_mask(64, size=64)
      mask128 = pto.vmi.create_mask(128, size=128)
      clamp_f = pto.vmi.vbrc(pto.f32(float.fromhex("0x1.a36e2eb1c432dp-14")), size=64)
      inv_qv = pto.vmi.vbrc(pto.f32(float.fromhex("0x1.2492492492492p-9")), size=64)
      u1 = pto.vmi.vbrc(pto.ui32(1), size=64)
      u254 = pto.vmi.vbrc(pto.ui32(254), size=64)
      zero_bf = pto.vmi.vbrc(pto.bf16(0), size=128)
      abs_mask = pto.vmi.vbrc(pto.ui16(32767), size=128)

      for group in pto.static_range(4):
        for chunk in pto.static_range(2):
          amax_bf = zero_bf
          input_base = (ping * 32768) + (group * 8192) + (chunk * 128)
          for offset in pto.static_range(32):
            x_bf = pto.vmi.vload(
                pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")), input_base + (offset * 256)),
                0, size=128,
            )
            abs_bf = pto.vmi.vinterpret_cast(
                pto.vmi.vand(pto.vmi.vinterpret_cast(x_bf, to_dtype=pto.ui16), abs_mask, mask128),
                to_dtype=pto.bf16,
            )
            amax_bf = pto.vmi.vmax(amax_bf, abs_bf, mask128)

          halves = pto.vmi.vintlv(zero_bf, amax_bf, mask128)
          amax0 = pto.vmi.vmax(pto.vmi.vinterpret_cast(halves[0], to_dtype=pto.f32), clamp_f, mask64)
          amax1 = pto.vmi.vmax(pto.vmi.vinterpret_cast(halves[1], to_dtype=pto.f32), clamp_f, mask64)
          scale_raw0 = pto.vmi.vmul(amax0, inv_qv, mask64)
          scale_raw1 = pto.vmi.vmul(amax1, inv_qv, mask64)
          bits0 = pto.vmi.vinterpret_cast(scale_raw0, to_dtype=pto.ui32)
          bits1 = pto.vmi.vinterpret_cast(scale_raw1, to_dtype=pto.ui32)
          exp0 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits0, u1, mask64), 23, mask64), 1, mask64)
          exp1 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits1, u1, mask64), 23, mask64), 1, mask64)
          inverse0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp0, mask64), 23, mask64), to_dtype=pto.f32)
          inverse1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp1, mask64), 23, mask64), to_dtype=pto.f32)
          scale0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp0, 23, mask64), to_dtype=pto.f32)
          scale1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp1, 23, mask64), to_dtype=pto.f32)
          scale_base = (ping * 1024) + (group * 256) + (chunk * 128)
          pto.vmi.vstore(scale0, pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")), scale_base + 49152), 0, mask64)
          pto.vmi.vstore(scale1, pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")), scale_base + 49216), 0, mask64)

          for row in pto.static_range(32):
            row_base = input_base + (row * 256)
            x_bf0 = pto.vmi.vload(pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")), row_base), 0, size=64)
            x_bf1 = pto.vmi.vload(pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")), row_base + 64), 0, size=64)
            y0 = pto.vmi.vcvt(pto.vmi.vmul(pto.vmi.vcvt(x_bf0, to_dtype=pto.f32), inverse0, mask64), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            y1 = pto.vmi.vcvt(pto.vmi.vmul(pto.vmi.vcvt(x_bf1, to_dtype=pto.f32), inverse1, mask64), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
            output_base = row_base + 131072
            pto.vmi.vstore(y0, pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")), output_base), 0, mask64)
            pto.vmi.vstore(y1, pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")), output_base + 64), 0, mask64)

      pto.set_flag("V", "MTE3", event_id=ping)
      pto.set_flag("V", "MTE2", event_id=ping)
      pto.wait_flag("V", "MTE3", event_id=ping)
      pto.mte_ub_gm(pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")), (ping * 32768) + 131072), pto.addptr(out, gm_offset), 256, nburst=(128, 256, 7168), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=ping)
    pto.set_flag("V", "MTE3", event_id=ping + 2)
    pto.wait_flag("V", "MTE3", event_id=ping + 2)
    if active:
      sf_offset = ((((w * 72) + pto.get_block_idx()) // 28) * 28672) + (((w * 72) + pto.get_block_idx()) % 28) * 256
      pto.mte_ub_gm(pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")), (ping * 1024) + 49152), pto.addptr(out_sf, sf_offset), 1024, nburst=(4, 1024, 28672), l2_cache="naci")
    pto.set_flag("MTE3", "V", event_id=ping + 2)

  for event in pto.static_range(2):
    pto.wait_flag("MTE3", "V", event_id=event)
    pto.wait_flag("V", "MTE2", event_id=event)
    pto.wait_flag("MTE3", "V", event_id=event + 2)
