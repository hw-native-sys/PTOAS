"""Standalone VMI BF16 -> E4M3 -> BF16 round trip.

This is a deliberately explicit, two-stage persistent pipeline for an
8192x2048 matrix.  The FP8 payload and FP32 group scales are caller-owned
device buffers.  Their allocation is outside the timed operation.
"""

from ptodsl import pto


ROWS, WIDTH, CORES = 8192, 2048, 72
GROUP, VECTOR = 32, 256


@pto.jit(name="float_scale_quantize", kernel_kind="vector", target="a5", mode="explicit")
def float_scale_quantize(
    source: pto.ptr(pto.bf16, "gm"),
    encoded: pto.ptr(pto.f8e4m3, "gm"),
    scales: pto.ptr(pto.f32, "gm"),
):
    """Encode 32x1024 tiles through alternating UB slots."""
    ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    pto.set_flag("MTE3", "V", event_id=0)
    pto.set_flag("MTE3", "V", event_id=1)
    pto.set_flag("MTE3", "V", event_id=2)
    pto.set_flag("MTE3", "V", event_id=3)
    pto.set_flag("V", "MTE2", event_id=0)
    pto.set_flag("V", "MTE2", event_id=1)
    for wave in range(8):
        pto.wait_flag("V", "MTE2", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            pto.mte_gm_ub(
                pto.addptr(source, wave * 2359296 + (pto.get_block_idx() // 2) * 65536
                           + (pto.get_block_idx() % 2) * 1024),
                pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")), (wave & 1) * 32768),
                0, 2048, nburst=(32, 4096, 2048),
            )
        pto.set_flag("MTE2", "V", event_id=wave & 1)
        pto.wait_flag("MTE3", "V", event_id=(wave & 1) + 2)
        pto.wait_flag("MTE2", "V", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            mask256 = pto.vmi.create_mask(256, size=256)
            mask8 = pto.vmi.create_mask(8, size=8)
            fp8_max = pto.vmi.vbrc(pto.f32(448.0), size=8)
            clamp_min = pto.vmi.vbrc(pto.f32(1.0e-4), size=8)
            # Keep these materialized: they preserve the known low-level
            # allocation/scheduling form even though later PTOAS passes fold them.
            pto.vmi.vbrc(pto.ui32(1), size=8)
            pto.vmi.vbrc(pto.ui32(254), size=8)
            for row in range(32):
                for strip in range(4):
                    value = pto.vmi.vcvt(
                        pto.vmi.vload(
                            pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")),
                                       (wave & 1) * 32768 + row * 1024 + strip * 256),
                            0, size=256,
                        ), to_dtype=pto.f32,
                    )
                    maximum = pto.vmi.vmax(
                        pto.vmi.vcmax(pto.vmi.vabs(value, mask256), mask256, group=8),
                        clamp_min, mask8,
                    )
                    scale = pto.vmi.vdiv(maximum, fp8_max, mask8)
                    reciprocal = pto.vmi.vdiv(fp8_max, maximum, mask8)
                    pto.vmi.vstore(
                        scale,
                        pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")),
                                   (wave & 1) * 1024 + row * 32 + strip * 8 + 49152),
                        0, group=8, stride=1,
                    )
                    pto.vmi.vstore(
                        reciprocal,
                        pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")),
                                   row * 128 + strip * 32 + 51200),
                        0, group=8, stride=1,
                    )
        pto.set_flag("V", "MTE3", event_id=(wave & 1) + 2)
        pto.wait_flag("MTE3", "V", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            mask256 = pto.vmi.create_mask(256, size=256)
            for row in range(32):
                for strip in range(4):
                    reciprocal = pto.vmi.vload(
                        pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")),
                                   row * 128 + strip * 32 + 51200),
                        0, dist_mode="brc", group=8, size=256, stride=1,
                    )
                    value = pto.vmi.vcvt(
                        pto.vmi.vload(
                            pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")),
                                       (wave & 1) * 32768 + row * 1024 + strip * 256),
                            0, size=256,
                        ), to_dtype=pto.f32,
                    )
                    pto.vmi.vstore(
                        pto.vmi.vcvt(
                            pto.vmi.vmul(value, reciprocal, mask256), rounding="R",
                            saturate="SAT", to_dtype=pto.f8e4m3,
                        ),
                        pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")),
                                   (wave & 1) * 32768 + row * 1024 + strip * 256 + 131072),
                        0, mask256,
                    )
        pto.set_flag("V", "MTE3", event_id=wave & 1)
        pto.set_flag("V", "MTE2", event_id=wave & 1)
        pto.wait_flag("V", "MTE3", event_id=(wave & 1) + 2)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            pto.mte_ub_gm(
                pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")),
                           (wave & 1) * 1024 + 49152),
                pto.addptr(scales, wave * 73728 + (pto.get_block_idx() // 2) * 2048
                           + (pto.get_block_idx() % 2) * 32),
                128, nburst=(32, 128, 256), l2_cache="naci",
            )
        pto.set_flag("MTE3", "V", event_id=(wave & 1) + 2)
        pto.wait_flag("V", "MTE3", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            pto.mte_ub_gm(
                pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")),
                           (wave & 1) * 32768 + 131072),
                pto.addptr(encoded, wave * 2359296 + (pto.get_block_idx() // 2) * 65536
                           + (pto.get_block_idx() % 2) * 1024),
                1024, nburst=(32, 1024, 2048), l2_cache="naci",
            )
        pto.set_flag("MTE3", "V", event_id=wave & 1)
    pto.wait_flag("MTE3", "V", event_id=0)
    pto.wait_flag("MTE3", "V", event_id=1)
    pto.wait_flag("MTE3", "V", event_id=2)
    pto.wait_flag("MTE3", "V", event_id=3)
    pto.wait_flag("V", "MTE2", event_id=0)
    pto.wait_flag("V", "MTE2", event_id=1)


@pto.jit(name="float_scale_dequantize", kernel_kind="vector", target="a5", mode="explicit")
def float_scale_dequantize(
    destination: pto.ptr(pto.bf16, "gm"),
    encoded: pto.ptr(pto.f8e4m3, "gm"),
    scales: pto.ptr(pto.f32, "gm"),
):
    """Decode 64x256 tiles through alternating UB slots."""
    ub = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    pto.set_flag("MTE3", "V", event_id=0)
    pto.set_flag("MTE3", "V", event_id=1)
    pto.set_flag("V", "MTE2", event_id=0)
    pto.set_flag("V", "MTE2", event_id=1)
    for wave in range(8):
        pto.wait_flag("V", "MTE2", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            pto.mte_gm_ub(
                pto.addptr(scales, wave * 73728 + (pto.get_block_idx() // 4) * 4096
                           + (pto.get_block_idx() % 4) * 16),
                pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")),
                           (wave & 1) * 1024 + 16384),
                0, 64, nburst=(64, 256, 64),
            )
            pto.mte_gm_ub(
                pto.addptr(encoded, wave * 2359296 + (pto.get_block_idx() // 4) * 131072
                           + (pto.get_block_idx() % 4) * 512),
                pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")), (wave & 1) * 32768),
                0, 512, nburst=(64, 2048, 512),
            )
        pto.set_flag("MTE2", "V", event_id=wave & 1)
        pto.wait_flag("MTE3", "V", event_id=wave & 1)
        pto.wait_flag("MTE2", "V", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            for row in range(64):
                for half in range(2):
                    mask256 = pto.vmi.create_mask(256, size=256)
                    scale = pto.vmi.vbrc(
                        pto.vmi.vload(
                            pto.addptr(pto.castptr(ub, pto.ptr(pto.f32, "ub")),
                                       (wave & 1) * 1024 + row * 16 + half * 8 + 16384),
                            0, group=8, size=8, stride=1,
                        ), group=8, size=256,
                    )
                    value = pto.vmi.vcvt(
                        pto.vmi.vload(
                            pto.addptr(pto.castptr(ub, pto.ptr(pto.f8e4m3, "ub")),
                                       (wave & 1) * 32768 + row * 512 + half * 256),
                            0, size=256,
                        ), to_dtype=pto.f32,
                    )
                    pto.vmi.vstore(
                        pto.vmi.vcvt(pto.vmi.vmul(value, scale, mask256), to_dtype=pto.bf16),
                        pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")),
                                   (wave & 1) * 32768 + row * 512 + half * 256 + 36864),
                        0, mask256,
                    )
        pto.set_flag("V", "MTE3", event_id=wave & 1)
        pto.set_flag("V", "MTE2", event_id=wave & 1)
        pto.wait_flag("V", "MTE3", event_id=wave & 1)
        if wave * 9 + pto.get_block_idx() // 8 < 64:
            pto.mte_ub_gm(
                pto.addptr(pto.castptr(ub, pto.ptr(pto.bf16, "ub")),
                           (wave & 1) * 32768 + 36864),
                pto.addptr(destination, wave * 2359296 + (pto.get_block_idx() // 4) * 131072
                           + (pto.get_block_idx() % 4) * 512),
                1024, nburst=(64, 1024, 4096), l2_cache="naci",
            )
        pto.set_flag("MTE3", "V", event_id=wave & 1)
    pto.wait_flag("MTE3", "V", event_id=0)
    pto.wait_flag("MTE3", "V", event_id=1)
    pto.wait_flag("V", "MTE2", event_id=0)
    pto.wait_flag("V", "MTE2", event_id=1)
