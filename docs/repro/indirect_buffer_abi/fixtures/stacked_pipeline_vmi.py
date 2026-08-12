"""One dense-buffer VMI recurrence stage for the pointer-table ABI reproducer."""

from ptodsl import pto

TOKENS, CORES, LANES, HIDDEN = 8192, 72, 4, 4096
TILE_VALUES = LANES * HIDDEN
TILE_BYTES = TILE_VALUES * 2


@pto.jit(name="dense_recurrence_stage", kernel_kind="vector", target="a5", mode="explicit")
def dense_recurrence_stage(
    residual_in: pto.ptr(pto.bf16, "gm"),
    pre_mix: pto.ptr(pto.f32, "gm"),
    layer_output: pto.ptr(pto.bf16, "gm"),
    post_mix: pto.ptr(pto.f32, "gm"),
    comb_mix: pto.ptr(pto.f32, "gm"),
    layer_input: pto.ptr(pto.bf16, "gm"),
    residual_out: pto.ptr(pto.bf16, "gm"),
):
    """Apply one 4-lane recurrence to a dense 8192 x 4096 tensor.

    The host invokes this fixed-formal kernel ten times in stream order.  That
    is the portable replacement for a runtime pointer-table loop: its extra
    dense staging is intentionally part of the VMI timing.
    """
    ub_u8 = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
    ub_bf16 = pto.castptr(ub_u8, pto.ptr(pto.bf16, "ub"))
    ub_f32 = pto.castptr(ub_u8, pto.ptr(pto.f32, "ub"))
    # UB: residual [0, 32768), activation [32768, 40960), input [40960,
    # 45056) in BF16 elements; 24 float coefficients begin at byte 98304.
    # Keep this offset identical to the compact CCE layout.
    coeff = 24576
    for wave in range(114):
        tile = wave * CORES + pto.get_block_idx()
        if tile < TOKENS:
            pto.mte_gm_ub(pto.addptr(residual_in, tile * TILE_VALUES), ub_bf16, 0, TILE_BYTES,
                          nburst=(1, TILE_BYTES, TILE_BYTES))
            pto.mte_gm_ub(pto.addptr(layer_output, tile * HIDDEN), pto.addptr(ub_bf16, 32768), 0, 8192,
                          nburst=(1, 8192, 8192))
            pto.mte_gm_ub(pto.addptr(pre_mix, tile * LANES), pto.addptr(ub_f32, coeff), 0, 16,
                          nburst=(1, 16, 16))
            pto.mte_gm_ub(pto.addptr(post_mix, tile * LANES), pto.addptr(ub_f32, coeff + 4), 0, 16,
                          nburst=(1, 16, 16))
            pto.mte_gm_ub(pto.addptr(comb_mix, tile * LANES * LANES), pto.addptr(ub_f32, coeff + 8), 0, 64,
                          nburst=(1, 64, 64))
        # Keep the standalone form deliberately conservative.  These complete
        # barriers make the load/compute/store dependencies unambiguous; the
        # workload is large enough that the measured gap remains device work,
        # not host-launch latency.
        pto.pipe_barrier(pto.Pipe.ALL)
        if tile < TOKENS:
            mask = pto.vmi.create_mask(64, size=64)
            pre = [pto.vmi.vload(pto.addptr(ub_f32, coeff + lane), 0, dist_mode="brc", size=64)
                   for lane in pto.static_range(LANES)]
            post = [pto.vmi.vload(pto.addptr(ub_f32, coeff + 4 + lane), 0, dist_mode="brc", size=64)
                    for lane in pto.static_range(LANES)]
            mix = [pto.vmi.vload(pto.addptr(ub_f32, coeff + 8 + index), 0, dist_mode="brc", size=64)
                   for index in pto.static_range(LANES * LANES)]
            for chunk in range(64):
                offset = chunk * 64
                residual = [
                    pto.vmi.vcvt(pto.vmi.vload(pto.addptr(ub_bf16, lane * HIDDEN + offset), 0, size=64),
                                to_dtype=pto.f32)
                    for lane in pto.static_range(LANES)
                ]
                layer_in = pto.vmi.vmul(residual[0], pre[0], mask)
                for lane in pto.static_range(1, LANES):
                    layer_in = pto.vmi.vmula(layer_in, residual[lane], pre[lane], mask)
                pto.vmi.vstore(pto.vmi.vcvt(layer_in, to_dtype=pto.bf16),
                               pto.addptr(ub_bf16, 40960 + offset), 0, mask)
                activation = pto.vmi.vcvt(
                    pto.vmi.vload(pto.addptr(ub_bf16, 32768 + offset), 0, size=64), to_dtype=pto.f32)
                # Keep the old residual tile intact until every output lane
                # has consumed it. Storing each lane as it is computed would
                # overwrite a later lane's input in this in-place UB buffer.
                result = [pto.vmi.vmul(activation, post[output_lane], mask)
                          for output_lane in pto.static_range(LANES)]
                for input_lane in pto.static_range(LANES):
                    for output_lane in pto.static_range(LANES):
                        result[output_lane] = pto.vmi.vmula(
                            result[output_lane], residual[input_lane],
                            mix[input_lane * LANES + output_lane], mask,
                        )
                for output_lane in pto.static_range(LANES):
                    pto.vmi.vstore(pto.vmi.vcvt(result[output_lane], to_dtype=pto.bf16),
                                   pto.addptr(ub_bf16, output_lane * HIDDEN + offset), 0, mask)
        pto.pipe_barrier(pto.Pipe.ALL)
        if tile < TOKENS:
            pto.mte_ub_gm(pto.addptr(ub_bf16, 40960), pto.addptr(layer_input, tile * HIDDEN), 8192,
                          nburst=(1, 8192, 8192), l2_cache="naci")
            pto.mte_ub_gm(ub_bf16, pto.addptr(residual_out, tile * TILE_VALUES), TILE_BYTES,
                          nburst=(1, TILE_BYTES, TILE_BYTES), l2_cache="naci")
        pto.pipe_barrier(pto.Pipe.ALL)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["--emit-mlir"]:
        print(dense_recurrence_stage.compile().mlir_text())
    else:
        raise SystemExit("usage: stacked_pipeline_vmi.py --emit-mlir")
