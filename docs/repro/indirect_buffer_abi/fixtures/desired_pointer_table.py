"""Design target for a typed GM pointer-table VMI kernel.

This file is intentionally a *non-executable* PTOAS feature sketch.  It is
kept beside the negative fixture so the requested API is concrete enough to
review, while ``check.sh`` still verifies that today's compiler rejects the
nested pointer type.  The names below describe a reasonable lowering target,
not an API currently provided by PTOAS.

The important property is that pointer lookup replaces only the address
materialization.  It must not replace the production tile schedule with ten
host launches or with a stack/unstack copy pass.
"""

# DESIRED API (illustrative names; this source is not compiled):
#
# from ptodsl import pto
#
# TOKENS = 8192
# LAYERS = 10
# CORES = 72
# LANES = 4
# HIDDEN = 4096
# TILE = LANES * HIDDEN                 # BF16 elements per residual tile
# TILE_BYTES = TILE * 2
# WAVES = (TOKENS + CORES - 1) // CORES
#
# # Every table entry is a GM address supplied by the caller.  A pointer load
# # is an address computation and must be hoisted out of the 64-vector inner
# # loop; it must not copy the pointed-to tensor through an intermediate table.
# GmF32Table = pto.ptr(pto.ptr(pto.f32, "gm"), "gm")
# GmBF16Table = pto.ptr(pto.ptr(pto.bf16, "gm"), "gm")
#
# @pto.jit(name="typed_table_recurrence", target="a5", backend="vpto",
#          mode="explicit", kernel_kind="vector")
# def typed_table_recurrence(
#     comb_table: GmF32Table,
#     initial_residual: pto.ptr(pto.bf16, "gm"),
#     input_table: GmBF16Table,
#     activation_table: GmBF16Table,
#     post_table: GmF32Table,
#     pre_table: GmF32Table,
#     residual_table: GmBF16Table,
#     layer_count: pto.i32,
# ):
#     """One 72-core persistent launch for all ten recurrence stages."""
#     ub = pto.castptr(pto.const(0, pto.i64), pto.ptr(pto.ui8, "ub"))
#     # Two slots are sufficient: while slot s is consumed by VMI, slot 1-s
#     # can receive the next layer's coefficients and activation tile.  The
#     # exact byte offsets are implementation details, but the live regions
#     # must fit the requested dynamic UB and must not alias residual inputs.
#     residual_ub = [pto.addptr(ub, s * (TILE_BYTES + 32768)) for s in range(2)]
#     activation_ub = [pto.addptr(ub, s * 8192 + 65536) for s in range(2)]
#     coeff_ub = [pto.addptr(ub, s * 256 + 98304) for s in range(2)]
#     input_ub = [pto.addptr(ub, s * 8192 + 114688) for s in range(2)]
#     output_ub = [pto.addptr(ub, s * TILE_BYTES + 131072) for s in range(2)]
#
#     for wave in range(WAVES):
#         tile = wave * CORES + pto.get_block_idx()
#         if tile >= TOKENS:
#             continue
#
#         # The initial residual is a single dense tensor.  Each subsequent
#         # stage reads the prior stage's address from residual_table[layer-1].
#         pto.mte_gm_ub(
#             pto.addptr(initial_residual, tile * TILE), residual_ub[0], 0,
#             TILE_BYTES, nburst=(1, TILE_BYTES, TILE_BYTES))
#         pto.pipe_barrier(pto.Pipe.MTE2_V)
#
#         for layer in range(LAYERS):
#             s = layer & 1
#             next_s = 1 - s
#
#             # Typed pointer loads happen once per stage and are valid GM
#             # sources/destinations for DMA.  No data is stacked in a dense
#             # workspace and no host-side loop is needed.
#             pre = pto.load_ptr(pre_table, layer)
#             post = pto.load_ptr(post_table, layer)
#             comb = pto.load_ptr(comb_table, layer)
#             layer_in = pto.load_ptr(input_table, layer)
#             activation = pto.load_ptr(activation_table, layer)
#             residual_out = pto.load_ptr(residual_table, layer)
#
#             # Preload the next stage while this stage's vector work runs.
#             # A real lowering may use explicit MTE/V barriers or equivalent
#             # dependency tokens; the dependency must remain asynchronous.
#             if layer + 1 < LAYERS:
#                 pto.mte_gm_ub(
#                     pto.addptr(pto.load_ptr(pre_table, layer + 1), tile * LANES),
#                     coeff_ub[next_s], 0, 16,
#                     nburst=(1, 16, 16))
#                 pto.mte_gm_ub(
#                     pto.addptr(pto.load_ptr(post_table, layer + 1), tile * LANES),
#                     pto.addptr(coeff_ub[next_s], 64), 0, 16,
#                     nburst=(1, 16, 16))
#                 pto.mte_gm_ub(
#                     pto.addptr(pto.load_ptr(comb_table, layer + 1), tile * LANES * LANES),
#                     pto.addptr(coeff_ub[next_s], 128), 0, 64,
#                     nburst=(1, 64, 64))
#
#             pto.mte_gm_ub(pto.addptr(activation, tile * HIDDEN),
#                          activation_ub[s], 0, 8192,
#                          nburst=(1, 8192, 8192))
#             pto.mte_gm_ub(pto.addptr(pre, tile * LANES),
#                          pto.addptr(coeff_ub[s], 0), 0, 16,
#                          nburst=(1, 16, 16))
#             pto.mte_gm_ub(pto.addptr(post, tile * LANES),
#                          pto.addptr(coeff_ub[s], 64), 0, 16,
#                          nburst=(1, 16, 16))
#             pto.mte_gm_ub(pto.addptr(comb, tile * LANES * LANES),
#                          pto.addptr(coeff_ub[s], 128), 0, 64,
#                          nburst=(1, 64, 64))
#             pto.pipe_barrier(pto.Pipe.MTE2_V)
#
#             mask = pto.vmi.create_mask(64, size=64)
#             pre_v = [pto.vmi.vload(pto.addptr(coeff_ub[s], i), 0,
#                                    dist_mode="brc", size=64)
#                      for i in pto.static_range(LANES)]
#             post_v = [pto.vmi.vload(pto.addptr(coeff_ub[s], 64 + i), 0,
#                                     dist_mode="brc", size=64)
#                       for i in pto.static_range(LANES)]
#             mix_v = [pto.vmi.vload(pto.addptr(coeff_ub[s], 128 + i), 0,
#                                    dist_mode="brc", size=64)
#                      for i in pto.static_range(LANES * LANES)]
#
#             for chunk in range(HIDDEN // 64):
#                 off = chunk * 64
#                 old = [pto.vmi.vcvt(
#                     pto.vmi.vload(pto.addptr(residual_ub[s], lane * HIDDEN + off), 0, size=64),
#                     to_dtype=pto.f32)
#                        for lane in pto.static_range(LANES)]
#                 layer_v = pto.vmi.vmul(old[0], pre_v[0], mask)
#                 for lane in pto.static_range(1, LANES):
#                     layer_v = pto.vmi.vmula(layer_v, old[lane], pre_v[lane], mask)
#                 pto.vmi.vstore(pto.vmi.vcvt(layer_v, to_dtype=pto.bf16),
#                                pto.addptr(input_ub[s], off), 0, mask)
#
#                 act = pto.vmi.vcvt(
#                     pto.vmi.vload(pto.addptr(activation_ub[s], off), 0, size=64),
#                     to_dtype=pto.f32)
#                 result = [pto.vmi.vmul(act, post_v[o], mask)
#                           for o in pto.static_range(LANES)]
#                 # Preserve every old residual register until all 16 products
#                 # have been accumulated. This is the recurrence dependency.
#                 for i in pto.static_range(LANES):
#                     for o in pto.static_range(LANES):
#                         result[o] = pto.vmi.vmula(
#                             result[o], old[i], mix_v[i * LANES + o], mask)
#                 for o in pto.static_range(LANES):
#                     pto.vmi.vstore(
#                         pto.vmi.vcvt(result[o], to_dtype=pto.bf16),
#                         pto.addptr(output_ub[s], o * HIDDEN + off), 0, mask)
#
#             pto.pipe_barrier(pto.Pipe.V_MTE3)
#             pto.mte_ub_gm(input_ub[s], pto.addptr(layer_in, tile * HIDDEN),
#                           8192, nburst=(1, 8192, 8192))
#             pto.mte_ub_gm(output_ub[s],
#                           pto.addptr(residual_out, tile * TILE), TILE_BYTES,
#                           nburst=(1, TILE_BYTES, TILE_BYTES))
#             pto.pipe_barrier(pto.Pipe.MTE3_MTE2)
#             residual_ub[next_s] = output_ub[s]
#
# # Host ABI target:
# # typed_table_recurrence<<<CORES, dynamic_ub_bytes, stream>>>(
# #     comb_table, initial_residual, input_table, activation_table,
# #     post_table, pre_table, residual_table, LAYERS);
# # The pointer-table entries are device addresses, not host pointers.  PTOAS
# # must preserve their GM address space through load_ptr and through MTE2/MTE3
# # lowering, with bounds/effect metadata sufficient for dependency analysis.
