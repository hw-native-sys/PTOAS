import mlir.dialects.pto as pto


@pto.vkernel(target="a5", name="template_abs_kernel")
def template_abs_kernel(src: pto.Tile, dst: pto.Tile):
    total = src.shape[0] * src.shape[1]
    step = 256 // src.ub_ptr.elem_bytes

    with pto.strict_vecscope(src.ub_ptr, dst.ub_ptr, 0, total, step, total) as (
        vin,
        vout,
        lb,
        ub,
        vec_step,
        remaining,
    ):
        for offset in range(lb, ub, vec_step):
            mask, remaining = pto.plt_b32(remaining)
            vec_in = pto.vlds(vin, offset)
            vec_out = pto.vabs(vec_in, mask)
            pto.vsts(vec_out, vout, offset, mask)


template_abs_kernel_f32 = template_abs_kernel.jit(
    src=pto.Tile(
        ub_ptr=pto.ptr(pto.f32, "ub"),
        shape=pto.const([32, 32]),
    ),
    dst=pto.Tile(
        ub_ptr=pto.ptr(pto.f32, "ub"),
        shape=pto.const([32, 32]),
    ),
)

template_abs_kernel_f16 = template_abs_kernel.jit(
    src=pto.Tile(
        ub_ptr=pto.ptr(pto.f16, "ub"),
        shape=pto.const([32, 32]),
    ),
    dst=pto.Tile(
        ub_ptr=pto.ptr(pto.f16, "ub"),
        shape=pto.const([32, 32]),
    ),
)


if __name__ == "__main__":
    print(template_abs_kernel_f32.mlir_text(), end="")
