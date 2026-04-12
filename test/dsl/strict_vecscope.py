import mlir.dialects.pto as pto


@pto.vkernel(target="a5", name="abs_strict_vecscope_kernel_2d")
def abs_strict_vecscope_kernel_2d(
    inp: pto.ptr(pto.f32, "gm"), out: pto.ptr(pto.f32, "gm")
):
    ub_in = pto.castptr(0, pto.ptr(pto.f32, "ub"))
    ub_out = pto.castptr(4096, pto.ptr(pto.f32, "ub"))

    pto.set_loop_size_outtoub(1, 1)
    pto.copy_gm_to_ubuf(inp, ub_in, 0, 32, 128, 0, 0, False, 0, 128, 128)

    pto.set_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")
    pto.wait_flag("PIPE_MTE2", "PIPE_V", "EVENT_ID0")

    with pto.strict_vecscope(ub_in, ub_out, 0, 1024, 64, 1024) as (
        src,
        dst,
        lb,
        ub,
        step,
        remaining,
    ):
        for offset in range(lb, ub, step):
            mask, remaining = pto.plt_b32(remaining)
            vec_in = pto.vlds(src, offset)
            vec_out = pto.vabs(vec_in, mask)
            pto.vsts(vec_out, dst, offset, mask)

    pto.set_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")
    pto.wait_flag("PIPE_V", "PIPE_MTE3", "EVENT_ID0")

    pto.set_loop_size_ubtoout(1, 1)
    pto.copy_ubuf_to_gm(ub_out, out, 0, 32, 128, 0, 128, 128)
    pto.barrier("PIPE_ALL")

    return


if __name__ == "__main__":
    print(abs_strict_vecscope_kernel_2d.mlir_text(), end="")
