"""Non-executable sketch of the requested generic surface."""

# DESIRED API (names are illustrative):
#
# @pto.jit(target="a5", backend="vpto")
# def indirect_buffers(
#     table: pto.ptr(pto.ptr(pto.f32, "gm"), "gm"),
#     dst: pto.ptr(pto.f32, "gm"),
#     count: pto.i32,
# ):
#     with pto.for_(0, count) as i:
#         src = pto.load_ptr(table, i)
#         ... DMA from src without first stacking the pointed-to data ...
