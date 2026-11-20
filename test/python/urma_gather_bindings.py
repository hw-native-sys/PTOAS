# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from ptoas.mlir.dialects import pto
from ptoas.mlir.ir import Context, InsertionPoint, IntegerType, Location, Module, Type


def check_gather_bindings() -> None:
    """Check shared URMA handles, Gather/Scatter builders and zero-valued enums."""
    session_type = pto.AsyncSessionType.get_urma()
    event_type = pto.AsyncEventType.get_urma()
    if session_type != Type.parse("!pto.async_session<urma>") or event_type != Type.parse("!pto.async_event<urma>"):
        raise RuntimeError("URMA handle types do not round trip")
    if str(pto.AsyncSessionType.get()) != "!pto.async_session" or str(pto.AsyncEventType.get()) != "!pto.async_event":
        raise RuntimeError("legacy async handle constructors changed")

    module = Module.parse("""
        module attributes {pto.target_arch = "a5"} {
          func.func @gather(%workspace: !pto.ptr<i8, gm>, %dsts: !pto.ptr<i8, gm>,
                            %segs: !pto.ptr<i8, gm>, %n: ui32, %scratch: !pto.tile_buf<vec, 1x1280xi8>)
              attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
            return
          }
        }
    """)
    block = module.body.operations[0].regions[0].blocks[0]
    workspace, dsts, segs, count, scratch = block.arguments
    i1 = IntegerType.get_signless(1)
    with InsertionPoint.at_block_begin(block):
        scalar = pto.BuildUrmaSessionOp(session_type, i1, workspace)
        pto.TPutAsyncGatherOp(
            event_type, i1, scalar.session,
            destinations=dsts, destinationCount=count, segments=segs, segmentCount=count,
        )
        vector = pto.BuildUrmaSessionOp(session_type, i1, workspace, scratch=scratch)
        prefix = pto.TPutAsyncGatherOp(
            event_type, i1, vector.session,
            destinations=dsts, destinationCount=count, segments=segs, segmentCount=count,
            scratch=scratch, cqe_policy=0, submit_mode=0,
        )
        final = pto.TGetAsyncScatterOp(
            event_type, i1, vector.session, scratch=scratch,
            cqe_policy=pto.UrmaCqePolicy.Final, submit_mode=pto.AsyncSubmitMode.DeferAndSubmit,
        )
        pto.WaitAsyncEventOp(final.event, vector.session)
        get = pto.TGetAsyncScatterOp(
            event_type, i1, scalar.session,
            sources=dsts, sourceCount=count, segments=segs, segmentCount=count,
            cqe_policy=0, submit_mode=0,
        )
        put_final = pto.TPutAsyncGatherOp(
            event_type, i1, scalar.session,
            cqe_policy=pto.UrmaCqePolicy.Final, submit_mode=pto.AsyncSubmitMode.DeferAndSubmit,
        )
        pto.TestAsyncEventOp(put_final.event, scalar.session)
    if any(str(op.cqe_policy) != "#pto.urma_cqe_policy<no_cqe>" for op in (prefix, get)):
        raise RuntimeError("explicit zero-valued no_cqe policy was dropped")
    if pto.AsyncSubmitModeAttr.get(0).value != pto.AsyncSubmitMode.Immediate.value:
        raise RuntimeError("explicit zero-valued immediate mode was dropped")
    if not module.operation.verify() or not Module.parse(str(module)).operation.verify():
        raise RuntimeError("URMA Gather/Scatter bindings produced invalid IR")


def check_legacy_async_builders() -> None:
    """Check that legacy builders still infer their unparameterized results."""
    module = Module.parse("""
        module {
          func.func @legacy(%workspace: !pto.ptr<i8, gm>, %scratch: !pto.tile_buf<vec, 1x1280xi8>,
                            %view: !pto.partition_tensor_view<128xf32>, %ctx: !pto.prefetch_async_context) {
            return
          }
        }
    """)
    block = module.body.operations[0].regions[0].blocks[0]
    workspace, scratch, view, prefetch_context = block.arguments
    with InsertionPoint.at_block_begin(block):
        session = pto.BuildAsyncSessionOp(scratch, workspace).result
        prefetch_session = pto.GetPrefetchAsyncSessionOp(prefetch_context).result
        put = pto.TPutAsyncOp(view, view, session).result
        get = pto.TGetAsyncOp(view, view, session).result
        prefetch = pto.TPrefetchAsyncOp(view, prefetch_context).result
    session_type = pto.AsyncSessionType.get()
    event_type = pto.AsyncEventType.get()
    if session.type != session_type or prefetch_session.type != session_type:
        raise RuntimeError("legacy async session result inference changed")
    if any(event.type != event_type for event in (put, get, prefetch)):
        raise RuntimeError("legacy async event result inference changed")
    if not module.operation.verify():
        raise RuntimeError("legacy async builders produced invalid IR")


def main() -> None:
    with Context() as context, Location.unknown(context):
        pto.register_dialect(context, load=True)
        check_gather_bindings()
        check_legacy_async_builders()


if __name__ == "__main__":
    main()
