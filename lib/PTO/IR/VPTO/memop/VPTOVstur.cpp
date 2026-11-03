// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//===- VPTOVstur.cpp - pto.Vstur methods ----------------------------------===//
//===----------------------------------------------------------------------===//

#include "VPTOMemOpInternal.h"

using namespace mlir;
using namespace mlir::pto;
using namespace mlir::pto::memop_detail;

namespace {

// VSTUR's effective store length is not an SSA operand: it is read from SPR
// SQZN. Only VSQZ writes SQZN, and pto.init_align only creates the align-state
// carrier -- it does not initialize SQZN (whose reset value is Undefined). The
// VSQZ result must reach this store's value operand directly or through
// pto.vbitcast, which is exactly the form for which the VPTO emitter sets
// VSQZ #st=1 and pushes SQZN. The VPTO op does not carry #st as an attribute:
// the emitter derives it from this value chain, so checking the chain is
// equivalent to checking #st. Otherwise the store length -- and therefore
// whether and what the store writes -- is undefined, so reject the sequence.
//
// This is a value-chain check rather than a dominance or function-wide scan: a
// dominating but unrelated VSQZ is emitted with #st=0 and drops its SQZN, and
// a VSQZ in a non-dominating sibling branch cannot define SQZN on every path
// to the store.
bool hasSqznProducingValue(Value value) {
  Operation *definingOp = value.getDefiningOp();
  while (auto bitcast = dyn_cast_or_null<VbitcastOp>(definingOp)) {
    definingOp = bitcast.getInput().getDefiningOp();
  }
  return isa_and_nonnull<VsqzOp>(definingOp);
}

} // namespace

void VsturOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
  // pto.vstur writes UB through %base, so the write effect is load-bearing
  // even though the updated align state is also returned as an SSA result.
  // Without it, DCE and other effect-based consumers treat the op as free of
  // side effects and can delete it when the caller drops that result (issue
  // 1716: the store disappeared from the emitted LLVM IR). The effect also
  // lets effect-based classifications of the UB access see a store.
  effects.emplace_back(MemoryEffects::Write::get(), &getBaseMutable());
}

LogicalResult VsturOp::verify() {
  if (failed(verifyUbStoreAlignBase(*this))) {
    return failure();
  }
  if (!isSupportedPostMode(getMode())) {
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  }
  if (!hasSqznProducingValue(getValue())) {
    return emitOpError()
           << "requires its value to be produced by a pto.vsqz (directly or "
              "through pto.vbitcast): VSTUR's store length comes from SPR "
              "SQZN, only VSQZ writes SQZN and pto.init_align does not "
              "initialize it; emit a pto.vsqz for this value so the emitter "
              "sets #st=1, or use pto.vstus for a non-squeezed unaligned "
              "store. Otherwise SQZN is Undefined and the store length is "
              "undefined";
  }
  return success();
}
