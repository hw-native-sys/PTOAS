// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// State verification for the initial straight-line local-tile profile.
#include "PTO/IR/Linx/LinxDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
using namespace mlir;
using namespace mlir::pto::linx;

static LogicalResult checkFunctionSignature(func::FuncOp func)
{
    if (func.isDeclaration() || !llvm::hasSingleElement(func.getBody()) || func.getNumResults() != 0) {
        return func.emitError("Linx requires a defined straight-line void function");
    }
    for (Type type : func.getArgumentTypes()) {
        if (!isa<GMViewType>(type)) {
            return func.emitError("Linx entry arguments must be GM views");
        }
    }
    for (NamedAttribute attr : func->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name != "sym_name" && name != "function_type" && name != "sym_visibility") {
            return func.emitError("unsupported Linx function attribute: ") << name;
        }
    }
    return success();
}

static LogicalResult checkTileOrigins(Operation& op, const llvm::DenseSet<Value>& allocated)
{
    for (Value value : op.getOperands()) {
        if (isa<TileBufType>(value.getType()) && !allocated.contains(value)) {
            return op.emitError("tile must originate from Linx alloc_tile in this function");
        }
    }
    return success();
}

static SmallVector<Value> getReadTiles(Operation& op)
{
    return llvm::TypeSwitch<Operation*, SmallVector<Value>>(&op)
        .Case<TAddOp, TMatmulOp>([](auto binary) {
            return SmallVector<Value>{binary.getLhs(), binary.getRhs()};
        })
        .Case<TRowExpandSubOp, TRowExpandDivOp>([](auto expand) {
            return SmallVector<Value>{expand.getSrc(), expand.getRow()};
        })
        .Case<TStoreOp, TExpOp, TRowMaxOp, TRowSumOp>([](auto unary) {
            return SmallVector<Value>{unary.getSrc()};
        })
        .Default([](Operation*) { return SmallVector<Value>{}; });
}

static Value getWrittenTile(Operation& op)
{
    return llvm::TypeSwitch<Operation*, Value>(&op)
        .Case<TLoadOp, TAddOp, TMatmulOp, TExpOp, TRowMaxOp, TRowSumOp,
              TRowExpandSubOp, TRowExpandDivOp>([](auto writer) { return writer.getDst(); })
        .Default([](Operation*) { return Value{}; });
}

static LogicalResult checkTileAccess(Operation& op, const llvm::DenseSet<Value>& allocated,
                                     llvm::DenseSet<Value>& initialized)
{
    if (!isa<TLoadOp, TAddOp, TMatmulOp, TStoreOp, TExpOp, TRowMaxOp, TRowSumOp,
                 TRowExpandSubOp, TRowExpandDivOp>(op)) {
        return op.emitError("unsupported Linx operation");
    }
    if (failed(checkTileOrigins(op, allocated))) {
        return failure();
    }
    for (Value value : getReadTiles(op)) {
        if (!initialized.contains(value)) {
            return op.emitError("reads an uninitialized Linx tile");
        }
    }
    if (Value written = getWrittenTile(op)) {
        initialized.insert(written);
    }
    return success();
}

static LogicalResult checkFunctionBody(func::FuncOp func)
{
    llvm::DenseSet<Value> allocated;
    llvm::DenseSet<Value> initialized;
    for (Operation& op : func.getBody().front()) {
        if (!op.getAttrs().empty()) {
            return op.emitError("Linx tile operations do not support attributes in this profile");
        }
        if (auto alloc = dyn_cast<AllocTileOp>(op)) {
            allocated.insert(alloc.getTile());
        } else if (!isa<func::ReturnOp>(op)) {
            if (failed(checkTileAccess(op, allocated, initialized))) {
                return failure();
            }
        }
    }
    return success();
}

LogicalResult mlir::pto::linx::verifyModule(ModuleOp module)
{
    auto arch = module->getAttrOfType<StringAttr>("pto.target_arch");
    if (!arch || arch.getValue() != "linx") {
        return module.emitError("expected pto.target_arch = linx");
    }
    if (module->hasAttr("pto.linx_profile") && !module->getAttrOfType<StringAttr>("pto.linx_profile")) {
        return module.emitError("Linx profile must be a string");
    }
    for (NamedAttribute attr : module->getAttrs()) {
        StringRef name = attr.getName().strref();
        if (name != "pto.target_arch" && name != "pto.linx_profile") {
            return module.emitError("unsupported Linx module attribute: ") << name;
        }
    }
    auto profile = module->getAttrOfType<StringAttr>("pto.linx_profile");
    if (profile && profile.getValue() != "cpu-sim") {
        return module.emitError("unsupported Linx profile");
    }
    for (Operation& op : module.getBody()->getOperations()) {
        auto func = dyn_cast<func::FuncOp>(op);
        if (!func) {
            return op.emitError("Linx module supports only function definitions");
        }
        if (failed(checkFunctionSignature(func)) || failed(checkFunctionBody(func))) {
            return failure();
        }
    }
    return success();
}
