// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Validate the supported straight-line subset, then construct target-independent EmitC.
#include "PTO/Conversion/Linx/LinxToEmitC.h"
#include "PTO/IR/Linx/LinxDialect.h"
#include "PTO/Analysis/Linx/LinxTileRoleAnalysis.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::pto::linx;

static std::string tileName(TileBufType type, TileRole role)
{
    StringRef roleName = role == TileRole::Left ? "Left" : role == TileRole::Right ? "Right" : "Vec";
    StringRef layoutName;
    if (type.getLayout() == TileLayout::RowMajor) {
        layoutName = "RowMajor";
    } else if (type.getLayout() == TileLayout::M16) {
        layoutName = "CubeM16";
    } else if (type.getLayout() == TileLayout::M32) {
        layoutName = "CubeM32";
    } else if (type.getLayout() == TileLayout::N8) {
        layoutName = "CubeN8";
    } else {
        llvm_unreachable("verified Linx layout");
    }
    return "pto::Tile<pto::TileType::" + roleName.str() + ", float, " + std::to_string(type.getRows()) + ", " +
           std::to_string(type.getCols()) + ", pto::BLayout::" + layoutName.str() + ", " +
           std::to_string(type.getRows()) + ", " + std::to_string(type.getCols()) +
           ", pto::SLayout::NoneBox, 512, pto::PadValue::Null, pto::CompactMode::Null>";
}

static Value call(OpBuilder& b, Location loc, StringRef name, TypeRange results, ValueRange args)
{
    auto op = b.create<emitc::CallOpaqueOp>(loc, results, name, ArrayAttr{}, ArrayAttr{}, args);
    return results.empty() ? Value{} : op.getResult(0);
}

static Value buildGM(OpBuilder& b, Location loc, GMViewType type, Value ptr, Value stride)
{
    std::string shape =
        "pto::Shape<1, 1, 1, " + std::to_string(type.getRows()) + ", " + std::to_string(type.getCols()) + ">";
    std::string strides = "pto::Stride<-1, -1, -1, -1, -1>";
    auto opaque = [&](StringRef text) { return emitc::OpaqueType::get(b.getContext(), text); };
    Value shapeValue = call(b, loc, shape, {opaque(shape)}, {});
    auto i32 = b.getI32Type();
    Value one = b.create<emitc::ConstantOp>(loc, i32, b.getI32IntegerAttr(1));
    // Singleton outer dimensions are never indexed; their strides can be one.
    Value strideValue = call(b, loc, strides, {opaque(strides)}, {one, one, one, stride, one});
    std::string gm = "pto::GlobalTensor<float, " + shape + ", " + strides + ", pto::Layout::ND>";
    return call(b, loc, gm, {opaque(gm)}, {ptr, shapeValue, strideValue});
}

static StringRef apiName(Operation& op)
{
    return llvm::TypeSwitch<Operation*, StringRef>(&op)
        .Case<TLoadOp>([](auto) { return "::TLOAD"; })
        .Case<TStoreOp>([](auto) { return "::TSTORE"; })
        .Case<TAddOp>([](auto) { return "::TADD"; })
        .Case<TMatmulOp>([](auto) { return "::TMATMUL"; })
        .Case<TExpOp>([](auto) { return "::TEXP"; })
        .Case<TRowMaxOp>([](auto) { return "::TROWMAX"; })
        .Case<TRowSumOp>([](auto) { return "::TROWSUM"; })
        .Case<TRowExpandSubOp>([](auto) { return "::TROWEXPANDSUB"; })
        .Case<TRowExpandDivOp>([](auto) { return "::TROWEXPANDDIV"; })
        .Default([](Operation*) -> StringRef { llvm_unreachable("verified Linx operation"); });
}

static void lowerOperation(Operation& op, OpBuilder& b, llvm::DenseMap<Value, Value>& values,
                           const LinxTileRoleAnalysis& roles)
{
    if (auto alloc = dyn_cast<AllocTileOp>(op)) {
        auto tileType = cast<TileBufType>(alloc.getTile().getType());
        auto type = emitc::OpaqueType::get(b.getContext(), tileName(tileType, roles.getRole(alloc.getTile())));
        values[alloc.getTile()] = b.create<emitc::VariableOp>(op.getLoc(), type,
                                                          emitc::OpaqueAttr::get(b.getContext(), ""));
        return;
    }
    if (isa<func::ReturnOp>(op)) {
        b.create<emitc::ReturnOp>(op.getLoc(), Value{});
        return;
    }
    // Linx IR is destination-last; the public C++ API is destination-first.
    SmallVector<Value> args{values.lookup(op.getOperands().back())};
    for (Value input : op.getOperands().drop_back()) {
        args.push_back(values.lookup(input));
    }
    call(b, op.getLoc(), apiName(op), {}, args);
}

static void lowerFunction(func::FuncOp source, OpBuilder& b, const LinxTileRoleAnalysis& roles)
{
    Location loc = source.getLoc();
    SmallVector<Type> arguments;
    for (unsigned i = 0; i < source.getNumArguments(); ++i) {
        arguments.push_back(emitc::PointerType::get(b.getF32Type()));
        arguments.push_back(b.getI32Type());
    }
    auto func = b.create<emitc::FuncOp>(loc, source.getName(), b.getFunctionType(arguments, {}));
    func.setSpecifiersAttr(b.getStrArrayAttr({"extern \"C\""}));
    Block* body = func.addEntryBlock();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToStart(body);
    llvm::DenseMap<Value, Value> values;
    for (auto item : llvm::enumerate(source.getArguments())) {
        values[item.value()] = buildGM(
            b, loc, cast<GMViewType>(item.value().getType()), body->getArgument(2 * item.index()),
            body->getArgument(2 * item.index() + 1));
    }
    for (Operation& op : source.getBody().front()) {
        lowerOperation(op, b, values, roles);
    }
}

LogicalResult mlir::pto::linx::emitCpp(ModuleOp module, llvm::raw_ostream& output)
{
    if (failed(verifyModule(module))) {
        return failure();
    }
    WalkResult supported = module.walk([](AllocTileOp alloc) {
        auto type = cast<TileBufType>(alloc.getTile().getType());
        if (type.getLocation() != TileLocation::Local) {
            alloc.emitOpError("shared tile C++ lowering is not implemented");
            return WalkResult::interrupt();
        }
        return WalkResult::advance();
    });
    if (supported.wasInterrupted()) {
        return failure();
    }
    LinxTileRoleAnalysis roles;
    if (failed(roles.run(module))) {
        return failure();
    }
    module.getContext()->getOrLoadDialect<emitc::EmitCDialect>();
    OwningOpRef<ModuleOp> converted = ModuleOp::create(module.getLoc());
    OpBuilder b(module.getContext());
    b.setInsertionPointToStart(converted->getBody());
    b.create<emitc::IncludeOp>(module.getLoc(), "pto/pto-inst.hpp", false);
    for (auto func : module.getOps<func::FuncOp>()) {
        lowerFunction(func, b, roles);
    }
    if (failed(verify(*converted))) {
        return failure();
    }
    return emitc::translateToCpp(*converted, output);
}
