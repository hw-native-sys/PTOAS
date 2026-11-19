// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Lower resolved bridge IR to func.call at the start of VPTO LLVM lowering.
// Declarations are derived from the registry descriptor, never from call-site
// operand types, and object handles become their storage pointers directly.
#include "PTO/Support/CodeConstants.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/IR/VPTOBridgeRegistry.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::pto {
#define GEN_PASS_DEF_VPTOBRIDGECALLLOWERING
#include "PTO/Transforms/Passes.h.inc"

namespace {
Type getBridgeABIType(MLIRContext *context, BridgeValueType value) {
  switch (value.kind) {
  case BridgeValueKind::F32:
    return Float32Type::get(context);
  case BridgeValueKind::I32:
      return IntegerType::get(context, mlir::pto::kValue32);
  case BridgeValueKind::I64:
  case BridgeValueKind::U64:
      return IntegerType::get(context, mlir::pto::kValue64);
  case BridgeValueKind::Pointer:
    return LLVM::LLVMPointerType::get(
        context, value.space == BridgePointerSpace::Global ? 1 : 0);
  case BridgeValueKind::ObjectHandle:
    return LLVM::LLVMPointerType::get(context);
  }
  return Type();
}

FunctionType getBridgeFunctionType(MLIRContext *context,
                                   const BridgeFunctionDesc &desc) {
  SmallVector<Type> arguments;
  SmallVector<Type> results;
  for (BridgeValueType value : desc.arguments) {
    arguments.push_back(getBridgeABIType(context, value));
  }
  for (BridgeValueType value : desc.results) {
    results.push_back(getBridgeABIType(context, value));
  }
  return FunctionType::get(context, arguments, results);
}

// Declare the callee in the module that owns the call; after CV split this is
// the per-core kernel module rather than the container.
FailureOr<func::FuncOp> declareBridgeFunction(Operation *anchor,
                                              StringRef symbol,
                                              FunctionType type) {
  auto module = anchor->getParentOfType<ModuleOp>();
  auto function = module.lookupSymbol<func::FuncOp>(symbol);
  if (function) {
    const bool incompatible = function.getFunctionType() != type;
    if (incompatible) {
      anchor->emitError() << "resolved bridge symbol '" << symbol
                          << "' is already declared with type "
                          << function.getFunctionType() << ", expected "
                          << type;
      return failure();
    }
    return function;
  }
  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  function = builder.create<func::FuncOp>(module.getLoc(), symbol, type);
  function.setPrivate();
  return function;
}

// Instance symbols name the wrapper functions, so the table is consumed here:
// references are resolved first, then the instance ops are erased to make
// room for the func.func declarations of the same names.
StringAttr resolveInstanceSymbol(Operation *op, BridgeEntryId entry) {
  auto reference = op->getAttrOfType<FlatSymbolRefAttr>("instance");
  if (!reference) {
    return StringAttr();
  }
  auto instance =
      SymbolTable::lookupNearestSymbolFrom<BridgeInstanceOp>(op, reference);
  const bool unusable = !instance || instance.getEntry() != entry;
  if (unusable) {
    return StringAttr();
  }
  return instance.getSymNameAttr();
}

struct LowerBridgeCalls final
    : impl::VPTOBridgeCallLoweringBase<LowerBridgeCalls> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerBridgeCalls)

  // Storage is a fixed-size alloca hoisted to the function entry block, so a
  // create inside a loop does not grow the stack; the void init stays in place.
  LogicalResult lowerCreate(BridgeObjectCreateOp create, StringAttr initSymbol,
                            func::FuncOp function,
                            DenseMap<Value, Value> &storageByHandle) const {
    const BridgeFunctionDesc *desc = findBridgeFunction(create.getEntry());
    const bool validContract = desc && desc->createsObject &&
                               desc->storageSize != 0 && desc->storageAlign != 0;
    if (!validContract) {
      return create.emitError("bridge object has an invalid registry contract");
    }
    MLIRContext *context = create.getContext();
    auto initFunc = declareBridgeFunction(
        create, initSymbol.getValue(), getBridgeFunctionType(context, *desc));
    if (failed(initFunc)) {
      return failure();
    }
    OpBuilder entryBuilder = OpBuilder::atBlockBegin(&function.front());
    auto size = entryBuilder.create<LLVM::ConstantOp>(
        create.getLoc(), entryBuilder.getI64Type(),
        entryBuilder.getI64IntegerAttr(static_cast<int64_t>(desc->storageSize)));
    auto storage = entryBuilder.create<LLVM::AllocaOp>(
        create.getLoc(), LLVM::LLVMPointerType::get(context),
        entryBuilder.getI8Type(), size,
        static_cast<unsigned>(desc->storageAlign));
    OpBuilder builder(create);
    SmallVector<Value> initArgs{storage.getResult()};
    llvm::append_range(initArgs, create.getArgs());
    builder.create<func::CallOp>(create.getLoc(), *initFunc, initArgs);
    storageByHandle[create.getResult()] = storage.getResult();
    return success();
  }

  LogicalResult lowerCall(BridgeCallOp call, StringAttr symbol,
                          const DenseMap<Value, Value> &storageByHandle) const {
    const BridgeFunctionDesc *desc = findBridgeFunction(call.getEntry());
    if (!desc) {
      return call.emitError("bridge call has no registered entry");
    }
    FunctionType type = getBridgeFunctionType(call.getContext(), *desc);
    OpBuilder builder(call);
    SmallVector<Value> arguments;
    for (auto [index, arg] : llvm::enumerate(call.getArgs())) {
      const BridgeValueKind kind = desc->arguments[index].kind;
      if (kind == BridgeValueKind::ObjectHandle) {
        auto storage = storageByHandle.find(arg);
        if (storage == storageByHandle.end()) {
          return call.emitError() << "bridge argument #" << index
                                  << " is not an object created in this function";
        }
        arguments.push_back(storage->second);
        continue;
      }
      // PTO GM pointers are converted with the other VPTO types later in the
      // LLVM pipeline; this cast is reconciled there.
      Type abiType = type.getInput(index);
      const bool needsCast =
          kind == BridgeValueKind::Pointer && arg.getType() != abiType;
      if (needsCast) {
        arg = builder
                  .create<UnrealizedConversionCastOp>(call.getLoc(), abiType, arg)
                  .getResult(0);
      }
      arguments.push_back(arg);
    }
    auto function = declareBridgeFunction(call, symbol.getValue(), type);
    if (failed(function)) {
      return failure();
    }
    auto replacement =
        builder.create<func::CallOp>(call.getLoc(), *function, arguments);
    call.replaceAllUsesWith(replacement.getResults());
    call.erase();
    return success();
  }

  using SymbolsByOp = DenseMap<Operation *, StringAttr>;

  /// Resolves every reference before the instance table is erased: the
  /// instance symbols name the wrapper functions this pass declares.
  LogicalResult collectInstanceSymbols(SymbolsByOp &symbols) {
    bool hadError = false;
    getOperation().walk([&symbols, &hadError](Operation *op) {
      if (!isa<BridgeCallOp, BridgeObjectCreateOp>(op)) {
        return;
      }
      auto entry = op->getAttrOfType<BridgeEntryAttr>("entry");
      StringAttr symbol = resolveInstanceSymbol(op, entry.getValue());
      if (!symbol) {
        op->emitError("bridge op has no resolved instance");
        hadError = true;
        return;
      }
      symbols[op] = symbol;
    });
    return success(!hadError);
  }

  void eraseInstanceTable() {
    SmallVector<BridgeInstanceOp> instances;
    getOperation().walk(
        [&instances](BridgeInstanceOp op) { instances.push_back(op); });
    for (BridgeInstanceOp instance : instances) {
      instance.erase();
    }
  }

  /// Lowers one function body. Handles never escape their function, so the
  /// storage of each object is tracked here rather than module-wide.
  LogicalResult lowerFunction(func::FuncOp function,
                              const SymbolsByOp &symbols) const
  {
    DenseMap<Value, Value> storageByHandle;
    SmallVector<BridgeObjectCreateOp> creates;
    SmallVector<BridgeCallOp> calls;
    function.walk([&creates, &calls](Operation *op) {
      if (auto create = dyn_cast<BridgeObjectCreateOp>(op)) {
        creates.push_back(create);
      } else if (auto call = dyn_cast<BridgeCallOp>(op)) {
        calls.push_back(call);
      }
    });
    bool hadError = false;
    for (BridgeObjectCreateOp create : creates) {
      if (failed(lowerCreate(create, symbols.lookup(create), function,
                             storageByHandle))) {
        hadError = true;
      }
    }
    for (BridgeCallOp call : calls) {
      if (failed(lowerCall(call, symbols.lookup(call), storageByHandle))) {
        hadError = true;
      }
    }
    if (hadError) {
      return failure();
    }
    for (BridgeObjectCreateOp create : creates) {
      if (!create.getResult().use_empty()) {
        create.emitError(
            "bridge object handle is used outside Pipe lifecycle calls");
        return failure();
      }
      create.erase();
    }
    return success();
  }

  void runOnOperation() override {
    SymbolsByOp symbols;
    if (failed(collectInstanceSymbols(symbols))) {
      signalPassFailure();
      return;
    }
    eraseInstanceTable();
    SmallVector<func::FuncOp> functions;
    getOperation().walk(
        [&functions](func::FuncOp function) { functions.push_back(function); });
    for (func::FuncOp function : functions) {
      if (failed(lowerFunction(function, symbols))) {
        signalPassFailure();
        return;
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> createVPTOBridgeCallLoweringPass() {
  return std::make_unique<LowerBridgeCalls>();
}
} // namespace mlir::pto
