// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- VPTOPrintDTDataABI.cpp - print DebugTunnel ABI wiring -------------===//
//===----------------------------------------------------------------------===//
//
// A kernel that prints needs a DebugTunnelData pointer, and the tunnel has to
// be initialized on entry and finished on every return. Two conventions
// deliver that pointer, and which one applies is a property of the runtime, so
// it is selected explicitly rather than guessed from the signature:
//
//  * implicit (real NPU): the CCE launch machinery fills a hidden trailing
//    kernel argument, which object emission renames to `.DTData`.
//  * explicit (simulator): the host runner opens the tunnel itself and passes
//    the pointer as the kernel's first argument, because the simulator launch
//    runtime does not fill the `.DTData` slot.
//
// The hidden trailing argument is appended in both modes: the CCE device
// compile with --cce-enable-print requires that slot to exist.
//
// This runs at module level before the CV split, so both core modules carry
// the wired ABI, and before instance resolution, so the lifecycle calls get
// their instances like any other bridge call.
//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "PTO/IR/PTO.h"
#include "PTO/IR/VPTOBridgeRegistry.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::pto {
#define GEN_PASS_DEF_VPTOPRINTDTDATAABI
#include "PTO/Transforms/Passes.h.inc"

namespace {
bool usesPrintBridge(func::FuncOp func) {
  bool found = false;
  func.walk([&](BridgeCallOp call) {
    const BridgeFunctionDesc *desc = findBridgeFunction(call.getEntry());
    if (desc && desc->family == BridgeFamily::Print) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// The tunnel pointer reaches one function per launch, so a printing function
/// that something else calls would need the pointer threaded through its call
/// sites. Report that instead of wiring a signature its callers do not pass.
bool isKernelEntry(func::FuncOp func, SymbolTableCollection &symbolTables,
                   ModuleOp module) {
  auto uses = symbolTables.getSymbolTable(module).getSymbolUses(func, module);
  return !uses || uses->empty();
}

struct PrintDTDataABI final : impl::VPTOPrintDTDataABIBase<PrintDTDataABI> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrintDTDataABI)

  PrintDTDataABI() = default;
  explicit PrintDTDataABI(const VPTOPrintDTDataABIOptions &options)
      : impl::VPTOPrintDTDataABIBase<PrintDTDataABI>(options) {}

  LogicalResult wireFunction(func::FuncOp func, bool explicitMode) const {
    MLIRContext *context = func.getContext();
    auto tunnelType = LLVM::LLVMPointerType::get(context, 1);
    const unsigned hiddenIndex = func.getNumArguments();
    Value tunnel;
    if (explicitMode) {
      const bool hasExplicitArgument =
          hiddenIndex > 0 &&
          isa<PtrType>(func.getArgumentTypes().front()) &&
          cast<PtrType>(func.getArgumentTypes().front())
                  .getMemorySpace()
                  .getAddressSpace() == AddressSpace::GM;
      if (!hasExplicitArgument) {
        return func.emitError()
               << "explicit print DTData delivery needs a global-memory "
                  "pointer as the kernel's first argument";
      }
    }
    // Appended in both modes: the device compile requires the slot.
    func.insertArgument(hiddenIndex, tunnelType, {}, func.getLoc());
    func.setFunctionType(FunctionType::get(context, func.getArgumentTypes(),
                                           func.getResultTypes()));
    func->setAttr(
        "pto.print_dtdata_index", IntegerAttr::get(IntegerType::get(context, mlir::pto::kValue32), hiddenIndex));
    tunnel = explicitMode ? func.getArgument(0) : func.getArgument(hiddenIndex);

    Region &body = func.getBody();
    if (body.empty()) {
      return func.emitError()
             << "a printing kernel entry must have a body to wire the print "
                "DebugTunnel lifecycle into";
    }
    Location loc = func.getLoc();
    Block &entry = body.front();
    Block *rest = entry.splitBlock(entry.begin());
    OpBuilder builder(&entry, entry.begin());
    builder.create<BridgeCallOp>(loc, TypeRange{}, BridgeEntryId::PrintInit,
                                 /*spec=*/nullptr, /*instance=*/nullptr,
                                 ValueRange{tunnel});
    builder.create<cf::BranchOp>(loc, rest);

    SmallVector<func::ReturnOp> returns;
    body.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
    for (func::ReturnOp ret : returns) {
      OpBuilder returnBuilder(ret);
      returnBuilder.create<BridgeCallOp>(
          loc, TypeRange{}, BridgeEntryId::PrintFinish, /*spec=*/nullptr,
          /*instance=*/nullptr, ValueRange{tunnel});
    }
    return success();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const bool explicitMode = mode == "explicit";
    const bool knownMode = explicitMode || mode == "implicit";
    if (!knownMode) {
      module.emitError() << "unknown print DTData delivery mode '" << mode
                         << "'; expected 'implicit' or 'explicit'";
      signalPassFailure();
      return;
    }
    SymbolTableCollection symbolTables;
    SmallVector<func::FuncOp> printing;
    module.walk([&](func::FuncOp func) {
      if (usesPrintBridge(func)) {
        printing.push_back(func);
      }
    });
    for (func::FuncOp func : printing) {
      if (!isKernelEntry(func, symbolTables, module)) {
        func.emitError()
            << "a printing function that is called from this module cannot "
               "receive the print DebugTunnel pointer; print from the kernel "
               "entry instead";
        signalPassFailure();
        return;
      }
      if (failed(wireFunction(func, explicitMode))) {
        signalPassFailure();
        return;
      }
    }
    if (!printing.empty()) {
      module.walk([&](func::FuncOp func) {
        const bool skipEntry = func.isDeclaration() || func.isPrivate() ||
            func->hasAttr("pto.print_dtdata_index") ||
            !isKernelEntry(func, symbolTables, module);
        if (skipEntry) {
          return;
        }
        const unsigned hiddenIndex = func.getNumArguments();
        auto tunnelType = LLVM::LLVMPointerType::get(func.getContext(), 1);
        func.insertArgument(hiddenIndex, tunnelType, {}, func.getLoc());
        func->setAttr(
            "pto.print_dtdata_index",
            IntegerAttr::get(IntegerType::get(func.getContext(), mlir::pto::kValue32), hiddenIndex));
      });
    }
  }
};
} // namespace

std::unique_ptr<Pass> createVPTOPrintDTDataABIPass(llvm::StringRef mode) {
  VPTOPrintDTDataABIOptions options;
  options.mode = mode.str();
  return std::make_unique<PrintDTDataABI>(options);
}
} // namespace mlir::pto
