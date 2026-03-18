//===- A5VMTextEmitter.cpp - A5VM textual LLVM-like emitter --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMTextEmitter.h"

#include "PTO/IR/A5VM.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;

namespace mlir::pto {
namespace {

static LogicalResult writeReportFile(const A5VMEmissionOptions &options,
                                     ArrayRef<std::string> unresolved) {
  if (options.unresolvedReportPath.empty())
    return success();

  std::error_code ec;
  llvm::raw_fd_ostream reportOS(options.unresolvedReportPath, ec,
                                llvm::sys::fs::OF_Text);
  if (ec)
    return failure();
  for (const std::string &record : unresolved)
    reportOS << record << "\n";
  return success();
}

static constexpr llvm::StringLiteral kSupportedA5VMPrimitives[] = {
    "a5vm.copy_gm_to_ubuf", "a5vm.vlds", "a5vm.vabs",
    "a5vm.vsts",            "a5vm.copy_ubuf_to_gm"};

static bool isSupportedA5VMPrimitive(Operation *op) {
  if (!isa<a5vm::CopyGmToUbufOp, a5vm::VldsOp, a5vm::VabsOp, a5vm::VstsOp,
           a5vm::CopyUbufToGmOp>(op))
    return false;

  return llvm::is_contained(kSupportedA5VMPrimitives,
                            op->getName().getStringRef());
}

static void collectUnsupportedA5VMOps(Operation *op,
                                      llvm::SmallVectorImpl<std::string> &out) {
  if (op != op->getParentOfType<ModuleOp>() &&
      op->getName().getDialectNamespace() == "a5vm" &&
      !isSupportedA5VMPrimitive(op)) {
    out.push_back(("unsupported-op=" + op->getName().getStringRef()).str());
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nested : block)
        collectUnsupportedA5VMOps(&nested, out);
    }
  }
}

} // namespace

LogicalResult translateA5VMModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const A5VMEmissionOptions &options,
                                        llvm::raw_ostream &diagOS) {
  llvm::SmallVector<std::string, 8> unsupportedOps;
  collectUnsupportedA5VMOps(module.getOperation(), unsupportedOps);

  if (failed(writeReportFile(options, unsupportedOps))) {
    diagOS << "A5VM emission failed: could not write unresolved report to '"
           << options.unresolvedReportPath << "'\n";
    return failure();
  }

  if (options.printIntrinsicSelections) {
    diagOS << "A5VM emission note: preserving raw corrected A5VM backend text "
              "at the Phase 1 seam; intrinsic selection is deferred.\n";
  }

  if (!unsupportedOps.empty() && !options.allowUnresolved) {
    for (const std::string &record : unsupportedOps)
      diagOS << "A5VM emission failed: " << record << "\n";
    return failure();
  }

  module->print(os);
  if (!unsupportedOps.empty()) {
    os << "\n";
    for (const std::string &record : unsupportedOps)
      os << "// A5VM-UNSUPPORTED: " << record << "\n";
  }

  return success();
}

} // namespace mlir::pto
