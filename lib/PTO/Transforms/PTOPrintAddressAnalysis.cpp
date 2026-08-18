// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOPrintAddressAnalysis.cpp - Address analysis debug printer ------===//

#include "PTO/Analysis/PTOAddressAnalysis.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOPRINTADDRESSANALYSIS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static void printValue(llvm::raw_ostream &os, Value value) {
  value.printAsOperand(os, OpPrintingFlags());
}

static void printEvolution(llvm::raw_ostream &os,
                           pto::PTOValueEvolutionAnalysis &analysis,
                           Value value, scf::ForOp loop) {
  os << "  value ";
  printValue(os, value);
  os << " type=" << value.getType() << " evolution=";
  auto evolution = analysis.getEvolution(value, loop);
  if (!evolution) {
    os << "unknown("
       << pto::stringifyPTOAnalysisUnknownReason(evolution.reason) << ")\n";
    return;
  }
  os << "initial=";
  pto::printPTOTypedExpr(evolution.value->initial, os);
  os << " step=";
  pto::printPTOTypedExpr(evolution.value->step, os);
  os << " trip-count=" << evolution.value->tripCount;
  if (evolution.value->rangeKnown) {
    const pto::PTOFiniteRange &range = evolution.value->range;
    os << " range="
       << (range.unsignedInterpretation ? "unsigned[" : "signed[");
    if (range.unsignedInterpretation) {
      os << range.lowerInclusive.getZExtValue() << ","
         << range.upperInclusive.getZExtValue();
    } else {
      os << range.lowerInclusive.getSExtValue() << ","
         << range.upperInclusive.getSExtValue();
    }
    os << "]";
  } else {
    os << " range=unknown";
  }
  os
     << " no-wrap=" << (evolution.value->noWrap ? "true" : "false")
     << "\n";
}

static void printConvertedDelta(llvm::raw_ostream &os,
                                pto::PTOAddressAnalysis &analysis,
                                const pto::PTOTypedExprRef &deltaBytes,
                                int64_t unitBytes) {
  os << " unit" << unitBytes << "=";
  auto converted = analysis.convertDeltaToUnit(deltaBytes, unitBytes);
  if (!converted) {
    os << "unknown("
       << pto::stringifyPTOAnalysisUnknownReason(converted.reason) << ")";
    return;
  }
  pto::printPTOTypedExpr(*converted.value, os);
}

struct PTOPrintAddressAnalysisPass
    : public pto::impl::PTOPrintAddressAnalysisBase<
          PTOPrintAddressAnalysisPass> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    auto &valueAnalysis = getAnalysis<pto::PTOValueEvolutionAnalysis>();
    auto &addressAnalysis = getAnalysis<pto::PTOAddressAnalysis>();
    // Function passes may execute concurrently. Buffer one function's report
    // and publish it atomically so FileCheck sees deterministic whole lines.
    llvm::SmallString<1024> storage;
    llvm::raw_svector_ostream os(storage);

    os << "address-analysis @" << function.getSymName() << "\n";
    function.walk([&](scf::ForOp loop) {
      os << " loop\n";
      printEvolution(os, valueAnalysis, loop.getInductionVar(), loop);
      for (BlockArgument iterArg : loop.getRegionIterArgs()) {
        printEvolution(os, valueAnalysis, iterArg, loop);
      }
      loop.getBody()->walk([&](Operation *operation) {
        if (
            operation->getParentOp() != loop.getOperation()) {
          return;
        }
        if (isa<arith::IndexCastOp, arith::IndexCastUIOp, arith::TruncIOp,
                arith::ExtSIOp, arith::ExtUIOp>(operation)) {
          printEvolution(os, valueAnalysis, operation->getResult(0), loop);
        }

        auto addresses = addressAnalysis.getAddresses(operation);
        if (!addresses) {
          return;
        }
        for (const pto::PTOAddressExpr &address : *addresses.value) {
          os << "  op=" << operation->getName() << " root=";
          printValue(os, address.rootOrBase);
          os << " element-offset=";
          pto::printPTOTypedExpr(address.elementOffset, os);
          os << " current-offset=";
          if (!address.offset) {
            os << "none";
          } else {
            os << pto::stringifyVPTOAddressUnit(address.offset->unit) << ":";
            printValue(os, address.offset->sourceValue);
          }
          os << " delta-bytes=";
          auto delta = addressAnalysis.getDeltaBytes(address, loop);
          if (!delta) {
            os << "unknown("
               << pto::stringifyPTOAnalysisUnknownReason(delta.reason)
               << ")\n";
            continue;
          }
          pto::printPTOTypedExpr(*delta.value, os);
          for (int64_t unitBytes : {int64_t{1}, int64_t{2}, int64_t{4},
                                    int64_t{32}}) {
            printConvertedDelta(os, addressAnalysis, *delta.value, unitBytes);
          }
          os << "\n";
        }
      });
    });
    llvm::errs() << storage;
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOPrintAddressAnalysisPass() {
  return std::make_unique<PTOPrintAddressAnalysisPass>();
}
