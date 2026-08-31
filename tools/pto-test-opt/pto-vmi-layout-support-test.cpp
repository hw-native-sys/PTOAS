// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- pto-vmi-layout-support-test.cpp -----------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VMILayoutSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool check(bool condition, const Twine &message) {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
  }
  return condition;
}

static Operation *createOperation(Block &block, Location loc, StringRef name,
                                  ArrayRef<Type> operandTypes,
                                  ArrayRef<Type> resultTypes) {
  SmallVector<Value> operands;
  for (Type type : operandTypes) {
    operands.push_back(block.addArgument(type, loc));
  }
  OperationState state(loc, name);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  return Operation::create(state);
}

static bool testSameAndFreeRelations(MLIRContext &context) {
  Location loc = UnknownLoc::get(&context);
  Type type =
      VMIVRegType::get(&context, 256, Float32Type::get(&context), Attribute{});
  VMILayoutAttr c = VMILayoutAttr::getContiguous(&context);
  VMILayoutAttr d2 = VMILayoutAttr::getDeinterleaved(&context, 2);
  VMILayoutAttr lhsLayouts[] = {c, d2};
  VMILayoutAttr d2Only[] = {d2};
  VMILayoutRelationPortDomain domains[] = {
      {VMILayoutRelationPortKind::Operand, 0, lhsLayouts},
      {VMILayoutRelationPortKind::Operand, 1, d2Only},
      {VMILayoutRelationPortKind::Result, 0, lhsLayouts}};

  Block block;
  Operation *add = createOperation(block, loc, VMIAddFOp::getOperationName(),
                                   {type, type}, {type});
  SmallVector<VMILayoutRelationFact> facts;
  VMILayoutSupport support;
  FailureOr<size_t> count = support.visitLayoutRelationFacts(
      add, domains,
      [&](const VMILayoutRelationFact &fact) { facts.push_back(fact); });
  bool ok = check(succeeded(count) && *count == 1 && facts.size() == 1,
                  "same-layout relation must intersect all finite domains");
  ok &= check(facts.front().operandLayouts[0] == d2 &&
                  facts.front().operandLayouts[1] == d2 &&
                  facts.front().resultLayouts[0] == d2,
              "same-layout relation must bind every layout port");
  std::string reason;
  count = support.visitLayoutRelationFacts(
      add, {}, [](const VMILayoutRelationFact &) {}, &reason);
  ok &= check(failed(count) && reason.find("finite") != std::string::npos,
              "same-layout relation must reject an unbounded query");
  add->destroy();

  Operation *constant = createOperation(
      block, loc, VMIConstantOp::getOperationName(), {}, {type});
  VMILayoutRelationPortDomain resultDomain{VMILayoutRelationPortKind::Result, 0,
                                           lhsLayouts};
  count = support.visitLayoutRelationFacts(
      constant, resultDomain, [](const VMILayoutRelationFact &) {});
  ok &= check(succeeded(count) && *count == 2,
              "free-result relation must enumerate its finite result domain");
  constant->destroy();
  return ok;
}

static bool testRawPortsAndInvalidDomains(MLIRContext &context) {
  Location loc = UnknownLoc::get(&context);
  Type valueType =
      VMIVRegType::get(&context, 128, Float32Type::get(&context), Attribute{});
  Type maskType = VMIMaskType::get(&context, 128, "b32", Attribute{});
  Type indexType = IndexType::get(&context);
  VMILayoutAttr c = VMILayoutAttr::getContiguous(&context);
  VMILayoutAttr cOnly[] = {c};
  VMILayoutRelationPortDomain domains[] = {
      {VMILayoutRelationPortKind::Operand, 2, cOnly},
      {VMILayoutRelationPortKind::Operand, 3, cOnly},
      {VMILayoutRelationPortKind::Result, 0, cOnly}};

  Block block;
  Operation *load =
      createOperation(block, loc, VMIMaskedLoadOp::getOperationName(),
                      {indexType, indexType, maskType, valueType}, {valueType});
  VMILayoutRelationFact fact;
  VMILayoutSupport support;
  FailureOr<size_t> count = support.visitLayoutRelationFacts(
      load, domains,
      [&](const VMILayoutRelationFact &candidate) { fact = candidate; });
  bool ok = check(succeeded(count) && *count == 1,
                  "masked-load relation must enumerate a legal row");
  ok &= check(fact.operandLayouts.size() == 4 && !fact.operandLayouts[0] &&
                  !fact.operandLayouts[1] && fact.operandLayouts[2] == c &&
                  fact.operandLayouts[3] == c,
              "relation facts must retain raw non-layout operand slots");

  VMILayoutRelationPortDomain badDomain{VMILayoutRelationPortKind::Operand, 0,
                                        cOnly};
  std::string reason;
  count = support.visitLayoutRelationFacts(
      load, badDomain, [](const VMILayoutRelationFact &) {}, &reason);
  ok &=
      check(failed(count) &&
                reason == "layout relation domain references a non-layout port",
            "non-layout domain must fail explicitly");
  load->destroy();
  return ok;
}

static bool testUnsupportedStructure(MLIRContext &context) {
  Location loc = UnknownLoc::get(&context);
  Type type =
      VMIVRegType::get(&context, 128, Float32Type::get(&context), Attribute{});
  VMILayoutAttr c = VMILayoutAttr::getContiguous(&context);
  VMILayoutAttr cOnly[] = {c};
  VMILayoutRelationPortDomain domain{VMILayoutRelationPortKind::Operand, 0,
                                     cOnly};
  Block block;
  Operation *ret = createOperation(
      block, loc, func::ReturnOp::getOperationName(), {type}, {});
  std::string reason;
  FailureOr<size_t> count = VMILayoutSupport().visitLayoutRelationFacts(
      ret, domain, [](const VMILayoutRelationFact &) {}, &reason);
  bool ok =
      check(failed(count) && reason.find("no finite") != std::string::npos,
            "structural transport must not become a VMI relation");
  ret->destroy();
  return ok;
}

static bool testRegisteredRuleRelations(MLIRContext &context) {
  Location loc = UnknownLoc::get(&context);
  Type f32 =
      VMIVRegType::get(&context, 64, Float32Type::get(&context), Attribute{});
  Type i32 = VMIVRegType::get(&context, 64, IntegerType::get(&context, 32),
                              Attribute{});
  Type mask = VMIMaskType::get(&context, 64, "b32", Attribute{});
  VMILayoutAttr d2 = VMILayoutAttr::getDeinterleaved(&context, 2);
  VMILayoutAttr d2Only[] = {d2};
  VMILayoutRelationPortDomain unaryDomains[] = {
      {VMILayoutRelationPortKind::Operand, 0, d2Only},
      {VMILayoutRelationPortKind::Result, 0, d2Only}};

  Block block;
  Operation *cast = createOperation(block, loc, VMIFPToSIOp::getOperationName(),
                                    {f32}, {i32});
  VMILayoutSupport support;
  FailureOr<size_t> count = support.visitLayoutRelationFacts(
      cast, unaryDomains, [](const VMILayoutRelationFact &) {});
  bool ok = check(succeeded(count) && *count == 1,
                  "same-width numeric cast must use same-layout support");
  cast->destroy();

  VMILayoutRelationPortDomain vexpdifDomains[] = {
      {VMILayoutRelationPortKind::Operand, 0, d2Only},
      {VMILayoutRelationPortKind::Operand, 1, d2Only},
      {VMILayoutRelationPortKind::Operand, 2, d2Only},
      {VMILayoutRelationPortKind::Result, 0, d2Only}};
  Operation *vexpdif = createOperation(
      block, loc, VMIVexpdifOp::getOperationName(), {f32, f32, mask}, {f32});
  count = support.visitLayoutRelationFacts(
      vexpdif, vexpdifDomains, [](const VMILayoutRelationFact &) {});
  ok &= check(succeeded(count) && *count == 1,
              "f32 vexpdif must retain its same-layout relation");
  vexpdif->destroy();
  return ok;
}

static bool testConversionFacts(MLIRContext &context) {
  Type type =
      VMIVRegType::get(&context, 256, Float32Type::get(&context), Attribute{});
  VMILayoutAttr c = VMILayoutAttr::getContiguous(&context);
  VMILayoutAttr d4 = VMILayoutAttr::getDeinterleaved(&context, 4);
  VMILayoutAttr ls2 = VMILayoutAttr::getContiguous(&context, 2);
  VMILayoutAttr cOnly[] = {c};
  VMILayoutAttr d4Only[] = {d4};
  VMILayoutAttr ls2Only[] = {ls2};
  VMILayoutSupport support;
  auto count = support.visitLayoutConversionFacts(
      type, cOnly, d4Only, [](const VMILayoutConversionFact &) {});
  bool ok = check(succeeded(count) && *count == 1,
                  "registered c-to-d4 conversion must enumerate");
  count = support.visitLayoutConversionFacts(
      type, d4Only, ls2Only, [](const VMILayoutConversionFact &) {});
  ok &= check(succeeded(count) && *count == 0,
              "unregistered conversion must not default to support");
  count = support.visitLayoutConversionFacts(
      type, cOnly, cOnly, [](const VMILayoutConversionFact &) {});
  ok &= check(succeeded(count) && *count == 1,
              "identity conversion relation must enumerate once");
  std::string reason;
  count = support.visitLayoutConversionFacts(
      type, {}, {}, [](const VMILayoutConversionFact &) {}, &reason);
  ok &= check(failed(count) &&
                  reason.find("finite endpoint") != std::string::npos,
              "polymorphic identity conversion needs a finite anchor");
  return ok;
}

} // namespace

int main() {
  DialectRegistry registry;
  registry.insert<PTODialect, func::FuncDialect>();
  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool ok = testSameAndFreeRelations(context);
  ok &= testRawPortsAndInvalidDomains(context);
  ok &= testUnsupportedStructure(context);
  ok &= testRegisteredRuleRelations(context);
  ok &= testConversionFacts(context);
  if (!ok) {
    return 1;
  }
  llvm::outs() << "VMI layout support enumeration tests passed\n";
  return 0;
}
