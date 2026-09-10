// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_IR_VPTOINTERFACES
#define MLIR_DIALECT_PTO_IR_VPTOINTERFACES

include "mlir/IR/OpBase.td"

def VPTOAddressSemanticsOpInterface
    : OpInterface<"VPTOAddressSemanticsOpInterface"> {
  let description = [{
    Exposes the complete VPTO addressing contract. Current-access base,
    optional offset, and unit are structurally separate from the optional
    post-access advance operand, unit, target constraint, and updated-base
    result. Consumers must not infer either contract from operand or result
    positions.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<
      "Return normalized current-access and post-update address semantics.",
      "::mlir::pto::VPTOAddressSemantics",
      "getVPTOAddressSemantics", (ins), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::pto::getDefaultVPTOAddressSemantics(
            $_op.getOperation());
      }]>
  ];
}

def VPTOSchedulingOpInterface : OpInterface<"VPTOSchedulingOpInterface"> {
  let description = [{
    Exposes operation-local semantics needed by the emission-stage VPTO
    scheduler when ordinary SSA def-use and MemoryEffectOpInterface are not
    sufficient. The operation reports only its own effects; region formation,
    alias analysis, and dependency construction are scheduler responsibilities.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<
      "Return the complete normalized semantics for VPTO scheduling.",
      "::mlir::pto::VPTOSchedulingSemantics", "getVPTOSchedulingSemantics",
      (ins), [{}],
      /*defaultImplementation=*/[{
        return ::mlir::pto::getDefaultVPTOSchedulingSemantics(
            $_op.getOperation());
      }]>
  ];
}

def PTO_StateAccessOpInterface : OpInterface<"StateAccessOpInterface"> {
  let description = [{
    Describes one access point to an implicit PTO hardware state resource.
    The interface separates the accessed resource from the access kind, so a
    state-analysis pass never has to match op names or the SSA shape of
    get/set helper sequences. `Query` reads the full state and produces an
    SSA result, `Write` installs the operand value, `Consume` implicitly
    depends on the current state without producing one, and `Clobber` may
    change the state arbitrarily. This interface only describes ordered
    semantics; whether a specific read can be forwarded is decided by the
    state-analysis pass, so implementors must not add the `Pure` trait on
    the strength of this interface alone.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the accessed state resource.",
      "::mlir::pto::StateResource", "getStateResource">,
    InterfaceMethod<"Return the access kind.",
      "::mlir::pto::StateAccessKind", "getStateAccessKind">,
    InterfaceMethod<
      "Return the SSA value read from or written to the state; empty for Consume and Clobber.",
      "::mlir::Value", "getAccessedStateValue">
  ];
}

def PTO_StateGuardOpInterface : OpInterface<"StateGuardOpInterface"> {
  let description = [{
    Describes a temporary state scope: the guard overrides a statically known
    subset of one implicit state resource while its body executes, and the
    logical state is restored after the guard exits. Bits outside
    `controlledStateBits` keep their entry values; controlled bits take the
    value implied by `requiredStateBits`. The guard stores no saved value and
    does not decide where configuration or restore hardware accesses are
    emitted; those decisions belong to the state-optimization pass. The
    `uint64_t` bit sets only model compile-time-known partial coverage of at
    most 64 bits.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the guarded state resource.",
      "::mlir::pto::StateResource", "getStateResource">,
    InterfaceMethod<"Return the state bits owned by this guard.",
      "uint64_t", "getControlledStateBits">,
    InterfaceMethod<
      "Return the controlled bits whose required value is one.",
      "uint64_t", "getRequiredStateBits">,
    InterfaceMethod<"Return the guarded body.",
      "::mlir::Region &", "getGuardedBody">
  ];
}

def PTO_MadSemanticOpInterface : OpInterface<"MadSemanticOpInterface"> {
  let description = [{
    Interface for semantic MAD-family ops. The interface is only a uniform view
    over the existing IR operands and attributes; it does not duplicate operand
    state into a separate lowering descriptor.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the left matrix operand.",
      "::mlir::Value", "getLhs", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getLhs(); }]>,
    InterfaceMethod<"Return the right matrix operand.",
      "::mlir::Value", "getRhs", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getRhs(); }]>,
    InterfaceMethod<"Return the accumulator/destination operand.",
      "::mlir::Value", "getDst", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getDst(); }]>,
    InterfaceMethod<"Return M.",
      "::mlir::Value", "getM", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getM(); }]>,
    InterfaceMethod<"Return N.",
      "::mlir::Value", "getN", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getN(); }]>,
    InterfaceMethod<"Return K.",
      "::mlir::Value", "getK", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getK(); }]>,
    InterfaceMethod<"Return true for MX MAD forms.",
      "bool", "isMadMxFamily">,
    InterfaceMethod<"Return true if the op carries a bias operand.",
      "bool", "hasBiasOperand">,
    InterfaceMethod<"Return true if the op reads the existing accumulator.",
      "bool", "readsAccumulator">,
    InterfaceMethod<"Return true if TF32 mode is a legal semantic attribute.",
      "bool", "supportsTf32Mode">,
    InterfaceMethod<"Return the bias operand, or null for non-bias forms.",
      "::mlir::Value", "getBiasOrNull">,
    InterfaceMethod<"Return true if the accumulator source is bias.",
      "bool", "initializesAccumulatorWithBias", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.hasBiasOperand(); }]>,
    InterfaceMethod<"Return true if the accumulator is initialized to zero.",
      "bool", "initializesAccumulatorWithZero", (ins), [{}],
      /*defaultImplementation=*/[{
        return !$_op.readsAccumulator() && !$_op.hasBiasOperand();
      }]>,
    InterfaceMethod<"Return unit_flag_mode.",
      "::mlir::Attribute", "getUnitFlagModeAttr", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getUnitFlagModeAttr(); }]>,
    InterfaceMethod<"Return true if disable_gemv is set.",
      "bool", "getDisableGemv", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getDisableGemv(); }]>,
    InterfaceMethod<"Return sat_mode.",
      "::mlir::Attribute", "getSatModeAttr", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getSatModeAttr(); }]>,
    InterfaceMethod<"Return tf32_mode.",
      "::mlir::Attribute", "getTf32ModeAttr">,
    InterfaceMethod<"Return true if n_dir is set.",
      "bool", "getNDir", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getNDir(); }]>
  ];
}

def PTO_MadRawOpInterface : OpInterface<"MadRawOpInterface"> {
  let description = [{
    Interface for raw MAD-family ops. Raw ops carry already-materialized X_t
    and optionally a bias operand; intrinsic selection is derived from this
    interface and the pointer element types.
  }];
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"Return the left matrix operand.",
      "::mlir::Value", "getLhs", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getLhs(); }]>,
    InterfaceMethod<"Return the right matrix operand.",
      "::mlir::Value", "getRhs", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getRhs(); }]>,
    InterfaceMethod<"Return the accumulator/destination operand.",
      "::mlir::Value", "getDst", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getDst(); }]>,
    InterfaceMethod<"Return X_t.",
      "::mlir::Value", "getXt", (ins), [{}],
      /*defaultImplementation=*/[{ return $_op.getXt(); }]>,
    InterfaceMethod<"Return true for MX MAD forms.",
      "bool", "isMadMxFamily">,
    InterfaceMethod<"Return true if the op carries a bias operand.",
      "bool", "hasBiasOperand">,
    InterfaceMethod<"Return the bias operand, or null for non-bias forms.",
      "::mlir::Value", "getBiasOrNull">
  ];
}

#endif // MLIR_DIALECT_PTO_IR_VPTOINTERFACES
