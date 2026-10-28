// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#pragma once
//===- VMIToVPTOPatternInternals9.inc - VMIToVPTO internals -*- C++ -*-===//
//===----------------------------------------------------------------------===//

/// Parses the pass option load-safety.
static FailureOr<VMILoadSafetyPolicy> parseLoadSafetyPolicy(StringRef value) {
  if (value == "policy") {
    return VMILoadSafetyPolicy::Policy;
  }
  if (value == "warn") {
    return VMILoadSafetyPolicy::Warn;
  }
  if (value == "error") {
    return VMILoadSafetyPolicy::Error;
  }
  return failure();
}

struct VMIToVPTOPass : public mlir::pto::impl::VMIToVPTOBase<VMIToVPTOPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(VMIToVPTOPass)

  explicit VMIToVPTOPass(const VMIToVPTOOptions &options)
      : mlir::pto::impl::VMIToVPTOBase<VMIToVPTOPass>(options) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // A pipeline that still passes the removed pass-local option must fail with
    // an explanation instead of silently running with the module state.  The
    // occurrence count is the only reliable "was parsed" signal: the options
    // struct's default value is copied into the pass through operator=, which
    // marks hasValue() but is not a user-given value.
    if (deprecatedVmiFastmath.getNumOccurrences() > 0) {
      module.emitError()
          << "-vmi-to-vpto=vmi-fastmath is no longer supported: the VMI "
             "conversion state is pipeline-wide. Use --vmi-fastmath on the "
             "driver, or record it with "
             "-vmi-set-fastmath=vmi-fastmath=<bool> before the VMI passes.";
      signalPassFailure();
      return;
    }
    // The VMI conversion state is the module attribute the driver recorded, not
    // a pass option, so this pass and vmi-layout-assignment read one value.
    bool enableVmiFastmath = getVMIFastmathState(module);
    FailureOr<VMILoadSafetyPolicy> loadSafetyPolicy =
        parseLoadSafetyPolicy(loadSafety);
    if (failed(loadSafetyPolicy)) {
      module.emitError() << "invalid load-safety value \"" << loadSafety
                         << "\"; expected \"policy\", \"warn\" or \"error\"";
      signalPassFailure();
      return;
    }
    if (failed(verifyVMIToVPTOInputIR(module))) {
      signalPassFailure();
      return;
    }
    if (failed(verifySupportedVMIToVPTOOps(
        module, enableStableGatherMaskedLoad, *loadSafetyPolicy,
        enableVmiFastmath))) {
      signalPassFailure();
      return;
    }

    MLIRContext *context = module.getContext();
    VMIToVPTOTypeConverter typeConverter;
    RewritePatternSet patterns(context);

    populateVMIConversionPatterns(typeConverter, patterns, *loadSafetyPolicy,
                                  enableVmiFastmath);
    if (failed(applyPartialOneToNConversion(module, typeConverter,
                                            std::move(patterns)))) {
      module.emitError() << kVMIDiagResidualOpPrefix
                         << "failed to convert all VMI ops/types to VPTO";
      signalPassFailure();
      return;
    }
    if (failed(verifyNoResidualVMIIR(module))) {
      signalPassFailure();
    }
  }
};
