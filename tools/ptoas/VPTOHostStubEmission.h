// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTOAS_VPTO_HOST_STUB_EMISSION_H
#define PTOAS_VPTO_HOST_STUB_EMISSION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include <string>

namespace llvm {
class raw_ostream;
}

namespace mlir::pto {

LogicalResult emitVPTOHostStubSource(ModuleOp module, std::string &stubSource,
                                     llvm::raw_ostream &diagOS);

LogicalResult emitVPTOHostStubSource(ArrayRef<ModuleOp> modules,
                                     std::string &stubSource,
                                     llvm::raw_ostream &diagOS);

/// Generate a device-side `kernel_entry(__gm__ int64_t* args)` wrapper source
/// (cpp) that unpacks the simpler dispatch payload's args[] (tensors-first
/// ChipTensor* → buffer.addr + start_offset, then scalar union uint64 punning)
/// and forwards to the VPTO body entry (`<logicalName>_mix_aiv`/`_mix_aic`).
///
/// This mirrors the EmitC wrapper codegen (pypto pto_backend.py
/// _generate_kernel_wrapper) but is emitted from the PTO IR at VPTOHostStub
/// emission time (before MLIR→LLVM lowering/rename). Parameter order is
/// preserved because normalizeFuncSignaturesForOfficialLLVMLowering only
/// normalizes types, not arg count/order.
LogicalResult emitVPTODeviceWrapperSource(ModuleOp module,
                                          std::string &wrapperSource,
                                          llvm::raw_ostream &diagOS);

LogicalResult emitVPTODeviceWrapperSource(ArrayRef<ModuleOp> modules,
                                          std::string &wrapperSource,
                                          llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif
