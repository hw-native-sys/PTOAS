#ifndef MLIR_DIALECT_PTO_TRANSFORMS_A5VMTEXTEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_A5VMTEXTEMITTER_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

struct A5VMEmissionOptions {
  // Phase 1 keeps the A5VM seam at raw backend-text granularity, so intrinsic
  // selection tracing is intentionally unused until the later HIVM-emission work.
  bool printIntrinsicSelections = false;
  bool allowUnresolved = false;
  std::string unresolvedReportPath;
};

LogicalResult translateA5VMModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const A5VMEmissionOptions &options,
                                        llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_A5VMTEXTEMITTER_H
