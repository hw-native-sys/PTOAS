#ifndef MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H
#define MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
class ModuleOp;
}

namespace llvm {
class raw_ostream;
}

namespace mlir::pto {

struct VPTOEmissionOptions {
  bool dumpVPTOIR = false;
  bool printIntrinsicSelections = false;
  bool allowUnresolved = true;
  std::string unresolvedReportPath;
  std::string targetTriple;
  std::string march;
  std::string aicoreArch;
  std::string defaultTargetCPU;
  std::string defaultTargetFeatures;
};

LogicalResult
translateVPTOModuleToLLVMText(ModuleOp module, llvm::raw_ostream &os,
                              const VPTOEmissionOptions &options,
                              llvm::raw_ostream &diagOS);

LogicalResult
translateVPTOModuleToLLVMBitcode(ModuleOp module, llvm::raw_ostream &os,
                                 const VPTOEmissionOptions &options,
                                 llvm::raw_ostream &diagOS);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_VPTOLLVMEMITTER_H
