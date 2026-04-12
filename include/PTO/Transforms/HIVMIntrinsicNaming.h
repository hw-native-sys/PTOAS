#ifndef MLIR_DIALECT_PTO_TRANSFORMS_HIVMINTRINSICNAMING_H
#define MLIR_DIALECT_PTO_TRANSFORMS_HIVMINTRINSICNAMING_H

#include <string>
#include <vector>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace mlir::pto {

struct NamingInputs {
  std::string sourceOpName;
  std::string family;
  std::string vectorShape;
  std::string elementType;
  std::vector<std::string> usedFields;
  std::vector<std::string> missingFields;
};

struct UnresolvedEmissionRecord {
  std::string sourceOpName;
  std::string placeholderName;
  std::string candidateName;
  std::vector<std::string> usedFields;
  std::vector<std::string> missingFields;
  std::string resultTypeFragment;
  std::string location;
};

struct IntrinsicSelection {
  bool resolved = false;
  std::string sourceOpName;
  std::string calleeName;
  std::string placeholderName;
  std::string candidateName;
  std::vector<std::string> usedFields;
  std::vector<std::string> missingFields;
  std::string resultTypeFragment;
  std::string location;

  std::string getEmittedCallee() const {
    return resolved ? calleeName : placeholderName;
  }

  UnresolvedEmissionRecord asUnresolvedRecord() const {
    return UnresolvedEmissionRecord{sourceOpName, placeholderName, candidateName,
                                    usedFields, missingFields, resultTypeFragment,
                                    location};
  }
};

FailureOr<IntrinsicSelection> selectIntrinsic(Operation *op);
FailureOr<IntrinsicSelection> selectLoadIntrinsic(Operation *op);
FailureOr<IntrinsicSelection> selectUnaryIntrinsic(Operation *op);
FailureOr<IntrinsicSelection> selectStoreIntrinsic(Operation *op);

} // namespace mlir::pto

#endif // MLIR_DIALECT_PTO_TRANSFORMS_HIVMINTRINSICNAMING_H
