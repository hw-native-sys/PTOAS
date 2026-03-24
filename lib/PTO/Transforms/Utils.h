#ifndef MLIR_DIALECT_PTO_UTILS_UTILS_H
#define MLIR_DIALECT_PTO_UTILS_UTILS_H
#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <cassert>
#include <queue>
#include <set>
#include <type_traits>

namespace mlir {
namespace pto {
  /// Address spaces treated as local reusable buffer scopes in planning.
  const std::set<pto::AddressSpace> LocalBufferSpace{
    pto::AddressSpace::VEC, pto::AddressSpace::MAT, pto::AddressSpace::ACC, pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT, pto::AddressSpace::BIAS, pto::AddressSpace::SCALING};
  constexpr const uint8_t kBitsToByte = 8;
  /// Returns the only `func.return` in a function, otherwise null.
  func::ReturnOp getAssumedUniqueReturnOp(func::FuncOp funcOp);
  /// Returns (result, source) when `op` is a generic alias/view-like op.
  /// TileBuffer-specific alias semantics are provided by TileBufferSemantics.
  std::optional<std::pair<Value, Value>> getOperationAliasInfo(Operation *op);
  /// Reads PTO address-space from memref values.
  std::optional<AddressSpaceAttr> GetBufferSpaceAttr(Value operand);
  /// Returns true when the address-space belongs to local memory.
  bool isLocalBuffer(std::optional<AddressSpaceAttr> memorySpaceAttr);
  /// Traces memref aliases/views to alloc-like roots.
  Value tracebackMemRef(Value memrefVal);
  /// Computes static product of shape dims; returns nullopt on dynamic dims.
  std::optional<int64_t> getStaticTotalSize(const ArrayRef<int64_t> &shapes);
  /// Rounds `lhs` up to `rhs` alignment.
  uint64_t AlignUp(uint64_t lhs, uint64_t rhs);
  /// Returns nearest parent loop that semantically owns the value.
  LoopLikeOpInterface getParentLoop(Value val);
  /// Gets the top-most module containing `op`.
  ModuleOp getTopLevelModuleOp(Operation *op);
  /// Rewrites memref value's address-space while preserving shape/element type.
  void setBaseMemRefTypeScope(Value val, AddressSpaceAttr targetMemScope);
  /// Builds a memref type with the same payload and a new address-space.
  BaseMemRefType getBaseMemRefTypeWithNewScope(BaseMemRefType type,
                                             AddressSpaceAttr targetMemScope);
  /// Traces a memref value to `memref.alloc` when possible.
  std::optional<memref::AllocOp> tracebackMemRefToAlloc(Value memrefVal);
  /// Returns true when the value ultimately comes from function arguments.
  bool isFromFunctionArg(mlir::Value v);
  /// Returns true when an operation touches any local buffer value.
  bool isOpTouchLocalBuffer(Operation *op);
}
}
#endif
