// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOViewToMemrefCompute.cpp ----------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTOViewToMemrefInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::pto {

namespace {

constexpr unsigned kComputeInlineCapacity = 8;

template <typename T>
using DefaultInlineVector = SmallVector<T, kComputeInlineCapacity>;

constexpr unsigned kThirdOperandIndex = 2;
constexpr unsigned kFourthOperandIndex = 3;
constexpr unsigned kFifthOperandIndex = 4;
constexpr unsigned kSixthOperandIndex = 5;

template <typename OpTy>
static DefaultInlineVector<OpTy> collectComputeOps(func::FuncOp func) {
  DefaultInlineVector<OpTy> ops;
  func.walk([&](OpTy op) { ops.push_back(op); });
  return ops;
}

template <typename OpTy, typename RewriteFn>
static LogicalResult rewriteComputeOps(func::FuncOp func, MLIRContext *ctx,
                                       RewriteFn &&rewriteFn) {
  for (OpTy op : collectComputeOps<OpTy>(func)) {
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    if (failed(rewriteFn(rewriter, op)))
      return failure();
  }
  return success();
}

template <typename... Values>
static LogicalResult requireMemRefs(Operation *op, StringRef message,
                                    Values... values) {
  Value operands[] = {values...};
  for (Value value : operands) {
    if (!isa<MemRefType>(value.getType())) {
      op->emitError(message);
      return failure();
    }
  }
  return success();
}

static LogicalResult requireVectorType(Operation *op, Value value,
                                       StringRef message) {
  if (!isa<VectorType>(value.getType())) {
    op->emitError(message);
    return failure();
  }
  return success();
}

static bool isMemRefLikeValue(Value value) {
  Type type = value.getType();
  return isa<MemRefType, RankedTensorType, PartitionTensorViewType, TileBufType>(
      type);
}

static LogicalResult lowerLoadStoreOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TLoadOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TLoadOp op) {
        auto newOp = rewriter.create<TLoadOp>(op.getLoc(), TypeRange{},
                                              op->getOperand(0),
                                              op->getOperand(1));
        newOp->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newOp->getResults());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TStoreOp>(func, ctx, [](IRRewriter &rewriter,
                                                       TStoreOp op) {
        Value preQuant = op.getPreQuantScalar();
        TStoreOp newOp = rewriter.create<TStoreOp>(
            op.getLoc(), TypeRange{}, op->getOperand(0), op->getOperand(1),
            preQuant ? preQuant : Value{});
        newOp->setAttrs(op->getAttrs());
        rewriter.replaceOp(op, newOp->getResults());
        return success();
      })))
    return failure();
  return rewriteComputeOps<TTransOp>(func, ctx, [](IRRewriter &rewriter,
                                                   TTransOp op) {
    rewriter.replaceOpWithNewOp<TTransOp>(
        op, TypeRange{}, op->getOperand(0), op->getOperand(1),
        op->getOperand(kThirdOperandIndex));
    return success();
  });
}

static LogicalResult lowerBasicArithmeticOps(func::FuncOp func,
                                             MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TExpOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TExpOp op) {
        rewriter.replaceOpWithNewOp<TExpOp>(op, TypeRange{}, op->getOperand(0),
                                            op->getOperand(1));
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TMulOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TMulOp op) {
        rewriter.replaceOpWithNewOp<TMulOp>(op, op->getOperand(0),
                                            op.getOperand(1),
                                            op->getOperand(kThirdOperandIndex));
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TMulSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TMulSOp op) {
        rewriter.replaceOpWithNewOp<TMulSOp>(op, op->getOperand(0),
                                             op.getScalar(),
                                             op->getOperand(kThirdOperandIndex));
        return success();
      })))
    return failure();
  return rewriteComputeOps<TAddOp>(func, ctx, [](IRRewriter &rewriter,
                                                 TAddOp op) {
    rewriter.replaceOpWithNewOp<TAddOp>(op, TypeRange{}, op->getOperand(0),
                                        op->getOperand(1),
                                        op->getOperand(kThirdOperandIndex));
    return success();
  });
}

static LogicalResult lowerLoadStoreAndBasicOps(func::FuncOp func,
                                               MLIRContext *ctx) {
  if (failed(lowerLoadStoreOps(func, ctx)) ||
      failed(lowerBasicArithmeticOps(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerMatmulBaseOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TMatmulOp>(func, ctx, [](IRRewriter &rewriter,
                                                        TMatmulOp op) {
        rewriter.replaceOpWithNewOp<TMatmulOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1),
            op->getOperand(kThirdOperandIndex), op.getAccPhaseAttr());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TMatmulAccOp>(
          func, ctx, [](IRRewriter &rewriter, TMatmulAccOp op) {
            rewriter.replaceOpWithNewOp<TMatmulAccOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex), op.getAccPhaseAttr());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TMatmulBiasOp>(
          func, ctx, [](IRRewriter &rewriter, TMatmulBiasOp op) {
            rewriter.replaceOpWithNewOp<TMatmulBiasOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex));
            return success();
          })))
    return failure();
  return success();
}

static LogicalResult lowerMatmulMxOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TMatmulMxOp>(
          func, ctx, [](IRRewriter &rewriter, TMatmulMxOp op) {
            rewriter.replaceOpWithNewOp<TMatmulMxOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex),
                op->getOperand(kFifthOperandIndex));
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TMatmulMxAccOp>(
          func, ctx, [](IRRewriter &rewriter, TMatmulMxAccOp op) {
            rewriter.replaceOpWithNewOp<TMatmulMxAccOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex),
                op->getOperand(kFifthOperandIndex),
                op->getOperand(kSixthOperandIndex));
            return success();
          })))
    return failure();
  return rewriteComputeOps<TMatmulMxBiasOp>(
      func, ctx, [](IRRewriter &rewriter, TMatmulMxBiasOp op) {
        rewriter.replaceOpWithNewOp<TMatmulMxBiasOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1),
            op->getOperand(kThirdOperandIndex),
            op->getOperand(kFourthOperandIndex),
            op->getOperand(kFifthOperandIndex),
            op->getOperand(kSixthOperandIndex));
        return success();
      });
}

static LogicalResult lowerMatmulOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(lowerMatmulBaseOps(func, ctx)) ||
      failed(lowerMatmulMxOps(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerGemvBaseOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TGemvOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TGemvOp op) {
        rewriter.replaceOpWithNewOp<TGemvOp>(op, TypeRange{}, op->getOperand(0),
                                             op->getOperand(1),
                                             op->getOperand(kThirdOperandIndex));
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TGemvAccOp>(func, ctx, [](IRRewriter &rewriter,
                                                         TGemvAccOp op) {
        rewriter.replaceOpWithNewOp<TGemvAccOp>(
            op, TypeRange{}, op->getOperand(0), op->getOperand(1),
            op->getOperand(kThirdOperandIndex),
            op->getOperand(kFourthOperandIndex));
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TGemvBiasOp>(
          func, ctx, [](IRRewriter &rewriter, TGemvBiasOp op) {
            rewriter.replaceOpWithNewOp<TGemvBiasOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex));
            return success();
          })))
    return failure();
  return success();
}

static LogicalResult lowerGemvMxAndMovOps(func::FuncOp func,
                                          MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TGemvMxOp>(
          func, ctx, [](IRRewriter &rewriter, TGemvMxOp op) {
            rewriter.replaceOpWithNewOp<TGemvMxOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex),
                op->getOperand(kFifthOperandIndex));
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TGemvMxAccOp>(
          func, ctx, [](IRRewriter &rewriter, TGemvMxAccOp op) {
            rewriter.replaceOpWithNewOp<TGemvMxAccOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex),
                op->getOperand(kFifthOperandIndex),
                op->getOperand(kSixthOperandIndex));
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TGemvMxBiasOp>(
          func, ctx, [](IRRewriter &rewriter, TGemvMxBiasOp op) {
            rewriter.replaceOpWithNewOp<TGemvMxBiasOp>(
                op, TypeRange{}, op->getOperand(0), op->getOperand(1),
                op->getOperand(kThirdOperandIndex),
                op->getOperand(kFourthOperandIndex),
                op->getOperand(kFifthOperandIndex),
                op->getOperand(kSixthOperandIndex));
            return success();
          })))
    return failure();
  return rewriteComputeOps<TMovOp>(func, ctx, [](IRRewriter &rewriter,
                                                 TMovOp op) {
    rewriter.replaceOpWithNewOp<TMovOp>(
        op, TypeRange{}, op.getSrc(), op.getDst(), op.getFp(),
        op.getPreQuantScalar(), op.getAccToVecModeAttr(),
        op.getReluPreModeAttr());
    return success();
  });
}

static LogicalResult lowerGemvAndMovOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(lowerGemvBaseOps(func, ctx)) ||
      failed(lowerGemvMxAndMovOps(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerElementwiseOpsA1(func::FuncOp func,
                                           MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TAbsOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TAbsOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TAbsOp>(op, TypeRange{}, op.getSrc(),
                                            op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TAddCOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TAddCOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getSrc2(),
                                  op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TAddCOp>(op, TypeRange{}, op.getSrc0(),
                                             op.getSrc1(), op.getSrc2(),
                                             op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TAddSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TAddSOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TAddSOp>(op, TypeRange{}, op.getSrc(),
                                             op.getScalar(), op.getDst());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerElementwiseOpsA2(func::FuncOp func,
                                           MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TAddSCOp>(
          func, ctx, [](IRRewriter &rewriter, TAddSCOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc0(), op.getSrc1(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TAddSCOp>(
                op, TypeRange{}, op.getSrc0(), op.getScalar(), op.getSrc1(),
                op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TAndOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TAndOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TAndOp>(op, TypeRange{}, op.getSrc0(),
                                            op.getSrc1(), op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TConcatOp>(func, ctx, [](IRRewriter &rewriter,
                                                        TConcatOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TConcatOp>(op, TypeRange{}, op.getSrc0(),
                                               op.getSrc1(), op.getDst());
        return success();
      })))
    return failure();
  return rewriteComputeOps<TConcatidxOp>(
      func, ctx, [](IRRewriter &rewriter, TConcatidxOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getSrc0Idx(),
                                  op.getSrc1Idx(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TConcatidxOp>(
            op, TypeRange{}, op.getSrc0(), op.getSrc1(), op.getSrc0Idx(),
            op.getSrc1Idx(), op.getDst());
        return success();
      });
}

static LogicalResult lowerElementwiseOpsA(func::FuncOp func,
                                          MLIRContext *ctx) {
  if (failed(lowerElementwiseOpsA1(func, ctx)) ||
      failed(lowerElementwiseOpsA2(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerElementwiseOpsB1a(func::FuncOp func,
                                            MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TAndSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TAndSOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TAndSOp>(op, TypeRange{}, op.getSrc(),
                                             op.getScalar(), op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TCIOp>(func, ctx, [](IRRewriter &rewriter,
                                                    TCIOp op) {
        if (!isa<IntegerType>(op->getOperand(0).getType()) ||
            failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getDst())))
          return op.emitError("ins/outs are not memref yet"), failure();
        rewriter.replaceOpWithNewOp<TCIOp>(op, TypeRange{}, op->getOperand(0),
                                           op.getDst(), op.getDescending());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerElementwiseOpsB1b(func::FuncOp func,
                                            MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TCmpOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TCmpOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getDst())))
          return failure();
        auto newOp = rewriter.create<TCmpOp>(op.getLoc(), TypeRange{},
                                             op.getSrc0(), op.getSrc1(),
                                             op.getDst());
        if (auto attr = op.getCmpModeAttr())
          newOp->setAttr("cmpMode", attr);
        rewriter.replaceOp(op, newOp->getResults());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TCmpSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TCmpSOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        if (!isa<IntegerType, FloatType>(op.getScalar().getType())) {
          op.emitError("expects scalar to be an integer or float type");
          return failure();
        }
        auto newOp = rewriter.create<TCmpSOp>(
            op.getLoc(), TypeRange{}, op.getSrc(), op.getScalar(),
            op.getCmpModeAttr(), op.getDst());
        rewriter.replaceOp(op, newOp->getResults());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerElementwiseOpsB2(func::FuncOp func,
                                           MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TColExpandOp>(
          func, ctx, [](IRRewriter &rewriter, TColExpandOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TColExpandOp>(op, TypeRange{},
                                                      op.getSrc(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TColMaxOp>(
          func, ctx, [](IRRewriter &rewriter, TColMaxOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TColMaxOp>(op, TypeRange{},
                                                   op.getSrc(), op.getDst());
            return success();
          })))
    return failure();
  return rewriteComputeOps<TColMinOp>(
      func, ctx, [](IRRewriter &rewriter, TColMinOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TColMinOp>(op, TypeRange{}, op.getSrc(),
                                               op.getDst());
        return success();
      });
}

static LogicalResult lowerElementwiseOpsB(func::FuncOp func,
                                          MLIRContext *ctx) {
  if (failed(lowerElementwiseOpsB1a(func, ctx)) ||
      failed(lowerElementwiseOpsB1b(func, ctx)) ||
      failed(lowerElementwiseOpsB2(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerColumnAndConversionOps1(func::FuncOp func,
                                                  MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TColExpandMulOp>(
          func, ctx, [](IRRewriter &rewriter, TColExpandMulOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc0(), op.getSrc1(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TColExpandMulOp>(
                op, TypeRange{}, op.getSrc0(), op.getSrc1(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TColExpandMaxOp>(
          func, ctx, [](IRRewriter &rewriter, TColExpandMaxOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc0(), op.getSrc1(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TColExpandMaxOp>(
                op, TypeRange{}, op.getSrc0(), op.getSrc1(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TColExpandMinOp>(
          func, ctx, [](IRRewriter &rewriter, TColExpandMinOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc0(), op.getSrc1(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TColExpandMinOp>(
                op, TypeRange{}, op.getSrc0(), op.getSrc1(), op.getDst());
            return success();
          })))
    return failure();
  return success();
}

static LogicalResult lowerTColSumOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TColSumOp>(func, ctx, [ctx](IRRewriter &rewriter,
                                                            TColSumOp op) {
        if (failed(requireMemRefs(op.getOperation(), "src/dst are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        Value tmp = op.getTmp();
        if (tmp) {
          if (failed(requireMemRefs(op.getOperation(), "tmp is not memref yet",
                                    tmp)))
            return failure();
          BoolAttr isBinaryAttr = op.getIsBinaryAttr();
          if (!isBinaryAttr)
            isBinaryAttr = BoolAttr::get(ctx, false);
          rewriter.replaceOpWithNewOp<TColSumOp>(op, TypeRange{}, op.getSrc(),
                                                 tmp, op.getDst(),
                                                 isBinaryAttr);
          return success();
        }
        SmallVector<Value> operands{op.getSrc(), op.getDst()};
        SmallVector<NamedAttribute> attrs;
        for (auto attr : op->getAttrs()) {
          if (attr.getName() != "isBinary")
            attrs.push_back(attr);
        }
        rewriter.replaceOpWithNewOp<TColSumOp>(op, TypeRange{}, operands,
                                               attrs);
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerBasicConversionOps(func::FuncOp func,
                                             MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TCvtOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TCvtOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        auto newOp = rewriter.create<TCvtOp>(
            op.getLoc(), TypeRange{}, op.getSrc(), op.getDst(),
            op.getRmodeAttr(), op.getSatModeAttr());
        rewriter.replaceOp(op, newOp->getResults());
        return success();
      })))
    return failure();
  return rewriteComputeOps<TDivOp>(func, ctx, [](IRRewriter &rewriter,
                                                 TDivOp op) {
    if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                              op.getSrc0(), op.getSrc1(), op.getDst())))
      return failure();
    rewriter.replaceOpWithNewOp<TDivOp>(op, TypeRange{}, op.getSrc0(),
                                        op.getSrc1(), op.getDst());
    return success();
  });
}

static LogicalResult lowerColumnAndConversionOps(func::FuncOp func,
                                                 MLIRContext *ctx) {
  if (failed(lowerColumnAndConversionOps1(func, ctx)) ||
      failed(lowerTColSumOps(func, ctx)) ||
      failed(lowerBasicConversionOps(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerDivScalarOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TDivSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TDivSOp op) {
        bool srcIsMemref = isMemRefLikeValue(op.getSrc());
        bool scaleIsMemref = isMemRefLikeValue(op.getScalar());
        if (!srcIsMemref && !scaleIsMemref) {
          op.emitError(
              "at least one operand (src or scale) must be tile_buf or memref");
          return failure();
        }
        if (srcIsMemref && scaleIsMemref) {
          op.emitError(
              "exactly one operand (src or scale) must be tile_buf or memref, the other must be scalar");
          return failure();
        }
        if (!isMemRefLikeValue(op.getDst())) {
          op.emitError("dst operand must be tile_buf or memref");
          return failure();
        }
        rewriter.replaceOpWithNewOp<TDivSOp>(op, TypeRange{}, op.getSrc(),
                                             op.getScalar(), op.getDst());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerExpandExtractOps(func::FuncOp func,
                                           MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TExpandsOp>(
          func, ctx, [](IRRewriter &rewriter, TExpandsOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TExpandsOp>(op, TypeRange{},
                                                    op.getScalar(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TExtractOp>(
          func, ctx, [](IRRewriter &rewriter, TExtractOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not correct yet",
                                      op.getSrc(), op.getDst())) ||
                !isa<IndexType>(op.getIndexRow().getType()) ||
                !isa<IndexType>(op.getIndexCol().getType())) {
              op.emitError("ins/outs are not correct yet");
              return failure();
            }
            rewriter.replaceOpWithNewOp<TExtractOp>(
                op, TypeRange{}, op.getSrc(), op.getIndexRow(),
                op.getIndexCol(), op.getDst());
            return success();
          })))
    return failure();
  return success();
}

static LogicalResult lowerScalarAndPadOps2(func::FuncOp func,
                                           MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TFillPadOp>(
          func, ctx, [](IRRewriter &rewriter, TFillPadOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TFillPadOp>(op, TypeRange{},
                                                    op.getSrc(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TFillPadInplaceOp>(
          func, ctx, [](IRRewriter &rewriter, TFillPadInplaceOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TFillPadInplaceOp>(
                op, TypeRange{}, op.getSrc(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TSetValOp>(func, ctx, [](IRRewriter &rewriter,
                                                        TSetValOp op) {
        if (failed(requireMemRefs(op.getOperation(), "dst is not memref yet",
                                  op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TSetValOp>(op, TypeRange{}, op.getDst(),
                                               op.getOffset(), op.getVal());
        return success();
      })))
    return failure();
  return rewriteComputeOps<TGetValOp>(func, ctx, [](IRRewriter &rewriter,
                                                    TGetValOp op) {
    if (failed(requireMemRefs(op.getOperation(), "src is not memref yet",
                              op.getSrc())))
      return failure();
    auto newOp = rewriter.create<TGetValOp>(op.getLoc(), op.getDst().getType(),
                                            op.getSrc(), op.getOffset());
    rewriter.replaceOp(op, newOp.getDst());
    return success();
  });
}

static LogicalResult lowerGatherOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TGatherOp>(func, ctx, [](IRRewriter &rewriter,
                                                        TGatherOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        if (auto maskPattern = op.getMaskPatternAttr()) {
          rewriter.replaceOpWithNewOp<TGatherOp>(
              op, TypeRange{}, op.getSrc(), op.getDst(), Value(), Value(),
              Value(), Value(), maskPattern, CmpModeAttr(), IntegerAttr());
          return success();
        }
        if (op.getCdst() || op.getKValue()) {
          if (failed(requireMemRefs(op.getOperation(),
                                    "compare-form tgather expects cdst/tmp to be memref yet",
                                    op.getCdst(), op.getTmp())))
            return failure();
          rewriter.replaceOpWithNewOp<TGatherOp>(
              op, TypeRange{}, op.getSrc(), op.getDst(), op.getCdst(), Value(),
              op.getTmp(), op.getKValue(), MaskPatternAttr(),
              op.getCmpModeAttr(), op.getOffsetAttr());
          return success();
        }
        if (op.getIndices() || op.getTmp()) {
          if (failed(requireMemRefs(op.getOperation(),
                                    "index-form tgather expects indices/tmp to be memref yet",
                                    op.getIndices(), op.getTmp())))
            return failure();
          rewriter.replaceOpWithNewOp<TGatherOp>(
              op, TypeRange{}, op.getSrc(), op.getDst(), Value(),
              op.getIndices(), op.getTmp(), Value(), MaskPatternAttr(),
              CmpModeAttr(), IntegerAttr());
          return success();
        }
        op.emitError(
            "expects tgather to be in mask, index+tmp, or compare+tmp form");
        return failure();
      })))
    return failure();
  return rewriteComputeOps<TGatherBOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TGatherBOp op) {
    if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                              op.getSrc(), op.getOffsets(), op.getDst())))
      return failure();
    rewriter.replaceOpWithNewOp<TGatherBOp>(op, TypeRange{}, op.getSrc(),
                                            op.getOffsets(), op.getDst());
    return success();
  });
}

static LogicalResult lowerScalarAndPadOps(func::FuncOp func,
                                          MLIRContext *ctx) {
  if (failed(lowerDivScalarOps(func, ctx)) ||
      failed(lowerExpandExtractOps(func, ctx)) ||
      failed(lowerScalarAndPadOps2(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerLogAndReluOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TLogOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TLogOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TLogOp>(op, TypeRange{}, op.getSrc(),
                                            op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TLReluOp>(func, ctx, [](IRRewriter &rewriter,
                                                       TLReluOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not correct type yet",
                                  op.getSrc(), op.getDst())) ||
            !isa<FloatType>(op.getSlope().getType())) {
          op.emitError("ins/outs are not correct type yet");
          return failure();
        }
        rewriter.replaceOpWithNewOp<TLReluOp>(op, TypeRange{}, op.getSrc(),
                                              op.getSlope(), op.getDst());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerMaxLikeOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TMaxOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TMaxOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TMaxOp>(op, TypeRange{}, op.getSrc0(),
                                            op.getSrc1(), op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TMaxSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TMaxSOp op) {
        if (failed(requireMemRefs(op.getOperation(),
                                  "expects src/dst to be memref and scalar to be integer/float",
                                  op.getSrc(), op.getDst())) ||
            !isa<IntegerType, FloatType>(op.getScalar().getType())) {
          op.emitError(
              "expects src/dst to be memref and scalar to be integer/float");
          return failure();
        }
        rewriter.replaceOpWithNewOp<TMaxSOp>(op, TypeRange{}, op.getSrc(),
                                             op.getScalar(), op.getDst());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerMinLikeOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TMinOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TMinOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TMinOp>(op, TypeRange{}, op.getSrc0(),
                                            op.getSrc1(), op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TMinSOp>(func, ctx, [](IRRewriter &rewriter,
                                                      TMinSOp op) {
        if (failed(requireMemRefs(op.getOperation(),
                                  "expects src/dst to be memref and scalar to be integer/float",
                                  op.getSrc(), op.getDst())) ||
            !isa<IntegerType, FloatType>(op.getScalar().getType())) {
          op.emitError(
              "expects src/dst to be memref and scalar to be integer/float");
          return failure();
        }
        rewriter.replaceOpWithNewOp<TMinSOp>(op, TypeRange{}, op.getSrc(),
                                             op.getScalar(), op.getDst());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerQuantOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TMovFPOp>(func, ctx, [](IRRewriter &rewriter,
                                                       TMovFPOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getFp(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TMovFPOp>(op, TypeRange{}, op.getSrc(),
                                              op.getFp(), op.getDst());
        return success();
      })))
    return failure();
  return rewriteComputeOps<TQuantOp>(func, ctx, [](IRRewriter &rewriter,
                                                   TQuantOp op) {
    if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                              op.getSrc(), op.getFp(), op.getDst())))
      return failure();
    Value offset = op.getOffset();
    if (offset && failed(requireMemRefs(op.getOperation(), "offset is not memref yet",
                                        offset)))
      return failure();
    rewriter.replaceOpWithNewOp<TQuantOp>(op, TypeRange{}, op.getSrc(),
                                          op.getFp(), offset, op.getDst(),
                                          op.getQuantTypeAttr());
    return success();
  });
}

static LogicalResult lowerExtremaAndQuantOps(func::FuncOp func,
                                             MLIRContext *ctx) {
  if (failed(lowerLogAndReluOps(func, ctx)) ||
      failed(lowerMaxLikeOps(func, ctx)) ||
      failed(lowerMinLikeOps(func, ctx)) ||
      failed(lowerQuantOps(func, ctx))) {
    return failure();
  }
  return success();
}

static LogicalResult lowerMergeSortOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TMrgSortOp>(func, ctx, [](IRRewriter &rewriter,
                                                         TMrgSortOp op) {
        if (op.isFormat1()) {
          if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                    op.getSrc(), op.getDst())))
            return failure();
          rewriter.replaceOpWithNewOp<TMrgSortOp>(
              op, TypeRange{}, ValueRange{op.getSrc()}, op.getBlockLen(),
              ValueRange{op.getDst()}, Value(), Value(), op.getExhaustedAttr());
          return success();
        }
        if (op.isFormat2()) {
          for (Value src : op.getSrcs()) {
            if (!isa<MemRefType>(src.getType())) {
              op.emitError("format2 ins/outs are not memref yet");
              return failure();
            }
          }
          if (op.getDsts().size() != 1u || !op.getTmp()) {
            op.emitError("format2 expects outs(dst) and ins(tmp)");
            return failure();
          }
          if (failed(requireMemRefs(op.getOperation(),
                                    "format2 dst/tmp must be memref", op.getDst(),
                                    op.getTmp())))
            return failure();
          if (failed(requireVectorType(op.getOperation(), op.getExcuted(),
                                       "format2 outs(excuted) must be vector")))
            return failure();
          rewriter.replaceOpWithNewOp<TMrgSortOp>(
              op, TypeRange{}, op.getSrcs(), Value(), ValueRange{op.getDst()},
              op.getTmp(), op.getExcuted(), op.getExhaustedAttr());
          return success();
        }
        op.emitError("tmrgsort must be format1 or format2");
        return failure();
      })))
    return failure();
  return success();
}

static LogicalResult lowerLogicOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TNegOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TNegOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TNegOp>(op, TypeRange{}, op.getSrc(),
                                            op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TNotOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TNotOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TNotOp>(op, TypeRange{}, op.getSrc(),
                                            op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TOrOp>(func, ctx, [](IRRewriter &rewriter,
                                                    TOrOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc0(), op.getSrc1(), op.getDst())))
          return failure();
        rewriter.replaceOpWithNewOp<TOrOp>(op, op.getSrc0(), op.getSrc1(),
                                           op.getDst());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<TOrSOp>(func, ctx, [](IRRewriter &rewriter,
                                                     TOrSOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getDst())) ||
            !isa<IntegerType>(op.getScalar().getType())) {
          op.emitError("ins/outs are not memref yet");
          return failure();
        }
        rewriter.replaceOpWithNewOp<TOrSOp>(op, TypeRange{}, op.getSrc(),
                                            op.getScalar(), op.getDst());
        return success();
      })))
    return failure();
  return success();
}

static LogicalResult lowerPartOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(rewriteComputeOps<TPartAddOp>(
          func, ctx, [](IRRewriter &rewriter, TPartAddOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc0(), op.getSrc1(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TPartAddOp>(op, op.getSrc0(),
                                                    op.getSrc1(), op.getDst());
            return success();
          })))
    return failure();
  if (failed(rewriteComputeOps<TPartMulOp>(
          func, ctx, [](IRRewriter &rewriter, TPartMulOp op) {
            if (failed(requireMemRefs(op.getOperation(),
                                      "ins/outs are not memref yet",
                                      op.getSrc0(), op.getSrc1(), op.getDst())))
              return failure();
            rewriter.replaceOpWithNewOp<TPartMulOp>(op, op.getSrc0(),
                                                    op.getSrc1(), op.getDst());
            return success();
          })))
    return failure();
  return success();
}

static LogicalResult lowerGatherPrintOps(func::FuncOp func,
                                         MLIRContext *ctx) {
  if (failed(rewriteComputeOps<MGatherOp>(func, ctx, [](IRRewriter &rewriter,
                                                        MGatherOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getDst(), op.getIdx(), op.getMem())))
          return failure();
        rewriter.replaceOpWithNewOp<MGatherOp>(op, TypeRange{}, op.getMem(),
                                               op.getIdx(), op.getDst(),
                                               op.getGatherOobAttr());
        return success();
      })))
    return failure();
  if (failed(rewriteComputeOps<MScatterOp>(func, ctx, [](IRRewriter &rewriter,
                                                         MScatterOp op) {
        if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                                  op.getSrc(), op.getIdx(), op.getMem())))
          return failure();
        rewriter.replaceOpWithNewOp<MScatterOp>(
            op, TypeRange{}, op.getSrc(), op.getIdx(), op.getMem(),
            op.getScatterAtomicOpAttr(), op.getScatterOobAttr());
        return success();
      })))
    return failure();
  return rewriteComputeOps<TPrintOp>(func, ctx, [](IRRewriter &rewriter,
                                                   TPrintOp op) {
    if (failed(requireMemRefs(op.getOperation(), "ins/outs are not memref yet",
                              op.getSrc())))
      return failure();
    rewriter.replaceOpWithNewOp<TPrintOp>(op, TypeRange{}, op.getSrc());
    return success();
  });
}

static LogicalResult lowerMiscComputeOps(func::FuncOp func,
                                         MLIRContext *ctx) {
  if (failed(lowerMergeSortOps(func, ctx)) ||
      failed(lowerLogicOps(func, ctx)) ||
      failed(lowerPartOps(func, ctx)) ||
      failed(lowerGatherPrintOps(func, ctx))) {
    return failure();
  }
  return success();
}

} // namespace

LogicalResult lowerViewToMemrefComputeOps(func::FuncOp func, MLIRContext *ctx) {
  if (failed(lowerLoadStoreAndBasicOps(func, ctx)) ||
      failed(lowerMatmulOps(func, ctx)) ||
      failed(lowerGemvAndMovOps(func, ctx)) ||
      failed(lowerElementwiseOpsA(func, ctx)) ||
      failed(lowerElementwiseOpsB(func, ctx)) ||
      failed(lowerColumnAndConversionOps(func, ctx)) ||
      failed(lowerScalarAndPadOps(func, ctx)) ||
      failed(lowerGatherOps(func, ctx)) ||
      failed(lowerExtremaAndQuantOps(func, ctx)) ||
      failed(lowerMiscComputeOps(func, ctx))) {
    return failure();
  }
  return success();
}

} // namespace mlir::pto
