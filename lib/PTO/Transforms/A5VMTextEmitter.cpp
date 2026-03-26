//===- A5VMTextEmitter.cpp - A5VM textual LLVM-like emitter --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMTextEmitter.h"

#include "PTO/IR/A5VM.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/HIVMIntrinsicNaming.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>

using namespace mlir;

namespace mlir::pto {
namespace {

static std::string joinList(llvm::ArrayRef<std::string> items) {
  std::string out = "[";
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0)
      out += ", ";
    out += items[i];
  }
  out += "]";
  return out;
}

static std::string serializeRecord(const UnresolvedEmissionRecord &record) {
  std::string line = "op=" + record.sourceOpName;
  line += " placeholder=" + record.placeholderName;
  if (!record.candidateName.empty())
    line += " candidate=" + record.candidateName;
  if (!record.resultTypeFragment.empty())
    line += " result_type=" + record.resultTypeFragment;
  line += " used_fields=" + joinList(record.usedFields);
  line += " missing_fields=" + joinList(record.missingFields);
  line += " loc=\"" + record.location + "\"";
  return line;
}

static std::string formatLocationString(Location loc) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  loc.print(os);
  return storage;
}

static LogicalResult writeReportFile(const A5VMEmissionOptions &options,
                                     ArrayRef<UnresolvedEmissionRecord> records,
                                     llvm::raw_ostream &diagOS) {
  if (options.unresolvedReportPath.empty())
    return success();

  std::error_code ec;
  llvm::raw_fd_ostream reportOS(options.unresolvedReportPath, ec,
                                llvm::sys::fs::OF_Text);
  if (ec) {
    diagOS << "A5VM emission failed: could not write unresolved report to '"
           << options.unresolvedReportPath << "': " << ec.message() << "\n";
    return failure();
  }

  for (const auto &record : records)
    reportOS << serializeRecord(record) << "\n";
  return success();
}

class LLVMTextEmitter {
public:
  LLVMTextEmitter(ModuleOp sourceModule, llvm::raw_ostream &diagOS,
                  const A5VMEmissionOptions &options)
      : sourceModule(sourceModule), diagOS(diagOS), options(options),
        llvmModule(std::make_unique<llvm::Module>("ptoas.hivm", llvmContext)),
        builder(llvmContext) {}

  LogicalResult emitTo(llvm::raw_ostream &os) {
    for (auto func : sourceModule.getOps<func::FuncOp>()) {
      if (failed(emitFunction(func)))
        return failure();
    }

    if (failed(writeReportFile(options, unresolvedRecords, diagOS)))
      return failure();

    if (options.printIntrinsicSelections) {
      for (const auto &selection : intrinsicSelections) {
        diagOS << "intrinsic-selection op=" << selection.sourceOpName
               << " emitted=" << selection.getEmittedCallee();
        if (!selection.candidateName.empty())
          diagOS << " candidate=" << selection.candidateName;
        diagOS << "\n";
      }
    }

    llvmModule->print(os, nullptr);
    return success();
  }

private:
  ModuleOp sourceModule;
  llvm::raw_ostream &diagOS;
  const A5VMEmissionOptions &options;
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule;
  llvm::IRBuilder<> builder;
  llvm::DenseMap<Value, llvm::Value *> values;
  std::vector<UnresolvedEmissionRecord> unresolvedRecords;
  std::vector<IntrinsicSelection> intrinsicSelections;
  std::unordered_map<std::string, llvm::Function *> declarations;

  llvm::Type *getIntegerType(unsigned width) {
    return llvm::Type::getIntNTy(llvmContext, width);
  }

  llvm::Value *castIntegerLikeToI64(llvm::Value *value) {
    if (!value)
      return nullptr;
    if (value->getType() == getIntegerType(64))
      return value;
    if (!value->getType()->isIntegerTy())
      return nullptr;
    unsigned width = value->getType()->getIntegerBitWidth();
    if (width < 64)
      return builder.CreateZExt(value, getIntegerType(64), "i64.zext");
    if (width > 64)
      return builder.CreateTrunc(value, getIntegerType(64), "i64.trunc");
    return value;
  }

  llvm::Value *castIntegerLikeToI32(llvm::Value *value) {
    if (!value)
      return nullptr;
    if (value->getType() == getIntegerType(32))
      return value;
    if (!value->getType()->isIntegerTy())
      return nullptr;
    unsigned width = value->getType()->getIntegerBitWidth();
    if (width < 32)
      return builder.CreateZExt(value, getIntegerType(32), "i32.zext");
    if (width > 32)
      return builder.CreateTrunc(value, getIntegerType(32), "i32.trunc");
    return value;
  }

  llvm::Value *castIntegerLikeToI16(llvm::Value *value) {
    if (!value)
      return nullptr;
    if (value->getType() == getIntegerType(16))
      return value;
    if (!value->getType()->isIntegerTy())
      return nullptr;
    unsigned width = value->getType()->getIntegerBitWidth();
    if (width < 16)
      return builder.CreateZExt(value, getIntegerType(16), "i16.zext");
    if (width > 16)
      return builder.CreateTrunc(value, getIntegerType(16), "i16.trunc");
    return value;
  }

  std::optional<uint64_t> parsePipeImmediate(llvm::StringRef pipe) {
    if (pipe == "PIPE_S")
      return 0;
    if (pipe == "PIPE_V")
      return 1;
    if (pipe == "PIPE_M")
      return 2;
    if (pipe == "PIPE_MTE1")
      return 3;
    if (pipe == "PIPE_MTE2")
      return 4;
    if (pipe == "PIPE_MTE3")
      return 5;
    if (pipe == "PIPE_ALL")
      return 6;
    if (pipe == "PIPE_MTE4")
      return 7;
    if (pipe == "PIPE_MTE5")
      return 8;
    if (pipe == "PIPE_V2")
      return 9;
    if (pipe == "PIPE_FIX")
      return 10;
    if (pipe == "VIRTUAL_PIPE_MTE2_L1A")
      return 11;
    if (pipe == "VIRTUAL_PIPE_MTE2_L1B")
      return 12;
    return std::nullopt;
  }

  std::optional<uint64_t> parseEventImmediate(llvm::StringRef event) {
    if (!event.consume_front("EVENT_ID"))
      return std::nullopt;
    uint64_t value = 0;
    if (event.getAsInteger(10, value))
      return std::nullopt;
    return value;
  }

  std::optional<uint64_t> parseLoadDistImmediate(llvm::StringRef dist) {
    if (dist.empty() || dist == "NORM")
      return 0;
    if (dist == "BLK")
      return 15;
    if (dist == "UNPK_B16")
      return 14;
    if (dist == "DINTLV_B32")
      return 19;
    return std::nullopt;
  }

  std::optional<uint64_t> parseStoreDistImmediate(Type valueType,
                                                  llvm::StringRef dist) {
    auto vecType = dyn_cast<a5vm::VecType>(valueType);
    if (!vecType)
      return std::nullopt;

    if (dist.empty()) {
      unsigned bitWidth = 0;
      if (auto intType = dyn_cast<IntegerType>(vecType.getElementType()))
        bitWidth = intType.getWidth();
      else if (auto floatType = dyn_cast<FloatType>(vecType.getElementType()))
        bitWidth = floatType.getWidth();
      switch (bitWidth) {
      case 8:
        return 0;
      case 16:
        return 1;
      case 32:
        return 2;
      default:
        return std::nullopt;
      }
    }

    if (dist == "NORM_B8")
      return 0;
    if (dist == "NORM_B16")
      return 1;
    if (dist == "NORM_B32")
      return 2;
    if (dist == "ONEPT_B8")
      return 3;
    if (dist == "ONEPT_B16")
      return 4;
    if (dist == "ONEPT_B32")
      return 5;
    if (dist == "PK_B16")
      return 6;
    if (dist == "PK_B32")
      return 7;
    if (dist == "INTLV_B8")
      return 8;
    if (dist == "INTLV_B16")
      return 9;
    if (dist == "PK_B64")
      return 10;
    if (dist == "INTLV_B32")
      return 11;
    if (dist == "PK4_B32")
      return 12;
    if (dist == "MRG4CHN_B8")
      return 13;
    if (dist == "MRG2CHN_B8")
      return 14;
    if (dist == "MRG2CHN_B16")
      return 15;
    return std::nullopt;
  }

  llvm::Type *convertScalarType(Type type) {
    if (type.isIndex())
      return getIntegerType(64);
    if (auto intType = dyn_cast<IntegerType>(type))
      return getIntegerType(intType.getWidth());
    if (type.isF32())
      return llvm::Type::getFloatTy(llvmContext);
    if (type.isF16())
      return llvm::Type::getHalfTy(llvmContext);
    if (type.isBF16())
      return llvm::Type::getBFloatTy(llvmContext);
    return nullptr;
  }

  llvm::Type *convertType(Type type) {
    if (auto vecType = dyn_cast<a5vm::VecType>(type))
      return llvm::FixedVectorType::get(
          convertScalarType(vecType.getElementType()), vecType.getElementCount());

    if (isa<a5vm::MaskType>(type))
      return llvm::FixedVectorType::get(getIntegerType(1), 256);
    if (isa<a5vm::AlignType>(type))
      return getIntegerType(64);

    if (auto ptrType = dyn_cast<LLVM::LLVMPointerType>(type))
      return llvm::PointerType::get(llvmContext, ptrType.getAddressSpace());

    if (auto memrefType = dyn_cast<BaseMemRefType>(type)) {
      unsigned addressSpace = 0;
      Attribute memorySpace = memrefType.getMemorySpace();
      if (auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(memorySpace))
        addressSpace = static_cast<unsigned>(addrSpace.getAddressSpace());
      else if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
        addressSpace = static_cast<unsigned>(intAttr.getInt());
      if (addressSpace == 0)
        addressSpace = static_cast<unsigned>(pto::AddressSpace::GM);
      return llvm::PointerType::get(llvmContext, addressSpace);
    }

    return convertScalarType(type);
  }

  llvm::Value *convertElementOffsetToBytes(llvm::Value *offset, Type elementType) {
    llvm::Value *offsetI32 = castIntegerLikeToI32(offset);
    if (!offsetI32)
      return nullptr;

    unsigned bitWidth = 0;
    if (auto intType = dyn_cast<IntegerType>(elementType))
      bitWidth = intType.getWidth();
    else if (auto floatType = dyn_cast<FloatType>(elementType))
      bitWidth = floatType.getWidth();
    if (bitWidth == 0 || bitWidth % 8 != 0)
      return nullptr;

    return builder.CreateMul(
        offsetI32, llvm::ConstantInt::get(getIntegerType(32), bitWidth / 8),
        "offset.bytes");
  }

  llvm::Value *lookup(Value value) {
    auto it = values.find(value);
    return it == values.end() ? nullptr : it->second;
  }

  void bind(Value key, llvm::Value *value) { values[key] = value; }

  LogicalResult emitFunction(func::FuncOp func) {
    llvm::SmallVector<llvm::Type *, 4> argTypes;
    for (Type argType : func.getFunctionType().getInputs()) {
      llvm::Type *llvmType = convertType(argType);
      if (!llvmType) {
        diagOS << "A5VM emission failed: unsupported function argument type in "
               << func.getName() << "\n";
        return failure();
      }
      argTypes.push_back(llvmType);
    }

    auto *fnType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext), argTypes, false);
    llvm::Function *llvmFunc = llvm::Function::Create(
        fnType, llvm::GlobalValue::ExternalLinkage, func.getName().str(), llvmModule.get());

    auto argIt = llvmFunc->arg_begin();
    for (BlockArgument arg : func.getArguments())
      bind(arg, &*argIt++);

    llvm::BasicBlock *entry =
        llvm::BasicBlock::Create(llvmContext, "entry", llvmFunc);
    builder.SetInsertPoint(entry);

    if (func.getBlocks().size() != 1) {
      diagOS << "A5VM emission failed: only single-block func.func is currently "
                "supported for Abs path\n";
      return failure();
    }

    for (Operation &op : func.getBody().front()) {
      if (failed(emitOperation(&op)))
        return failure();
    }
    if (!builder.GetInsertBlock()->getTerminator())
      builder.CreateRetVoid();

    return success();
  }

  llvm::Function *getOrCreateDeclaration(const std::string &name,
                                         llvm::FunctionType *type) {
    auto it = declarations.find(name);
    if (it != declarations.end())
      return it->second;

    auto *fn = llvm::Function::Create(type, llvm::GlobalValue::ExternalLinkage,
                                      name, llvmModule.get());
    declarations[name] = fn;
    return fn;
  }

  llvm::Value *convertAttrToValue(Attribute attr) {
    if (auto intAttr = dyn_cast<IntegerAttr>(attr))
      return llvm::ConstantInt::get(convertScalarType(intAttr.getType()),
                                    intAttr.getValue());
    if (auto boolAttr = dyn_cast<BoolAttr>(attr))
      return llvm::ConstantInt::get(getIntegerType(1), boolAttr.getValue());
    return nullptr;
  }

  void recordUnresolvedHelper(llvm::StringRef sourceOpName,
                              llvm::StringRef placeholderName,
                              llvm::StringRef candidateName,
                              llvm::ArrayRef<std::string> usedFields,
                              llvm::ArrayRef<std::string> missingFields,
                              llvm::StringRef resultTypeFragment,
                              Operation *anchorOp) {
    unresolvedRecords.push_back(UnresolvedEmissionRecord{
        sourceOpName.str(), placeholderName.str(), candidateName.str(),
        std::vector<std::string>(usedFields.begin(), usedFields.end()),
        std::vector<std::string>(missingFields.begin(), missingFields.end()),
        resultTypeFragment.str(),
        anchorOp ? formatLocationString(anchorOp->getLoc()) : std::string()});
  }

  llvm::Value *emitConstant(arith::ConstantOp op) {
    if (auto intAttr = dyn_cast<IntegerAttr>(op.getValue())) {
      llvm::Type *type = convertScalarType(op.getType());
      return llvm::ConstantInt::get(type, intAttr.getValue());
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(op.getValue())) {
      llvm::Type *type = convertScalarType(op.getType());
      return llvm::ConstantFP::get(type, floatAttr.getValue().convertToDouble());
    }
    return nullptr;
  }

  llvm::Value *emitCastLike(llvm::Value *input, Type resultType) {
    llvm::Type *dstType = convertType(resultType);
    if (!dstType || !input)
      return nullptr;

    if (input->getType() == dstType)
      return input;
    if (input->getType()->isPointerTy() && dstType->isIntegerTy())
      return builder.CreatePtrToInt(input, dstType);
    if (input->getType()->isIntegerTy() && dstType->isPointerTy())
      return builder.CreateIntToPtr(input, dstType);
    if (input->getType()->isPointerTy() && dstType->isPointerTy())
      return builder.CreateAddrSpaceCast(input, dstType);
    if (input->getType()->isIntegerTy() && dstType->isIntegerTy()) {
      unsigned srcWidth = input->getType()->getIntegerBitWidth();
      unsigned dstWidth = dstType->getIntegerBitWidth();
      if (srcWidth == dstWidth)
        return input;
      if (srcWidth < dstWidth)
        return builder.CreateZExt(input, dstType);
      return builder.CreateTrunc(input, dstType);
    }
    return nullptr;
  }

  void attachLoopMetadata(llvm::BranchInst *branch) {
    llvm::Metadata *ops[] = {nullptr,
                             llvm::MDNode::get(llvmContext,
                                               llvm::MDString::get(llvmContext,
                                                                   "llvm.loop.aivector_scope"))};
    auto *loopID = llvm::MDNode::getDistinct(llvmContext, ops);
    loopID->replaceOperandWith(0, loopID);
    branch->setMetadata(llvm::LLVMContext::MD_loop, loopID);
  }

  LogicalResult emitForOp(scf::ForOp op) {
    llvm::Function *fn = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *condBB =
        llvm::BasicBlock::Create(llvmContext, "for.cond", fn);
    llvm::BasicBlock *bodyBB =
        llvm::BasicBlock::Create(llvmContext, "for.body", fn);
    llvm::BasicBlock *exitBB =
        llvm::BasicBlock::Create(llvmContext, "for.exit", fn);

    llvm::Value *lower = lookup(op.getLowerBound());
    llvm::Value *upper = lookup(op.getUpperBound());
    llvm::Value *step = lookup(op.getStep());
    if (!lower || !upper || !step) {
      diagOS << "A5VM emission failed: unresolved loop bounds in scf.for\n";
      return failure();
    }

    builder.CreateBr(condBB);

    builder.SetInsertPoint(condBB);
    auto *iv = builder.CreatePHI(lower->getType(), 2, "iv");
    iv->addIncoming(lower, builder.GetInsertBlock()->getSinglePredecessor());
    llvm::Value *cond = builder.CreateICmpSLT(iv, upper);
    builder.CreateCondBr(cond, bodyBB, exitBB);

    auto saved = lookup(op.getInductionVar());
    bind(op.getInductionVar(), iv);

    builder.SetInsertPoint(bodyBB);
    for (Operation &nested : op.getBody()->without_terminator()) {
      if (failed(emitOperation(&nested)))
        return failure();
    }

    if (!builder.GetInsertBlock()->getTerminator()) {
      llvm::Value *next = builder.CreateAdd(iv, step, "iv.next");
      auto *backedge = builder.CreateBr(condBB);
      if (op->hasAttr("llvm.loop.aivector_scope"))
        attachLoopMetadata(backedge);
      iv->addIncoming(next, builder.GetInsertBlock());
    }

    if (saved)
      bind(op.getInductionVar(), saved);

    builder.SetInsertPoint(exitBB);
    return success();
  }

  LogicalResult emitIfOp(scf::IfOp op) {
    llvm::Value *cond = lookup(op.getCondition());
    if (!cond) {
      diagOS << "A5VM emission failed: unresolved condition in scf.if\n";
      return failure();
    }

    llvm::Function *fn = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock *thenBB =
        llvm::BasicBlock::Create(llvmContext, "if.then", fn);
    llvm::BasicBlock *elseBB =
        llvm::BasicBlock::Create(llvmContext, "if.else", fn);
    llvm::BasicBlock *mergeBB =
        llvm::BasicBlock::Create(llvmContext, "if.end", fn);

    builder.CreateCondBr(cond, thenBB, elseBB);

    builder.SetInsertPoint(thenBB);
    for (Operation &nested : op.getThenRegion().front().without_terminator()) {
      if (failed(emitOperation(&nested)))
        return failure();
    }
    if (!builder.GetInsertBlock()->getTerminator())
      builder.CreateBr(mergeBB);

    builder.SetInsertPoint(elseBB);
    if (!op.getElseRegion().empty()) {
      for (Operation &nested : op.getElseRegion().front().without_terminator()) {
        if (failed(emitOperation(&nested)))
          return failure();
      }
    }
    if (!builder.GetInsertBlock()->getTerminator())
      builder.CreateBr(mergeBB);

    builder.SetInsertPoint(mergeBB);
    return success();
  }

  LogicalResult emitA5VMCall(Operation *op) {
    auto selectionOr = selectIntrinsic(op);
    if (failed(selectionOr)) {
      diagOS << "A5VM emission failed: could not select intrinsic for "
             << op->getName().getStringRef() << "\n";
      return failure();
    }

    IntrinsicSelection selection = *selectionOr;
    intrinsicSelections.push_back(selection);
    if (!selection.resolved)
      unresolvedRecords.push_back(selection.asUnresolvedRecord());

    llvm::SmallVector<llvm::Type *, 8> argTypes;
    llvm::SmallVector<llvm::Value *, 8> argValues;
    llvm::SmallVector<llvm::Value *, 8> rawOperands;
    for (Value operand : op->getOperands()) {
      llvm::Value *llvmOperand = lookup(operand);
      if (!llvmOperand) {
        diagOS << "A5VM emission failed: unresolved operand for "
               << op->getName().getStringRef() << "\n";
        return failure();
      }
      rawOperands.push_back(llvmOperand);
    }

    auto appendArg = [&](llvm::Value *value) {
      argTypes.push_back(value->getType());
      argValues.push_back(value);
    };

    auto packLoopPair = [&](llvm::Value *low, llvm::Value *high) -> llvm::Value * {
      llvm::Value *lhs = builder.CreateShl(
          high, llvm::ConstantInt::get(high->getType(), 40), "loop.pack.hi");
      return builder.CreateOr(lhs, low, "loop.pack");
    };

    auto packLoopSize = [&](llvm::Value *loop2, llvm::Value *loop1)
        -> llvm::Value * {
      llvm::Value *loop2I64 = castIntegerLikeToI64(loop2);
      llvm::Value *loop1I64 = castIntegerLikeToI64(loop1);
      if (!loop2I64 || !loop1I64)
        return nullptr;
      llvm::Value *lhs = builder.CreateShl(
          loop2I64, llvm::ConstantInt::get(getIntegerType(64), 21),
          "loop.size.hi");
      return builder.CreateOr(lhs, loop1I64, "loop.size");
    };

    auto packCopyGmToUbConfig0 = [&](llvm::ArrayRef<llvm::Value *> ops)
        -> llvm::Value * {
      if (ops.size() != 12)
        return nullptr;
      llvm::Value *sid = castIntegerLikeToI64(ops[4]);
      llvm::Value *nBurst = castIntegerLikeToI64(ops[5]);
      llvm::Value *lenBurst = castIntegerLikeToI64(ops[6]);
      llvm::Value *leftPadding = castIntegerLikeToI64(ops[7]);
      llvm::Value *rightPadding = castIntegerLikeToI64(ops[8]);
      llvm::Value *cacheCtl = castIntegerLikeToI64(ops[9]);
      if (!sid || !nBurst || !lenBurst || !leftPadding || !rightPadding ||
          !cacheCtl)
        return nullptr;

      llvm::Value *dataSelect = llvm::ConstantInt::get(getIntegerType(64), 0);
      if (auto attr = dyn_cast_or_null<BoolAttr>(op->getAttr("data_select_bit")))
        dataSelect = llvm::ConstantInt::get(getIntegerType(64), attr.getValue());
      else if (auto attr =
                   dyn_cast_or_null<BoolAttr>(op->getAttr("a5vm.data_select_bit")))
        dataSelect = llvm::ConstantInt::get(getIntegerType(64), attr.getValue());

      llvm::Value *config = sid;
      config = builder.CreateOr(
          config,
          builder.CreateShl(nBurst,
                            llvm::ConstantInt::get(getIntegerType(64), 4)),
          "copy.cfg0.nb");
      config = builder.CreateOr(
          config,
          builder.CreateShl(lenBurst,
                            llvm::ConstantInt::get(getIntegerType(64), 25)),
          "copy.cfg0.lb");
      config = builder.CreateOr(
          config,
          builder.CreateShl(leftPadding,
                            llvm::ConstantInt::get(getIntegerType(64), 46)),
          "copy.cfg0.lpad");
      config = builder.CreateOr(
          config,
          builder.CreateShl(rightPadding,
                            llvm::ConstantInt::get(getIntegerType(64), 52)),
          "copy.cfg0.rpad");
      config = builder.CreateOr(
          config,
          builder.CreateShl(dataSelect,
                            llvm::ConstantInt::get(getIntegerType(64), 58)),
          "copy.cfg0.ds");
      config = builder.CreateOr(
          config,
          builder.CreateShl(cacheCtl,
                            llvm::ConstantInt::get(getIntegerType(64), 60)),
          "copy.cfg0.l2");
      return config;
    };

    auto packCopyGmToUbConfig1 = [&](llvm::ArrayRef<llvm::Value *> ops)
        -> llvm::Value * {
      if (ops.size() != 12)
        return nullptr;
      llvm::Value *gmStride = castIntegerLikeToI64(ops[10]);
      llvm::Value *ubStride = castIntegerLikeToI64(ops[11]);
      if (!gmStride || !ubStride)
        return nullptr;
      return packLoopPair(gmStride, ubStride);
    };

    auto packCopyUbToGmConfig0 = [&](llvm::ArrayRef<llvm::Value *> ops)
        -> llvm::Value * {
      if (ops.size() != 10)
        return nullptr;
      llvm::Value *sid = castIntegerLikeToI64(ops[4]);
      llvm::Value *nBurst = castIntegerLikeToI64(ops[5]);
      llvm::Value *lenBurst = castIntegerLikeToI64(ops[6]);
      llvm::Value *reserved = castIntegerLikeToI64(ops[7]);
      if (!sid || !nBurst || !lenBurst || !reserved)
        return nullptr;

      llvm::Value *config = sid;
      config = builder.CreateOr(
          config,
          builder.CreateShl(nBurst,
                            llvm::ConstantInt::get(getIntegerType(64), 4)),
          "copy.cfg0.nb");
      config = builder.CreateOr(
          config,
          builder.CreateShl(lenBurst,
                            llvm::ConstantInt::get(getIntegerType(64), 25)),
          "copy.cfg0.lb");
      config = builder.CreateOr(
          config,
          builder.CreateShl(reserved,
                            llvm::ConstantInt::get(getIntegerType(64), 60)),
          "copy.cfg0.rsrv");
      return config;
    };

    auto packCopyUbToGmConfig1 = [&](llvm::ArrayRef<llvm::Value *> ops)
        -> llvm::Value * {
      if (ops.size() != 10)
        return nullptr;
      llvm::Value *dstStride = castIntegerLikeToI64(ops[8]);
      llvm::Value *srcStride = castIntegerLikeToI64(ops[9]);
      if (!dstStride || !srcStride)
        return nullptr;
      return packLoopPair(dstStride, srcStride);
    };

    auto appendSyncImmediateTriplet =
        [&](llvm::StringRef srcPipe, llvm::StringRef dstPipe,
            llvm::StringRef eventId) -> LogicalResult {
      auto srcImm = parsePipeImmediate(srcPipe);
      auto dstImm = parsePipeImmediate(dstPipe);
      auto eventImm = parseEventImmediate(eventId);
      if (!srcImm || !dstImm || !eventImm) {
        diagOS << "A5VM emission failed: could not encode sync immediates for "
               << op->getName().getStringRef() << "\n";
        return failure();
      }
      appendArg(llvm::ConstantInt::get(getIntegerType(64), *srcImm));
      appendArg(llvm::ConstantInt::get(getIntegerType(64), *dstImm));
      appendArg(llvm::ConstantInt::get(getIntegerType(64), *eventImm));
      return success();
    };

    auto appendBarrierImmediate = [&](llvm::StringRef pipe) -> LogicalResult {
      auto pipeImm = parsePipeImmediate(pipe);
      if (!pipeImm) {
        diagOS << "A5VM emission failed: could not encode barrier immediate for "
               << op->getName().getStringRef() << "\n";
        return failure();
      }
      appendArg(llvm::ConstantInt::get(getIntegerType(64), *pipeImm));
      return success();
    };

    if (isa<a5vm::SetLoop2StrideOutToUbOp, a5vm::SetLoop1StrideOutToUbOp,
            a5vm::SetLoop2StrideUbToOutOp, a5vm::SetLoop1StrideUbToOutOp>(op)) {
      if (rawOperands.size() != 2) {
        diagOS << "A5VM emission failed: expected two operands for loop config op\n";
        return failure();
      }
      appendArg(packLoopPair(rawOperands[0], rawOperands[1]));
    } else if (isa<a5vm::SetLoopSizeOutToUbOp, a5vm::SetLoopSizeUbToOutOp>(op)) {
      if (rawOperands.size() != 2) {
        diagOS << "A5VM emission failed: expected two operands for loop size op\n";
        return failure();
      }
      llvm::Value *packed = packLoopSize(rawOperands[0], rawOperands[1]);
      if (!packed) {
        diagOS << "A5VM emission failed: could not pack loop size op\n";
        return failure();
      }
      appendArg(packed);
    } else if (isa<a5vm::CopyGmToUbufOp>(op)) {
      if (rawOperands.size() != 12) {
        diagOS << "A5VM emission failed: expected twelve operands for copy_gm_to_ubuf\n";
        return failure();
      }
      llvm::Value *config0 = packCopyGmToUbConfig0(rawOperands);
      llvm::Value *config1 = packCopyGmToUbConfig1(rawOperands);
      if (!config0 || !config1) {
        diagOS << "A5VM emission failed: could not pack copy_gm_to_ubuf config\n";
        return failure();
      }
      appendArg(rawOperands[1]);
      appendArg(rawOperands[0]);
      appendArg(config0);
      appendArg(config1);
    } else if (isa<a5vm::CopyUbufToGmOp>(op)) {
      if (rawOperands.size() != 10) {
        diagOS << "A5VM emission failed: expected ten operands for copy_ubuf_to_gm\n";
        return failure();
      }
      llvm::Value *config0 = packCopyUbToGmConfig0(rawOperands);
      llvm::Value *config1 = packCopyUbToGmConfig1(rawOperands);
      if (!config0 || !config1) {
        diagOS << "A5VM emission failed: could not pack copy_ubuf_to_gm config\n";
        return failure();
      }
      appendArg(rawOperands[1]);
      appendArg(rawOperands[0]);
      appendArg(config0);
      appendArg(config1);
    } else if (auto setFlag = dyn_cast<a5vm::SetFlagOp>(op)) {
      if (failed(appendSyncImmediateTriplet(setFlag.getSrcPipe(),
                                            setFlag.getDstPipe(),
                                            setFlag.getEventId())))
        return failure();
    } else if (auto waitFlag = dyn_cast<a5vm::WaitFlagOp>(op)) {
      if (failed(appendSyncImmediateTriplet(waitFlag.getSrcPipe(),
                                            waitFlag.getDstPipe(),
                                            waitFlag.getEventId())))
        return failure();
    } else if (auto barrier = dyn_cast<a5vm::PipeBarrierOp>(op)) {
      if (failed(appendBarrierImmediate(barrier.getPipe())))
        return failure();
    } else if (isa<a5vm::PltB32Op>(op)) {
      if (rawOperands.size() != 1) {
        diagOS << "A5VM emission failed: expected one operand for plt_b32\n";
        return failure();
      }
      llvm::Value *laneCount = castIntegerLikeToI32(rawOperands[0]);
      if (!laneCount) {
        diagOS << "A5VM emission failed: could not cast plt_b32 lane count\n";
        return failure();
      }
      appendArg(laneCount);
    } else if (auto vlds = dyn_cast<a5vm::VldsOp>(op)) {
      if (rawOperands.size() != 2) {
        diagOS << "A5VM emission failed: expected two operands for vlds\n";
        return failure();
      }
      llvm::Value *offsetBytes =
          convertElementOffsetToBytes(rawOperands[1],
                                      cast<a5vm::VecType>(vlds.getResult().getType())
                                          .getElementType());
      auto distImm = parseLoadDistImmediate(vlds.getDistAttr() ? *vlds.getDist() : "");
      if (!offsetBytes || !distImm) {
        diagOS << "A5VM emission failed: could not encode vlds offset/dist\n";
        return failure();
      }
      appendArg(rawOperands[0]);
      appendArg(offsetBytes);
      appendArg(llvm::ConstantInt::get(getIntegerType(32), *distImm));
      appendArg(llvm::ConstantInt::get(getIntegerType(32), 0));
    } else if (auto vldsPost = dyn_cast<a5vm::VldsPostOp>(op)) {
      if (rawOperands.size() != 2) {
        diagOS << "A5VM emission failed: expected two operands for vlds_post\n";
        return failure();
      }
      llvm::Value *offsetBytes =
          convertElementOffsetToBytes(rawOperands[1],
                                      cast<a5vm::VecType>(vldsPost.getResult().getType())
                                          .getElementType());
      auto distImm =
          parseLoadDistImmediate(vldsPost.getDistAttr() ? *vldsPost.getDist() : "NORM");
      if (!offsetBytes || !distImm) {
        diagOS << "A5VM emission failed: could not encode vlds_post offset/dist\n";
        return failure();
      }
      appendArg(rawOperands[0]);
      appendArg(offsetBytes);
      appendArg(llvm::ConstantInt::get(getIntegerType(32), *distImm));
      appendArg(llvm::ConstantInt::get(getIntegerType(32), 1));
    } else if (auto vsts = dyn_cast<a5vm::VstsOp>(op)) {
      if (rawOperands.size() != 4) {
        diagOS << "A5VM emission failed: expected four operands for vsts\n";
        return failure();
      }
      llvm::Value *offsetBytes =
          convertElementOffsetToBytes(rawOperands[2],
                                      cast<a5vm::VecType>(vsts.getValue().getType())
                                          .getElementType());
      auto distImm = parseStoreDistImmediate(
          vsts.getValue().getType(), vsts.getDistAttr() ? *vsts.getDist() : "");
      if (!offsetBytes || !distImm) {
        diagOS << "A5VM emission failed: could not encode vsts offset/dist\n";
        return failure();
      }
      appendArg(rawOperands[0]);
      appendArg(rawOperands[1]);
      appendArg(offsetBytes);
      appendArg(llvm::ConstantInt::get(getIntegerType(32), *distImm));
      appendArg(llvm::ConstantInt::get(getIntegerType(32), 0));
      appendArg(rawOperands[3]);
    } else if (auto vstsPost = dyn_cast<a5vm::VstsPostOp>(op)) {
      if (rawOperands.size() != 4) {
        diagOS << "A5VM emission failed: expected four operands for vsts_post\n";
        return failure();
      }
      llvm::Value *offsetBytes =
          convertElementOffsetToBytes(rawOperands[2],
                                      cast<a5vm::VecType>(vstsPost.getValue().getType())
                                          .getElementType());
      auto distImm = parseStoreDistImmediate(
          vstsPost.getValue().getType(),
          vstsPost.getDistAttr() ? *vstsPost.getDist() : "");
      if (!offsetBytes || !distImm) {
        diagOS << "A5VM emission failed: could not encode vsts_post offset/dist\n";
        return failure();
      }
      appendArg(rawOperands[0]);
      appendArg(rawOperands[1]);
      appendArg(offsetBytes);
      appendArg(llvm::ConstantInt::get(getIntegerType(32), *distImm));
      appendArg(llvm::ConstantInt::get(getIntegerType(32), 1));
      appendArg(rawOperands[3]);
    } else {
      for (llvm::Value *operand : rawOperands)
        appendArg(operand);
    }

    auto addAttrOperand = [&](Attribute attr) {
      if (!attr)
        return;
      if (llvm::Value *value = convertAttrToValue(attr)) {
        appendArg(value);
      }
    };

    bool skipGenericAttrs =
        isa<a5vm::CopyGmToUbufOp, a5vm::CopyUbufToGmOp, a5vm::SetFlagOp,
            a5vm::WaitFlagOp, a5vm::PipeBarrierOp, a5vm::PltB32Op,
            a5vm::VldsOp, a5vm::VldsPostOp, a5vm::VstsOp, a5vm::VstsPostOp>(op);
    for (NamedAttribute attr : op->getAttrs()) {
      if (skipGenericAttrs)
        continue;
      if (attr.getName() == "operandSegmentSizes")
        continue;
      addAttrOperand(attr.getValue());
    }

    llvm::Type *resultType = nullptr;
    if (auto plt = dyn_cast<a5vm::PltB32Op>(op)) {
      llvm::Type *maskType = convertType(plt.getMask().getType());
      llvm::Type *scalarOutType = convertType(plt.getScalarOut().getType());
      if (!maskType || !scalarOutType) {
        diagOS << "A5VM emission failed: unsupported result type for "
               << op->getName().getStringRef() << "\n";
        return failure();
      }
      resultType = llvm::StructType::get(llvmContext, {maskType, scalarOutType});
    } else if (auto vldsPost = dyn_cast<a5vm::VldsPostOp>(op)) {
      llvm::Type *vecType = convertType(vldsPost.getResult().getType());
      llvm::Type *ptrType = convertType(vldsPost.getUpdatedSource().getType());
      if (!vecType || !ptrType) {
        diagOS << "A5VM emission failed: unsupported result type for "
               << op->getName().getStringRef() << "\n";
        return failure();
      }
      resultType = llvm::StructType::get(llvmContext, {vecType, ptrType});
    } else if (op->getNumResults() == 1) {
      resultType = convertType(op->getResult(0).getType());
      if (!resultType) {
        diagOS << "A5VM emission failed: unsupported result type for "
               << op->getName().getStringRef() << "\n";
        return failure();
      }
    } else if (op->getNumResults() > 1) {
      diagOS << "A5VM emission failed: multi-result A5VM calls are not yet "
                "supported in the Abs path\n";
      return failure();
    } else {
      resultType = llvm::Type::getVoidTy(llvmContext);
    }

    auto *fnType = llvm::FunctionType::get(resultType, argTypes, false);
    llvm::Function *callee =
        getOrCreateDeclaration(selection.getEmittedCallee(), fnType);
    llvm::CallInst *call = builder.CreateCall(callee, argValues);
    if (auto plt = dyn_cast<a5vm::PltB32Op>(op)) {
      bind(plt.getMask(), builder.CreateExtractValue(call, {0}));
      bind(plt.getScalarOut(), builder.CreateExtractValue(call, {1}));
    } else if (auto vldsPost = dyn_cast<a5vm::VldsPostOp>(op)) {
      bind(vldsPost.getResult(), builder.CreateExtractValue(call, {0}));
      bind(vldsPost.getUpdatedSource(), builder.CreateExtractValue(call, {1}));
    } else if (op->getNumResults() == 1) {
      bind(op->getResult(0), call);
    }
    return success();
  }

  LogicalResult emitOperation(Operation *op) {
    if (isa<scf::YieldOp>(op))
      return success();

    if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
      llvm::Value *value = emitConstant(constant);
      if (!value) {
        diagOS << "A5VM emission failed: unsupported arith.constant in Abs path\n";
        return failure();
      }
      bind(constant.getResult(), value);
      return success();
    }

    if (auto extract = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(op)) {
      llvm::Value *source = lookup(extract.getSource());
      if (!source) {
        diagOS << "A5VM emission failed: unresolved memref source for "
                  "extract_aligned_pointer_as_index\n";
        return failure();
      }
      bind(extract.getResult(),
           builder.CreatePtrToInt(source, getIntegerType(64), "ptrtoint"));
      return success();
    }

    if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(op)) {
      if (reinterpretCast.getResult().use_empty()) {
        bind(reinterpretCast.getResult(), nullptr);
        return success();
      }
      llvm::Value *source = lookup(reinterpretCast.getSource());
      if (!source)
        return failure();
      bind(reinterpretCast.getResult(), source);
      return success();
    }

    if (auto cast = dyn_cast<arith::IndexCastUIOp>(op)) {
      llvm::Value *input = lookup(cast.getIn());
      llvm::Value *result = emitCastLike(input, cast.getType());
      if (!result) {
        diagOS << "A5VM emission failed: unsupported arith.index_castui\n";
        return failure();
      }
      bind(cast.getResult(), result);
      return success();
    }

    if (auto muli = dyn_cast<arith::MulIOp>(op)) {
      bind(muli.getResult(),
           builder.CreateMul(lookup(muli.getLhs()), lookup(muli.getRhs()), "mul"));
      return success();
    }

    if (auto addi = dyn_cast<arith::AddIOp>(op)) {
      bind(addi.getResult(),
           builder.CreateAdd(lookup(addi.getLhs()), lookup(addi.getRhs()), "add"));
      return success();
    }

    if (auto subi = dyn_cast<arith::SubIOp>(op)) {
      bind(subi.getResult(),
           builder.CreateSub(lookup(subi.getLhs()), lookup(subi.getRhs()), "sub"));
      return success();
    }

    if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
      llvm::CmpInst::Predicate predicate;
      switch (cmpi.getPredicate()) {
      case arith::CmpIPredicate::slt:
        predicate = llvm::CmpInst::ICMP_SLT;
        break;
      default:
        diagOS << "A5VM emission failed: unsupported arith.cmpi predicate in Abs "
                  "path\n";
        return failure();
      }
      bind(cmpi.getResult(),
           builder.CreateICmp(predicate, lookup(cmpi.getLhs()),
                              lookup(cmpi.getRhs()), "icmp"));
      return success();
    }

    if (auto select = dyn_cast<arith::SelectOp>(op)) {
      bind(select.getResult(),
           builder.CreateSelect(lookup(select.getCondition()),
                                lookup(select.getTrueValue()),
                                lookup(select.getFalseValue()), "select"));
      return success();
    }

    if (auto ptrToInt = dyn_cast<LLVM::PtrToIntOp>(op)) {
      llvm::Value *input = lookup(ptrToInt.getArg());
      llvm::Value *result = emitCastLike(input, ptrToInt.getType());
      if (!result)
        return failure();
      bind(ptrToInt.getRes(), result);
      return success();
    }

    if (auto intToPtr = dyn_cast<LLVM::IntToPtrOp>(op)) {
      llvm::Value *input = lookup(intToPtr.getArg());
      llvm::Value *result = emitCastLike(input, intToPtr.getType());
      if (!result)
        return failure();
      bind(intToPtr.getRes(), result);
      return success();
    }

    if (auto forOp = dyn_cast<scf::ForOp>(op))
      return emitForOp(forOp);

    if (auto ifOp = dyn_cast<scf::IfOp>(op))
      return emitIfOp(ifOp);

    if (isa<func::ReturnOp>(op)) {
      if (!builder.GetInsertBlock()->getTerminator())
        builder.CreateRetVoid();
      return success();
    }

    if (op->getName().getDialectNamespace() == "a5vm")
      return emitA5VMCall(op);

    diagOS << "A5VM emission failed: unsupported op in Abs path: "
           << op->getName().getStringRef() << "\n";
    return failure();
  }
};

} // namespace

LogicalResult translateA5VMModuleToText(ModuleOp module, llvm::raw_ostream &os,
                                        const A5VMEmissionOptions &options,
                                        llvm::raw_ostream &diagOS) {
  LLVMTextEmitter emitter(module, diagOS, options);
  return emitter.emitTo(os);
}

} // namespace mlir::pto
