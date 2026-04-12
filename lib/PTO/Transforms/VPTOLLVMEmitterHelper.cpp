//===- VPTOLLVMEmitterHelper.cpp - VPTO LLVM emission helpers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOLLVMEmitterHelper.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace mlir;

namespace mlir::pto {
namespace {

constexpr StringLiteral kAIVScopeDummyCallee = "aivscope_dummy";

struct QueriedTargetAttrs {
  std::string targetCPU;
  std::string targetFeatures;
};

static bool hasPtoMemRefMemorySpace(Type type) {
  if (auto memRefType = dyn_cast<MemRefType>(type))
    return isa<pto::AddressSpaceAttr>(memRefType.getMemorySpace());
  if (auto functionType = dyn_cast<FunctionType>(type))
    return llvm::any_of(functionType.getInputs(), hasPtoMemRefMemorySpace) ||
           llvm::any_of(functionType.getResults(), hasPtoMemRefMemorySpace);
  return false;
}

static bool hasPtoMemRefMemorySpace(TypeRange types) {
  return llvm::any_of(types, [](Type type) {
    return hasPtoMemRefMemorySpace(type);
  });
}

struct ConvertPtoMemRefSpaceCarrierOp final : ConversionPattern {
  ConvertPtoMemRefSpaceCarrierOp(TypeConverter &typeConverter,
                                 MLIRContext *context)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!hasPtoMemRefMemorySpace(op->getOperandTypes()) &&
        !hasPtoMemRefMemorySpace(op->getResultTypes()))
      return failure();
    if (op->getNumRegions() != 0)
      return rewriter.notifyMatchFailure(
          op, "region ops with PTO memref spaces are handled structurally");

    FailureOr<Operation *> converted =
        convertOpResultTypes(op, operands, *typeConverter, rewriter);
    if (failed(converted))
      return failure();
    return success();
  }
};

struct ConvertMemRefReinterpretCastSpaceOp final
    : OpConversionPattern<memref::ReinterpretCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getType());
    auto memRefResultType = dyn_cast_or_null<MemRefType>(convertedResultType);
    if (!memRefResultType)
      return rewriter.notifyMatchFailure(op, "expected memref result type");

    rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
        op, memRefResultType, adaptor.getSource(), adaptor.getOffsets(),
        adaptor.getSizes(), adaptor.getStrides(), op.getStaticOffsets(),
        op.getStaticSizes(), op.getStaticStrides());
    return success();
  }
};

struct ConvertMemRefSubViewSpaceOp final
    : OpConversionPattern<memref::SubViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedResultType = getTypeConverter()->convertType(op.getType());
    auto memRefResultType = dyn_cast_or_null<MemRefType>(convertedResultType);
    if (!memRefResultType)
      return rewriter.notifyMatchFailure(op, "expected memref result type");

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op, memRefResultType, adaptor.getSource(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    return success();
  }
};

struct ConvertMemRefSpaceUnrealizedCastOp final
    : OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    if (!hasPtoMemRefMemorySpace(op->getOperandTypes()) &&
        !hasPtoMemRefMemorySpace(op->getResultTypes()))
      return failure();

    Type convertedResultType =
        getTypeConverter()->convertType(op.getResult(0).getType());
    if (!convertedResultType)
      return failure();

    Value input = adaptor.getOperands().front();
    if (input.getType() == convertedResultType) {
      rewriter.replaceOp(op, input);
      return success();
    }
    return failure();
  }
};

static void ensureAIVScopeDummyDecl(ModuleOp module) {
  SymbolTable symbolTable(module);
  if (symbolTable.lookup<func::FuncOp>(kAIVScopeDummyCallee))
    return;

  OpBuilder builder(module.getBodyRegion());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType = builder.getFunctionType(TypeRange{}, TypeRange{});
  auto dummy = builder.create<func::FuncOp>(module.getLoc(),
                                            kAIVScopeDummyCallee, funcType);
  dummy.setPrivate();
}

static bool satisfiesAIVectorScopeLatchPostcondition(llvm::Loop *loop) {
  llvm::BasicBlock *latch = loop->getLoopLatch();
  if (!latch)
    return false;

  llvm::SmallVector<llvm::BasicBlock *, 4> preds(llvm::predecessors(latch));
  if (preds.size() != 1)
    return false;

  auto *predTerm = preds.front()->getTerminator();
  return predTerm && predTerm->getNumSuccessors() == 1 &&
         predTerm->getSuccessor(0) == latch;
}

static LogicalResult ensureDummyPredForAIVectorScopeLatch(
    llvm::Loop *loop, llvm::raw_ostream &diagOS) {
  if (satisfiesAIVectorScopeLatchPostcondition(loop))
    return success();

  llvm::BasicBlock *latch = loop->getLoopLatch();
  if (!latch) {
    diagOS << "VPTO LLVM emission failed: aivscope loop is missing a latch\n";
    return failure();
  }

  llvm::SmallVector<llvm::BasicBlock *, 4> preds(llvm::predecessors(latch));
  if (preds.empty()) {
    diagOS << "VPTO LLVM emission failed: aivscope latch has no predecessor\n";
    return failure();
  }

  auto *dummy = llvm::SplitBlockPredecessors(
      latch, preds, "aivscope.dummy", static_cast<llvm::DominatorTree *>(nullptr),
      static_cast<llvm::LoopInfo *>(nullptr), nullptr, /*PreserveLCSSA=*/false);
  if (!dummy) {
    diagOS << "VPTO LLVM emission failed: failed to normalize aivscope latch "
              "predecessors\n";
    return failure();
  }

  if (!satisfiesAIVectorScopeLatchPostcondition(loop)) {
    diagOS << "VPTO LLVM emission failed: normalized aivscope latch still does "
              "not satisfy the single-predecessor/single-successor contract\n";
    return failure();
  }
  return success();
}

static FailureOr<std::string> extractQuotedLLVMFnAttr(llvm::StringRef ir,
                                                      llvm::StringRef key) {
  std::string pattern = "\"";
  pattern += key.str();
  pattern += "\"=\"";
  size_t start = ir.find(pattern);
  if (start == llvm::StringRef::npos)
    return failure();
  start += pattern.size();
  size_t end = ir.find('"', start);
  if (end == llvm::StringRef::npos || end <= start)
    return failure();
  return ir.slice(start, end).str();
}

static FailureOr<QueriedTargetAttrs>
queryDefaultTargetAttrs(const VPTOEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  static llvm::StringMap<QueriedTargetAttrs> cache;

  if (options.targetTriple.empty() || options.march.empty() ||
      options.aicoreArch.empty()) {
    diagOS << "VPTO LLVM emission failed: missing target query options\n";
    return failure();
  }

  std::string cacheKey =
      options.targetTriple + "|" + options.march + "|" + options.aicoreArch;
  if (auto it = cache.find(cacheKey); it != cache.end())
    return it->second;

  auto bisheng = llvm::sys::findProgramByName("bisheng");
  if (!bisheng) {
    diagOS << "VPTO LLVM emission failed: unable to find 'bisheng' in PATH\n";
    return failure();
  }
  const std::string &bishengPath = *bisheng;

  llvm::SmallString<64> inputPath;
  llvm::SmallString<64> outputPath;
  int inputFD = -1;
  int outputFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-vpto-target-query",
                                                   "c", inputFD, inputPath)) {
    diagOS << "VPTO LLVM emission failed: cannot create bisheng query input: "
           << ec.message() << "\n";
    return failure();
  }
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-vpto-target-query",
                                                   "ll", outputFD, outputPath)) {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
    diagOS << "VPTO LLVM emission failed: cannot create bisheng query output: "
           << ec.message() << "\n";
    return failure();
  }

  auto cleanup = llvm::make_scope_exit([&]() {
    llvm::sys::fs::remove(inputPath);
    llvm::sys::fs::remove(outputPath);
  });

  {
    llvm::raw_fd_ostream inputOS(inputFD, /*shouldClose=*/false);
    inputOS << "void f(void) {}\n";
  }
  llvm::sys::Process::SafelyCloseFileDescriptor(inputFD);
  llvm::sys::Process::SafelyCloseFileDescriptor(outputFD);

  llvm::SmallString<128> stderrPath;
  int stderrFD = -1;
  if (auto ec = llvm::sys::fs::createTemporaryFile("ptoas-vpto-target-query",
                                                   "stderr", stderrFD,
                                                   stderrPath)) {
    diagOS << "VPTO LLVM emission failed: cannot create bisheng query stderr: "
           << ec.message() << "\n";
    return failure();
  }
  auto stderrCleanup = llvm::make_scope_exit([&]() {
    llvm::sys::fs::remove(stderrPath);
  });
  llvm::sys::Process::SafelyCloseFileDescriptor(stderrFD);

  llvm::SmallVector<std::string> argStorage = {
      bishengPath,
      ("--target=" + options.targetTriple),
      ("-march=" + options.march),
      ("--cce-aicore-arch=" + options.aicoreArch),
      "--cce-aicore-only",
      "-x",
      "c",
      inputPath.str().str(),
      "-S",
      "-emit-llvm",
      "-o",
      outputPath.str().str(),
  };
  llvm::SmallVector<llvm::StringRef> args;
  args.reserve(argStorage.size());
  for (const std::string &arg : argStorage)
    args.push_back(arg);

  std::string execErr;
  bool execFailed = false;
  int rc = llvm::sys::ExecuteAndWait(
      bishengPath, args, std::nullopt,
      {std::nullopt, std::nullopt, llvm::StringRef(stderrPath)}, 0, 0,
      &execErr, &execFailed);

  auto stderrBuffer = llvm::MemoryBuffer::getFile(stderrPath);
  llvm::StringRef stderrText =
      stderrBuffer ? stderrBuffer.get()->getBuffer() : llvm::StringRef();

  if (execFailed || rc != 0) {
    diagOS << "VPTO LLVM emission failed: bisheng target query failed\n";
    diagOS << "Command:";
    for (llvm::StringRef arg : args)
      diagOS << " " << arg;
    diagOS << "\n";
    if (!execErr.empty())
      diagOS << execErr << "\n";
    if (!stderrText.empty())
      diagOS << stderrText << "\n";
    return failure();
  }

  auto outputBuffer = llvm::MemoryBuffer::getFile(outputPath);
  if (!outputBuffer) {
    diagOS << "VPTO LLVM emission failed: cannot read bisheng query output\n";
    return failure();
  }

  FailureOr<std::string> targetCPU =
      extractQuotedLLVMFnAttr(outputBuffer.get()->getBuffer(), "target-cpu");
  FailureOr<std::string> targetFeatures =
      extractQuotedLLVMFnAttr(outputBuffer.get()->getBuffer(), "target-features");
  if (failed(targetCPU) || failed(targetFeatures)) {
    diagOS << "VPTO LLVM emission failed: cannot parse bisheng target attrs\n";
    diagOS << outputBuffer.get()->getBuffer() << "\n";
    return failure();
  }

  QueriedTargetAttrs attrs{*targetCPU, *targetFeatures};
  cache[cacheKey] = attrs;
  return attrs;
}

} // namespace

LogicalResult normalizePtoMemRefSpaces(ModuleOp module,
                                       llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion([&](MemRefType type) -> Type {
    auto addrSpace = dyn_cast_or_null<pto::AddressSpaceAttr>(type.getMemorySpace());
    if (!addrSpace)
      return type;
    return MemRefType::get(
        type.getShape(), type.getElementType(), type.getLayout(),
        IntegerAttr::get(IntegerType::get(context, 64),
                         static_cast<int64_t>(addrSpace.getAddressSpace())));
  });
  typeConverter.addTypeAttributeConversion(
      [](MemRefType, pto::AddressSpaceAttr attr) -> Attribute {
        return IntegerAttr::get(IntegerType::get(attr.getContext(), 64),
                                static_cast<int64_t>(attr.getAddressSpace()));
      });
  auto materializeMemRefCast = [](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    return builder
        .create<UnrealizedConversionCastOp>(loc, TypeRange{resultType}, inputs)
        .getResult(0);
  };
  typeConverter.addSourceMaterialization(materializeMemRefCast);
  typeConverter.addTargetMaterialization(materializeMemRefCast);
  typeConverter.addArgumentMaterialization(materializeMemRefCast);

  ConversionTarget target(*context);
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp>(
      [&](func::CallOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<func::ReturnOp>(
      [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) {
        return isLegalForBranchOpInterfaceTypeConversionPattern(op,
                                                                typeConverter);
      });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });

  RewritePatternSet patterns(context);
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
  patterns.add<ConvertMemRefReinterpretCastSpaceOp, ConvertMemRefSubViewSpaceOp,
               ConvertMemRefSpaceUnrealizedCastOp>(typeConverter, context);
  patterns.add<ConvertPtoMemRefSpaceCarrierOp>(typeConverter, context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: memref address-space normalization "
              "failed\n";
    return failure();
  }

  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1)
      return;
    if (!hasPtoMemRefMemorySpace(castOp->getOperandTypes()) &&
        !hasPtoMemRefMemorySpace(castOp->getResultTypes()))
      return;
    Type convertedResultType =
        typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType &&
        convertedResultType == castOp.getOperand(0).getType())
      castsToFold.push_back(castOp);
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }

  WalkResult leftover = module.walk([&](Operation *op) {
    if (hasPtoMemRefMemorySpace(op->getOperandTypes()) ||
        hasPtoMemRefMemorySpace(op->getResultTypes())) {
      diagOS << "VPTO LLVM emission failed: residual PTO memref address space "
                "on op "
             << op->getName().getStringRef() << "\n";
      op->print(diagOS);
      diagOS << "\n";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (leftover.wasInterrupted())
    return failure();
  return success();
}

void materializeVecScopeCarrierLoops(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  (void)ctx->getOrLoadDialect<arith::ArithDialect>();
  (void)ctx->getOrLoadDialect<scf::SCFDialect>();
  ensureAIVScopeDummyDecl(module);

  SmallVector<pto::VecScopeOp, 16> scopes;
  module.walk([&](pto::VecScopeOp vecScope) { scopes.push_back(vecScope); });

  IRRewriter rewriter(module.getContext());
  for (pto::VecScopeOp vecScope : llvm::reverse(scopes)) {
    if (!vecScope || vecScope.getBody().empty())
      continue;

    rewriter.setInsertionPoint(vecScope);
    auto loc = vecScope.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp carrier = rewriter.create<scf::ForOp>(loc, c0, c1, c1);

    Block &vecScopeBody = vecScope.getBody().front();
    Block *carrierBody = carrier.getBody();
    Operation *yield = carrierBody->getTerminator();
    carrierBody->getOperations().splice(Block::iterator(yield),
                                        vecScopeBody.getOperations(),
                                        vecScopeBody.begin(),
                                        vecScopeBody.end());
    rewriter.setInsertionPoint(yield);
    rewriter.create<func::CallOp>(loc, kAIVScopeDummyCallee, TypeRange{},
                                  ValueRange{});
    rewriter.eraseOp(vecScope);
  }

  SmallVector<pto::StrictVecScopeOp, 16> strictScopes;
  module.walk([&](pto::StrictVecScopeOp strictVecScope) {
    strictScopes.push_back(strictVecScope);
  });

  for (pto::StrictVecScopeOp strictVecScope : llvm::reverse(strictScopes)) {
    if (!strictVecScope || strictVecScope.getBody().empty())
      continue;

    rewriter.setInsertionPoint(strictVecScope);
    auto loc = strictVecScope.getLoc();
    Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    scf::ForOp carrier = rewriter.create<scf::ForOp>(loc, c0, c1, c1);

    Block &strictBody = strictVecScope.getBody().front();
    Block *carrierBody = carrier.getBody();
    Operation *yield = carrierBody->getTerminator();

    IRMapping mapping;
    for (auto [blockArg, capture] :
         llvm::zip(strictBody.getArguments(), strictVecScope.getCaptures()))
      mapping.map(blockArg, capture);

    rewriter.setInsertionPoint(yield);
    for (Operation &nested : strictBody.getOperations())
      rewriter.clone(nested, mapping);
    rewriter.create<func::CallOp>(loc, kAIVScopeDummyCallee, TypeRange{},
                                  ValueRange{});

    rewriter.eraseOp(strictVecScope);
  }
}

LogicalResult attachAIVectorScopeMetadata(llvm::Module &llvmModule,
                                          llvm::raw_ostream &diagOS) {
  llvm::Function *dummyCallee = llvmModule.getFunction(kAIVScopeDummyCallee);
  if (!dummyCallee)
    return success();

  for (llvm::Function &function : llvmModule) {
    if (function.isDeclaration())
      continue;
    llvm::DominatorTree dt(function);
    llvm::LoopInfo loopInfo(dt);

    llvm::SmallVector<llvm::CallInst *, 4> dummyCalls;
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto *call = dyn_cast<llvm::CallInst>(&inst);
        if (call && call->getCalledFunction() == dummyCallee)
          dummyCalls.push_back(call);
      }
    }

    for (llvm::CallInst *dummyCall : dummyCalls) {
      llvm::BasicBlock *markedBlock = dummyCall->getParent();
      llvm::Loop *loop = loopInfo.getLoopFor(markedBlock);
      if (!loop) {
        diagOS << "VPTO LLVM emission failed: aivscope_dummy in function "
               << function.getName() << " does not belong to an LLVM loop\n";
        return failure();
      }

      if (markedBlock == loop->getLoopLatch() &&
          dummyCall != markedBlock->getTerminator()) {
        markedBlock->splitBasicBlock(dummyCall->getIterator(), "aivscope.latch");
        dt.recalculate(function);
        loopInfo.releaseMemory();
        loopInfo.analyze(dt);
        markedBlock = dummyCall->getParent();
        loop = loopInfo.getLoopFor(markedBlock);
        if (!loop) {
          diagOS << "VPTO LLVM emission failed: split aivscope latch in "
                 << function.getName()
                 << " no longer belongs to an LLVM loop\n";
          return failure();
        }
      }

      if (failed(ensureDummyPredForAIVectorScopeLatch(loop, diagOS)))
        return failure();

      dt.recalculate(function);
      loopInfo.releaseMemory();
      loopInfo.analyze(dt);
      loop = loopInfo.getLoopFor(markedBlock);
      if (!loop) {
        diagOS << "VPTO LLVM emission failed: aivscope_dummy in function "
               << function.getName()
               << " lost its loop after latch normalization\n";
        return failure();
      }

      llvm::BasicBlock *latch = loop->getLoopLatch();
      auto *branch = dyn_cast_or_null<llvm::BranchInst>(
          latch ? latch->getTerminator() : nullptr);
      if (!branch || branch->isConditional()) {
        diagOS << "VPTO LLVM emission failed: normalized aivscope loop in "
               << function.getName()
               << " does not have an unconditional latch backedge\n";
        return failure();
      }

      llvm::LLVMContext &ctx = llvmModule.getContext();
      llvm::Metadata *ops[] = {
          nullptr, llvm::MDNode::get(ctx, llvm::MDString::get(ctx, "llvm.loop.aivector_scope"))};
      auto *loopID = llvm::MDNode::getDistinct(ctx, ops);
      loopID->replaceOperandWith(0, loopID);
      branch->setMetadata(llvm::LLVMContext::MD_loop, loopID);
      dummyCall->eraseFromParent();
    }
  }

  if (dummyCallee->use_empty())
    dummyCallee->eraseFromParent();
  return success();
}

void attachHIVMKernelAnnotations(llvm::Module &llvmModule) {
  llvm::NamedMDNode *annotations =
      llvmModule.getOrInsertNamedMetadata("hivm.annotations");
  llvm::LLVMContext &ctx = llvmModule.getContext();
  llvm::Type *i32Ty = llvm::Type::getInt32Ty(ctx);
  llvm::Constant *one = llvm::ConstantInt::get(i32Ty, 1);

  auto addAnnotation = [&](llvm::Function &function, llvm::StringRef kind) {
    llvm::Metadata *ops[] = {
        llvm::ValueAsMetadata::get(&function),
        llvm::MDString::get(ctx, kind),
        llvm::ConstantAsMetadata::get(one)};
    annotations->addOperand(llvm::MDNode::get(ctx, ops));
  };

  for (llvm::Function &function : llvmModule) {
    if (function.isDeclaration())
      continue;
    if (function.getLinkage() != llvm::GlobalValue::ExternalLinkage)
      continue;

    llvm::StringRef name = function.getName();
    if (name.contains(".extracted") || name.contains(".vector.thread"))
      continue;

    addAnnotation(function, "kernel");
    addAnnotation(function, "kernel_with_simd");
  }
}

LogicalResult
applyQueriedTargetAttrs(ModuleOp module, const VPTOEmissionOptions &options,
                        llvm::raw_ostream &diagOS) {
  FailureOr<QueriedTargetAttrs> attrs = queryDefaultTargetAttrs(options, diagOS);
  if (failed(attrs)) {
    if (options.defaultTargetCPU.empty() ||
        options.defaultTargetFeatures.empty())
      return failure();
    diagOS << "VPTO LLVM emission: falling back to configured default target "
              "attributes\n";
    attrs = QueriedTargetAttrs{options.defaultTargetCPU,
                               options.defaultTargetFeatures};
  }

  MLIRContext *ctx = module.getContext();
  StringAttr cpuAttr = StringAttr::get(ctx, attrs->targetCPU);
  LLVM::TargetFeaturesAttr featureAttr =
      LLVM::TargetFeaturesAttr::get(ctx, attrs->targetFeatures);
  module.walk([&](LLVM::LLVMFuncOp funcOp) {
    funcOp.setTargetCpuAttr(cpuAttr);
    funcOp.setTargetFeaturesAttr(featureAttr);
  });
  return success();
}

} // namespace mlir::pto
