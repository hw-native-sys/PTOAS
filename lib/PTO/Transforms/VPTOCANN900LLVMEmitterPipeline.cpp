// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOCANN900LLVMEmitterInternal.h"

namespace mlir::pto::detail {

void populateVPTOOpLoweringPatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                    LoweringState &state) {
  populateVPTOArithmeticPatterns(typeConverter, patterns, state);
  populateVPTOVectorMemoryPatterns(typeConverter, patterns, state);
  populateVPTOScalarPatterns(typeConverter, patterns, state);
}

void markIllegalVPTOSyncOps(ConversionTarget &target) {
  target.addIllegalOp<pto::SetFlagOp, pto::WaitFlagOp, pto::SetFlagDynOp, pto::WaitFlagDynOp, pto::SyncSetOp,
                      pto::SyncWaitOp, pto::SetIntraBlockOp, pto::WaitIntraBlockOp, pto::BarrierOp, pto::MemBarOp,
                      pto::CmoCacheInvalidOp, pto::FenceBarrierAllOp, pto::DsbOp, pto::DcciOp, pto::GetBufOp,
                      pto::RlsBufOp, pto::GetBufDynOp, pto::RlsBufDynOp>();
}

void markIllegalVPTOSimtOps(ConversionTarget &target) {
  target.addIllegalOp<
      pto::GetBlockIdxOp, pto::GetSubBlockIdxOp, pto::GetBlockNumOp, pto::GetSubBlockNumOp, pto::GetCtrlOp,
      pto::GetVms4SrOp, pto::GetTidXOp, pto::GetTidYOp, pto::GetTidZOp, pto::GetBlockDimXOp, pto::GetBlockDimYOp,
      pto::GetBlockDimZOp, pto::GetGridDimXOp, pto::GetGridDimYOp, pto::GetGridDimZOp, pto::GetBlockIdxXOp,
      pto::GetBlockIdxYOp, pto::GetBlockIdxZOp, pto::GetVecCoreIdOp, pto::GetLaneIdOp, pto::GetClock32Op,
      pto::GetClock64Op, pto::GetLaneMaskEqOp, pto::GetLaneMaskLeOp, pto::GetLaneMaskLtOp, pto::GetLaneMaskGeOp,
      pto::GetLaneMaskGtOp, pto::VoteAllOp, pto::VoteAnyOp, pto::VoteUniOp, pto::VoteBallotOp, pto::ShuffleIdxOp,
      pto::ShuffleUpOp, pto::ShuffleDownOp, pto::ShuffleBflyOp, pto::ReduxAddOp, pto::ReduxMaxOp, pto::ReduxMinOp,
      pto::AtomicCasOp, pto::AtomicExchOp, pto::AtomicAddOp, pto::AtomicSubOp, pto::AtomicMinOp, pto::AtomicMaxOp,
      pto::AtomicAndOp, pto::AtomicOrOp, pto::AtomicXorOp, pto::TrapOp, pto::PrmtOp, pto::MulhiOp, pto::MulI32ToI64Op,
      pto::SqrtOp, pto::AbsFOp, pto::ExpOp, pto::LogOp, pto::CeilOp, pto::FloorOp, pto::RintOp, pto::RoundOp,
      pto::FMinOp, pto::FMaxOp, pto::PowOp, pto::FmaOp, pto::ConvertOp, pto::SyncthreadsOp, pto::ThreadfenceOp,
      pto::ThreadfenceBlockOp, pto::KeepOp, pto::ResumeOp>();
}

void markIllegalVPTOConfigOps(ConversionTarget &target) {
  target.addIllegalOp<pto::SetLoop2StrideOutToUbOp, pto::SetLoop1StrideOutToUbOp, pto::SetLoopSizeOutToUbOp,
                      pto::SetLoop2StrideUbToOutOp, pto::SetLoop1StrideUbToOutOp, pto::SetLoopSizeUbToOutOp,
                      pto::SetLoop3ParaOp, pto::SetChannelParaOp, pto::SetLoop2StrideOutToL1Op,
                      pto::SetLoop1StrideOutToL1Op, pto::SetLoopSizeOutToL1Op, pto::SetMte2NzParaOp,
                      pto::SetPadValOutToL1Op, pto::SetReluAlphaOp, pto::SetFixClipReluOp, pto::SetFpcOp,
                      pto::SetStoreAtomicCfgOp, pto::SetAtomicS32Op, pto::SetAtomicS8Op, pto::SetCtrlOp,
                      pto::StoreVfSimtInfoOp, pto::SetMovPadValOp, pto::SetQuantPreOp>();
  target.addIllegalOp<pto::Sbitset0Op, pto::Sbitset1Op>();
}

void markIllegalVPTOMemoryOps(ConversionTarget &target) {
  target.addIllegalOp<pto::VldsOp, pto::Vldsx2Op, pto::VsldbOp, pto::VldasOp, pto::InitAlignOp, pto::VldusOp,
                      pto::SprclrOp, pto::SprstiOp, pto::SprstsOp, pto::VstsOp, pto::VsstbOp, pto::Vstsx2Op,
                      pto::VstarOp, pto::VstasOp, pto::Vgather2Op, pto::Vgather2BcOp, pto::VgatherbOp, pto::VscatterOp,
                      pto::PldiOp, pto::PldsOp, pto::PstiOp, pto::PstsOp, pto::PstuOp, pto::VstusOp, pto::VsturOp>();
}

void markIllegalVPTOPredicateOps(ConversionTarget &target) {
  target.addIllegalOp<pto::PltB8Op, pto::PltB16Op, pto::PltB32Op, pto::PltmB8Op, pto::PltmB16Op, pto::PltmB32Op,
                      pto::PsetB8Op, pto::PsetB16Op, pto::PsetB32Op, pto::PgeB8Op, pto::PgeB16Op, pto::PgeB32Op>();
}

void markIllegalVPTOArithmeticAndCopyOps(ConversionTarget &target) {
  target.addIllegalOp<
      pto::VabsOp, pto::VexpOp, pto::VlnOp, pto::VnegOp, pto::VsqrtOp, pto::VreluOp, pto::VnotOp, pto::VsqzOp,
      pto::VusqzOp, pto::VmulaOp, pto::VmullOp, pto::VaddOp, pto::VsubOp, pto::VmulOp, pto::VdivOp, pto::VmaxOp,
      pto::VminOp, pto::VandOp, pto::VorOp, pto::VxorOp, pto::VmaddOp, pto::VaddcOp, pto::VsubcOp, pto::VaddcsOp,
      pto::VsubcsOp, pto::VshlOp, pto::VshrOp, pto::VmulsOp, pto::VaddsOp, pto::VmaxsOp, pto::VminsOp, pto::VlreluOp,
      pto::VshlsOp, pto::VshrsOp, pto::VcaddOp, pto::VcmaxOp, pto::VcminOp, pto::VcgaddOp, pto::VcgmaxOp, pto::VcgminOp,
      pto::VcpaddOp, pto::Chistv2Op, pto::Dhistv2Op, pto::VcbmaxOp, pto::VcbminOp, pto::VdupOp, pto::VbrOp,
      pto::PpackOp, pto::PunpackOp, pto::PbitcastOp, pto::VselOp, pto::VselrOp, pto::PnotOp, pto::PselOp, pto::PandOp,
      pto::PorOp, pto::PxorOp, pto::PdintlvB8Op, pto::PdintlvB16Op, pto::PdintlvB32Op, pto::PintlvB8Op,
      pto::PintlvB16Op, pto::PintlvB32Op, pto::VsunpackOp, pto::VzunpackOp, pto::VpackOp, pto::VintlvOp, pto::VdintlvOp,
      pto::VpreluOp, pto::VaxpyOp, pto::VmulscvtOp, pto::VciOp, pto::VexpdifOp, pto::VbitsortOp, pto::Vmrgsort4Op,
      pto::VtrcOp, pto::VcvtOp, pto::VbitcastOp, pto::VcmpOp, pto::VcmpsOp, pto::CopyGmToUbufOp, pto::CopyUbufToGmOp,
      pto::CopyUbufToUbufOp, pto::CopyCbufToUbufOp, pto::CopyUbufToCbufOp, pto::CopyGmToCbufOp, pto::CreateCbufMatrixOp,
      pto::LoadCbufToCaOp, pto::LoadCbufToCbOp, pto::LoadCbufToCaS4Op, pto::LoadCbufToCbS4Op, pto::LoadCbufToCaMxOp,
      pto::LoadCbufToCbMxOp, pto::CopyMatrixCcToGmOp, pto::CopyMatrixCcToCbufOp, pto::CopyMatrixCcToUbOp,
      pto::CopyCbufToBtOp, pto::CopyCbufToFbufOp, pto::CopyGmToCbufMultiNd2NzOp, pto::CopyGmToCbufMultiDn2NzOp,
      pto::MadOp, pto::MadAccOp, pto::MadBiasOp, pto::MadMxOp, pto::MadMxAccOp, pto::MadMxBiasOp, pto::MadRawOp,
      pto::MadBiasRawOp, pto::MadMxRawOp, pto::MadMxBiasRawOp>();
}

void configureVPTOOpLoweringTarget(ConversionTarget &target, VPTOTypeConverter &typeConverter) {
  (void)typeConverter;
  target.addLegalOp<ModuleOp>();
  target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect, LLVM::LLVMDialect, func::FuncDialect,
                         scf::SCFDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();
  markIllegalVPTOSyncOps(target);
  markIllegalVPTOSimtOps(target);
  markIllegalVPTOConfigOps(target);
  markIllegalVPTOMemoryOps(target);
  markIllegalVPTOPredicateOps(target);
  markIllegalVPTOArithmeticAndCopyOps(target);
  target.markUnknownOpDynamicallyLegal([](Operation *op) { return !isa<pto::TrapOp>(op); });
}

void populateVPTOStructuralTypePatterns(VPTOTypeConverter &typeConverter, RewritePatternSet &patterns,
                                        ConversionTarget &target) {
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns, target);
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);
}

void configureVPTOTypeLoweringTarget(ConversionTarget &target, VPTOTypeConverter &typeConverter) {
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) && typeConverter.isLegal(&op.getBody());
  });
  target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>([&](Operation *op) { return typeConverter.isLegal(op); });
  target.addDynamicallyLegalOp<cf::BranchOp, cf::CondBranchOp>(
      [&](Operation *op) { return isLegalForBranchOpInterfaceTypeConversionPattern(op, typeConverter); });
  target.addDynamicallyLegalOp<arith::SelectOp>([&](arith::SelectOp op) {
    return typeConverter.isLegal(op->getOperandTypes()) && typeConverter.isLegal(op->getResultTypes());
  });
  target.addIllegalOp<pto::AddPtrOp, pto::CastPtrOp, pto::LoadScalarOp, pto::StoreScalarOp, pto::PTOLoadOp,
                      pto::PTOStoreOp, pto::PTOLdgOp, pto::PTOStgOp, pto::PTOLdDevOp, pto::PTOStDevOp,
                      pto::DeclareStructOp, pto::StructGetOp, pto::StructSetOp>();
}

void configureVPTOCarrierTypeLegality(ConversionTarget &target, VPTOTypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>([&](UnrealizedConversionCastOp op) {
    return !hasVPTOConvertibleType(op->getOperandTypes()) && !hasVPTOConvertibleType(op->getResultTypes());
  });
  target.addDynamicallyLegalOp<LLVM::AllocaOp>([&](LLVM::AllocaOp op) {
    return typeConverter.isLegal(op->getOperandTypes()) && typeConverter.isLegal(op->getResultTypes()) &&
           typeConverter.isLegal(op.getElemType());
  });
  target.addDynamicallyLegalOp<LLVM::GEPOp>([&](LLVM::GEPOp op) {
    return typeConverter.isLegal(op->getOperandTypes()) && typeConverter.isLegal(op->getResultTypes()) &&
           typeConverter.isLegal(op.getElemType());
  });
  target.markUnknownOpDynamicallyLegal([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) && typeConverter.isLegal(op->getResultTypes());
  });
}
void foldVPTOTypeCasts(ModuleOp module, TypeConverter &typeConverter) {
  SmallVector<UnrealizedConversionCastOp> castsToFold;
  module.walk([&](UnrealizedConversionCastOp castOp) {
    if (castOp->getNumOperands() != 1 || castOp->getNumResults() != 1) {
      return;
    }
    if (!hasVPTOConvertibleType(castOp->getOperandTypes()) && !hasVPTOConvertibleType(castOp->getResultTypes())) {
      return;
    }
    Type convertedResultType = typeConverter.convertType(castOp.getResult(0).getType());
    if (convertedResultType && convertedResultType == castOp.getOperand(0).getType()) {
      castsToFold.push_back(castOp);
    }
  });
  for (UnrealizedConversionCastOp castOp : castsToFold) {
    castOp.getResult(0).replaceAllUsesWith(castOp.getOperand(0));
    castOp.erase();
  }
}

LogicalResult lowerVPTOOps(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  configureVPTOOpLoweringTarget(target, typeConverter);
  populateVPTOOpLoweringPatterns(typeConverter, patterns, state);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO op lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS))) {
    return failure();
  }
  return success();
}

LogicalResult lowerVPTOTypes(ModuleOp module, llvm::raw_ostream &diagOS) {
  MLIRContext *context = module.getContext();
  VPTOTypeConverter typeConverter(context);
  ConversionTarget target(*context);
  RewritePatternSet patterns(context);
  LoweringState state;

  configureVPTOTypeLoweringTarget(target, typeConverter);
  configureVPTOCarrierTypeLegality(target, typeConverter);
  populateVPTOStructuralTypePatterns(typeConverter, patterns, target);
  populateVPTOTypePatterns(typeConverter, patterns, target, state);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    diagOS << "VPTO LLVM emission failed: VPTO type lowering failed\n";
    return failure();
  }
  if (failed(materializeDecls(module, state.plannedDecls, diagOS))) {
    return failure();
  }
  foldVPTOTypeCasts(module, typeConverter);
  return success();
}

Type normalizeTypeForOfficialLLVMLowering(Type type, Builder &builder) {
  type = convertVPTOType(type, builder);
  return type;
}

void normalizeFuncSignaturesForOfficialLLVMLowering(ModuleOp module) {
  Builder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    FunctionType oldType = funcOp.getFunctionType();
    SmallVector<Type> newInputs;
    SmallVector<Type> newResults;
    bool changed = false;

    for (Type input : oldType.getInputs()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(input, builder);
      changed |= (normalized != input);
      newInputs.push_back(normalized);
    }
    for (Type result : oldType.getResults()) {
      Type normalized = normalizeTypeForOfficialLLVMLowering(result, builder);
      changed |= (normalized != result);
      newResults.push_back(normalized);
    }

    if (!changed) {
      continue;
    }

    auto newType = builder.getFunctionType(newInputs, newResults);
    funcOp.setFunctionTypeAttr(TypeAttr::get(newType));

    if (funcOp.isExternal()) {
      continue;
    }
    Block &entry = funcOp.getBody().front();
    for (auto [arg, newType] : llvm::zip(entry.getArguments(), newInputs)) {
      if (arg.getType() != newType) {
        arg.setType(newType);
      }
    }
  }
}

void forceV300CtrlModeForVPTOFuncs(ModuleOp module) {
  OpBuilder builder(module.getContext());

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!needsV300CtrlModeForVPTOFunc(funcOp)) {
      continue;
    }

    Block &entry = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entry);
    auto i64Type = builder.getI64Type();
    auto bit60 = builder.create<arith::ConstantOp>(funcOp.getLoc(), i64Type, builder.getI64IntegerAttr(60));
    Value ctrl = builder.create<pto::GetCtrlOp>(funcOp.getLoc(), i64Type).getResult();
    Value ctrlV300 = builder.create<pto::Sbitset0Op>(funcOp.getLoc(), i64Type, ctrl, bit60.getResult()).getResult();
    builder.create<pto::SetCtrlOp>(funcOp.getLoc(), ctrlV300);
  }
}

std::optional<FunctionKernelKind> getKernelKind(ModuleOp module) {
  auto kernelKind = module->getAttrOfType<FunctionKernelKindAttr>(FunctionKernelKindAttr::name);
  if (!kernelKind) {
    return std::nullopt;
  }
  return kernelKind.getKernelKind();
}

VPTOEmissionOptions makeDeviceEmissionOptions(const VPTOEmissionOptions &baseOptions, FunctionKernelKind kind) {
  VPTOEmissionOptions options = baseOptions;
  constexpr llvm::StringLiteral kVecTargetFeatures =
      "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
      "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
      "+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec";
  constexpr llvm::StringLiteral kCubeTargetFeatures =
      "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
      "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
      "+MOVX8,+SPR7bits,+SyncV,+dav-c310-cube";
  if (kind == FunctionKernelKind::Vector) {
    options.march = "dav-c310-vec";
    options.aicoreArch = "dav-c310-vec";
    options.defaultTargetCPU = "dav-c310-vec";
    options.defaultTargetFeatures = kVecTargetFeatures.str();
  } else if (kind == FunctionKernelKind::Cube) {
    options.march = "dav-c310-cube";
    options.aicoreArch = "dav-c310-cube";
    options.defaultTargetCPU = "dav-c310-cube";
    options.defaultTargetFeatures = kCubeTargetFeatures.str();
  }
  return options;
}

FailureOr<ModuleOp> getUniqueDeviceModuleByKernelKind(ModuleOp module, FunctionKernelKind kind,
                                                      llvm::raw_ostream &diagOS) {
  ModuleOp matched;
  for (ModuleOp child : module.getOps<ModuleOp>()) {
    auto kernelKind = getKernelKind(child);
    if (!kernelKind) {
      continue;
    }
    if (*kernelKind != kind) {
      continue;
    }
    if (matched) {
      diagOS << "VPTO LLVM emission failed: duplicate device module with " << FunctionKernelKindAttr::name << "\n";
      return failure();
    }
    matched = child;
  }
  return matched;
}

LogicalResult renameKernelFunctionsForKernelKind(ModuleOp module, llvm::raw_ostream &diagOS) {
  auto kernelKind = getKernelKind(module);
  if (!kernelKind) {
    diagOS << "VPTO LLVM emission failed: device module missing " << FunctionKernelKindAttr::name << "\n";
    return failure();
  }

  StringRef suffix;
  if (*kernelKind == FunctionKernelKind::Vector) {
    suffix = kVectorSuffix;
  } else if (*kernelKind == FunctionKernelKind::Cube) {
    suffix = kCubeSuffix;
  } else {
    diagOS << "VPTO LLVM emission failed: unsupported " << FunctionKernelKindAttr::name << "\n";
    return failure();
  }

  for (func::FuncOp funcOp : module.getOps<func::FuncOp>()) {
    if (!pto::hasExplicitPTOEntryAttr(funcOp)) {
      continue;
    }
    if (funcOp.getSymName().ends_with(suffix)) {
      continue;
    }
    funcOp.setSymName((funcOp.getSymName() + suffix).str());
  }
  return success();
}

struct LowerVPTOOpsPass final : public PassWrapper<LowerVPTOOpsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVPTOOpsPass)

  void runOnOperation() override {
    materializeVecScopeCarrierLoops(getOperation());
    if (failed(lowerVPTOOps(getOperation(), llvm::errs()))) {
      signalPassFailure();
    }
  }
};

struct LowerVPTOTypesPass final : public PassWrapper<LowerVPTOTypesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVPTOTypesPass)

  void runOnOperation() override {
    if (failed(lowerVPTOTypes(getOperation(), llvm::errs()))) {
      signalPassFailure();
    }
  }
};

struct NormalizeFuncSignaturesForLLVMLoweringPass final
    : public PassWrapper<NormalizeFuncSignaturesForLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(NormalizeFuncSignaturesForLLVMLoweringPass)

  void runOnOperation() override { normalizeFuncSignaturesForOfficialLLVMLowering(getOperation()); }
};

struct PrepareVPTOLLVMLoweringPass final : public PassWrapper<PrepareVPTOLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareVPTOLLVMLoweringPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    pto::annotatePTOEntryFunctions(module);
    forceV300CtrlModeForVPTOFuncs(module);
    if (failed(renameKernelFunctionsForKernelKind(module, llvm::errs()))) {
      signalPassFailure();
    }
  }
};

llvm::StringSet<llvm::BumpPtrAllocator> collectSimtEntryFunctionNames(ModuleOp module) {
  llvm::StringSet<llvm::BumpPtrAllocator> simtEntries;
  module.walk([&](func::FuncOp funcOp) {
    if (funcOp->hasAttr(pto::kPTOSimtEntryAttrName)) {
      simtEntries.insert(funcOp.getSymName());
    }
  });
  return simtEntries;
}

void applyArtifactVisibilityLinkage(ModuleOp sourceModule, llvm::Module &llvmModule) {
  llvm::StringMap<bool> externalByName;
  sourceModule.walk([&](func::FuncOp funcOp) {
    if (funcOp.isDeclaration()) {
      return;
    }
    externalByName[funcOp.getSymName()] = pto::hasExternalArtifactVisibility(funcOp);
  });

  for (llvm::Function &function : llvmModule) {
    auto it = externalByName.find(function.getName());
    if (it == externalByName.end()) {
      continue;
    }
    if (it->second) {
      function.setLinkage(llvm::GlobalValue::ExternalLinkage);
      continue;
    }
    function.setLinkage(llvm::GlobalValue::InternalLinkage);
  }
}

void applySimtEntryCallingConvention(llvm::Module &llvmModule,
                                     const llvm::StringSet<llvm::BumpPtrAllocator> &simtEntryNames) {
  for (llvm::Function &function : llvmModule) {
    if (simtEntryNames.contains(function.getName())) {
      function.setCallingConv(llvm::CallingConv::SimtEntry);
      function.addFnAttr(llvm::Attribute::NoInline);
      // Match Bisheng's C++ frontend shape for SIMT outlined bodies. The
      // exported wrapper owns the real kernel metadata, while the SIMT body is
      // an ODR helper called with the SIMT calling convention. In CANN beta.1,
      // leaving the SIMT body as a strong GLOBAL FUNC makes the runtime count it
      // as an extra kernel without matching .ascend.meta, which can corrupt the
      // selected kernel metadata. linkonce_odr lowers to a weak helper symbol
      // and avoids that beta.1 metadata mismatch.
      function.setLinkage(llvm::GlobalValue::LinkOnceODRLinkage);
    }
  }

  for (llvm::Function &function : llvmModule) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&inst);
        if (!call) {
          continue;
        }
        auto *callee = call->getCalledFunction();
        if (!callee || !simtEntryNames.contains(callee->getName())) {
          continue;
        }
        call->setCallingConv(llvm::CallingConv::SimtEntry);
      }
    }
  }
}

FailureOr<EmittedLLVMModule> emitDeviceLLVMModule(ModuleOp deviceModule, StringRef kernelKind,
                                                  const VPTOEmissionOptions &options,
                                                  const llvm::StringSet<llvm::BumpPtrAllocator> &simtEntryNames,
                                                  llvm::raw_ostream &diagOS) {
  if (!deviceModule) {
    return EmittedLLVMModule{};
  }
  if (failed(applyQueriedTargetAttrs(deviceModule, options, diagOS))) {
    return failure();
  }

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  registerBuiltinDialectTranslation(*deviceModule.getContext());
  registerLLVMDialectTranslation(*deviceModule.getContext());
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(deviceModule.getOperation(), *llvmContext);
  if (!llvmModule) {
    diagOS << "VPTO LLVM emission failed: LLVM IR export failed for " << kernelKind << " module\n";
    return failure();
  }

  applyArtifactVisibilityLinkage(deviceModule, *llvmModule);
  for (llvm::Function &func : *llvmModule) {
    if (!func.getName().starts_with("llvm.hivm.vscatter.")) {
      continue;
    }
    // Work around a bug in older Bisheng releases: vscatter was not modeled
    // as writing through its destination pointer, so EarlyCSE could eliminate
    // a load after vscatter as redundant.
    func.setOnlyAccessesArgMemory();
    func.addFnAttr(llvm::Attribute::NoUnwind);
    func.addFnAttr(llvm::Attribute::WriteOnly);
  }
  applySimtEntryCallingConvention(*llvmModule, simtEntryNames);
  if (failed(attachAIVectorScopeMetadata(*llvmModule, diagOS))) {
    return failure();
  }
  attachHIVMKernelAnnotations(*llvmModule, deviceModule);
  llvmModule->setModuleIdentifier(("ptoas.hivm.official." + kernelKind).str());
  llvmModule->setSourceFileName(("ptoas.hivm.official." + kernelKind).str());
  return EmittedLLVMModule{std::move(llvmContext), std::move(llvmModule)};
}

template <typename EmitFn> LogicalResult runPipeline(ModuleOp module, llvm::raw_ostream &diagOS, EmitFn &&emit) {
  OwningOpRef<Operation *> clonedOp(module->clone());
  ModuleOp clonedModule = cast<ModuleOp>(*clonedOp);

  if (failed(validateVPTOAuthoringIR(clonedModule, &diagOS))) {
    diagOS << "VPTO LLVM emission failed: authoring-stage VPTO legality "
              "validation failed\n";
    return failure();
  }

  PassManager pm(clonedModule.getContext());
  pm.enableVerifier();
  auto &kernelModulePM = pm.nest<ModuleOp>();
  kernelModulePM.addPass(std::make_unique<PrepareVPTOLLVMLoweringPass>());
  kernelModulePM.addPass(std::make_unique<LowerVPTOOpsPass>());
  kernelModulePM.addPass(std::make_unique<LowerVPTOTypesPass>());
  kernelModulePM.addPass(std::make_unique<NormalizeFuncSignaturesForLLVMLoweringPass>());
  kernelModulePM.addPass(arith::createArithExpandOpsPass());
  // pto-convert-scf-to-cf-with-loop-hints performs the SCF-to-CF conversion for this pipeline:
  // it runs the upstream conversion patterns plus a higher-benefit lowering
  // for {pto.unroll = "enable"} loops that attaches llvm.loop_annotation to
  // the latch, so the !llvm.loop.unroll.enable metadata survives into the
  // emitted LLVM IR.  It replaces createConvertSCFToCFPass here; running both
  // would be redundant.
  kernelModulePM.addNestedPass<func::FuncOp>(pto::createPTOConvertSCFToCFWithLoopHintsPass());
  kernelModulePM.addPass(createArithToLLVMConversionPass());
  kernelModulePM.addPass(createConvertIndexToLLVMPass());
  kernelModulePM.addPass(createFinalizeMemRefToLLVMConversionPass());
  kernelModulePM.addPass(createConvertFuncToLLVMPass());
  kernelModulePM.addPass(createConvertControlFlowToLLVMPass());
  kernelModulePM.addPass(createReconcileUnrealizedCastsPass());
  if (failed(mlir::applyPassManagerCLOptions(pm))) {
    diagOS << "VPTO LLVM emission failed: unable to apply MLIR pass manager "
              "command-line options\n";
    return failure();
  }
  if (failed(pm.run(clonedModule))) {
    diagOS << "VPTO LLVM emission failed: official lowering pipeline failed\n";
    return failure();
  }
  return emit(clonedModule);
}

LogicalResult lowerCANN900Module(ModuleOp module, const VPTOEmissionOptions &options, EmittedLLVMModule &cubeModule,
                                 EmittedLLVMModule &vectorModule, llvm::raw_ostream &diagOS) {
  llvm::StringSet<llvm::BumpPtrAllocator> simtEntryNames = collectSimtEntryFunctionNames(module);
  cubeModule.context.reset();
  cubeModule.module.reset();
  vectorModule.context.reset();
  vectorModule.module.reset();
  return runPipeline(module, diagOS, [&](ModuleOp loweredModule) {
    auto vectorDeviceModule = getUniqueDeviceModuleByKernelKind(loweredModule, FunctionKernelKind::Vector, diagOS);
    if (failed(vectorDeviceModule)) {
      return failure();
    }
    auto cubeDeviceModule = getUniqueDeviceModuleByKernelKind(loweredModule, FunctionKernelKind::Cube, diagOS);
    if (failed(cubeDeviceModule)) {
      return failure();
    }

    if (*vectorDeviceModule) {
      auto vectorOptions = makeDeviceEmissionOptions(options, FunctionKernelKind::Vector);
      auto emitted = emitDeviceLLVMModule(*vectorDeviceModule, "vector", vectorOptions, simtEntryNames, diagOS);
      if (failed(emitted)) {
        return failure();
      }
      vectorModule.context = std::move(emitted->context);
      vectorModule.module = std::move(emitted->module);
    }
    if (*cubeDeviceModule) {
      auto cubeOptions = makeDeviceEmissionOptions(options, FunctionKernelKind::Cube);
      auto emitted = emitDeviceLLVMModule(*cubeDeviceModule, "cube", cubeOptions, simtEntryNames, diagOS);
      if (failed(emitted)) {
        return failure();
      }
      cubeModule.context = std::move(emitted->context);
      cubeModule.module = std::move(emitted->module);
    }
    return success();
  });
}

} // namespace mlir::pto::detail
