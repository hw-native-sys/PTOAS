//===- HIVMIntrinsicNaming.cpp - HIVM intrinsic selection -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/HIVMIntrinsicNaming.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <optional>

using namespace mlir;

namespace mlir::pto {
namespace {

static std::string getLocationString(Location loc) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  loc.print(os);
  return storage;
}

static std::string sanitizeNameFragment(llvm::StringRef text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '_')
      out.push_back(c);
    else
      out.push_back('_');
  }
  return out;
}

static std::string printAttrText(Attribute attr) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << attr;
  return storage;
}

static std::string getElementTypeFragment(Type type) {
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.isUnsigned() ? "u" : "s") + std::to_string(intType.getWidth());
  return "unknown";
}

static std::string getVectorTypeFragment(Type type) {
  auto vecType = dyn_cast<pto::VRegType>(type);
  if (!vecType)
    return {};
  return ("v" + std::to_string(vecType.getElementCount()) +
          getElementTypeFragment(vecType.getElementType()));
}

static std::string getCopyElementFragment(Type type) {
  auto ptrType = dyn_cast<pto::PtrType>(type);
  if (!ptrType)
    return {};
  Type elementType = ptrType.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    switch ((floatType.getWidth() + 7) / 8) {
    case 1:
      return "u8";
    case 2:
      return "u16";
    case 4:
    case 8:
      return "u32";
    default:
      return {};
    }
  }
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    switch ((intType.getWidth() + 7) / 8) {
    case 1:
      return "u8";
    case 2:
      return "u16";
    case 4:
    case 8:
      return "u32";
    default:
      return {};
    }
  }
  return {};
}

static std::string getOpMnemonic(Operation *op) {
  return op->getName().stripDialect().str();
}

static IntrinsicSelection makeResolved(Operation *op, llvm::StringRef calleeName,
                                       llvm::ArrayRef<std::string> usedFields,
                                       llvm::StringRef resultTypeFragment) {
  IntrinsicSelection selection;
  selection.resolved = true;
  selection.sourceOpName = op->getName().getStringRef().str();
  selection.calleeName = calleeName.str();
  selection.usedFields.assign(usedFields.begin(), usedFields.end());
  selection.resultTypeFragment = resultTypeFragment.str();
  selection.location = getLocationString(op->getLoc());
  return selection;
}

static IntrinsicSelection makeUnresolved(Operation *op,
                                         llvm::StringRef familyOrOp,
                                         llvm::StringRef candidateName,
                                         llvm::ArrayRef<std::string> usedFields,
                                         llvm::ArrayRef<std::string> missingFields,
                                         llvm::StringRef resultTypeFragment) {
  IntrinsicSelection selection;
  selection.resolved = false;
  selection.sourceOpName = op->getName().getStringRef().str();
  selection.candidateName = candidateName.str();
  selection.usedFields.assign(usedFields.begin(), usedFields.end());
  selection.missingFields.assign(missingFields.begin(), missingFields.end());
  selection.resultTypeFragment = resultTypeFragment.str();
  selection.location = getLocationString(op->getLoc());

  std::string name = "__ptoas_hivm_unresolved.";
  name += sanitizeNameFragment(familyOrOp);
  if (!resultTypeFragment.empty()) {
    name += ".";
    name += sanitizeNameFragment(resultTypeFragment);
  }
  selection.placeholderName = std::move(name);
  return selection;
}

static FailureOr<IntrinsicSelection> selectSyncLike(Operation *op) {
  llvm::SmallVector<std::string, 4> usedFields;
  usedFields.push_back("op=" + getOpMnemonic(op));

  if (auto setFlag = dyn_cast<pto::SetFlagOp>(op)) {
    usedFields.push_back("src_pipe=" + printAttrText(setFlag.getSrcPipe()));
    usedFields.push_back("dst_pipe=" + printAttrText(setFlag.getDstPipe()));
    usedFields.push_back("event=" + printAttrText(setFlag.getEventId()));
    return makeResolved(op, "llvm.hivm.SET.FLAG.IMM", usedFields, "");
  } else if (auto waitFlag = dyn_cast<pto::WaitFlagOp>(op)) {
    usedFields.push_back("src_pipe=" + printAttrText(waitFlag.getSrcPipe()));
    usedFields.push_back("dst_pipe=" + printAttrText(waitFlag.getDstPipe()));
    usedFields.push_back("event=" + printAttrText(waitFlag.getEventId()));
    return makeResolved(op, "llvm.hivm.WAIT.FLAG.IMM", usedFields, "");
  } else if (auto barrier = dyn_cast<pto::BarrierOp>(op)) {
    usedFields.push_back("pipe=" + printAttrText(barrier.getPipe()));
    return makeResolved(op, "llvm.hivm.BARRIER", usedFields, "");
  }

  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
  return makeUnresolved(op, getOpMnemonic(op), "", usedFields, missingFields, "");
}

static FailureOr<IntrinsicSelection> selectConfigLike(Operation *op) {
  llvm::SmallVector<std::string, 2> usedFields = {"op=" + getOpMnemonic(op)};

  if (isa<pto::SetLoop2StrideOutToUbOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB", usedFields,
                        "");
  if (isa<pto::SetLoop1StrideOutToUbOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB",
                        usedFields, "");
  if (isa<pto::SetLoopSizeOutToUbOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB", usedFields, "");
  if (isa<pto::SetLoop2StrideUbToOutOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT", usedFields,
                        "");
  if (isa<pto::SetLoop1StrideUbToOutOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT", usedFields,
                        "");
  if (isa<pto::SetLoopSizeUbToOutOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT", usedFields, "");

  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
  return makeUnresolved(op, getOpMnemonic(op), "", usedFields, missingFields,
                        "");
}

static FailureOr<IntrinsicSelection> selectPredicateIntrinsic(Operation *op) {
  llvm::SmallVector<std::string, 4> usedFields;
  if (auto pset = dyn_cast<pto::PsetB8Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(pset.getResult().getType());
    usedFields = {"family=pset", "bitwidth=8", "result=" + resultFragment,
                  "pattern=i32"};
    return makeResolved(op, "llvm.hivm.pset.b8", usedFields, resultFragment);
  }
  if (auto pset = dyn_cast<pto::PsetB16Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(pset.getResult().getType());
    usedFields = {"family=pset", "bitwidth=16", "result=" + resultFragment,
                  "pattern=i32"};
    return makeResolved(op, "llvm.hivm.pset.b16", usedFields, resultFragment);
  }
  if (auto pset = dyn_cast<pto::PsetB32Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(pset.getResult().getType());
    usedFields = {"family=pset", "bitwidth=32", "result=" + resultFragment,
                  "pattern=i32"};
    return makeResolved(op, "llvm.hivm.pset.b32", usedFields, resultFragment);
  }
  if (auto pge = dyn_cast<pto::PgeB8Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(pge.getResult().getType());
    usedFields = {"family=pge", "bitwidth=8", "result=" + resultFragment,
                  "pattern=i32", "variant=i32_zero"};
    return makeResolved(op, "llvm.hivm.pge.b8", usedFields, resultFragment);
  }
  if (auto pge = dyn_cast<pto::PgeB16Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(pge.getResult().getType());
    usedFields = {"family=pge", "bitwidth=16", "result=" + resultFragment,
                  "pattern=i32", "variant=i32_zero"};
    return makeResolved(op, "llvm.hivm.pge.b16", usedFields, resultFragment);
  }
  if (auto pge = dyn_cast<pto::PgeB32Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(pge.getResult().getType());
    usedFields = {"family=pge", "bitwidth=32", "result=" + resultFragment,
                  "pattern=i32", "variant=i32_zero"};
    return makeResolved(op, "llvm.hivm.pge.b32", usedFields, resultFragment);
  }
  if (auto plt = dyn_cast<pto::PltB8Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(plt.getMask().getType());
    usedFields = {"family=plt", "bitwidth=8", "result=" + resultFragment,
                  "variant=v300", "scalar=i32", "scalar_out=i32"};
    return makeResolved(op, "llvm.hivm.plt.b8.v300", usedFields, resultFragment);
  }
  if (auto plt = dyn_cast<pto::PltB16Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(plt.getMask().getType());
    usedFields = {"family=plt", "bitwidth=16", "result=" + resultFragment,
                  "variant=v300", "scalar=i32", "scalar_out=i32"};
    return makeResolved(op, "llvm.hivm.plt.b16.v300", usedFields, resultFragment);
  }
  if (auto plt = dyn_cast<pto::PltB32Op>(op)) {
    const std::string resultFragment =
        getVectorTypeFragment(plt.getMask().getType());
    usedFields = {"family=plt", "bitwidth=32", "result=" + resultFragment,
                  "variant=v300", "scalar=i32", "scalar_out=i32"};
    return makeResolved(op, "llvm.hivm.plt.b32.v300", usedFields, resultFragment);
  }

  return failure();
}

} // namespace

FailureOr<IntrinsicSelection> selectLoadIntrinsic(Operation *op) {
  if (auto vlds = dyn_cast<pto::VldsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vlds.getResult().getType());
    llvm::SmallVector<std::string, 4> usedFields = {
        "family=vldsx1", "vector=" + vecFragment, "mode=NO_POST_UPDATE"};
    if (vlds.getDistAttr())
      usedFields.push_back("dist=" + (*vlds.getDist()).str());

    if (vecFragment == "v64f32")
      return makeResolved(op, "llvm.hivm.vldsx1", usedFields, vecFragment);

    llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
    std::string candidate = "llvm.hivm.vldsx1";
    return makeUnresolved(op, "vldsx1", candidate, usedFields, missingFields,
                          vecFragment);
  }

  if (auto vldsPost = dyn_cast<pto::VldsPostOp>(op)) {
    const std::string vecFragment =
        getVectorTypeFragment(vldsPost.getResult().getType());
    llvm::SmallVector<std::string, 6> usedFields = {
        "family=vldsx1", "variant=post", "vector=" + vecFragment,
        "mode=POST_UPDATE"};
    if (vldsPost.getDistAttr())
      usedFields.push_back("dist=" + (*vldsPost.getDist()).str());

    if (vecFragment == "v64f32")
      return makeResolved(op, "llvm.hivm.vldsx1.post", usedFields, vecFragment);

    llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
    std::string candidate = "llvm.hivm.vldsx1.post";
    return makeUnresolved(op, "vldsx1.post", candidate, usedFields,
                          missingFields, vecFragment);
  }

  return failure();
}

FailureOr<IntrinsicSelection> selectUnaryIntrinsic(Operation *op) {
  auto vabs = dyn_cast<pto::VabsOp>(op);
  if (vabs) {
    const std::string vecFragment = getVectorTypeFragment(vabs.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vabs", "vector=" + vecFragment, "variant=x"};

    if (vecFragment == "v64f32")
      return makeResolved(op, "llvm.hivm.vabs.v64f32.x", usedFields, vecFragment);

    llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
    std::string candidate = "llvm.hivm.vabs";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeUnresolved(op, "vabs", candidate, usedFields, missingFields,
                          vecFragment);
  }

  if (auto vexp = dyn_cast<pto::VexpOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vexp.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vexp", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vexp";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto vdup = dyn_cast<pto::VdupOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vdup.getResult().getType());
    const bool vectorInput = isa<VectorType, pto::VRegType>(vdup.getInput().getType());
    const StringRef position = vdup.getPosition().value_or("LOWEST");
    const char *family =
        vectorInput ? (position == "HIGHEST" ? "vdupm" : "vdup") : "vdups";
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=" + std::string(family), "vector=" + vecFragment,
        "variant=z"};
    if (!vectorInput && !isa<FloatType, IntegerType>(vdup.getInput().getType())) {
      llvm::SmallVector<std::string, 1> missingFields = {"scalar_input_vdup_mapping"};
      return makeUnresolved(op, "vdup", "llvm.hivm.vdups", usedFields, missingFields,
                            vecFragment);
    }
    std::string candidate = "llvm.hivm.";
    candidate += family;
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".z";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VaddOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vadd", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vadd";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VsubOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vsub", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vsub";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VmulOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmul", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmul";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VmaxOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmax", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmax";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VmulsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmuls", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmuls";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VaddsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vadds", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vadds";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VmaxsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmaxs", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmaxs";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VminsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmins", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmins";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VlreluOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vlrelu", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vlrelu";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VshlsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vshls", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vshls";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<pto::VshrsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vshrs", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vshrs";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  return failure();
}

FailureOr<IntrinsicSelection> selectStoreIntrinsic(Operation *op) {
  llvm::SmallVector<std::string, 4> usedFields;
  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};

  if (auto vsts = dyn_cast<pto::VstsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vsts.getValue().getType());
    usedFields = {"family=vstsx1", "vector=" + vecFragment,
                  "predicate_source=explicit_mask", "mode=NO_POST_UPDATE"};
    if (vsts.getDistAttr())
      usedFields.push_back("dist=" + (*vsts.getDist()).str());
    if (vecFragment == "v64f32")
      return makeResolved(op, "llvm.hivm.vstsx1", usedFields, vecFragment);
    return makeUnresolved(op, "vstsx1", "llvm.hivm.vstsx1", usedFields, missingFields,
                          vecFragment);
  }

  if (auto vstsPost = dyn_cast<pto::VstsPostOp>(op)) {
    const std::string vecFragment =
        getVectorTypeFragment(vstsPost.getValue().getType());
    usedFields = {"family=vstsx1", "variant=post", "vector=" + vecFragment,
                  "predicate_source=explicit_mask", "mode=POST_UPDATE"};
    if (vstsPost.getDistAttr())
      usedFields.push_back("dist=" + (*vstsPost.getDist()).str());
    if (vecFragment == "v64f32")
      return makeResolved(op, "llvm.hivm.vstsx1.post", usedFields,
                          vecFragment);
    std::string candidate = "llvm.hivm.vstsx1.post";
    return makeUnresolved(op, "vstsx1.post", candidate, usedFields,
                          missingFields, vecFragment);
  }

  if (auto copy = dyn_cast<pto::CopyGmToUbufOp>(op)) {
    std::string elemFragment = getCopyElementFragment(copy.getSource().getType());
    usedFields = {"family=copy_gm_to_ubuf"};
    if (!elemFragment.empty())
      usedFields.push_back("element=" + elemFragment);
    if (elemFragment == "u8" || elemFragment == "u16" ||
        elemFragment == "u32" || elemFragment == "f32") {
      std::string callee = "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.";
      callee += elemFragment;
      callee += ".DV";
      return makeResolved(op, callee, usedFields, "");
    }
    std::string candidate = "llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2";
    if (!elemFragment.empty())
      candidate += "." + elemFragment + ".DV";
    missingFields.push_back("element_type_mapping");
    return makeUnresolved(op, "copy_gm_to_ubuf", candidate, usedFields,
                          missingFields, "");
  }

  if (auto copy = dyn_cast<pto::CopyUbufToGmOp>(op)) {
    std::string elemFragment = getCopyElementFragment(copy.getSource().getType());
    usedFields = {"family=copy_ubuf_to_gm"};
    if (!elemFragment.empty())
      usedFields.push_back("element=" + elemFragment);
    return makeResolved(op, "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV",
                        usedFields, "");
  }

  if (isa<pto::CopyUbufToUbufOp>(op)) {
    usedFields = {"family=copy_ubuf_to_ubuf"};
    return makeUnresolved(op, "copy_ubuf_to_ubuf", "copy_ubuf_to_ubuf",
                          usedFields, missingFields, "");
  }

  return failure();
}

FailureOr<IntrinsicSelection> selectIntrinsic(Operation *op) {
  if (isa<pto::SetFlagOp, pto::WaitFlagOp, pto::BarrierOp>(op))
    return selectSyncLike(op);

  if (isa<pto::SetLoop2StrideOutToUbOp, pto::SetLoop1StrideOutToUbOp,
          pto::SetLoopSizeOutToUbOp, pto::SetLoop2StrideUbToOutOp,
          pto::SetLoop1StrideUbToOutOp, pto::SetLoopSizeUbToOutOp>(op))
    return selectConfigLike(op);

  if (succeeded(selectLoadIntrinsic(op)))
    return *selectLoadIntrinsic(op);
  if (succeeded(selectUnaryIntrinsic(op)))
    return *selectUnaryIntrinsic(op);
  if (succeeded(selectPredicateIntrinsic(op)))
    return *selectPredicateIntrinsic(op);
  if (succeeded(selectStoreIntrinsic(op)))
    return *selectStoreIntrinsic(op);

  llvm::SmallVector<std::string, 2> usedFields = {"op=" + getOpMnemonic(op)};
  llvm::SmallVector<std::string, 2> missingFields = {"family_mapping",
                                                     "confirmed_hivm_name"};
  return makeUnresolved(op, getOpMnemonic(op), "", usedFields, missingFields,
                        "");
}

} // namespace mlir::pto
