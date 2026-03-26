//===- HIVMIntrinsicNaming.cpp - HIVM intrinsic selection -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/HIVMIntrinsicNaming.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>

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
  auto vecType = dyn_cast<a5vm::VecType>(type);
  if (!vecType)
    return {};
  return ("v" + std::to_string(vecType.getElementCount()) +
          getElementTypeFragment(vecType.getElementType()));
}

static std::string getCopyElementFragment(Operation *op) {
  if (auto attr = dyn_cast_or_null<StringAttr>(op->getAttr("a5vm.element_type")))
    return attr.getValue().str();
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

  if (auto setFlag = dyn_cast<a5vm::SetFlagOp>(op)) {
    usedFields.push_back("src_pipe=" + setFlag.getSrcPipe().str());
    usedFields.push_back("dst_pipe=" + setFlag.getDstPipe().str());
    usedFields.push_back("event=" + setFlag.getEventId().str());
    return makeResolved(op, "llvm.hivm.SET.FLAG.IMM", usedFields, "");
  } else if (auto waitFlag = dyn_cast<a5vm::WaitFlagOp>(op)) {
    usedFields.push_back("src_pipe=" + waitFlag.getSrcPipe().str());
    usedFields.push_back("dst_pipe=" + waitFlag.getDstPipe().str());
    usedFields.push_back("event=" + waitFlag.getEventId().str());
    return makeResolved(op, "llvm.hivm.WAIT.FLAG.IMM", usedFields, "");
  } else if (auto barrier = dyn_cast<a5vm::PipeBarrierOp>(op)) {
    usedFields.push_back("pipe=" + barrier.getPipe().str());
    return makeResolved(op, "llvm.hivm.BARRIER", usedFields, "");
  }

  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
  return makeUnresolved(op, getOpMnemonic(op), "", usedFields, missingFields, "");
}

static FailureOr<IntrinsicSelection> selectConfigLike(Operation *op) {
  llvm::SmallVector<std::string, 2> usedFields = {"op=" + getOpMnemonic(op)};

  if (isa<a5vm::SetLoop2StrideOutToUbOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB", usedFields,
                        "");
  if (isa<a5vm::SetLoop1StrideOutToUbOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB",
                        usedFields, "");
  if (isa<a5vm::SetLoopSizeOutToUbOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP.SIZE.OUTTOUB", usedFields, "");
  if (isa<a5vm::SetLoop2StrideUbToOutOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT", usedFields,
                        "");
  if (isa<a5vm::SetLoop1StrideUbToOutOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT", usedFields,
                        "");
  if (isa<a5vm::SetLoopSizeUbToOutOp>(op))
    return makeResolved(op, "llvm.hivm.SET.LOOP.SIZE.UBTOOUT", usedFields, "");

  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
  return makeUnresolved(op, getOpMnemonic(op), "", usedFields, missingFields,
                        "");
}

static FailureOr<IntrinsicSelection> selectPredicateIntrinsic(Operation *op) {
  llvm::SmallVector<std::string, 4> usedFields;
  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};

  if (auto plt = dyn_cast<a5vm::PltB32Op>(op)) {
    const std::string resultFragment = getVectorTypeFragment(plt.getMask().getType());
    usedFields = {"family=plt", "bitwidth=32", "result=" + resultFragment,
                  "variant=v300", "scalar=i32", "scalar_out=i32"};
    return makeResolved(op, "llvm.hivm.plt.b32.v300", usedFields, resultFragment);
  }

  return failure();
}

} // namespace

FailureOr<IntrinsicSelection> selectLoadIntrinsic(Operation *op) {
  auto vlds = dyn_cast<a5vm::VldsOp>(op);
  if (!vlds)
    return failure();

  const std::string vecFragment = getVectorTypeFragment(vlds.getResult().getType());
  llvm::SmallVector<std::string, 4> usedFields = {
      "family=vldsx1", "vector=" + vecFragment};
  if (vlds.getDistAttr())
    usedFields.push_back("dist=" + (*vlds.getDist()).str());

  if (vecFragment == "v64f32")
    return makeResolved(op, "llvm.hivm.vldsx1", usedFields, vecFragment);

  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};
  std::string candidate = "llvm.hivm.vldsx1";
  return makeUnresolved(op, "vldsx1", candidate, usedFields, missingFields,
                        vecFragment);
}

FailureOr<IntrinsicSelection> selectUnaryIntrinsic(Operation *op) {
  auto vabs = dyn_cast<a5vm::VabsOp>(op);
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

  if (auto vexp = dyn_cast<a5vm::VexpOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vexp.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vexp", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vexp";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto vdup = dyn_cast<a5vm::VdupOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vdup.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vdups", "vector=" + vecFragment, "variant=z"};
    if (!isa<FloatType, IntegerType>(vdup.getInput().getType())) {
      llvm::SmallVector<std::string, 2> missingFields = {"vector_input_vdup_mapping"};
      return makeUnresolved(op, "vdup", "llvm.hivm.vdups", usedFields, missingFields,
                            vecFragment);
    }
    std::string candidate = "llvm.hivm.vdups";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".z";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<a5vm::VaddOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vadd", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vadd";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<a5vm::VsubOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vsub", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vsub";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<a5vm::VmulOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmul", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmul";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  if (auto binary = dyn_cast<a5vm::VmaxOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(binary.getResult().getType());
    llvm::SmallVector<std::string, 3> usedFields = {
        "family=vmax", "vector=" + vecFragment, "variant=x"};
    std::string candidate = "llvm.hivm.vmax";
    if (!vecFragment.empty())
      candidate += "." + vecFragment + ".x";
    return makeResolved(op, candidate, usedFields, vecFragment);
  }

  return failure();
}

FailureOr<IntrinsicSelection> selectStoreIntrinsic(Operation *op) {
  llvm::SmallVector<std::string, 4> usedFields;
  llvm::SmallVector<std::string, 2> missingFields = {"confirmed_hivm_name"};

  if (auto vsts = dyn_cast<a5vm::VstsOp>(op)) {
    const std::string vecFragment = getVectorTypeFragment(vsts.getValue().getType());
    usedFields = {"family=vstsx1", "vector=" + vecFragment,
                  "predicate_source=explicit_mask"};
    if (vsts.getDistAttr())
      usedFields.push_back("dist=" + (*vsts.getDist()).str());
    if (vecFragment == "v64f32")
      return makeResolved(op, "llvm.hivm.vstsx1", usedFields,
                          vecFragment);
    std::string candidate = "llvm.hivm.vstsx1";
    return makeUnresolved(op, "vstsx1", candidate, usedFields, missingFields,
                          vecFragment);
  }

  if (auto copy = dyn_cast<a5vm::CopyGmToUbufOp>(op)) {
    std::string elemFragment = getCopyElementFragment(op);
    usedFields = {"family=copy_gm_to_ubuf"};
    if (!elemFragment.empty())
      usedFields.push_back("element=" + elemFragment);
    if (copy.getLayoutAttr())
      usedFields.push_back("layout=" + (*copy.getLayout()).str());
    if (copy.getDataSelectBitAttr())
      usedFields.push_back(std::string("data_select_bit=") +
                           (*copy.getDataSelectBit() ? "true" : "false"));
    if (copy.getUbPadAttr())
      usedFields.push_back(std::string("ub_pad=") +
                           (*copy.getUbPad() ? "true" : "false"));
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

  if (auto copy = dyn_cast<a5vm::CopyUbufToGmOp>(op)) {
    std::string elemFragment = getCopyElementFragment(op);
    usedFields = {"family=copy_ubuf_to_gm"};
    if (!elemFragment.empty())
      usedFields.push_back("element=" + elemFragment);
    if (copy.getLayoutAttr())
      usedFields.push_back("layout=" + (*copy.getLayout()).str());
    if (elemFragment == "f32")
      return makeResolved(op, "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV",
                          usedFields, "");
    std::string candidate = "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2";
    if (!elemFragment.empty())
      candidate += "." + elemFragment + ".DV";
    missingFields.push_back("element_type_mapping");
    return makeUnresolved(op, "copy_ubuf_to_gm", candidate, usedFields,
                          missingFields, "");
  }

  if (isa<a5vm::CopyUbufToUbufOp>(op)) {
    usedFields = {"family=copy_ubuf_to_ubuf"};
    return makeUnresolved(op, "copy_ubuf_to_ubuf", "copy_ubuf_to_ubuf",
                          usedFields, missingFields, "");
  }

  return failure();
}

FailureOr<IntrinsicSelection> selectIntrinsic(Operation *op) {
  if (isa<a5vm::SetFlagOp, a5vm::WaitFlagOp, a5vm::PipeBarrierOp>(op))
    return selectSyncLike(op);

  if (isa<a5vm::SetLoop2StrideOutToUbOp, a5vm::SetLoop1StrideOutToUbOp,
          a5vm::SetLoopSizeOutToUbOp, a5vm::SetLoop2StrideUbToOutOp,
          a5vm::SetLoop1StrideUbToOutOp, a5vm::SetLoopSizeUbToOutOp>(op))
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
