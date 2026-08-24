// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- mlir_encode.cpp ----------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "PTO/Support/CodeConstants.h"
#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_format.h"

#include "ptobc/leb128.h"
#include "ptobc_opcodes_v0.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <PTO/IR/PTO.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/SourceMgr.h>

#include <llvm/ADT/DenseMap.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <unordered_map>

namespace ptobc {

namespace {

constexpr unsigned kNamedAttributeInlineCapacity = 8;
constexpr unsigned kDigitInlineCapacity = 32;
constexpr unsigned kWordInlineCapacity = 4;
constexpr unsigned kFunctionInlineCapacity = 8;
constexpr unsigned kHexadecimalRadix = 16;
constexpr unsigned kDecimalRadix = 10;
constexpr uint8_t kDefaultModuleIndexWidth = 64;
constexpr size_t kSegmentedOperandImmediateCount = 3;
constexpr uint8_t kCmpPredicateEqEncoding = 0;
constexpr uint8_t kCmpPredicateNeEncoding = 1;
constexpr uint8_t kCmpPredicateSltEncoding = 2;
constexpr uint8_t kCmpPredicateSleEncoding = 3;
constexpr uint8_t kCmpPredicateSgtEncoding = 4;
constexpr uint8_t kCmpPredicateSgeEncoding = 5;
constexpr uint16_t kTExtractOpcode = 0x1021;
constexpr uint16_t kTExtractFpWireOpcode = 0x1022;
constexpr uint16_t kTInsertOpcode = 0x102D;
constexpr uint16_t kTInsertFpWireOpcode = 0x102E;
constexpr uint16_t kTMovOpcode = 0x1038;
constexpr uint16_t kTMovFpWireOpcode = 0x1039;
constexpr uint16_t kTStoreOpcode = 0x1065;
constexpr uint16_t kTStoreFpWireOpcode = 0x1066;

using NamedAttributeVector =
    llvm::SmallVector<mlir::NamedAttribute, kNamedAttributeInlineCapacity>;
using DigitBuffer = llvm::SmallVector<char, kDigitInlineCapacity>;
using WordVector = llvm::SmallVector<uint64_t, kWordInlineCapacity>;
using FunctionVector = llvm::SmallVector<mlir::func::FuncOp,
                                         kFunctionInlineCapacity>;

} // namespace

static bool canUseLegacyTMovFpWireOpcode(mlir::pto::TMovOp op) {
  return mlir::pto::classifyTMovForm(op.getFp()) ==
             mlir::pto::TMovForm::Fp &&
         op->getNumResults() == 0 && !op.getAccToVecModeAttr() &&
         op.getReluPreMode() == mlir::pto::ReluPreMode::NoRelu;
}

static bool canUseLegacyTStoreFpWireOpcode(mlir::pto::TStoreOp op) {
  return op->getNumResults() == 0 &&
         op.getAtomicType() == mlir::pto::AtomicType::AtomicNone &&
         op.getReluPreMode() == mlir::pto::ReluPreMode::NoRelu;
}

static bool shouldEncodeViaGenericV0CompatibilityShim(mlir::Operation &op) {
  // PTOBC v0 already shipped fixed-width known-op payloads for legacy
  // pto.tci / pto.trowexpandadd forms without tmp. Newer tmp-operand forms
  // must not reuse those schemas or older .ptobc files would become
  // undecodable, so serialize the new forms through the generic v0 opcode.
  if (auto tci = llvm::dyn_cast<mlir::pto::TCIOp>(&op)) {
    return static_cast<bool>(tci.getTmp());
  }
  if (auto trowexpandadd = llvm::dyn_cast<mlir::pto::TRowExpandAddOp>(&op)) {
    return static_cast<bool>(trowexpandadd.getTmp());
  }
  // The shipped tfree schemas have no operands. Preserve them for the legacy
  // form and use the generic record when the optional pipe entry is present.
  if (auto tfreeFromAic = llvm::dyn_cast<mlir::pto::TFreeFromAicOp>(&op)) {
    return static_cast<bool>(tfreeFromAic.getEntry());
  }
  if (auto tfreeFromAiv = llvm::dyn_cast<mlir::pto::TFreeFromAivOp>(&op)) {
    return static_cast<bool>(tfreeFromAiv.getEntry());
  }
  // tquant.mx intentionally has no v0 known-op schema. Preserve its complete
  // operand and attribute dictionary through the existing generic record even
  // when PTOBC_ALLOW_GENERIC is not enabled globally.
  if (llvm::isa<mlir::pto::TQuantMxOp>(&op)) {
    return true;
  }
  // The compact v0 schemas for these ops predate their optional pre-quant
  // operands. Keep the shipped fixed payloads unchanged and use the generic
  // opcode for forms that cannot be represented by those schemas.
  if (auto textract = llvm::dyn_cast<mlir::pto::TExtractOp>(&op)) {
    return static_cast<bool>(textract.getPreQuantScalar());
  }
  if (auto tinsert = llvm::dyn_cast<mlir::pto::TInsertOp>(&op)) {
    return static_cast<bool>(tinsert.getPreQuantScalar());
  }
  if (auto tmov = llvm::dyn_cast<mlir::pto::TMovOp>(&op)) {
    if (mlir::pto::classifyTMovForm(tmov.getFp()) ==
        mlir::pto::TMovForm::XToZz) {
      return true;
    }
    if (tmov.getPreQuantScalar()) {
      return true;
    }
    // The removed pto.tmov.fp op carried only src/fp/dst. Its legacy opcode
    // would silently discard mode/relu semantics when read by an older PTOAS.
    // The pre-unification pto.tmov op already understood fp plus these attrs,
    // so use the generic v0 record for the extended unified form.
    return tmov.getFp() && !canUseLegacyTMovFpWireOpcode(tmov);
  }
  // The fixed pto.tstore schema also predates its optional tensor result.
  // Result-bearing non-fp forms are readable by older unified pto.tstore
  // readers through generic v0. The fp form is rejected separately because
  // no pre-unification operation represented both fp and a result.
  if (auto tstore = llvm::dyn_cast<mlir::pto::TStoreOp>(&op)) {
    return tstore->getNumResults() != 0;
  }
  if (llvm::isa<mlir::pto::CmoCacheInvalidOp, mlir::pto::FenceBarrierAllOp>(
          &op)) {
    return true;
  }
  // The aiv_subblockid operand extends pipe transfer forms after the v0
  // known-op schemas were fixed. Keep the legacy compact payloads unchanged
  // and route only the new optional-operand forms through generic encoding.
  if (auto push = llvm::dyn_cast<mlir::pto::TPushToAicOp>(&op)) {
    return static_cast<bool>(push.getAivSubblockid());
  }
  if (auto pop = llvm::dyn_cast<mlir::pto::TPopFromAicOp>(&op)) {
    return static_cast<bool>(pop.getAivSubblockid());
  }
  if (auto push = llvm::dyn_cast<mlir::pto::TPushOp>(&op)) {
    return static_cast<bool>(push.getAivSubblockid());
  }
  if (auto pop = llvm::dyn_cast<mlir::pto::TPopOp>(&op)) {
    return static_cast<bool>(pop.getAivSubblockid());
  }
  return false;
}

static std::optional<uint16_t> getLegacyFpWireOpcode(mlir::Operation &op) {
  if (auto textract = llvm::dyn_cast<mlir::pto::TExtractOp>(&op)) {
    return textract.getFp() ? std::optional<uint16_t>(kTExtractFpWireOpcode)
                            : std::nullopt;
  }
  if (auto tinsert = llvm::dyn_cast<mlir::pto::TInsertOp>(&op)) {
    return tinsert.getFp() ? std::optional<uint16_t>(kTInsertFpWireOpcode)
                           : std::nullopt;
  }
  if (auto tmov = llvm::dyn_cast<mlir::pto::TMovOp>(&op)) {
    return tmov.getFp() && canUseLegacyTMovFpWireOpcode(tmov)
               ? std::optional<uint16_t>(kTMovFpWireOpcode)
               : std::nullopt;
  }
  if (auto tstore = llvm::dyn_cast<mlir::pto::TStoreOp>(&op)) {
    return tstore.getFp() && canUseLegacyTStoreFpWireOpcode(tstore)
               ? std::optional<uint16_t>(kTStoreFpWireOpcode)
               : std::nullopt;
  }
  return std::nullopt;
}

static std::optional<std::string>
getUnsupportedV0EncodingReason(mlir::Operation &op) {
  if (auto tmov = llvm::dyn_cast<mlir::pto::TMovOp>(&op);
      tmov && mlir::pto::classifyTMovForm(tmov.getFp()) ==
                  mlir::pto::TMovForm::Fp &&
      tmov->getNumResults() != 0) {
    return "pto.tmov fp with a result cannot be represented safely in "
           "PTO-BC v0; legacy opcode 0x1039 would silently drop the result, "
           "and PTOAS backends do not lower the generic result-bearing form";
  }

  auto tstore = llvm::dyn_cast<mlir::pto::TStoreOp>(&op);
  if (!tstore || !tstore.getFp() || canUseLegacyTStoreFpWireOpcode(tstore)) {
    return std::nullopt;
  }

  return "pto.tstore fp with a result or non-default atomicType/reluPreMode "
         "cannot be represented safely in PTO-BC v0; legacy opcode 0x1066 "
         "would silently drop those semantics";
}

static bool omitsDerivedOperandSegmentsInV0(uint16_t opcode) {
  switch (opcode) {
  case kTExtractOpcode:
  case kTExtractFpWireOpcode:
  case kTInsertOpcode:
  case kTInsertFpWireOpcode:
  case kTMovOpcode:
  case kTMovFpWireOpcode:
  case kTStoreOpcode:
  case kTStoreFpWireOpcode:
    return true;
  default:
    return false;
  }
}

static uint64_t internType(PTOBCFile& f, mlir::Type t) {
  std::string s = printType(t);
  f.strings.intern(s);
  // type ids are 1-based
  for (size_t i = 0; i < f.typeAsm.size(); ++i) {
    if (f.typeAsm[i] == s) {
      return i + 1;
    }
  }
  f.typeAsm.push_back(s);
  return f.typeAsm.size();
}

static mlir::DictionaryAttr stripAttrs(mlir::MLIRContext *ctx,
                                       mlir::DictionaryAttr dict,
                                       llvm::ArrayRef<llvm::StringRef> keys) {
  if (!dict || dict.empty() || keys.empty()) {
    return dict;
  }

  llvm::SmallVector<mlir::NamedAttribute, mlir::pto::kValue8> keep;
  keep.reserve(dict.size());
  for (auto na : dict) {
    if (llvm::is_contained(keys, na.getName().getValue())) {
      continue;
    }
    keep.push_back(na);
  }
  if (keep.size() == dict.size()) {
    return dict;
  }
  return mlir::DictionaryAttr::get(ctx, keep);
}

static mlir::DictionaryAttr stripKnownImmediateAttrs(
    mlir::MLIRContext *ctx, mlir::DictionaryAttr dict,
    const ptobc::v0::OpInfo &info) {
  switch (info.imm_kind) {
  case 0x01:
    return stripAttrs(ctx, dict, {"predicate"});
  case 0x02:
    return stripAttrs(ctx, dict, {"src_op", "dst_op", "event_id"});
  case 0x05:
    return stripAttrs(ctx, dict, {"value"});
  default:
    return dict;
  }
}

static bool isFrontendPipeOpWithDefaultId(mlir::Operation &op) {
  llvm::StringRef name = op.getName().getStringRef();
  return name == "pto.aic_initialize_pipe" ||
         name == "pto.aiv_initialize_pipe" ||
         name == "pto.talloc_to_aiv" || name == "pto.talloc_to_aic" ||
         name == "pto.tpush_to_aiv" || name == "pto.tpush_to_aic" ||
         name == "pto.tpop_from_aic" || name == "pto.tpop_from_aiv" ||
         name == "pto.tfree_from_aic" || name == "pto.tfree_from_aiv";
}

static mlir::DictionaryAttr
dropDefaultZeroPipeIdForV0Encoding(mlir::Operation &op,
                                   mlir::DictionaryAttr dict) {
  if (!dict || dict.empty() || !isFrontendPipeOpWithDefaultId(op)) {
    return dict;
  }

  auto idAttr = dict.getAs<mlir::IntegerAttr>("id");
  if (!idAttr || idAttr.getInt() != 0) {
    return dict;
  }

  llvm::SmallVector<mlir::NamedAttribute, kNamedAttributeInlineCapacity> keep;
  keep.reserve(dict.size());
  for (auto attr : dict) {
    if (attr.getName() == "id") {
      continue;
    }
    keep.push_back(attr);
  }
  return mlir::DictionaryAttr::get(op.getContext(), keep);
}

static uint64_t internAttr(PTOBCFile& f, mlir::DictionaryAttr dict) {
  if (!dict || dict.empty()) {
    return 0;
  }
  std::string s = printAttrDict(dict);
  f.strings.intern(s);
  for (size_t i = 0; i < f.attrAsm.size(); ++i) {
    if (f.attrAsm[i] == s) {
      return i + 1;
    }
  }
  f.attrAsm.push_back(s);
  return f.attrAsm.size();
}

static std::string hexFloatLiteral(mlir::FloatAttr a) {
  DigitBuffer digits;
  llvm::APInt bits = a.getValue().bitcastToAPInt();
  bits.toString(digits, /*Radix=*/kHexadecimalRadix, /*Signed=*/false,
                /*formatAsCLiteral=*/true);
  return std::string(digits.data(), digits.size());
}

static std::string apIntToSignedDecimal(const llvm::APInt &v) {
  DigitBuffer digits;
  v.toString(digits, /*Radix=*/kDecimalRadix, /*Signed=*/true,
             /*formatAsCLiteral=*/false);
  return std::string(digits.data(), digits.size());
}

static WordVector copyAPIntWords(const llvm::APInt &bits) {
  return WordVector(bits.getRawData(), bits.getRawData() + bits.getNumWords());
}

static void appendAPIntBytesLE(Buffer &buffer, const llvm::APInt &bits) {
  const unsigned byteLen = (bits.getBitWidth() + 7) / 8;
  writeULEB128(byteLen, buffer.bytes);

  WordVector words = copyAPIntWords(bits);
  for (unsigned i = 0; i < byteLen; ++i) {
    unsigned word = i / 8;
    unsigned off = (i % 8) * 8;
    uint8_t byte = uint8_t((words[word] >> off) & 0xFFU);
    buffer.bytes.push_back(byte);
  }
}

static std::optional<std::string>
buildScalarConstantDebugName(mlir::Value value,
                             std::unordered_map<std::string, int> &constCounts) {
  auto cst = llvm::dyn_cast_or_null<mlir::arith::ConstantOp>(
      value.getDefiningOp());
  if (!cst) {
    return std::nullopt;
  }

  mlir::Attribute attr = cst.getValue();
  std::string typeName = printType(value.getType());
  std::string baseName;
  if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    baseName = "c" + hexFloatLiteral(floatAttr) + "_" + typeName;
  } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    baseName = "c" + apIntToSignedDecimal(intAttr.getValue());
    if (typeName != "index") {
      baseName += "_" + typeName;
    }
  } else {
    return std::nullopt;
  }

  int &count = constCounts[baseName];
  std::string name = baseName;
  if (count > 0) {
    name += "_" + std::to_string(count);
  }
  ++count;
  return name;
}

struct Encoder {
  PTOBCFile file;

  bool emitDebugInfo = false;
  bool allowGeneric = false;

  // constpool dedup: key is raw bytes: tag + payload
  std::unordered_map<std::string, uint64_t> constIdByKey;

  // Per-function numbering state.
  uint64_t funcId = 0;
  uint64_t nextOpId = 0;
  llvm::DenseMap<mlir::Value, uint64_t> valueId;
  std::vector<mlir::Value> valueById;

  // Module-wide debug file table state.
  std::unordered_map<std::string, uint64_t> dbgFileIdByPath;

  uint64_t getValueId(mlir::Value v) {
    auto it = valueId.find(v);
    if (it == valueId.end()) {
      throw std::runtime_error("operand references undefined value");
    }
    return it->second;
  }

  uint64_t allocValueId(mlir::Value v) {
    uint64_t id = valueId.size();
    auto [it, inserted] = valueId.try_emplace(v, id);
    if (!inserted) {
      throw std::runtime_error("value already has an id");
    }
    valueById.push_back(v);
    return it->second;
  }

  uint64_t internDbgFile(llvm::StringRef path) {
    auto p = path.str();
    auto it = dbgFileIdByPath.find(p);
    if (it != dbgFileIdByPath.end()) {
      return it->second;
    }

    uint64_t sid = file.strings.intern(p);
    uint64_t fileId = file.dbgFiles.size();
    file.dbgFiles.push_back(DebugFileEntry{sid, /*hashKind=*/0, {}});
    dbgFileIdByPath.emplace(std::move(p), fileId);
    return fileId;
  }

  void recordOpLocation(uint64_t opId, mlir::Operation &op) {
    if (!emitDebugInfo) {
      return;
    }
    auto loc = op.getLoc();
    auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(loc);
    if (!flc) {
      return;
    }

    uint64_t fileId = internDbgFile(flc.getFilename().getValue());
    uint64_t sl = flc.getLine();
    uint64_t sc = flc.getColumn();
    uint64_t el = sl;
    uint64_t ec = sc + 1; // point-range

    file.dbgLocations.push_back(DebugLocationEntry{funcId, opId, fileId, sl, sc, el, ec});
  }

  void finalizeValueNamesForFunction() {
    if (!emitDebugInfo) {
      return;
    }
    // Deterministic value names for DebugInfo.
    std::unordered_map<std::string, int> constCounts;

    for (uint64_t vid = 0; vid < valueById.size(); ++vid) {
      uint64_t nameSid = file.strings.intern(
          buildScalarConstantDebugName(valueById[vid], constCounts)
              .value_or(std::to_string(vid)));
      file.dbgValueNames.push_back(DebugValueNameEntry{funcId, vid, nameSid});
    }
  }

  uint64_t internConst(uint8_t tag, const std::vector<uint8_t> &payload) {
    std::string key;
    key.resize(1 + payload.size());
    key[0] = char(tag);
    if (!payload.empty()) {
      std::copy(payload.begin(), payload.end(), key.begin() + 1);
    }
    auto it = constIdByKey.find(key);
    if (it != constIdByKey.end()) {
      return it->second;
    }
    uint64_t id = file.consts.size();
    file.consts.push_back(ConstEntry{tag, payload});
    constIdByKey.emplace(std::move(key), id);
    return id;
  }

  uint64_t internConstInt64(uint64_t typeId, int64_t value) {
    Buffer p;
    writeULEB128(typeId, p.bytes);
    writeSLEB128(value, p.bytes);
    return internConst(/*tag=*/0x01, p.bytes);
  }

  uint64_t internConstBits(uint8_t tag, uint64_t typeId,
                           const llvm::APInt &bits) {
    Buffer p;
    writeULEB128(typeId, p.bytes);
    appendAPIntBytesLE(p, bits);
    return internConst(tag, p.bytes);
  }

  uint64_t internConstIntBits(uint64_t typeId, const llvm::APInt &bits) {
    return internConstBits(/*tag=*/0x04, typeId, bits);
  }

  uint64_t internConstFloatBits(uint64_t dtypeId, const llvm::APInt &bits) {
    return internConstBits(/*tag=*/0x02, dtypeId, bits);
  }

  void resetForFunction(uint64_t fid) {
    funcId = fid;
    nextOpId = 0;
    valueId.clear();
    valueById.clear();
  }

  void encodeKnownOpImmediates(mlir::Operation &op, Buffer &out,
                               const ptobc::v0::OpInfo &info,
                               const ptobc::v0::OpcodeAndVariant &variantInfo,
                               llvm::SmallVectorImpl<uint64_t> &imms);
  void encodeKnownOpOperands(mlir::Operation &op, Buffer &out,
                             const ptobc::v0::OpInfo &info,
                             const ptobc::v0::OpcodeAndVariant &variantInfo,
                             llvm::ArrayRef<uint64_t> imms);
  void encodeKnownOp(mlir::Operation &op, Buffer &out,
                     const ptobc::v0::OpInfo &info,
                     const ptobc::v0::OpcodeAndVariant &variantInfo);
  void encodeGenericOp(mlir::Operation &op, Buffer &out);
  void encodeRegion(mlir::Region& region, Buffer& out);
  void encodeBlock(mlir::Block& block, Buffer& out);
  void encodeOp(mlir::Operation& op, Buffer& out);
};

void Encoder::encodeRegion(mlir::Region& region, Buffer& out) {
  writeULEB128(region.getBlocks().size(), out.bytes);
  for (auto& block : region.getBlocks()) {
    encodeBlock(block, out);
  }
}

void Encoder::encodeBlock(mlir::Block& block, Buffer& out) {
  // block args
  writeULEB128(block.getNumArguments(), out.bytes);
  for (auto arg : block.getArguments()) {
    writeULEB128(internType(file, arg.getType()), out.bytes);
    allocValueId(arg);
  }

  // ops count
  size_t opCount = 0;
  for (auto& op : block.getOperations()) {
    (void)op, ++opCount;
  }
  writeULEB128(opCount, out.bytes);

  for (auto& op : block.getOperations()) {
    encodeOp(op, out);
  }
}

void Encoder::encodeKnownOpImmediates(
    mlir::Operation &op, Buffer &out, const ptobc::v0::OpInfo &info,
    const ptobc::v0::OpcodeAndVariant &variantInfo,
    llvm::SmallVectorImpl<uint64_t> &imms) {
  switch (info.imm_kind) {
  case 0x00:
    return;
  case 0x01: {
    auto cmp = llvm::dyn_cast<mlir::arith::CmpIOp>(&op);
    if (!cmp) {
      throw std::runtime_error("imm_kind=cmpi_pred but op is not arith.cmpi");
    }
    uint8_t predicate = 0;
    switch (cmp.getPredicate()) {
    case mlir::arith::CmpIPredicate::eq:
      predicate = kCmpPredicateEqEncoding;
      break;
    case mlir::arith::CmpIPredicate::ne:
      predicate = kCmpPredicateNeEncoding;
      break;
    case mlir::arith::CmpIPredicate::slt:
      predicate = kCmpPredicateSltEncoding;
      break;
    case mlir::arith::CmpIPredicate::sle:
      predicate = kCmpPredicateSleEncoding;
      break;
    case mlir::arith::CmpIPredicate::sgt:
      predicate = kCmpPredicateSgtEncoding;
      break;
    case mlir::arith::CmpIPredicate::sge:
      predicate = kCmpPredicateSgeEncoding;
      break;
    default:
      throw std::runtime_error(
          "unsupported arith.cmpi predicate (v0 supports only eq/ne/slt/sle/sgt/sge)");
    }
    out.appendU8(predicate);
    imms.push_back(predicate);
    return;
  }
  case 0x02: {
    auto src = op.getAttrOfType<mlir::pto::SyncOpTypeAttr>("src_op");
    auto dst = op.getAttrOfType<mlir::pto::SyncOpTypeAttr>("dst_op");
    auto event = op.getAttrOfType<mlir::pto::EventAttr>("event_id");
    if (!src || !dst || !event) {
      throw std::runtime_error("event op missing src_op/dst_op/event_id attrs");
    }
    uint8_t srcValue = uint8_t(src.getOpType());
    uint8_t dstValue = uint8_t(dst.getOpType());
    uint8_t eventValue = uint8_t(event.getEvent());
    out.appendU8(srcValue);
    out.appendU8(dstValue);
    out.appendU8(eventValue);
    imms.append({srcValue, dstValue, eventValue});
    return;
  }
  case 0x05: {
    auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(&op);
    if (!cst) {
      throw std::runtime_error("imm_kind=const_id but op is not arith.constant");
    }

    mlir::Attribute attr = cst.getValue();
    uint64_t constId = 0;
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      uint64_t typeId = internType(file, cst.getType());
      const llvm::APInt &value = intAttr.getValue();
      constId = value.getBitWidth() <= 64 ? internConstInt64(typeId, value.getSExtValue())
                                          : internConstIntBits(typeId, value);
    } else if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
      uint64_t typeId = internType(file, cst.getType());
      constId = internConstFloatBits(typeId,
                                     floatAttr.getValue().bitcastToAPInt());
    } else {
      throw std::runtime_error(
          "unsupported arith.constant attribute kind for compact v0");
    }
    writeULEB128(constId, out.bytes);
    imms.push_back(constId);
    return;
  }
  case 0x06: {
    auto mtv = llvm::dyn_cast<mlir::pto::MakeTensorViewOp>(&op);
    if (!mtv) {
      throw std::runtime_error(
          "imm_kind=make_tensor_view but op is not pto.make_tensor_view");
    }
    uint8_t listMode = 0;
    out.appendU8(listMode);
    writeULEB128(mtv.getShape().size(), out.bytes);
    writeULEB128(mtv.getStrides().size(), out.bytes);
    imms.append({listMode, uint64_t(mtv.getShape().size()),
                 uint64_t(mtv.getStrides().size())});
    return;
  }
  case 0x07: {
    auto pv = llvm::dyn_cast<mlir::pto::PartitionViewOp>(&op);
    if (!pv) {
      throw std::runtime_error(
          "imm_kind=partition_view but op is not pto.partition_view");
    }
    uint8_t listMode = 0;
    out.appendU8(listMode);
    writeULEB128(pv.getOffsets().size(), out.bytes);
    writeULEB128(pv.getSizes().size(), out.bytes);
    imms.append({listMode, uint64_t(pv.getOffsets().size()),
                 uint64_t(pv.getSizes().size())});
    return;
  }
  case 0x08: {
    auto at = llvm::dyn_cast<mlir::pto::AllocTileOp>(&op);
    if (!at) {
      throw std::runtime_error(
          "imm_kind=alloc_tile but op is not pto.alloc_tile");
    }
    uint8_t mask = 0;
    if (at.getValidRow()) {
      mask |= 0x1;
    }
    if (at.getValidCol()) {
      mask |= 0x2;
    }
    if (at.getAddr()) {
      mask |= 0x4;
    }
    out.appendU8(mask);
    imms.push_back(mask);
    return;
  }
  default:
    (void)variantInfo;
    throw std::runtime_error("unknown imm_kind in v0 schema");
  }
}

void Encoder::encodeKnownOpOperands(
    mlir::Operation &op, Buffer &out, const ptobc::v0::OpInfo &info,
    const ptobc::v0::OpcodeAndVariant &variantInfo,
    llvm::ArrayRef<uint64_t> imms) {
  auto emitOperands = [&](size_t count) {
    if (op.getNumOperands() != count) {
      throw std::runtime_error("operand count mismatch for op: " +
                               op.getName().getStringRef().str());
    }
    for (auto value : op.getOperands()) {
      writeULEB128(getValueId(value), out.bytes);
    }
  };
  auto emitLegacyIndexedTscatterOperands = [&]() {
    auto tscatter = llvm::dyn_cast<mlir::pto::TScatterOp>(&op);
    if (!tscatter || tscatter.getMaskPatternAttr() ||
        variantInfo.opcode != 0x1056) {
      return false;
    }
    if (op.getNumOperands() != 3) {
      throw std::runtime_error("operand count mismatch for op: " +
                               op.getName().getStringRef().str());
    }
    // Preserve the historical v0 wire layout for indexed tscatter:
    //   (src, indexes, dst)
    writeULEB128(getValueId(tscatter.getSrc()), out.bytes);
    writeULEB128(getValueId(tscatter.getIndexes()), out.bytes);
    writeULEB128(getValueId(tscatter.getDst()), out.bytes);
    return true;
  };
  auto emitLegacyFpOperands = [&]() {
    auto emit = [&](llvm::ArrayRef<mlir::Value> operands) {
      for (mlir::Value value : operands) {
        writeULEB128(getValueId(value), out.bytes);
      }
    };
    switch (variantInfo.opcode) {
    case kTExtractFpWireOpcode: {
      auto fpOp = llvm::cast<mlir::pto::TExtractOp>(&op);
      emit({fpOp.getSrc(), fpOp.getFp(), fpOp.getIndexRow(), fpOp.getIndexCol(),
            fpOp.getDst()});
      return true;
    }
    case kTInsertFpWireOpcode: {
      auto fpOp = llvm::cast<mlir::pto::TInsertOp>(&op);
      emit({fpOp.getSrc(), fpOp.getFp(), fpOp.getIndexRow(), fpOp.getIndexCol(),
            fpOp.getDst()});
      return true;
    }
    case kTMovFpWireOpcode: {
      auto fpOp = llvm::cast<mlir::pto::TMovOp>(&op);
      emit({fpOp.getSrc(), fpOp.getFp(), fpOp.getDst()});
      return true;
    }
    case kTStoreFpWireOpcode: {
      auto fpOp = llvm::cast<mlir::pto::TStoreOp>(&op);
      emit({fpOp.getSrc(), fpOp.getFp(), fpOp.getDst()});
      return true;
    }
    default:
      return false;
    }
  };
  if (emitLegacyFpOperands()) {
    return;
  }

  switch (info.operand_mode) {
  case 0x00:
    if (emitLegacyIndexedTscatterOperands()) {
      return;
    }
    emitOperands(info.num_operands);
    return;
  case 0x01: {
    auto count =
        ptobc::v0::lookupOperandsByVariant(variantInfo.opcode, variantInfo.variant);
    if (!count) {
      throw std::runtime_error("missing by-variant operand count");
    }
    emitOperands(*count);
    return;
  }
  case 0x02:
    writeULEB128(op.getNumOperands(), out.bytes);
    for (auto value : op.getOperands()) {
      writeULEB128(getValueId(value), out.bytes);
    }
    return;
  case 0x03: {
    if (imms.size() < kSegmentedOperandImmediateCount) {
      throw std::runtime_error("segmented operands missing immediates");
    }
    if (imms[0] != 0) {
      throw std::runtime_error(
          "list_mode=1 not implemented in ptobc encoder yet");
    }
    emitOperands(size_t(info.num_operands) + size_t(imms[1]) + size_t(imms[mlir::pto::kValue2]));
    return;
  }
  case 0x04: {
    if (imms.empty()) {
      throw std::runtime_error("optmask operands missing immediate");
    }
    uint8_t mask = uint8_t(imms.front());
    emitOperands(((mask & 0x1) ? 1 : 0) + ((mask & 0x2) ? 1 : 0) +
                 ((mask & 0x4) ? 1 : 0));
    return;
  }
  default:
    throw std::runtime_error("unknown operand_mode in v0 schema");
  }
}

void Encoder::encodeKnownOp(mlir::Operation &op, Buffer &out,
                            const ptobc::v0::OpInfo &info,
                            const ptobc::v0::OpcodeAndVariant &variantInfo) {
  for (auto result : op.getResults()) {
    allocValueId(result);
  }

  out.appendU16LE(variantInfo.opcode);
  mlir::DictionaryAttr dict = op.getAttrDictionary();
  dict = stripKnownImmediateAttrs(op.getContext(), dict, info);
  if (omitsDerivedOperandSegmentsInV0(variantInfo.opcode)) {
    dict = stripAttrs(op.getContext(), dict, {"operandSegmentSizes"});
  }
  // The assembly printer omits default pipe ids on transfer/free ops. Keep v0
  // byte roundtrips stable by using the same representation while encoding.
  dict = dropDefaultZeroPipeIdForV0Encoding(op, dict);
  writeULEB128(internAttr(file, dict), out.bytes);

  if (info.has_variant_u8) {
    out.appendU8(variantInfo.variant);
  }

  WordVector imms;
  encodeKnownOpImmediates(op, out, info, variantInfo, imms);
  encodeKnownOpOperands(op, out, info, variantInfo, imms);

  if (info.result_type_mode == 0x01) {
    if (op.getNumResults() != info.num_results) {
      throw std::runtime_error("result count mismatch for op: " +
                               op.getName().getStringRef().str());
    }
    for (auto result : op.getResults()) {
      writeULEB128(internType(file, result.getType()), out.bytes);
    }
  } else if (info.result_type_mode == 0x02) {
    writeULEB128(op.getNumResults(), out.bytes);
    for (auto result : op.getResults()) {
      writeULEB128(internType(file, result.getType()), out.bytes);
    }
  } else if (info.result_type_mode != 0x00) {
    throw std::runtime_error("unknown result_type_mode in v0 schema");
  }

  if (op.getNumRegions() != info.num_regions) {
    throw std::runtime_error("region count mismatch for op: " +
                             op.getName().getStringRef().str());
  }
  for (auto &region : op.getRegions()) {
    encodeRegion(region, out);
  }
}

void Encoder::encodeGenericOp(mlir::Operation &op, Buffer &out) {
  out.appendU16LE(kOpcodeGeneric);
  writeULEB128(internAttr(file, op.getAttrDictionary()), out.bytes);

  auto opNameSid = file.strings.intern(op.getName().getStringRef().str());
  writeULEB128(opNameSid, out.bytes);

  writeULEB128(op.getNumResults(), out.bytes);
  for (auto result : op.getResults()) {
    allocValueId(result);
    writeULEB128(internType(file, result.getType()), out.bytes);
  }

  writeULEB128(op.getNumOperands(), out.bytes);
  for (auto operand : op.getOperands()) {
    writeULEB128(getValueId(operand), out.bytes);
  }

  writeULEB128(op.getNumRegions(), out.bytes);
  for (auto &region : op.getRegions()) {
    encodeRegion(region, out);
  }
}

void Encoder::encodeOp(mlir::Operation& op, Buffer& out) {
  if (emitDebugInfo) {
    uint64_t opId = nextOpId++;
    recordOpLocation(opId, op);
  }

  auto fullName = op.getName().getStringRef();
  if (auto reason = getUnsupportedV0EncodingReason(op)) {
    throw std::runtime_error(*reason);
  }

  if (auto tscatter = llvm::dyn_cast<mlir::pto::TScatterOp>(&op)) {
    uint16_t opcode = tscatter.getMaskPatternAttr()
                          ? ptobc::v0::kTscatterMaskOpcode
                          : uint16_t(0x1056);
    auto variantInfo =
        ptobc::v0::OpcodeAndVariant{opcode, /*hasVariant=*/0, /*variant=*/0};
    const auto *info = ptobc::v0::lookupByOpcode(opcode);
    if (!info) {
      throw std::runtime_error("missing v0 opcode schema for op: " +
                               fullName.str());
    }
    encodeKnownOp(op, out, *info, variantInfo);
    return;
  }

  if (shouldEncodeViaGenericV0CompatibilityShim(op)) {
    encodeGenericOp(op, out);
    return;
  }

  if (auto opcode = getLegacyFpWireOpcode(op)) {
    auto variantInfo =
        ptobc::v0::OpcodeAndVariant{*opcode, /*hasVariant=*/0, /*variant=*/0};
    const auto *info = ptobc::v0::lookupByOpcode(*opcode);
    if (!info) {
      throw std::runtime_error("missing legacy FP v0 opcode schema for op: " +
                               fullName.str());
    }
    encodeKnownOp(op, out, *info, variantInfo);
    return;
  }

  auto variantInfo = ptobc::v0::lookupOpcodeAndVariantByFullName(fullName);
  if (variantInfo) {
    const auto *info = ptobc::v0::lookupByOpcode(variantInfo->opcode);
    if (!info) {
      throw std::runtime_error("missing v0 opcode schema for op: " +
                               fullName.str());
    }
    encodeKnownOp(op, out, *info, *variantInfo);
    return;
  }

  if (!allowGeneric) {
    throw std::runtime_error(
        "op is not in v0 opcode table (and PTOBC_ALLOW_GENERIC is not set): " +
        fullName.str());
  }
  encodeGenericOp(op, out);
}

PTOBCFile encodeFromMLIRModule(mlir::ModuleOp module) {
  Encoder enc;
  enc.emitDebugInfo = (std::getenv("PTOBC_EMIT_DEBUGINFO") != nullptr);
  enc.allowGeneric = (std::getenv("PTOBC_ALLOW_GENERIC") != nullptr);

  // Pre-intern a few common strings to stabilize ids.
  enc.file.strings.intern("func.func");
  enc.file.strings.intern("func.return");

  // MODULE encoding
  Buffer m;
  // profile_id=0 (unspecified), index_width=64
  m.appendU8(0);
  m.appendU8(kDefaultModuleIndexWidth);

  // module_attr_id
  uint64_t modAttrId = internAttr(enc.file, module->getAttrDictionary());
  writeULEB128(modAttrId, m.bytes);

  // globals count
  writeULEB128(0, m.bytes);

  // function decls (top-level order)
  FunctionVector funcs;
  for (auto f : module.getOps<mlir::func::FuncOp>()) {
    funcs.push_back(f);
  }

  writeULEB128(funcs.size(), m.bytes);

  // encode decls
  for (auto f : funcs) {
    auto nameSid = enc.file.strings.intern(f.getName().str());
    // func type as opaque asm in type table
    auto funcTypeId = internType(enc.file, f.getFunctionType());
    // flags: bit0 import? (0)
    uint8_t flags = 0;
    auto funcAttrId = internAttr(enc.file, f->getAttrDictionary());

    writeULEB128(nameSid, m.bytes);
    writeULEB128(funcTypeId, m.bytes);
    m.appendU8(flags);
    writeULEB128(funcAttrId, m.bytes);
  }

  // bodies: for each function, encode its body region
  for (size_t i = 0; i < funcs.size(); ++i) {
    auto f = funcs[i];
    enc.resetForFunction(i);

    // function body is region #0
    enc.encodeRegion(f.getBody(), m);

    // DebugInfo: deterministic value names for this function.
    enc.finalizeValueNamesForFunction();
  }

  enc.file.moduleBytes = std::move(m.bytes);
  return enc.file;
}

mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path) {
  llvm::SourceMgr sm;
  std::string err;
  auto file = mlir::openInputFile(path, &err);
  if (!file) {
    throw std::runtime_error("failed to open input: " + path + (err.empty() ? "" : (": " + err)));
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &ctx);
  if (!module) {
    throw std::runtime_error("failed to parse MLIR file: " + path);
  }
  return module;
}

} // namespace ptobc
