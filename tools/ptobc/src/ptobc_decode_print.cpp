// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/Support/CodeConstants.h"
#include "ptobc/canonical_printer.h"
#include "ptobc/leb128.h"
#include "ptobc/mlir_codec.h"
#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_decode.h"
#include "ptobc/ptobc_format.h"
#include "ptobc_opcodes_v0.h"

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <PTO/IR/PTO.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/raw_ostream.h>

#include <climits>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <optional>
#include <stdexcept>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>

namespace ptobc {

static bool debugEnabled() { return std::getenv("PTOBC_DEBUG") != nullptr; }

constexpr unsigned kBitsPerByte = CHAR_BIT;
constexpr unsigned kWordInlineCapacity = 4;
constexpr unsigned kValueIdInlineCapacity = 8;
constexpr unsigned kOperandInlineCapacity = 8;
constexpr unsigned kResultTypeInlineCapacity = 4;
constexpr size_t kBytesPerWord = sizeof(uint64_t);
constexpr char kPTOBCMagic[] = "PTOBC\0";
constexpr size_t kPTOBCMagicSize = 6;
constexpr size_t kPTOBCHeaderSize = 14;
constexpr size_t kPTOBCVersionOffset = 6;
constexpr size_t kPTOBCPayloadLengthOffset = 10;
constexpr uint16_t kTExtractOpcode = 0x1021;
constexpr uint16_t kTExtractFpWireOpcode = 0x1022;
constexpr uint16_t kTInsertOpcode = 0x102D;
constexpr uint16_t kTInsertFpWireOpcode = 0x102E;
constexpr uint16_t kTMovOpcode = 0x1038;
constexpr uint16_t kTMovFpWireOpcode = 0x1039;
constexpr uint16_t kTStoreOpcode = 0x1065;
constexpr uint16_t kTStoreFpWireOpcode = 0x1066;

struct Reader {
  const uint8_t *p;
  const uint8_t *end;

  uint8_t readU8() {
    if (p >= end) {
      throw std::runtime_error("EOF");
    }
    return *p++;
  }
  uint16_t readU16LE() {
    uint16_t lo = readU8();
    uint16_t hi = readU8();
    return lo | (hi << kBitsPerByte);
  }
  uint32_t readU32LE() {
    uint32_t b0 = readU8();
    uint32_t b1 = readU8();
    uint32_t b2 = readU8();
    uint32_t b3 = readU8();
    return b0 | (b1 << kBitsPerByte) | (b2 << (kBitsPerByte * 2U)) |
           (b3 << (kBitsPerByte * 3U));
  }
  uint64_t readULEB() {
    uint64_t v;
    size_t n = readULEB128(p, size_t(end - p), v);
    p += n;
    return v;
  }

  int64_t readSLEB() {
    int64_t v;
    size_t n = readSLEB128(p, size_t(end - p), v);
    p += n;
    return v;
  }

  std::vector<uint8_t> readBytes(size_t n) {
    if (size_t(end - p) < n) {
      throw std::runtime_error("EOF");
    }
    std::vector<uint8_t> out(p, p + n);
    p += n;
    return out;
  }
};

static void parseStringsSection(const std::vector<uint8_t> &data,
                                std::vector<std::string> &strings) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  strings.clear();
  strings.reserve(cnt);
  for (uint64_t i = 0; i < cnt; ++i) {
    uint64_t len = r.readULEB();
    auto bs = r.readBytes(len);
    strings.emplace_back(bs.begin(), bs.end());
  }
  if (r.p != r.end) {
    throw std::runtime_error("trailing bytes in STRINGS");
  }
}

struct TypeEntry {
  uint8_t tag;
  std::string asmStr;
};
struct AttrEntry {
  uint8_t tag;
  std::string asmStr;
};

struct ConstEntryParsed {
  uint8_t tag;
  // tag=0x01 int: type_id + sLEB128
  uint64_t typeId = 0;
  int64_t intValue = 0;
  // tag=0x02 float bits: type_id + bytes
  std::vector<uint8_t> floatBytes;
  // tag=0x04 wide int bits: type_id + bytes
  std::vector<uint8_t> intBytes;
};

struct DbgFileEntry {
  uint64_t pathSid;
  uint8_t hashKind;
  std::vector<uint8_t> hashBytes;
};
struct DbgValueNameEntry {
  uint64_t funcId;
  uint64_t valueId;
  uint64_t nameSid;
};
struct DbgLocationEntry {
  uint64_t funcId;
  uint64_t opId;
  uint64_t fileId;
  uint64_t sl;
  uint64_t sc;
  uint64_t el;
  uint64_t ec;
};
struct DbgSnippetEntry {
  uint64_t funcId;
  uint64_t opId;
  uint64_t snippetSid;
};

struct DebugInfo {
  std::vector<DbgFileEntry> files;
  std::vector<DbgValueNameEntry> valueNames;
  std::vector<DbgLocationEntry> locations;
  std::vector<DbgSnippetEntry> snippets;
};

static DebugInfo parseDebugInfoSection(const std::vector<uint8_t> &data) {
  Reader r{data.data(), data.data() + data.size()};
  DebugInfo di;

  // FileTable
  uint64_t fcnt = r.readULEB();
  di.files.reserve(fcnt);
  for (uint64_t i = 0; i < fcnt; ++i) {
    uint64_t psid = r.readULEB();
    uint8_t hk = r.readU8();
    std::vector<uint8_t> hb;
    if (hk != 0) {
      uint64_t hlen = r.readULEB();
      hb = r.readBytes(hlen);
    }
    di.files.push_back({psid, hk, std::move(hb)});
  }

  // ValueNames
  uint64_t vcnt = r.readULEB();
  di.valueNames.reserve(vcnt);
  for (uint64_t i = 0; i < vcnt; ++i) {
    uint64_t fid = r.readULEB();
    uint64_t vid = r.readULEB();
    uint64_t nsid = r.readULEB();
    di.valueNames.push_back({fid, vid, nsid});
  }

  // OpLocations
  uint64_t lcnt = r.readULEB();
  di.locations.reserve(lcnt);
  for (uint64_t i = 0; i < lcnt; ++i) {
    uint64_t fid = r.readULEB();
    uint64_t opid = r.readULEB();
    uint64_t fileid = r.readULEB();
    uint64_t sl = r.readULEB();
    uint64_t sc = r.readULEB();
    uint64_t el = r.readULEB();
    uint64_t ec = r.readULEB();
    di.locations.push_back({fid, opid, fileid, sl, sc, el, ec});
  }

  // OpSnippets
  uint64_t scnt = r.readULEB();
  di.snippets.reserve(scnt);
  for (uint64_t i = 0; i < scnt; ++i) {
    uint64_t fid = r.readULEB();
    uint64_t opid = r.readULEB();
    uint64_t ssid = r.readULEB();
    di.snippets.push_back({fid, opid, ssid});
  }

  if (r.p != r.end) {
    throw std::runtime_error("trailing bytes in DEBUGINFO");
  }
  return di;
}

static void parseTypesSection(const std::vector<uint8_t> &data,
                              const std::vector<std::string> &strings,
                              std::vector<TypeEntry> &types) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  types.clear();
  types.reserve(cnt + 1);
  types.push_back({0, ""});
  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    uint8_t flags = r.readU8();
    if ((flags & 0x1) == 0) {
      throw std::runtime_error("type missing asm");
    }
    uint64_t sid = r.readULEB();
    if (sid >= strings.size()) {
      throw std::runtime_error("bad asm_sid");
    }
    types.push_back({tag, strings[sid]});
  }
  if (r.p != r.end) {
    throw std::runtime_error("trailing bytes in TYPES");
  }
}

static void parseAttrsSection(const std::vector<uint8_t> &data,
                              const std::vector<std::string> &strings,
                              std::vector<AttrEntry> &attrs) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  attrs.clear();
  attrs.reserve(cnt + 1);
  attrs.push_back({0, ""});
  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    uint8_t flags = r.readU8();
    if ((flags & 0x1) == 0) {
      throw std::runtime_error("attr missing asm");
    }
    uint64_t sid = r.readULEB();
    if (sid >= strings.size()) {
      throw std::runtime_error("bad asm_sid");
    }
    attrs.push_back({tag, strings[sid]});
  }
  if (r.p != r.end) {
    throw std::runtime_error("trailing bytes in ATTRS");
  }
}

static void parseConstPoolSection(const std::vector<uint8_t> &data,
                                  std::vector<ConstEntryParsed> &consts) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  consts.clear();
  consts.reserve(cnt);

  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    if (tag == 0x01) {
      uint64_t tid = r.readULEB();
      int64_t v = r.readSLEB();
      ConstEntryParsed e;
      e.tag = tag;
      e.typeId = tid;
      e.intValue = v;
      consts.push_back(std::move(e));
    } else if (tag == 0x02) {
      uint64_t tid = r.readULEB();
      uint64_t blen = r.readULEB();
      auto bytes = r.readBytes(blen);
      ConstEntryParsed e;
      e.tag = tag;
      e.typeId = tid;
      e.floatBytes = std::move(bytes);
      consts.push_back(std::move(e));
    } else if (tag == 0x03) {
      // index vec (not yet needed for sample decoding)
      uint64_t n = r.readULEB();
      for (uint64_t j = 0; j < n; ++j) {
        (void)r.readSLEB();
      }
      ConstEntryParsed e;
      e.tag = tag;
      consts.push_back(std::move(e));
    } else if (tag == 0x04) {
      // wide int bits: type_id(uLEB), blen(uLEB), bytes(blen)
      uint64_t tid = r.readULEB();
      uint64_t blen = r.readULEB();
      auto bytes = r.readBytes(blen);
      ConstEntryParsed e;
      e.tag = tag;
      e.typeId = tid;
      e.intBytes = std::move(bytes);
      consts.push_back(std::move(e));
    } else {
      throw std::runtime_error("unknown ConstEntry tag");
    }
  }

  if (r.p != r.end) {
    throw std::runtime_error("trailing bytes in CONSTPOOL");
  }
}

struct BuildCtx {
  mlir::MLIRContext *ctx;
  const std::vector<std::string> *strings;
  const std::vector<TypeEntry> *types;
  const std::vector<AttrEntry> *attrs;
  const std::vector<ConstEntryParsed> *consts;

  // Function-global value_id table.
  std::vector<mlir::Value> values;

  // Function-global op_id table (preorder DFS).
  uint64_t *nextOpId = nullptr;
  std::vector<mlir::Operation *> *opsById = nullptr;
};

static mlir::Type getType(BuildCtx &bc, uint64_t tid) {
  if (tid >= bc.types->size()) {
    throw std::runtime_error("bad type_id");
  }
  return parseType(*bc.ctx, (*bc.types)[tid].asmStr);
}

static mlir::DictionaryAttr getAttrDict(BuildCtx &bc, uint64_t aid) {
  if (aid == 0) {
    return mlir::DictionaryAttr::get(bc.ctx);
  }
  if (aid >= bc.attrs->size()) {
    throw std::runtime_error("bad attr_id");
  }
  return parseAttrDict(*bc.ctx, (*bc.attrs)[aid].asmStr);
}

static void buildRegionInto(BuildCtx &bc, Reader &r, mlir::Region &region);

static llvm::APInt rebuildAPIntFromBytes(llvm::ArrayRef<uint8_t> bytes,
                                         unsigned bitWidth) {
  const unsigned numWords = (bitWidth + 63) / 64;
  llvm::SmallVector<uint64_t, kWordInlineCapacity> words(numWords, 0);
  for (unsigned i = 0; i < bytes.size(); ++i) {
    unsigned word = i / kBytesPerWord;
    unsigned off = (i % kBytesPerWord) * kBitsPerByte;
    words[word] |= (uint64_t(bytes[i]) << off);
  }
  return llvm::APInt(bitWidth, words);
}

static mlir::Attribute buildFloatConstAttr(BuildCtx &bc,
                                           const ConstEntryParsed &entry) {
  auto ty = getType(bc, entry.typeId);
  auto floatType = mlir::dyn_cast<mlir::FloatType>(ty);
  if (!floatType) {
    throw std::runtime_error("ConstFloatBits type is not FloatType");
  }

  unsigned bitWidth = floatType.getWidth();
  unsigned byteLen = (bitWidth + 7) / 8;
  if (entry.floatBytes.size() != byteLen) {
    throw std::runtime_error("ConstFloatBits byte_len mismatch");
  }

  llvm::APInt bits = rebuildAPIntFromBytes(entry.floatBytes, bitWidth);
  llvm::APFloat value(floatType.getFloatSemantics(), bits);
  return mlir::FloatAttr::get(floatType, value);
}

static mlir::Attribute buildIntegerConstAttr(BuildCtx &bc,
                                             const ConstEntryParsed &entry) {
  auto ty = getType(bc, entry.typeId);
  auto intType = mlir::dyn_cast<mlir::IntegerType>(ty);
  if (!intType) {
    throw std::runtime_error("ConstIntBits type is not IntegerType");
  }

  unsigned bitWidth = intType.getWidth();
  unsigned byteLen = (bitWidth + 7) / 8;
  if (entry.intBytes.size() != byteLen) {
    throw std::runtime_error("ConstIntBits byte_len mismatch");
  }

  llvm::APInt bits = rebuildAPIntFromBytes(entry.intBytes, bitWidth);
  return mlir::IntegerAttr::get(intType, bits);
}

static mlir::Attribute buildConstAttr(BuildCtx &bc, uint64_t constId) {
  if (!bc.consts) {
    throw std::runtime_error("constpool not available");
  }
  if (constId >= bc.consts->size()) {
    throw std::runtime_error("const_id out of range");
  }
  const auto &e = (*bc.consts)[constId];

  if (e.tag == 0x01) {
    auto ty = getType(bc, e.typeId);
    auto it = mlir::dyn_cast<mlir::IntegerType>(ty);
    if (mlir::isa<mlir::IndexType>(ty)) {
      return mlir::IntegerAttr::get(ty, e.intValue);
    }
    if (!it) {
      throw std::runtime_error("ConstInt type is not integer/index");
    }
    return mlir::IntegerAttr::get(ty, e.intValue);
  }

  if (e.tag == 0x02) {
    return buildFloatConstAttr(bc, e);
  }

  if (e.tag == 0x04) {
    return buildIntegerConstAttr(bc, e);
  }

  throw std::runtime_error("unsupported const tag");
}

static void addAttrDictionary(mlir::OperationState &state,
                              mlir::DictionaryAttr dict) {
  for (auto attr : dict) {
    state.addAttribute(attr.getName(), attr.getValue());
  }
}

static void registerDecodedOp(BuildCtx &bc, uint64_t opId,
                              mlir::Operation *op) {
  if (!bc.opsById) {
    return;
  }
  if (opId >= bc.opsById->size()) {
    bc.opsById->resize(opId + 1, nullptr);
  }
  (*bc.opsById)[opId] = op;
}

static void assignDecodedResults(BuildCtx &bc, size_t resStart,
                                 mlir::Operation *op, size_t numResults) {
  for (size_t i = 0; i < numResults; ++i) {
    bc.values[resStart + i] = op->getResult(i);
  }
}

static llvm::SmallVector<uint64_t, kValueIdInlineCapacity>
readValueIds(Reader &r, size_t count) {
  llvm::SmallVector<uint64_t, kValueIdInlineCapacity> ids;
  ids.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    ids.push_back(r.readULEB());
  }
  return ids;
}

struct KnownOpImmediates {
  uint8_t cmpPred = 0;
  uint8_t evA = 0;
  uint8_t evB = 0;
  uint8_t evC = 0;
  uint64_t constId = 0;
  uint8_t listMode = 0;
  uint64_t n1 = 0;
  uint64_t n2 = 0;
  uint8_t optMask = 0;
};

static KnownOpImmediates readKnownOpImmediates(Reader &r,
                                               const ptobc::v0::OpInfo &info) {
  KnownOpImmediates imms;
  switch (info.imm_kind) {
  case 0x00:
    return imms;
  case 0x01:
    imms.cmpPred = r.readU8();
    return imms;
  case 0x02:
    imms.evA = r.readU8();
    imms.evB = r.readU8();
    imms.evC = r.readU8();
    return imms;
  case 0x05:
    imms.constId = r.readULEB();
    return imms;
  case 0x06:
  case 0x07:
    imms.listMode = r.readU8();
    imms.n1 = r.readULEB();
    imms.n2 = r.readULEB();
    return imms;
  case 0x08:
    imms.optMask = r.readU8();
    return imms;
  default:
    throw std::runtime_error("unknown imm_kind");
  }
}

static llvm::SmallVector<uint64_t, kValueIdInlineCapacity>
reorderLegacyIndexedTscatterOperandIds(
    uint16_t opcode, llvm::SmallVector<uint64_t, kValueIdInlineCapacity> ids) {
  const bool isIndexedTscatter = opcode == 0x1056 && ids.size() == 3;
  if (!isIndexedTscatter) {
    return ids;
  }
  // Historical v0 indexed tscatter payload is (src, indexes, dst), while the
  // current IR operand order is (src, dst, indexes).
  return llvm::SmallVector<uint64_t, kValueIdInlineCapacity>{ids[0], ids[2],
                                                             ids[1]};
}

static size_t getKnownOperandCount(Reader &r, uint16_t opcode, uint8_t variant,
                                   const ptobc::v0::OpInfo &info,
                                   const KnownOpImmediates &imms) {
  switch (info.operand_mode) {
  case 0x00:
    return info.num_operands;
  case 0x01: {
    auto count = ptobc::v0::lookupOperandsByVariant(opcode, variant);
    if (!count) {
      throw std::runtime_error("missing by-variant operand count");
    }
    return *count;
  }
  case 0x02:
    return r.readULEB();
  case 0x03:
    if (imms.listMode != 0) {
      throw std::runtime_error("list_mode=1 not supported yet");
    }
    return size_t(info.num_operands) + size_t(imms.n1) + size_t(imms.n2);
  case 0x04:
    return ((imms.optMask & 0x1) ? 1 : 0) + ((imms.optMask & 0x2) ? 1 : 0) +
           ((imms.optMask & 0x4) ? 1 : 0);
  default:
    throw std::runtime_error("unknown operand_mode");
  }
}

static llvm::SmallVector<uint64_t, kValueIdInlineCapacity>
readKnownOperandIds(Reader &r, uint16_t opcode, uint8_t variant,
                    const ptobc::v0::OpInfo &info,
                    const KnownOpImmediates &imms) {
  const size_t count = getKnownOperandCount(r, opcode, variant, info, imms);
  return reorderLegacyIndexedTscatterOperandIds(opcode, readValueIds(r, count));
}

static llvm::SmallVector<mlir::Value, kOperandInlineCapacity>
materializeOperands(BuildCtx &bc, llvm::ArrayRef<uint64_t> operandIds) {
  llvm::SmallVector<mlir::Value, kOperandInlineCapacity> operands;
  operands.reserve(operandIds.size());
  for (uint64_t valueId : operandIds) {
    if (valueId >= bc.values.size()) {
      throw std::runtime_error("operand value_id out of range");
    }
    operands.push_back(bc.values[valueId]);
  }
  return operands;
}

static void
normalizeLegacyFpOperandOrder(uint16_t opcode,
                              llvm::SmallVectorImpl<mlir::Value> &operands) {
  if (const bool hasExtractFpLayout =
          (opcode == kTExtractFpWireOpcode || opcode == kTInsertFpWireOpcode) &&
          operands.size() == mlir::pto::kValue5;
      hasExtractFpLayout) {
    // Legacy wire order: src, fp, row, col, dst.
    llvm::SmallVector<mlir::Value, mlir::pto::kValue5> reordered{
        operands[0], operands[2], operands[3], operands[4], operands[1]};
    operands.assign(reordered.begin(), reordered.end());
    return;
  }
  if ((opcode == kTMovFpWireOpcode || opcode == kTStoreFpWireOpcode) &&
      operands.size() == mlir::pto::kValue3) {
    // Legacy wire order: src, fp, dst.
    llvm::SmallVector<mlir::Value, mlir::pto::kValue3> reordered{
        operands[0], operands[2], operands[1]};
    operands.assign(reordered.begin(), reordered.end());
  }
}

static std::optional<llvm::SmallVector<int32_t, mlir::pto::kValue6>>
getUnifiedOperandSegments(uint16_t opcode,
                          llvm::ArrayRef<mlir::Value> operands) {
  switch (opcode) {
  case kTExtractOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 1, 1, 0, 0};
  case kTExtractFpWireOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 1, 1, 1, 0};
  case kTInsertOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 1, 1, 0, 0};
  case kTInsertFpWireOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 1, 1, 1, 0};
  case kTMovOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 0, 0};
  case kTMovFpWireOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 1, 0};
  case kTStoreOpcode:
    if (operands.size() == mlir::pto::kValue2) {
      return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 0, 0};
    }
    if (operands.size() == mlir::pto::kValue3) {
      const bool thirdIsFp =
          mlir::isa<mlir::pto::TileBufType>(operands.back().getType());
      return llvm::SmallVector<int32_t, mlir::pto::kValue6>{
          1, 1, thirdIsFp ? 1 : 0, thirdIsFp ? 0 : 1};
    }
    return std::nullopt;
  case kTStoreFpWireOpcode:
    return llvm::SmallVector<int32_t, mlir::pto::kValue6>{1, 1, 1, 0};
  default:
    return std::nullopt;
  }
}

static void setUnifiedOperandSegmentProperty(mlir::Operation *op,
                                             uint16_t opcode,
                                             llvm::ArrayRef<int32_t> segments) {
  switch (opcode) {
  case kTExtractOpcode:
  case kTExtractFpWireOpcode:
    llvm::cast<mlir::pto::TExtractOp>(op)
        .getProperties()
        .setOperandSegmentSizes(segments);
    return;
  case kTInsertOpcode:
  case kTInsertFpWireOpcode:
    llvm::cast<mlir::pto::TInsertOp>(op).getProperties().setOperandSegmentSizes(
        segments);
    return;
  case kTMovOpcode:
  case kTMovFpWireOpcode:
    llvm::cast<mlir::pto::TMovOp>(op).getProperties().setOperandSegmentSizes(
        segments);
    return;
  case kTStoreOpcode:
  case kTStoreFpWireOpcode:
    llvm::cast<mlir::pto::TStoreOp>(op).getProperties().setOperandSegmentSizes(
        segments);
    return;
  default:
    return;
  }
}

static mlir::Operation *buildGenericOpFromReader(BuildCtx &bc, Reader &r,
                                                 mlir::Block &block,
                                                 uint64_t opId,
                                                 uint64_t attrId) {
  uint64_t nameSid = r.readULEB();
  if (nameSid >= bc.strings->size()) {
    throw std::runtime_error("bad op_name sid");
  }
  std::string opName = (*bc.strings)[nameSid];

  uint64_t nres = r.readULEB();
  llvm::SmallVector<mlir::Type, kResultTypeInlineCapacity> resultTypes;
  resultTypes.reserve(nres);

  const size_t resStart = bc.values.size();
  for (uint64_t i = 0; i < nres; ++i) {
    resultTypes.push_back(getType(bc, r.readULEB()));
    bc.values.push_back(mlir::Value());
  }

  auto operandIds = readValueIds(r, r.readULEB());
  auto operands = materializeOperands(bc, operandIds);
  uint64_t nreg = r.readULEB();

  mlir::OperationState state(mlir::UnknownLoc::get(bc.ctx), opName);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  addAttrDictionary(state, getAttrDict(bc, attrId));
  for (uint64_t i = 0; i < nreg; ++i) {
    (void)state.addRegion();
  }

  mlir::Operation *op = mlir::Operation::create(state);
  block.getOperations().push_back(op);
  registerDecodedOp(bc, opId, op);
  assignDecodedResults(bc, resStart, op, nres);
  for (uint64_t i = 0; i < nreg; ++i) {
    buildRegionInto(bc, r, op->getRegion(i));
  }
  return op;
}

static void addImmediateAttrs(BuildCtx &bc, mlir::OperationState &state,
                              const ptobc::v0::OpInfo &info,
                              const KnownOpImmediates &imms) {
  switch (info.imm_kind) {
  case 0x01:
    state.addAttribute("predicate",
                       mlir::arith::CmpIPredicateAttr::get(
                           bc.ctx, mlir::arith::CmpIPredicate(imms.cmpPred)));
    return;
  case 0x02:
    state.addAttribute("src_op", mlir::pto::SyncOpTypeAttr::get(
                                     bc.ctx, mlir::pto::SyncOpType(imms.evA)));
    state.addAttribute("dst_op", mlir::pto::SyncOpTypeAttr::get(
                                     bc.ctx, mlir::pto::SyncOpType(imms.evB)));
    state.addAttribute("event_id", mlir::pto::EventAttr::get(
                                       bc.ctx, mlir::pto::EVENT(imms.evC)));
    return;
  case 0x05:
    state.addAttribute("value", buildConstAttr(bc, imms.constId));
    return;
  default:
    return;
  }
}

static llvm::SmallVector<mlir::Type, kResultTypeInlineCapacity>
readKnownResultTypes(BuildCtx &bc, Reader &r, const ptobc::v0::OpInfo &info,
                     uint64_t &numResults) {
  llvm::SmallVector<mlir::Type, kResultTypeInlineCapacity> resultTypes;
  switch (info.result_type_mode) {
  case 0x00:
    resultTypes.assign(numResults, mlir::NoneType::get(bc.ctx));
    return resultTypes;
  case 0x01:
    for (uint64_t i = 0; i < numResults; ++i) {
      resultTypes.push_back(getType(bc, r.readULEB()));
    }
    return resultTypes;
  case 0x02:
    numResults = r.readULEB();
    resultTypes.reserve(numResults);
    for (uint64_t i = 0; i < numResults; ++i) {
      resultTypes.push_back(getType(bc, r.readULEB()));
    }
    return resultTypes;
  default:
    throw std::runtime_error("unknown result_type_mode");
  }
}

static void reserveDecodedResults(BuildCtx &bc, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    bc.values.push_back(mlir::Value());
  }
}

static mlir::OperationState makeKnownOperationState(
    BuildCtx &bc, uint16_t opcode, uint8_t variant, uint64_t attrId,
    const ptobc::v0::OpInfo &info, llvm::ArrayRef<mlir::Value> operands,
    llvm::ArrayRef<mlir::Type> resultTypes, const KnownOpImmediates &imms) {
  const char *opNameC = ptobc::v0::fullNameFromOpcodeVariant(opcode, variant);
  if (!opNameC) {
    throw std::runtime_error("failed to map opcode->name");
  }
  mlir::OperationState state(mlir::UnknownLoc::get(bc.ctx), opNameC);
  state.addOperands(operands);
  state.addTypes(resultTypes);
  addAttrDictionary(state, getAttrDict(bc, attrId));
  addImmediateAttrs(bc, state, info, imms);
  for (unsigned i = 0; i < info.num_regions; ++i) {
    (void)state.addRegion();
  }
  return state;
}

static mlir::Operation *buildKnownOpFromReader(BuildCtx &bc, Reader &r,
                                               mlir::Block &block,
                                               uint64_t opId, uint16_t opcode,
                                               uint64_t attrId) {
  const auto *info = ptobc::v0::lookupByOpcode(opcode);
  if (!info) {
    throw std::runtime_error("missing opcode schema");
  }

  uint8_t variant = info->has_variant_u8 ? r.readU8() : 0;
  KnownOpImmediates imms = readKnownOpImmediates(r, *info);
  auto operandIds = readKnownOperandIds(r, opcode, variant, *info, imms);
  auto operands = materializeOperands(bc, operandIds);
  normalizeLegacyFpOperandOrder(opcode, operands);

  uint64_t numResults = info->num_results;
  auto resultTypes = readKnownResultTypes(bc, r, *info, numResults);

  const size_t resStart = bc.values.size();
  reserveDecodedResults(bc, numResults);

  mlir::OperationState state = makeKnownOperationState(
      bc, opcode, variant, attrId, *info, operands, resultTypes, imms);
  auto unifiedOperandSegments = getUnifiedOperandSegments(opcode, operands);

  mlir::Operation *op = mlir::Operation::create(state);
  // AttrSizedOperandSegments uses generated inherent property storage.
  // OperationState's generic attribute list does not initialize it, so update
  // that storage directly after creating the registered op.
  if (unifiedOperandSegments) {
    setUnifiedOperandSegmentProperty(op, opcode, *unifiedOperandSegments);
  }
  block.getOperations().push_back(op);
  registerDecodedOp(bc, opId, op);
  assignDecodedResults(bc, resStart, op, numResults);
  for (unsigned i = 0; i < info->num_regions; ++i) {
    buildRegionInto(bc, r, op->getRegion(i));
  }
  return op;
}

static void buildOpList(BuildCtx &bc, Reader &r, mlir::Block &block) {
  const bool dbg = debugEnabled();
  uint64_t opcnt = r.readULEB();
  if (dbg) {
    llvm::errs() << "[ptobc]   ops=" << opcnt << "\n";
  }

  for (uint64_t oi = 0; oi < opcnt; ++oi) {
    if (dbg) {
      llvm::errs() << "[ptobc]    op[" << oi << "]...\n";
    }
    const uint64_t opId = bc.nextOpId ? (*bc.nextOpId)++ : 0;

    uint16_t opcode = r.readU16LE();
    uint64_t attrId = r.readULEB();

    if (opcode == kOpcodeGeneric) {
      buildGenericOpFromReader(bc, r, block, opId, attrId);
      continue;
    }
    buildKnownOpFromReader(bc, r, block, opId, opcode, attrId);
  }
}

static void buildRegionInto(BuildCtx &bc, Reader &r, mlir::Region &region) {
  const bool dbg = debugEnabled();
  uint64_t bcnt = r.readULEB();
  if (dbg) {
    llvm::errs() << "[ptobc] region: blocks=" << bcnt << "\n";
  }
  region.getBlocks().clear();

  for (uint64_t bi = 0; bi < bcnt; ++bi) {
    if (dbg) {
      llvm::errs() << "[ptobc]  block[" << bi << "]...\n";
    }
    auto *block = new mlir::Block();

    uint64_t nargs = r.readULEB();
    if (dbg) {
      llvm::errs() << "[ptobc]   nargs=" << nargs << "\n";
    }
    for (uint64_t ai = 0; ai < nargs; ++ai) {
      uint64_t tid = r.readULEB();
      auto ty = getType(bc, tid);
      auto arg = block->addArgument(ty, mlir::UnknownLoc::get(bc.ctx));
      bc.values.push_back(arg);
    }

    buildOpList(bc, r, *block);
    region.push_back(block);
  }
}

struct FuncDecl {
  std::string name;
  mlir::FunctionType type;
  mlir::DictionaryAttr attrs;
  uint8_t flags = 0;
};

static uint64_t readModuleHeader(Reader &r, bool dbg) {
  uint8_t profile = r.readU8();
  uint8_t indexWidth = r.readU8();
  if (dbg) {
    llvm::errs() << "[ptobc] module: profile=" << unsigned(profile)
                 << " indexWidth=" << unsigned(indexWidth) << "\n";
  }

  uint64_t moduleAttrId = r.readULEB();
  uint64_t globalCount = r.readULEB();
  if (dbg) {
    llvm::errs() << "[ptobc] module: moduleAttrId=" << moduleAttrId
                 << " globals=" << globalCount << "\n";
  }
  if (globalCount != 0) {
    throw std::runtime_error("globals not supported");
  }
  return moduleAttrId;
}

static std::vector<FuncDecl> readFunctionDecls(BuildCtx &bc, Reader &r,
                                               bool dbg) {
  uint64_t funcCount = r.readULEB();
  if (dbg) {
    llvm::errs() << "[ptobc] module: funcs=" << funcCount << "\n";
  }

  std::vector<FuncDecl> decls;
  decls.reserve(funcCount);
  for (uint64_t i = 0; i < funcCount; ++i) {
    uint64_t nameSid = r.readULEB();
    uint64_t ftypeId = r.readULEB();
    uint8_t flags = r.readU8();
    uint64_t fattrId = r.readULEB();
    if (nameSid >= bc.strings->size()) {
      throw std::runtime_error("bad func name sid");
    }
    if (ftypeId >= bc.types->size()) {
      throw std::runtime_error("bad func type id");
    }

    if (dbg) {
      llvm::errs() << "[ptobc] func[" << i << "]: nameSid=" << nameSid
                   << " ftypeId=" << ftypeId << " flags=" << unsigned(flags)
                   << " fattrId=" << fattrId << "\n";
    }

    auto type = parseType(*bc.ctx, bc.types->at(ftypeId).asmStr);
    auto funcType = mlir::dyn_cast<mlir::FunctionType>(type);
    if (!funcType) {
      throw std::runtime_error("func type parse failed");
    }
    decls.push_back(
        {bc.strings->at(nameSid), funcType, getAttrDict(bc, fattrId), flags});
  }
  return decls;
}

static void applyAttrDictionary(mlir::Operation *op,
                                mlir::DictionaryAttr dict) {
  for (auto attr : dict) {
    op->setAttr(attr.getName(), attr.getValue());
  }
}

static void recordImportedFunction(
    std::vector<std::vector<mlir::Operation *>> *opsByFuncOut) {
  if (opsByFuncOut) {
    opsByFuncOut->push_back({});
  }
}

static void buildDefinedFunctionBody(
    BuildCtx &bc, Reader &r, mlir::func::FuncOp fn, bool dbg,
    std::vector<std::vector<mlir::Operation *>> *opsByFuncOut) {
  bc.values.clear();
  uint64_t nextOpId = 0;
  std::vector<mlir::Operation *> opsById;
  bc.nextOpId = &nextOpId;
  bc.opsById = &opsById;
  buildRegionInto(bc, r, fn.getBody());
  if (dbg) {
    llvm::errs() << "[ptobc] func body built ok: values=" << bc.values.size()
                 << " ops=" << opsById.size() << "\n";
  }
  if (opsByFuncOut) {
    opsByFuncOut->push_back(std::move(opsById));
  }
}

static void
buildFunctionBody(BuildCtx &bc, Reader &r, mlir::func::FuncOp fn, uint8_t flags,
                  bool dbg,
                  std::vector<std::vector<mlir::Operation *>> *opsByFuncOut) {
  const bool isImported = (flags & 0x1) != 0;
  if (isImported) {
    recordImportedFunction(opsByFuncOut);
    return;
  }
  buildDefinedFunctionBody(bc, r, fn, dbg, opsByFuncOut);
}

static void appendDecodedFunctions(
    BuildCtx &bc, Reader &r, mlir::ModuleOp module,
    const std::vector<FuncDecl> &decls, bool dbg,
    std::vector<std::vector<mlir::Operation *>> *opsByFuncOut) {
  for (const auto &decl : decls) {
    if (dbg) {
      llvm::errs() << "[ptobc] building func body: " << decl.name << "\n";
    }
    auto fn = mlir::func::FuncOp::create(mlir::UnknownLoc::get(bc.ctx),
                                         decl.name, decl.type);
    if (dbg) {
      llvm::errs() << "[ptobc] created func op\n";
    }
    applyAttrDictionary(fn, decl.attrs);
    buildFunctionBody(bc, r, fn, decl.flags, dbg, opsByFuncOut);
    module.push_back(fn);
  }
}

static mlir::ModuleOp
decodeToModule(mlir::MLIRContext &ctx, const std::vector<std::string> &strings,
               const std::vector<TypeEntry> &types,
               const std::vector<AttrEntry> &attrs,
               const std::vector<uint8_t> &constPool,
               const std::vector<uint8_t> &moduleBytes,
               std::vector<std::vector<mlir::Operation *>> *opsByFuncOut) {
  const bool dbg = debugEnabled();

  Reader r{moduleBytes.data(), moduleBytes.data() + moduleBytes.size()};
  std::vector<ConstEntryParsed> consts;
  parseConstPoolSection(constPool, consts);
  BuildCtx bc{&ctx, &strings, &types, &attrs, &consts, {}, nullptr, nullptr};
  uint64_t moduleAttrId = readModuleHeader(r, dbg);
  std::vector<FuncDecl> decls = readFunctionDecls(bc, r, dbg);

  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
  applyAttrDictionary(module.getOperation(), getAttrDict(bc, moduleAttrId));
  appendDecodedFunctions(bc, r, module, decls, dbg, opsByFuncOut);

  if (r.p != r.end) {
    throw std::runtime_error("trailing bytes in MODULE");
  }
  return module;
}

static std::pair<uint8_t, std::vector<uint8_t>> readSection(Reader &r,
                                                            bool dbg) {
  uint8_t sid = r.readU8();
  uint32_t sectionLen = r.readU32LE();
  auto bytes = r.readBytes(sectionLen);
  if (dbg) {
    llvm::errs() << "[ptobc] section id=" << unsigned(sid)
                 << " len=" << sectionLen << "\n";
  }
  return {sid, bytes};
}

struct DecodedSections {
  std::vector<uint8_t> strings;
  std::vector<uint8_t> types;
  std::vector<uint8_t> attrs;
  std::vector<uint8_t> constPool;
  std::vector<uint8_t> module;
  std::optional<DebugInfo> debugInfo;
};

static Reader validatePTOBCHeader(llvm::ArrayRef<uint8_t> fileBytes) {
  if (fileBytes.size() < kPTOBCHeaderSize) {
    throw std::runtime_error("file too small");
  }
  if (std::memcmp(fileBytes.data(), kPTOBCMagic, kPTOBCMagicSize) != 0) {
    throw std::runtime_error("bad magic");
  }

  Reader versionReader{fileBytes.data() + kPTOBCVersionOffset,
                       fileBytes.data() + fileBytes.size()};
  if (const uint16_t version = versionReader.readU16LE();
      version != kVersionV0) {
    throw std::runtime_error("unsupported version");
  }

  Reader payloadLengthReader{fileBytes.data() + kPTOBCPayloadLengthOffset,
                             fileBytes.data() + fileBytes.size()};
  const uint32_t payloadLen = payloadLengthReader.readU32LE();
  if (payloadLen != fileBytes.size() - kPTOBCHeaderSize) {
    throw std::runtime_error("payload_len mismatch");
  }
  return {fileBytes.data() + kPTOBCHeaderSize,
          fileBytes.data() + fileBytes.size()};
}

static void readOptionalSections(Reader &r, bool dbg,
                                 DecodedSections &sections) {
  while (r.p != r.end) {
    auto [sid, sec] = readSection(r, dbg);
    if (sid == kSectionDebugInfo) {
      if (sections.debugInfo) {
        throw std::runtime_error("duplicate DEBUGINFO section");
      }
      sections.debugInfo = parseDebugInfoSection(sec);
    } else if (sid != kSectionExtra) {
      throw std::runtime_error("unexpected trailing section id");
    }
  }
}

static DecodedSections readPTOBCSections(Reader &r, bool dbg) {
  auto [s1, d1] = readSection(r, dbg);
  auto [s2, d2] = readSection(r, dbg);
  auto [s3, d3] = readSection(r, dbg);
  auto [s4, d4] = readSection(r, dbg);
  auto [s6, d6] = readSection(r, dbg);
  if (s1 != kSectionStrings || s2 != kSectionTypes || s3 != kSectionAttrs ||
      s4 != kSectionConstPool || s6 != kSectionModule) {
    throw std::runtime_error("unexpected section order");
  }

  DecodedSections sections{std::move(d1), std::move(d2), std::move(d3),
                           std::move(d4), std::move(d6), std::nullopt};
  readOptionalSections(r, dbg, sections);
  return sections;
}

static void applyDebugLocations(
    mlir::MLIRContext &ctx, const std::vector<std::string> &strings,
    const DebugInfo &dbgInfo,
    const std::vector<std::vector<mlir::Operation *>> &opsByFunc) {
  for (const auto &location : dbgInfo.locations) {
    if (location.funcId >= opsByFunc.size()) {
      continue;
    }
    const auto &ops = opsByFunc[location.funcId];
    if (location.opId >= ops.size()) {
      continue;
    }
    mlir::Operation *op = ops[location.opId];
    if (!op || location.fileId >= dbgInfo.files.size()) {
      continue;
    }
    const auto &file = dbgInfo.files[location.fileId];
    if (file.pathSid >= strings.size()) {
      continue;
    }
    op->setLoc(mlir::FileLineColLoc::get(&ctx, strings[file.pathSid],
                                         unsigned(location.sl),
                                         unsigned(location.sc)));
  }
}

mlir::OwningOpRef<mlir::ModuleOp>
decodePTOBCToModule(llvm::ArrayRef<uint8_t> fileBytes, mlir::MLIRContext &ctx) {
  const bool dbg = debugEnabled();
  Reader r = validatePTOBCHeader(fileBytes);
  DecodedSections sections = readPTOBCSections(r, dbg);

  std::vector<std::string> strings;
  parseStringsSection(sections.strings, strings);

  std::vector<TypeEntry> types;
  parseTypesSection(sections.types, strings, types);

  std::vector<AttrEntry> attrs;
  parseAttrsSection(sections.attrs, strings, attrs);

  if (dbg) {
    llvm::errs() << "[ptobc] strings=" << strings.size()
                 << " types=" << types.size() << " attrs=" << attrs.size()
                 << " moduleBytes=" << sections.module.size() << "\n";
  }

  // Ensure dialects are loaded before we start materializing ops.
  (void)ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  (void)ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  (void)ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  (void)ctx.getOrLoadDialect<mlir::pto::PTODialect>();

  if (dbg) {
    llvm::errs() << "[ptobc] decoding module...\n";
  }

  std::vector<std::vector<mlir::Operation *>> opsByFunc;
  auto module = decodeToModule(ctx, strings, types, attrs, sections.constPool,
                               sections.module,
                               sections.debugInfo ? &opsByFunc : nullptr);

  // Apply op locations from DEBUGINFO (best-effort).
  if (sections.debugInfo) {
    applyDebugLocations(ctx, strings, *sections.debugInfo, opsByFunc);
  }

  return module;
}

void decodeFileToPTO(const std::string &inPath, const std::string &outPath) {
  const bool dbg = debugEnabled();
  if (dbg) {
    llvm::errs() << "[ptobc] decode: reading file: " << inPath << "\n";
  }
  auto data = readFile(inPath);

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::pto::PTODialect>();
  mlir::MLIRContext ctx(registry);
  ctx.allowUnregisteredDialects(true);

  auto module = decodePTOBCToModule(data, ctx);

  if (dbg) {
    llvm::errs() << "[ptobc] decoded module ok; printing...\n";
  }

  CanonicalPrintOptions opt;
  opt.generic = (std::getenv("PTOBC_PRINT_GENERIC") != nullptr);
  opt.keepMLIRFloatPrinting = (std::getenv("PTOBC_PRINT_PRETTY") != nullptr);
  opt.printDebugInfo = (std::getenv("PTOBC_PRINT_LOC") != nullptr);

  std::string out = printModuleCanonical(module.get(), opt);

  if (dbg) {
    llvm::errs() << "[ptobc] writing output: " << outPath << "\n";
  }
  std::ofstream ofs(outPath);
  ofs << out;
  if (!out.empty() && out.back() != '\n') {
    ofs << "\n";
  }
}

} // namespace ptobc
