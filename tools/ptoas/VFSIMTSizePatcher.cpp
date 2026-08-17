// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VFSIMTSizePatcher.h"

#include "PTO/Support/CodeConstants.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

namespace {

using llvm::StringRef;
using mlir::failure;
using mlir::FailureOr;
using mlir::LogicalResult;
using mlir::success;
using mlir::pto::VFSIMTSizeFixMode;
using mlir::pto::VFSIMTSizePatchResult;

constexpr uint16_t kHiIPUMachine = 0x1029;
constexpr uint64_t kInstructionBytes = 4;
constexpr uint64_t kVFSIMTCallSequenceBytes = 24;
constexpr uint64_t kMOVIOffsetFromVFSIMTBytes = 20;
constexpr uint64_t kFirstMOVKOffsetFromVFSIMTBytes = 16;
constexpr uint64_t kSecondMOVKOffsetFromVFSIMTBytes = 12;
constexpr uint64_t kSHLIOffsetFromVFSIMTBytes = 8;
constexpr uint64_t kADDOffsetFromVFSIMTBytes = 4;
constexpr uint64_t kInvalidVFSIMTSize = 0xffff;
constexpr unsigned kVFSIMTSizeShift = 37;
constexpr uint64_t kVFSIMTSizeMask = UINT64_C(0xffff) << kVFSIMTSizeShift;
constexpr unsigned kVFSIMTRegisterShift = 16;
constexpr unsigned kScalarDestinationShift = 17;
constexpr unsigned kADDSourceRegisterShift = 7;
constexpr unsigned kADDTargetRegisterShift = 12;
constexpr unsigned kMOVKChunkBitWidth = 16;
constexpr unsigned kRelativeWordOffsetBitWidth = 48;
constexpr uint64_t kScalarRegisterMask = UINT64_C(0x1f);
constexpr uint64_t kVFSIMTJoinOffsetMask = UINT64_C(0x3ff);
constexpr uint64_t kVFSIMTExitModeMask = UINT64_C(1) << 10;

// VF_SIMT scalar instructions are 64 bits. The two register operands, code
// size, join_ofst and exit_mode fields may vary between callsites.
constexpr uint64_t kVFSIMTVariableMask =
    kVFSIMTSizeMask | (kScalarRegisterMask << 16) |
    (kScalarRegisterMask << 11) | kVFSIMTJoinOffsetMask |
    kVFSIMTExitModeMask;
constexpr uint64_t kVFSIMTFixedMask = ~kVFSIMTVariableMask;
constexpr uint64_t kVFSIMTFixedBits =
    UINT64_C(0x15e0001c15200400) & kVFSIMTFixedMask;

struct SimtCallSite {
  std::string callerName;
  std::string calleeName;
  bool calleeContainsInlineAsm = false;
};

struct ELFFunction {
  std::string name;
  uint64_t address = 0;
  uint64_t size = 0;
  uint64_t sectionAddress = 0;
  uint64_t sectionSize = 0;
  uint64_t sectionFileOffset = 0;
  uint64_t sectionIndex = 0;
};

struct DecodedCallSite {
  uint64_t address = 0;
  uint64_t fileOffset = 0;
  uint64_t targetAddress = 0;
  uint16_t codeSize = 0;
  uint64_t instruction = 0;
};

struct PatchRecord {
  const SimtCallSite *manifest = nullptr;
  DecodedCallSite decoded;
  uint16_t expectedCodeSize = 0;
};

struct ObjectPatchAnalysis {
  std::unique_ptr<llvm::MemoryBuffer> buffer;
  llvm::SmallVector<PatchRecord, mlir::pto::kValue4> plan;
};

static void emitError(llvm::raw_ostream &diagOS, const llvm::Twine &message) {
  diagOS << "Error: VF_SIMT size patch: " << message << "\n";
}

// Convert LLVM's checked error carrier to an optional while preserving a
// useful diagnostic and, importantly, consuming any Error on the failure path.
template <typename T>
static std::optional<T> takeExpected(llvm::Expected<T> value,
                                     llvm::raw_ostream &diagOS,
                                     const llvm::Twine &context) {
  if (value) {
    return std::move(*value);
  }
  diagOS << "Error: VF_SIMT size patch: " << context << ": "
         << llvm::toString(value.takeError()) << "\n";
  return std::nullopt;
}

static bool functionContainsInlineAsm(const llvm::Function &function) {
  for (const llvm::BasicBlock &block : function) {
    for (const llvm::Instruction &instruction : block) {
      const auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
      if (call && call->isInlineAsm()) {
        return true;
      }
    }
  }
  return false;
}

static std::string getSIMTEntrySymbolName(llvm::StringRef functionName) {
  // BiSheng materializes the SIMT calling convention with this ELF suffix.
  return (functionName + "_simt_entry").str();
}

static FailureOr<llvm::SmallVector<SimtCallSite, mlir::pto::kValue4>>
collectManifest(llvm::Module &module, llvm::raw_ostream &diagOS) {
  // The LLVM module is the source of truth for allowed caller/callee pairs.
  // Refuse calls that cannot later be matched unambiguously to ELF symbols.
  llvm::SmallVector<SimtCallSite, mlir::pto::kValue4> manifest;
  for (llvm::Function &caller : module) {
    if (caller.isDeclaration() ||
        caller.getCallingConv() == llvm::CallingConv::SimtEntry) {
      continue;
    }
    for (llvm::BasicBlock &block : caller) {
      for (llvm::Instruction &instruction : block) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(&instruction);
        if (!call || call->getCallingConv() != llvm::CallingConv::SimtEntry) {
          continue;
        }
        llvm::Function *callee = call->getCalledFunction();
        if (!callee || callee->isDeclaration()) {
          emitError(diagOS,
                    llvm::Twine("caller '") + caller.getName() +
                        "' contains an indirect or undefined SIMT call");
          return failure();
        }
        if (callee->getCallingConv() != llvm::CallingConv::SimtEntry) {
          emitError(diagOS,
                    llvm::Twine("callee '") + callee->getName() +
                        "' does not use the SIMT entry calling convention");
          return failure();
        }
        manifest.push_back({caller.getName().str(),
                            getSIMTEntrySymbolName(callee->getName()),
                            functionContainsInlineAsm(*callee)});
      }
    }
  }
  return manifest;
}

static FailureOr<llvm::StringMap<ELFFunction>>
readFunctions(const llvm::object::ELFObjectFileBase &object,
              llvm::ArrayRef<SimtCallSite> manifest,
              llvm::raw_ostream &diagOS) {
  // Read only manifest-relevant functions, but first require the simple raw
  // object layout on which file-offset and symbol-range checks rely.
  llvm::StringMap<ELFFunction> functions;
  llvm::StringSet<> requiredFunctions;
  for (const SimtCallSite &call : manifest) {
    requiredFunctions.insert(call.callerName);
    requiredFunctions.insert(call.calleeName);
  }

  unsigned executableSections = 0;
  unsigned symbolTables = 0;
  for (const llvm::object::SectionRef &section : object.sections()) {
    llvm::object::ELFSectionRef elfSection(section);
    if (elfSection.getType() == llvm::ELF::SHT_SYMTAB) {
      ++symbolTables;
    }
    if ((elfSection.getFlags() & llvm::ELF::SHF_EXECINSTR) == 0) {
      continue;
    }
    ++executableSections;
  }
  if (symbolTables != 1) {
    emitError(diagOS, llvm::Twine("expected exactly one symbol table, found ") +
                          llvm::Twine(symbolTables));
    return failure();
  }
  if (executableSections != 1) {
    emitError(diagOS,
              llvm::Twine("expected exactly one executable section, found ") +
                  llvm::Twine(executableSections));
    return failure();
  }

  for (const llvm::object::ELFSymbolRef symbol : object.symbols()) {
    if (symbol.getELFType() != llvm::ELF::STT_FUNC) {
      continue;
    }
    auto name =
        takeExpected(symbol.getName(), diagOS, "failed to read symbol name");
    if (!name) {
      return failure();
    }
    if (!requiredFunctions.contains(*name)) {
      continue;
    }
    auto address =
        takeExpected(symbol.getAddress(), diagOS,
                     llvm::Twine("failed to read address for '") + *name + "'");
    auto sectionIt =
        takeExpected(symbol.getSection(), diagOS,
                     llvm::Twine("failed to read section for '") + *name + "'");
    if (!address || !sectionIt) {
      return failure();
    }
    if (*sectionIt == object.section_end()) {
      emitError(diagOS, llvm::Twine("function '") + *name + "' is undefined");
      return failure();
    }
    llvm::object::SectionRef section = **sectionIt;
    llvm::object::ELFSectionRef elfSection(section);
    if ((elfSection.getFlags() & llvm::ELF::SHF_EXECINSTR) == 0) {
      emitError(diagOS, llvm::Twine("function '") + *name +
                            "' is not in an executable section");
      return failure();
    }
    const uint64_t size = symbol.getSize();
    const uint64_t sectionAddress = section.getAddress();
    const uint64_t sectionSize = section.getSize();
    if (size == 0 || *address < sectionAddress ||
        *address - sectionAddress > sectionSize ||
        size > sectionSize - (*address - sectionAddress)) {
      emitError(diagOS, llvm::Twine("function '") + *name +
                            "' has an invalid ELF symbol range");
      return failure();
    }
    ELFFunction function{name->str(),       *address,    size,
                         sectionAddress,    sectionSize, elfSection.getOffset(),
                         section.getIndex()};
    if (!functions.try_emplace(function.name, std::move(function)).second) {
      emitError(diagOS,
                llvm::Twine("duplicate function symbol '") + *name + "'");
      return failure();
    }
  }
  return functions;
}

static bool isVFSIMT(uint64_t instruction) {
  return (instruction & kVFSIMTFixedMask) == kVFSIMTFixedBits;
}

static std::optional<uint16_t>
decodeMOVKChunk(uint32_t instruction, unsigned chunk, unsigned targetRegister) {
  constexpr uint32_t kNop = 0x41400000;
  if (instruction == kNop) {
    return 0;
  }

  constexpr uint32_t kMOVKFixedMask = 0xffc10000;
  const uint32_t expected = chunk == 1 ? 0x07410000 : 0x07810000;
  if ((instruction & kMOVKFixedMask) != expected ||
      ((instruction >> kScalarDestinationShift) & kScalarRegisterMask) !=
          targetRegister) {
    return std::nullopt;
  }
  return static_cast<uint16_t>(instruction);
}

static std::optional<uint64_t> decodeTargetAddress(StringRef bytes,
                                                   uint64_t callerAddress,
                                                   uint64_t instructionOffset,
                                                   uint64_t instruction) {
  // BiSheng materializes the target PC in the register consumed by VF_SIMT:
  //
  //   -24: MOV  pcRegister, PC
  //   -20: MOVI targetRegister, relativeWords[15:0]
  //   -16: MOVK targetRegister, relativeWords[31:16], #1  (or NOP)
  //   -12: MOVK targetRegister, relativeWords[47:32], #2  (or NOP)
  //    -8: SHLI targetRegister, targetRegister, #2
  //    -4: ADD  targetRegister, targetRegister, pcRegister
  //      0: VF_SIMT targetRegister, ...
  //
  // targetRegister comes from the VF_SIMT encoding. The sequence computes:
  //   target PC = address of MOV PC + sign_extend(relativeWords) * 4.
  // Reject any other sequence instead of guessing its target.
  if (instructionOffset < kVFSIMTCallSequenceBytes) {
    return std::nullopt;
  }
  // Follow the register named by VF_SIMT back through MOVI/MOVK/SHLI/ADD.
  const unsigned targetRegister =
      (instruction >> kVFSIMTRegisterShift) & kScalarRegisterMask;
  const uint64_t movPc = llvm::support::endian::read32le(
      reinterpret_cast<const uint8_t *>(bytes.data() + instructionOffset -
                                        kVFSIMTCallSequenceBytes));
  const uint64_t movi = llvm::support::endian::read32le(
      reinterpret_cast<const uint8_t *>(bytes.data() + instructionOffset -
                                        kMOVIOffsetFromVFSIMTBytes));
  const uint32_t movk1 = llvm::support::endian::read32le(
      reinterpret_cast<const uint8_t *>(bytes.data() + instructionOffset -
                                        kFirstMOVKOffsetFromVFSIMTBytes));
  const uint32_t movk2 = llvm::support::endian::read32le(
      reinterpret_cast<const uint8_t *>(bytes.data() + instructionOffset -
                                        kSecondMOVKOffsetFromVFSIMTBytes));
  const uint64_t shli = llvm::support::endian::read32le(
      reinterpret_cast<const uint8_t *>(bytes.data() + instructionOffset -
                                        kSHLIOffsetFromVFSIMTBytes));
  const uint64_t add = llvm::support::endian::read32le(
      reinterpret_cast<const uint8_t *>(bytes.data() + instructionOffset -
                                        kADDOffsetFromVFSIMTBytes));
  if ((movPc & 0xffc1ffff) != 0x02000880 ||
      (movi & 0xffc10000) != 0x07000000 ||
      (shli & 0xffc1ffff) != 0x02c00202 ||
      (add & 0xffc0007f) != 0x00000001) {
    return std::nullopt;
  }
  if (((movi >> kScalarDestinationShift) & kScalarRegisterMask) !=
          targetRegister ||
      ((shli >> kScalarDestinationShift) & kScalarRegisterMask) !=
          targetRegister ||
      ((add >> kScalarDestinationShift) & kScalarRegisterMask) !=
          targetRegister ||
      ((add >> kADDTargetRegisterShift) & kScalarRegisterMask) !=
          targetRegister) {
    return std::nullopt;
  }

  const unsigned pcRegister =
      (movPc >> kScalarDestinationShift) & kScalarRegisterMask;
  const unsigned addSourceRegister =
      (add >> kADDSourceRegisterShift) & kScalarRegisterMask;
  if (pcRegister != addSourceRegister) {
    return std::nullopt;
  }

  std::optional<uint16_t> upper1 = decodeMOVKChunk(movk1, 1, targetRegister);
  std::optional<uint16_t> upper2 = decodeMOVKChunk(movk2, mlir::pto::kValue2, targetRegister);
  if (!upper1 || !upper2) {
    return std::nullopt;
  }

  // MOVI and the optional MOVK instructions form a signed 48-bit word offset.
  const uint64_t encodedRelativeWords = (movi & 0xffff) |
                                        (static_cast<uint64_t>(*upper1)
                                         << kMOVKChunkBitWidth) |
                                        (static_cast<uint64_t>(*upper2)
                                         << 2 * kMOVKChunkBitWidth);
  const int64_t relativeWords =
      llvm::SignExtend64<kRelativeWordOffsetBitWidth>(encodedRelativeWords);
  // MOV PC observes the PC of the first instruction in this materialization
  // sequence; SHLI converts the signed word offset to a byte offset.
  const uint64_t pcAddress =
      callerAddress + instructionOffset - kVFSIMTCallSequenceBytes;
  const int64_t byteOffset = relativeWords * kInstructionBytes;
  if (byteOffset < 0) {
    const uint64_t magnitude = static_cast<uint64_t>(-byteOffset);
    if (magnitude > pcAddress) {
      return std::nullopt;
    }
    return pcAddress - magnitude;
  }
  if (static_cast<uint64_t>(byteOffset) >
      std::numeric_limits<uint64_t>::max() - pcAddress) {
    return std::nullopt;
  }
  return pcAddress + static_cast<uint64_t>(byteOffset);
}

static FailureOr<llvm::SmallVector<DecodedCallSite, mlir::pto::kValue4>>
decodeCallSites(const ELFFunction &caller, StringRef objectBytes,
                llvm::raw_ostream &diagOS) {
  // Restrict decoding to the caller symbol. Searching the whole text section
  // could associate a valid-looking instruction with the wrong LLVM caller.
  const uint64_t sectionRelative = caller.address - caller.sectionAddress;
  const uint64_t fileOffset = caller.sectionFileOffset + sectionRelative;
  if (fileOffset > objectBytes.size() ||
      caller.size > objectBytes.size() - fileOffset) {
    emitError(diagOS, llvm::Twine("caller '") + caller.name +
                          "' lies outside the object file");
    return failure();
  }
  StringRef bytes = objectBytes.substr(fileOffset, caller.size);
  llvm::SmallVector<DecodedCallSite, mlir::pto::kValue4> callSites;
  for (uint64_t offset = 0; offset + sizeof(uint64_t) <= bytes.size();
       offset += kInstructionBytes) {
    uint64_t instruction = llvm::support::endian::read64le(
        reinterpret_cast<const uint8_t *>(bytes.data() + offset));
    if (!isVFSIMT(instruction)) {
      continue;
    }
    std::optional<uint64_t> target =
        decodeTargetAddress(bytes, caller.address, offset, instruction);
    if (!target) {
      emitError(diagOS,
                llvm::Twine("caller '") + caller.name +
                    "' uses an unsupported VF_SIMT target sequence at 0x" +
                    llvm::Twine::utohexstr(caller.address + offset));
      return failure();
    }
    callSites.push_back(
        {caller.address + offset, fileOffset + offset, *target,
         static_cast<uint16_t>((instruction & kVFSIMTSizeMask) >>
                               kVFSIMTSizeShift),
         instruction});
    offset += sizeof(uint64_t) - kInstructionBytes;
  }
  return callSites;
}

static LogicalResult
validateObjectHeader(const llvm::object::ELFObjectFileBase &object,
                     llvm::raw_ostream &diagOS) {
  if (object.getEType() != llvm::ELF::ET_REL) {
    emitError(diagOS, "input is not a relocatable ELF object");
    return failure();
  }
  if (object.getEMachine() != kHiIPUMachine) {
    emitError(diagOS, llvm::Twine("unexpected ELF machine 0x") +
                          llvm::Twine::utohexstr(object.getEMachine()));
    return failure();
  }
  if (!object.is64Bit() || !object.isLittleEndian()) {
    emitError(diagOS, "expected a little-endian ELF64 object");
    return failure();
  }
  return success();
}

static FailureOr<llvm::SmallVector<PatchRecord, mlir::pto::kValue4>>
buildPatchPlan(llvm::ArrayRef<SimtCallSite> manifest,
               const llvm::StringMap<ELFFunction> &functions,
               StringRef objectBytes, llvm::raw_ostream &diagOS) {
  // Match each machine callsite to a manifest callee by its decoded target
  // address. Call order alone is not sufficient proof that a patch is safe.
  llvm::DenseMap<StringRef, llvm::SmallVector<const SimtCallSite *, 4>>
      callsByCaller;
  for (const SimtCallSite &call : manifest) {
    callsByCaller[call.callerName].push_back(&call);
  }

  llvm::SmallVector<PatchRecord, mlir::pto::kValue4> plan;
  for (auto &entry : callsByCaller) {
    auto callerIt = functions.find(entry.first);
    if (callerIt == functions.end()) {
      emitError(diagOS,
                llvm::Twine("missing ELF caller symbol '") + entry.first + "'");
      return failure();
    }
    auto decoded = decodeCallSites(callerIt->second, objectBytes, diagOS);
    if (failed(decoded)) {
      return failure();
    }
    auto &calls = entry.second;
    struct ResolvedCall {
      const SimtCallSite *manifest = nullptr;
      const ELFFunction *callee = nullptr;
      bool observed = false;
    };
    llvm::SmallVector<ResolvedCall, mlir::pto::kValue4> resolvedCalls;
    resolvedCalls.reserve(calls.size());
    for (const SimtCallSite *call : calls) {
      auto calleeIt = functions.find(call->calleeName);
      if (calleeIt == functions.end()) {
        emitError(diagOS, llvm::Twine("missing ELF callee symbol '") +
                              call->calleeName + "'");
        return failure();
      }
      const ELFFunction &callee = calleeIt->second;
      if (callee.size % kInstructionBytes != 0) {
        emitError(diagOS, llvm::Twine("callee '") + callee.name +
                              "' size is not a multiple of 4 bytes");
        return failure();
      }
      const uint64_t codeSize = callee.size / kInstructionBytes;
      if (codeSize == 0 || codeSize >= kInvalidVFSIMTSize) {
        emitError(diagOS, llvm::Twine("callee '") + callee.name +
                              "' code size cannot be encoded safely");
        return failure();
      }
      // The machine optimizer may duplicate a callsite, for example while
      // unrolling a loop. Keep one allowed entry per LLVM callee and match all
      // decoded callsites by their final target address.
      auto existing =
          llvm::find_if(resolvedCalls, [&](const ResolvedCall &item) {
            return item.callee->name == callee.name;
          });
      if (existing == resolvedCalls.end()) {
        resolvedCalls.push_back({call, &callee, false});
      }
    }

    for (const DecodedCallSite &decodedCall : *decoded) {
      ResolvedCall *matched = nullptr;
      for (ResolvedCall &candidate : resolvedCalls) {
        if (candidate.callee->address != decodedCall.targetAddress) {
          continue;
        }
        if (matched && matched->callee->name != candidate.callee->name) {
          emitError(diagOS,
                    llvm::Twine("caller '") + entry.first +
                        "' has an ambiguous VF_SIMT target at 0x" +
                        llvm::Twine::utohexstr(decodedCall.targetAddress));
          return failure();
        }
        if (!matched) {
          matched = &candidate;
        }
      }
      if (!matched) {
        emitError(diagOS,
                  llvm::Twine("caller '") + entry.first +
                      "' VF_SIMT target 0x" +
                      llvm::Twine::utohexstr(decodedCall.targetAddress) +
                      " does not match any LLVM direct-call callee");
        return failure();
      }
      matched->observed = true;
      plan.push_back(
          {matched->manifest, decodedCall,
           static_cast<uint16_t>(matched->callee->size / kInstructionBytes)});
    }
    for (const ResolvedCall &call : resolvedCalls) {
      if (call.observed) {
        continue;
      }
      emitError(diagOS, llvm::Twine("caller '") + entry.first +
                            "' has no decoded VF_SIMT callsite for callee '" +
                            call.callee->name + "'");
      return failure();
    }
  }
  llvm::sort(plan, [](const PatchRecord &lhs, const PatchRecord &rhs) {
    return lhs.decoded.fileOffset < rhs.decoded.fileOffset;
  });
  for (unsigned index = 1; index < plan.size(); ++index) {
    if (plan[index - 1].decoded.fileOffset + sizeof(uint64_t) >
        plan[index].decoded.fileOffset) {
      emitError(diagOS, "VF_SIMT patch ranges overlap");
      return failure();
    }
  }
  return plan;
}

static LogicalResult
validateNoRelocationOverlap(const llvm::object::ELFObjectFileBase &object,
                            llvm::ArrayRef<PatchRecord> plan,
                            llvm::raw_ostream &diagOS) {
  // Reject a VF_SIMT covered by a relocation: the linker could overwrite the
  // patched instruction and invalidate the checks performed on the raw object.
  for (const llvm::object::SectionRef &section : object.sections()) {
    if (section.relocation_begin() == section.relocation_end()) {
      continue;
    }
    auto relocatedSection =
        takeExpected(section.getRelocatedSection(), diagOS,
                     "failed to identify the section targeted by relocations");
    if (!relocatedSection) {
      return failure();
    }
    if (*relocatedSection == object.section_end()) {
      emitError(diagOS, "relocation section has no target section");
      return failure();
    }
    llvm::object::ELFSectionRef relocatedELFSection(**relocatedSection);
    for (const llvm::object::RelocationRef &relocation :
         section.relocations()) {
      // Relocation offsets are relative to their target section. Convert them
      // to ELF file offsets before comparing with the 8-byte VF_SIMT ranges.
      const uint64_t relocationFileOffset =
          relocatedELFSection.getOffset() + relocation.getOffset();
      for (const PatchRecord &record : plan) {
        if (relocationFileOffset >= record.decoded.fileOffset &&
            relocationFileOffset <
                record.decoded.fileOffset + sizeof(uint64_t)) {
          emitError(
              diagOS,
              llvm::Twine("relocation overlaps VF_SIMT at file offset 0x") +
                  llvm::Twine::utohexstr(record.decoded.fileOffset));
          return failure();
        }
      }
    }
  }
  return success();
}

static FailureOr<ObjectPatchAnalysis>
analyzeObject(llvm::ArrayRef<SimtCallSite> manifest, StringRef objectPath,
              llvm::raw_ostream &diagOS) {
  // Complete every structural, symbol, target and relocation check before an
  // output file is written. The returned buffer remains the immutable source
  // used for the final byte-diff validation.
  auto bufferOrError = llvm::MemoryBuffer::getFile(objectPath);
  if (!bufferOrError) {
    emitError(diagOS, llvm::Twine("failed to read '") + objectPath +
                          "': " + bufferOrError.getError().message());
    return failure();
  }
  std::unique_ptr<llvm::MemoryBuffer> buffer = std::move(*bufferOrError);
  auto objectOrError =
      llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef());
  if (!objectOrError) {
    emitError(diagOS, llvm::Twine("failed to parse '") + objectPath +
                          "': " + llvm::toString(objectOrError.takeError()));
    return failure();
  }
  auto *object =
      llvm::dyn_cast<llvm::object::ELFObjectFileBase>(objectOrError->get());
  if (!object) {
    emitError(diagOS, "input is not an ELF object");
    return failure();
  }
  if (failed(validateObjectHeader(*object, diagOS))) {
    return failure();
  }
  auto functions = readFunctions(*object, manifest, diagOS);
  if (failed(functions)) {
    return failure();
  }
  auto plan = buildPatchPlan(manifest, *functions, buffer->getBuffer(), diagOS);
  if (failed(plan) ||
      failed(validateNoRelocationOverlap(*object, *plan, diagOS))) {
    return failure();
  }
  return ObjectPatchAnalysis{std::move(buffer), std::move(*plan)};
}

static LogicalResult validatePatchedBytes(StringRef rawBytes,
                                          StringRef patchedBytes,
                                          llvm::ArrayRef<PatchRecord> plan,
                                          llvm::raw_ostream &diagOS) {
  // Verify both at instruction granularity and byte granularity: only the
  // registered VF_SIMT code-size fields may differ from the raw object.
  if (rawBytes.size() != patchedBytes.size()) {
    emitError(diagOS, "patched object size differs from the raw object");
    return failure();
  }
  for (const PatchRecord &record : plan) {
    const uint64_t rawInstruction =
        llvm::support::endian::read64le(reinterpret_cast<const uint8_t *>(
            rawBytes.data() + record.decoded.fileOffset));
    const uint64_t patchedInstruction =
        llvm::support::endian::read64le(reinterpret_cast<const uint8_t *>(
            patchedBytes.data() + record.decoded.fileOffset));
    if (((rawInstruction ^ patchedInstruction) & ~kVFSIMTSizeMask) != 0) {
      emitError(diagOS, "patch changes bits outside the VF_SIMT size field");
      return failure();
    }
  }
  for (size_t offset = 0; offset < rawBytes.size(); ++offset) {
    if (rawBytes[offset] == patchedBytes[offset]) {
      continue;
    }
    if (!llvm::any_of(plan, [offset](const PatchRecord &record) {
          return offset >= record.decoded.fileOffset &&
                 offset < record.decoded.fileOffset + sizeof(uint64_t);
        })) {
      emitError(diagOS,
                llvm::Twine("unexpected byte difference at file offset 0x") +
                    llvm::Twine::utohexstr(offset));
      return failure();
    }
  }
  return success();
}

static LogicalResult writePatchedObject(StringRef path, StringRef bytes,
                                        llvm::raw_ostream &diagOS) {
  // FileOutputBuffer may leave a partial file if allocation or commit fails.
  auto removeOutput =
      llvm::make_scope_exit([&]() { llvm::sys::fs::remove(path); });
  auto output = llvm::FileOutputBuffer::create(path, bytes.size());
  if (!output) {
    diagOS << "Error: VF_SIMT size patch: failed to create '" << path
           << "': " << llvm::toString(output.takeError()) << "\n";
    return failure();
  }
  std::copy(bytes.begin(), bytes.end(), (*output)->getBufferStart());
  if (llvm::Error error = (*output)->commit()) {
    diagOS << "Error: VF_SIMT size patch: failed to write '" << path
           << "': " << llvm::toString(std::move(error)) << "\n";
    return failure();
  }
  removeOutput.release();
  return success();
}

} // namespace

FailureOr<VFSIMTSizePatchResult> mlir::pto::verifyAndPatchVFSIMTSize(
    llvm::Module &module, llvm::StringRef rawObjectPath,
    llvm::StringRef patchedObjectPath, VFSIMTSizeFixMode mode,
    llvm::raw_ostream &diagOS) {
  // Build and validate the complete patch plan first, then apply it to an
  // in-memory copy. This keeps the raw object intact and prevents partial
  // output when any callsite is unsafe or inconsistent.
  VFSIMTSizePatchResult result;
  result.objectPath = rawObjectPath.str();
  if (mode == VFSIMTSizeFixMode::Off) {
    return result;
  }

  auto manifest = collectManifest(module, diagOS);
  if (failed(manifest)) {
    return failure();
  }
  if (manifest->empty()) {
    diagOS << "PTOAS: VF_SIMT size verification passed; no patch required\n";
    return result;
  }

  auto analysis = analyzeObject(*manifest, rawObjectPath, diagOS);
  if (failed(analysis)) {
    return failure();
  }

  std::string patchedBytes = analysis->buffer->getBuffer().str();
  for (const PatchRecord &record : analysis->plan) {
    ++result.verifiedCallSites;
    // A finite value was computed by BiSheng from the final machine function.
    // It need not equal ELF st_size because the symbol may include terminal or
    // alignment instructions that are not part of the VF_SIMT fetch range.
    if (record.decoded.codeSize != kInvalidVFSIMTSize) {
      if (record.decoded.codeSize == 0 ||
          record.decoded.codeSize > record.expectedCodeSize) {
        emitError(diagOS,
                  llvm::Twine("caller '") + record.manifest->callerName +
                      "' has VF_SIMT code size " +
                      llvm::Twine(record.decoded.codeSize) +
                      " outside callee '" + record.manifest->calleeName +
                      "' symbol size " + llvm::Twine(record.expectedCodeSize));
        return failure();
      }
      continue;
    }
    if (!record.manifest->calleeContainsInlineAsm) {
      emitError(diagOS, llvm::Twine("caller '") + record.manifest->callerName +
                            "' has VF_SIMT code size " +
                            llvm::Twine(record.decoded.codeSize) +
                            ", expected " +
                            llvm::Twine(record.expectedCodeSize));
      return failure();
    }
    if (mode == VFSIMTSizeFixMode::Verify) {
      emitError(
          diagOS,
          llvm::Twine("caller '") + record.manifest->callerName +
              "' has the known invalid VF_SIMT code size 0xffff for callee '" +
              record.manifest->calleeName + "'");
      return failure();
    }
    uint64_t patchedInstruction =
        (record.decoded.instruction & ~kVFSIMTSizeMask) |
        (static_cast<uint64_t>(record.expectedCodeSize) << kVFSIMTSizeShift);
    llvm::support::endian::write64le(
        reinterpret_cast<uint8_t *>(patchedBytes.data() +
                                    record.decoded.fileOffset),
        patchedInstruction);
    ++result.patchedCallSites;
  }

  if (result.patchedCallSites == 0) {
    diagOS << "PTOAS: VF_SIMT size verification passed; no patch required\n";
    return result;
  }
  if (patchedObjectPath.empty()) {
    emitError(diagOS, "patched object path is empty");
    return failure();
  }
  if (rawObjectPath == patchedObjectPath) {
    emitError(diagOS, "raw and patched object paths must be different");
    return failure();
  }
  if (failed(validatePatchedBytes(analysis->buffer->getBuffer(), patchedBytes,
                                  analysis->plan, diagOS))) {
    return failure();
  }
  if (failed(writePatchedObject(patchedObjectPath, patchedBytes, diagOS))) {
    return failure();
  }

  auto writtenBuffer = llvm::MemoryBuffer::getFile(patchedObjectPath);
  if (!writtenBuffer || writtenBuffer.get()->getBuffer() != patchedBytes) {
    llvm::sys::fs::remove(patchedObjectPath);
    emitError(diagOS, llvm::Twine("failed to verify written object '") +
                          patchedObjectPath + "'");
    return failure();
  }
  auto patchedAnalysis = analyzeObject(*manifest, patchedObjectPath, diagOS);
  if (failed(patchedAnalysis) ||
      patchedAnalysis->plan.size() != analysis->plan.size()) {
    llvm::sys::fs::remove(patchedObjectPath);
    emitError(diagOS, "patched object failed structural verification");
    return failure();
  }
  for (unsigned index = 0; index < analysis->plan.size(); ++index) {
    const PatchRecord &rawRecord = analysis->plan[index];
    const PatchRecord &patchedRecord = patchedAnalysis->plan[index];
    if (rawRecord.decoded.fileOffset != patchedRecord.decoded.fileOffset ||
        rawRecord.expectedCodeSize != patchedRecord.expectedCodeSize) {
      llvm::sys::fs::remove(patchedObjectPath);
      emitError(diagOS, "patched object has an unexpected VF_SIMT callsite");
      return failure();
    }
    const uint16_t expectedWrittenSize =
        rawRecord.decoded.codeSize == kInvalidVFSIMTSize
            ? rawRecord.expectedCodeSize
            : rawRecord.decoded.codeSize;
    if (patchedRecord.decoded.codeSize != expectedWrittenSize) {
      llvm::sys::fs::remove(patchedObjectPath);
      emitError(diagOS, "patched object has an unexpected VF_SIMT code size");
      return failure();
    }
  }

  for (const PatchRecord &record : analysis->plan) {
    if (record.decoded.codeSize != kInvalidVFSIMTSize) {
      continue;
    }
    diagOS << "PTOAS: patched VF_SIMT size\n"
           << "  caller: " << record.manifest->callerName << "\n"
           << "  callee: " << record.manifest->calleeName << "\n"
           << "  symbol size: "
           << static_cast<uint64_t>(record.expectedCodeSize) * kInstructionBytes
           << " bytes\n"
           << "  result: replaced known invalid size with size derived from "
              "the callee symbol\n";
  }
  result.changed = true;
  result.objectPath = patchedObjectPath.str();
  return result;
}
