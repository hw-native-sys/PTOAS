// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

// ---------------------------------------------------------------------------
// Print conversion kind — drives wrapper selection for the CCE cce::printf
// shims: SignedInt → pto_print_i64, UnsignedInt → pto_print_u64.
// ---------------------------------------------------------------------------
enum class PrintConversionKind : uint8_t {
  Float,       // %f, %F, %e, %E, %g, %G, %a, %A
  SignedInt,   // %d, %i
  UnsignedInt, // %u, %x, %X, %o
};

// ---------------------------------------------------------------------------
// Result of format-string analysis.
// ---------------------------------------------------------------------------
struct PrintFormatInfo {
  PrintConversionKind conversion;
};

// ---------------------------------------------------------------------------
// Parse a PrintOp format string and produce a PrintFormatInfo.
//
// The format string must contain exactly one conversion specifier (%% is
// treated as a literal '%' and does not count).  Returns failure() with a
// diagnostic message on parse errors.
// ---------------------------------------------------------------------------
inline FailureOr<PrintFormatInfo>
analyzePrintFormat(llvm::StringRef format) {
  if (format.empty())
    return failure();

  PrintFormatInfo info{};
  bool hasConversion = false;
  size_t pos = 0;

  while (pos < format.size()) {
    if (format[pos] != '%') { ++pos; continue; }

    ++pos;
    if (pos >= format.size())
      return failure(); // trailing '%'

    if (format[pos] == '%') { ++pos; continue; } // skip %%

    if (hasConversion)
      return failure(); // more than one conversion

    // Skip flags.
    while (pos < format.size() &&
           (format[pos] == '+' || format[pos] == '-' ||
            format[pos] == '0' || format[pos] == '#' ||
            format[pos] == ' '))
      ++pos;
    // Skip width.
    while (pos < format.size() && format[pos] >= '0' && format[pos] <= '9')
      ++pos;
    // Skip precision.
    if (pos < format.size() && format[pos] == '.') {
      ++pos;
      while (pos < format.size() && format[pos] >= '0' && format[pos] <= '9')
        ++pos;
    }
    // Skip length modifier; allow doubled forms (e.g. %lld, %hhx).
    if (pos < format.size() &&
        (format[pos] == 'l' || format[pos] == 'h' || format[pos] == 'z')) {
      char lengthMod = format[pos++];
      if (pos < format.size() && format[pos] == lengthMod)
        ++pos;
    }
    if (pos >= format.size())
      return failure(); // incomplete specifier

    char conv = format[pos++];
    hasConversion = true;

    // Classify conversion kind.
    switch (conv) {
    case 'f': case 'F': case 'e': case 'E':
    case 'g': case 'G': case 'a': case 'A':
      info.conversion = PrintConversionKind::Float;
      break;
    case 'd': case 'i':
      info.conversion = PrintConversionKind::SignedInt;
      break;
    case 'u': case 'x': case 'X': case 'o':
      info.conversion = PrintConversionKind::UnsignedInt;
      break;
    default:
      return failure(); // unsupported specifier
    }

    // Keep scanning to detect extra conversions (which we'll reject).
  }

  if (!hasConversion)
    return failure(); // no conversion specifier

  return info;
}

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PRINTENCODING_H
