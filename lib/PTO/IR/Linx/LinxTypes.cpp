// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Type storage/builders/accessors and dialect dispatch are generated from ODS.
#include "PTO/IR/Linx/LinxDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#include "PTO/IR/Linx/LinxEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "PTO/IR/Linx/LinxTypes.cpp.inc"

namespace mlir::pto::linx {

void PTODialect::initializeTypes()
{
    addTypes<
#define GET_TYPEDEF_LIST
#include "PTO/IR/Linx/LinxTypes.cpp.inc"
    >();
}

static LogicalResult verifyShape(function_ref<InFlightDiagnostic()> emitError,
                                 int64_t rows, int64_t cols, Type elementType)
{
    constexpr int64_t maxElements = 256 * 1024 / 4;
    if (rows <= 0 || cols <= 0) {
        return emitError() << "Linx requires positive rank-2 f32 shapes with capacity at most 256 KiB";
    }
    if (rows > maxElements || cols > maxElements || !elementType.isF32()) {
        return emitError() << "Linx requires positive rank-2 f32 shapes with capacity at most 256 KiB";
    }
    // Both dimensions are bounded above before multiplying, so int64_t cannot overflow.
    if (rows * cols > maxElements) {
        return emitError() << "Linx requires positive rank-2 f32 shapes with capacity at most 256 KiB";
    }
    return success();
}

LogicalResult TileBufType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  TileLocation location, TileLayout layout, int64_t rows, int64_t cols,
                                  Type elementType)
{
    if (location != TileLocation::Local && location != TileLocation::Shared) {
        return emitError() << "expected Linx tile location local or shared";
    }
    if (failed(verifyShape(emitError, rows, cols, elementType))) {
        return failure();
    }
    if (location == TileLocation::Shared && layout != TileLayout::RowMajor) {
        return emitError() << "shared tiles require row_major layout";
    }
    // f32 CELL geometry: M16=16x2, M32=32x1, N8=4x8.
    int64_t physicalRows = rows;
    int64_t physicalCols = cols;
    if (layout == TileLayout::M16 || layout == TileLayout::M32) {
        int64_t height = layout == TileLayout::M16 ? 16 : 32;
        int64_t width = layout == TileLayout::M16 ? 2 : 1;
        if (rows > height) {
            return emitError() << "matrix M layout rows must not exceed " << height;
        }
        physicalRows = height;
        physicalCols = ((cols + width - 1) / width) * width;
    } else if (layout == TileLayout::N8) {
        physicalRows = ((rows + 3) / 4) * 4;
        physicalCols = ((cols + 7) / 8) * 8;
    } else if (layout != TileLayout::RowMajor) {
        return emitError() << "invalid Linx tile layout";
    }
    if (physicalRows <= 0) {
        return emitError() << "Linx tile physical rows must be positive";
    }
    constexpr int64_t maxElements = 256 * 1024 / 4;
    if (physicalRows > maxElements || physicalCols > maxElements) {
        return emitError() << "Linx tile physical capacity exceeds 256 KiB";
    }
    if (physicalRows * physicalCols > maxElements) {
        return emitError() << "Linx tile physical capacity exceeds 256 KiB";
    }
    return success();
}

LogicalResult GMViewType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 int64_t rows, int64_t cols, Type elementType)
{
    return verifyShape(emitError, rows, cols, elementType);
}

static Type parseShapedType(AsmParser& parser, bool tile)
{
    auto loc = parser.getCurrentLocation();
    if (parser.parseLess()) {
        return {};
    }
    TileLocation location = TileLocation::Local;
    if (tile) {
        StringRef spelling;
        if (parser.parseKeyword(&spelling) || parser.parseComma()) {
            return {};
        }
        auto parsed = symbolizeTileLocation(spelling);
        if (!parsed) {
            parser.emitError(loc, "expected Linx tile location local or shared");
            return {};
        }
        location = *parsed;
    }
    SmallVector<int64_t> shape;
    Type elementType;
    if (parser.parseDimensionList(shape, false, true) || parser.parseType(elementType)) {
        return {};
    }
    TileLayout layout = TileLayout::RowMajor;
    if (tile && succeeded(parser.parseOptionalComma())) {
        StringRef spelling;
        if (parser.parseKeyword("blayout") || parser.parseEqual() || parser.parseKeyword(&spelling)) {
            return {};
        }
        auto parsed = symbolizeTileLayout(spelling);
        if (!parsed) {
            parser.emitError(loc, "expected Linx layout row_major, m16, m32 or n8");
            return {};
        }
        layout = *parsed;
    }
    if (parser.parseGreater()) {
        return {};
    }
    if (shape.size() != 2) {
        parser.emitError(parser.getCurrentLocation(), "Linx requires a rank-2 shape");
        return {};
    }
    if (tile) {
        return parser.getChecked<TileBufType>(
            loc, parser.getContext(), location, layout, shape[0], shape[1], elementType);
    }
    return parser.getChecked<GMViewType>(loc, parser.getContext(), shape[0], shape[1], elementType);
}

Type TileBufType::parse(AsmParser& parser) { return parseShapedType(parser, true); }
Type GMViewType::parse(AsmParser& parser) { return parseShapedType(parser, false); }

void TileBufType::print(AsmPrinter& printer) const
{
    printer << "<" << stringifyTileLocation(getLocation()) << ", " << getRows() << "x" << getCols() << "x"
            << getElementType();
    if (getLayout() != TileLayout::RowMajor) {
        printer << ", blayout=" << stringifyTileLayout(getLayout());
    }
    printer << ">";
}

void GMViewType::print(AsmPrinter& printer) const
{
    printer << "<" << getRows() << "x" << getCols() << "x" << getElementType() << ">";
}

} // namespace mlir::pto::linx
