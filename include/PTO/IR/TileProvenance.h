// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef PTO_IR_TILEPROVENANCE_H
#define PTO_IR_TILEPROVENANCE_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::pto {

// A source edge describes payload aliasing, never descriptor identity. View
// offsets must still be interpreted by the consumer's layout/range model.
struct TileSourceEdge {
    Value source;
    Value condition;
    bool whenTrue = true;
};

enum class TileSourceKind { LocalRoot, View, Selection, Loop, Unknown };

struct TileSources {
    TileSourceKind kind = TileSourceKind::Unknown;
    llvm::SmallVector<TileSourceEdge, 2> edges;
};

TileSources getTileSources(Value tile);
Value getTileViewSource(Value tile);

struct TileOrigins {
    llvm::SmallVector<Value, 2> roots;
    bool complete = true;
};

// Unknown producers remain explicit. In particular an incomplete/empty answer
// must not be interpreted as NoAlias by a synchronization consumer.
TileOrigins getTileOrigins(Value tile, bool followLoops = true);
bool isLocallyBoundTile(Value tile);

} // namespace mlir::pto
#endif
