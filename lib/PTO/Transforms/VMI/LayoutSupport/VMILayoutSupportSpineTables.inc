// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef VMILAYOUTSUPPORTSPINETABLES_INC
#define VMILAYOUTSUPPORTSPINETABLES_INC

//===- VMILayoutSupportSpineTables.inc - spine-scoped cast layout tables -*- C++ -*-===//
//===----------------------------------------------------------------------===//

// Legacy direction-spine preferences. The corresponding composite relations
// live in kLegalCastLayoutPatterns and are visible to every cast query. These
// preferences only retain the old framework's chain-specific ranking; the
// costed solver enumerates and prices the generic legal relations directly.
static constexpr PreferredCastLayoutPattern
    kSpineScopedCastLayoutPatterns[] = {
        // Integer 16<->32: unchanged dense round trip.
        {bits<32>(), bits<16>(), 0, d(4), d(2), CastTypeClass::Integer},
        {bits<16>(), bits<32>(), 0, d(2), d(4), CastTypeClass::Integer},
        // Floating-point 16<->32: composite one-part-per-chunk form.
        {bits<32>(), bits<16>(), 0, d(4), dls(4, 2), CastTypeClass::Float},
        {bits<16>(), bits<32>(), 0, dls(4, 2), d(4), CastTypeClass::Float},
        {bits<32>(), bits<8>(), 0, d(4), dls(4, 4), CastTypeClass::Float},
        {bits<8>(), bits<32>(), 0, dls(4, 4), d(4), CastTypeClass::Float},
        // Exact rows for the 128-element shapes the default rows above
        // mis-size: such a value fits in one part at 16-bit and two at 32-bit,
        // so the factor-4 composite leaves every carrier half empty and doubles
        // the carrier count.  Factor 2 keeps one part per chunk here.
        {bits<32>(), bits<16>(), 128, d(2), dls(2, 2), CastTypeClass::Float},
        {bits<16>(), bits<32>(), 128, dls(2, 2), d(2), CastTypeClass::Float},
        {bits<32>(), bits<8>(), 128, d(2), dls(2, 4), CastTypeClass::Float},
        {bits<8>(), bits<32>(), 128, dls(2, 4), d(2), CastTypeClass::Float},
};

//===----------------------------------------------------------------------===//

#endif // VMILAYOUTSUPPORTSPINETABLES_INC
