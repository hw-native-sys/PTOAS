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

// Spine-*scoped* extra cast layout rows.
//
// These rows are deliberately NOT part of kPreferredCastLayoutPatterns or
// kLegalCastLayoutPatterns: they are only ever reached through the
// getSpineScopedCastLayoutFact* queries, which the direction-spine peephole in
// VMILayoutAssignment calls for cast ops it has already proved to belong to one
// matched <up,down,up,down> chain *and* whose narrow result is handed straight
// to the closing widening leg (a trunc->ext handoff with no other consumer).
// A generic query - including every candidate query the layout propagator
// performs for a chain that did not match the peephole - can therefore never
// observe these rows.
//
//   (32->16): f32 deinterleaved=4 -> f16 deinterleaved=4, lane_stride=2
//   (16->32): f16 deinterleaved=4, lane_stride=2 -> f32 deinterleaved=4
//   (32->8):  f32 deinterleaved=4 -> f8  deinterleaved=4, lane_stride=4
//   (8->32):  f8  deinterleaved=4, lane_stride=4 -> f32 deinterleaved=4
//
// The narrow side then carries the conversion of wide part i in physical part i
// with the converted element sitting in the even lane of every 16-bit container
// (lane_stride=2) or in sub-lane 0 of every 32-bit container (lane_stride=4),
// which is exactly what one pto.vcvt per part already produces for one source
// chunk ({part = "EVEN"} for the 16-bit case, {part = "P0"} for the 8-bit one).
// The four physical parts are therefore produced and consumed one-to-one and no
// pto.vor is needed to assemble mutually exclusive part results into one dense
// value only to split it again at the next leg.
//
// The composite 16<->32 form is scoped to floating-point casts.  The integer
// narrowing lowering has no one-part-per-chunk route (it only knows the dense
// factor-2/factor-4 shapes), so an integer scoped leg must stay on the dense
// d(4) -> d(2) round-trip pair it already used.  The two type classes are
// disjoint, so the preferred lookup never sees more than one matching row.
static constexpr PreferredCastLayoutPattern
    kSpineScopedCastLayoutPatterns[] = {
        // Integer 16<->32: unchanged dense round trip.
        {bits<32>(), bits<16>(), 0, d(4), d(2), CastTypeClass::Integer},
        {bits<16>(), bits<32>(), 0, d(2), d(4), CastTypeClass::Integer},
        // Floating-point 16<->32: composite one-part-per-chunk form.
        {bits<32>(), bits<16>(), 0, d(4), dls(4, 2), CastTypeClass::Float},
        {bits<16>(), bits<32>(), 0, dls(4, 2), d(4), CastTypeClass::Float},
        {bits<32>(), bits<8>(), 0, d(4), dls(4, 4)},
        {bits<8>(), bits<32>(), 0, dls(4, 4), d(4)},
};

// The matching *legal* relations for the scoped rows above.  They are kept out
// of kLegalCastLayoutPatterns for the same reason: a legal row is a candidate
// generator for every query that starts from that side, so adding one globally
// would make the composite narrow form reachable from any 32<->16 or 32<->8
// cast.  The dense integer 16<->32 pair is already supplied by the generic
// legal table, so it is deliberately not repeated here (a duplicate would make
// the scoped verification ambiguous).
static constexpr LegalCastLayoutPattern kSpineScopedLegalCastLayoutPatterns[] =
    {
        {bits<32>(), bits<16>(), d(4), dls(4, 2), CastTypeClass::Float},
        {bits<16>(), bits<32>(), dls(4, 2), d(4), CastTypeClass::Float},
        {bits<32>(), bits<8>(), d(4), dls(4, 4)},
        {bits<8>(), bits<32>(), dls(4, 4), d(4)},
};

//===----------------------------------------------------------------------===//

#endif // VMILAYOUTSUPPORTSPINETABLES_INC
