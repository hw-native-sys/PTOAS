<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under
the terms and conditions of CANN Open Software License Agreement Version 2.0
(the "License").  Please refer to the License for details.  You may not use
this file except in compliance with the License.  THIS SOFTWARE IS PROVIDED
ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
express or implied.  See LICENSE in the root of the software repository.
-->

# VMI layout cost-solver conformance

This ledger defines the coverage contract for the new relation provider and
cost model.  A positive row is valid only when the conformance marker causes
the provider to enumerate the relation, the cost model accepts it, and the
lowering-conformance pass observes exactly the predicted number of physical
rearrangement operations.  A row is not considered covered by a business
regression that exercises only the final selected layout.

| Support family | Positive fixture | Relation/lowering check |
| --- | --- | --- |
| ensure data layout | `vmi_layout_cost_conformance_ensure_layout.pto` | yes |
| ensure mask layout | `vmi_layout_cost_conformance_ensure_mask_layout.pto` | yes |
| mask granularity | `vmi_layout_cost_conformance_mask_granularity.pto` | yes |
| cast (`truncf`, `extf`, `extsi`, `extui`, `trunci`) | `vmi_layout_cost_conformance_cast.pto` | yes |
| bitcast / vinterpret-cast | `vmi_layout_cost_conformance_bitcast.pto` | yes |
| channel split/merge | `vmi_layout_cost_conformance_channel.pto` | yes |
| vintlv/vdintlv | `vmi_layout_cost_conformance_vintlv.pto` | yes |
| deinterleave load and interleave store | `vmi_layout_cost_conformance_group_memory.pto` | yes; both use the shared `VMILayoutSupport` family query |
| dense/group/masked load/store | `vmi_layout_cost_conformance_load_store.pto`, `vmi_layout_cost_conformance_masked_load_store.pto`, `vmi_layout_cost_conformance_group_memory.pto` | yes |
| group broadcast and group-broadcast load | `vmi_layout_cost_conformance_group_broadcast.pto`, `vmi_layout_cost_conformance_group_broadcast_op.pto` | yes |
| group reduction (all add/max/min, float/integer) | `vmi_layout_cost_conformance_group_reduce.pto`, `vmi_layout_cost_conformance_group_reduce_quarter.pto` | yes |
| plain reduction | `vmi_layout_cost_conformance_reduce.pto`, `vmi_layout_cost_conformance_legacy_reduce.pto` | yes |
| histogram and `vselr` | `vmi_layout_cost_conformance_histogram_vselr.pto` | yes |
| generated masks and mask producers | `vmi_layout_cost_conformance_generated.pto`, `vmi_layout_cost_conformance_producers.pto` | yes |
| shuffle and compaction | `vmi_layout_cost_conformance_shuffle.pto`, `vmi_layout_cost_conformance_memory_compaction.pto` | yes |
| same-layout elementwise and unified operations | `vmi_layout_cost_conformance_elementwise.pto`, `vmi_layout_cost_conformance_unified.pto`, `vmi_layout_cost_conformance_vsel_zero.pto` | yes |

Explicit conversion edges are also checked outside the aggregate relation
pass: `vmi_layout_assignment_ensure_mask_layout.pto` verifies that assignment
does not union the source and result of `ensure_mask_layout`.  This is a
contract test for the distinction between an equivalence constraint and a
conversion edge.

The command below is the authoritative aggregate check (the build-tree test
configuration must be used so that `ptoir_obj_root` and LLVM lit settings are
available):

```sh
llvm-lit --config-prefix=lit .work/build-llvm19/test/lit \
  --filter=vmi_layout_cost_conformance_
```

The expected result is 32/32.  Each positive fixture runs the cost pass and a
second invocation that materializes every exposed relation, lowers it to
VPTO, and compares the predicted rearrangement count with emitted
`vintlv`/`vdintlv`/pack/unpack operations.

The following are intentional negative boundaries, not missing coverage:

* grouped `vci` with a non-contiguous result;
* multi-group `group_slot_load` with dynamic or unaligned source stride;
* unsupported numeric/group-slot and reduction physical recipes;
* unsupported unified merge/compare modes and the `vcvt` phase boundary.

These boundaries must remain rejected by both Support and lowering.  Adding a
case-specific relation to make one of them pass would violate the contract;
the correct change is a new complete physical recipe shared by Support, cost,
and lowering, with its own positive conformance row.
