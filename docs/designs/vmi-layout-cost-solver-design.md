# VMI Layout Cost Solver Design

## 1. Status and Design Lens

This document defines the implemented design contract for cost-based VMI layout
assignment. The planner, frontier solver, physical cost model, and immutable
plan commit path are implemented in `VMILayoutPlanner`,
`VMILayoutConflictSolver`, and `VMILayoutCostModel`; unsupported operation
families must fail explicitly until their shared Support relations are added.

The optimization problem is not "which support row should an operation use?"
in isolation. The target cases already contain layout requirements that cannot
all be made equal. The optimization problem is:

> Among complete legal whole-graph layout plans, choose where the unavoidable
> producer-to-use conversions occur and minimize their total static layout-
> conversion cost, including conversion cost hidden inside operation lowering.

This distinction fixes the ownership boundaries:

```text
VMILayoutSupport and relation queries
  own operation legality and explicit-conversion legality

solver constraint propagation
  removes choices that cannot participate in a legal completion
  never uses preference to commit a layout

DFS or Frontier DP
  chooses among the legal choices that remain after propagation

VMILayoutCostModel
  evaluates complete plans and admissible partial-state lower bounds

VMILayoutPlanValidator
  independently validates the selected complete plan before mutation

VMILayoutPlanApplier
  applies the validated plan to IR without propagation or layout decisions
```

The solver does not replace support and does not introduce a second set of
layout rules. Its propagation is read-only constraint filtering over the same
support queries; it is not the current priority/seed propagation algorithm.

## 2. Normative Contract

```text
C1. Existing VMILayoutSupport facts and the unified operation relation query are
    the sole source of operation and explicit-conversion legality. The planner
    must not copy support rules into a planner-only table.

C2. A complete plan assigns exactly one layout to every layout-bearing SSA
    value and exactly one required layout to every layout-bearing operand use.
    A value and each of its uses are separate variables.

C3. For an SSA value v used by a non-structural operand u:

      assignedLayout(v) == requiredLayout(u)
        cost 0

      assignedLayout(v) != requiredLayout(u)
        legal only when the registered ensure conversion is supported
        cost getConversionCost(type(v), assignedLayout(v),
                               requiredLayout(u))

    This producer-to-use factor is the representation of an explicit conversion
    position. It must not be hidden in the producer relation.

C4. An operation relation constrains all layout-bearing operand-required and
    result-assigned ports of one concrete operation. It may be table-backed,
    rule-backed, parameterized by an arbitrary layout, and dependent on element
    type, shape, attributes, or analyzable operand facts. All forms use the same
    relation-query interface.

C5. Relation queries may filter domains and report logical consequences common
    to every legal completion. They may not choose a preferred relation or
    layout. Preference affects deterministic branch ordering and final tie-
    breaking only.

C6. Relation enumeration is domain-stable. If a concrete relation is returned
    for a finite candidate domain, re-querying the same operation with the
    singleton domain formed from that relation's port layouts must return the
    identical canonical relation, including its recipe/direct-producer marker
    and intrinsic rearrangement cost. This prevents candidate-aware providers
    from silently dropping a Support row during domain narrowing.

C7. A selected operation relation may contain a layout conversion hidden by
    lowering. The plan pays getRelationCost(op, selectedRelation) exactly once.

C8. PlanCost is one checked non-negative scalar:

      PlanCost = sum(producer-to-use conversion costs)
               + sum(operation-relation costs)

    Only layout rearrangement and merge instructions contribute. Native memory
    instruction count, semantic compute, vcvt, E2B/BRC, and physical-noop
    bitcasts are zero by themselves.

C9. Cost is not weighted by loop depth, trip count, latency, throughput, or
    execution frequency in version one.

C10. Every preassigned layout on a layout-bearing SSA value, including an
    internal block argument or result, and every verified enclosing-IR boundary
    layout is a fixed constraint. Unassigned VReg/mask function boundaries are
    outside the version-one cost-solver input contract. Production end-to-end
    input does not contain such layout-bearing function ABI values.

C11. Structural SCF, CFG, direct-call, return, and signature relationships use
     the same hard layout equalities as current layout assignment. For a
     structural operand u carrying source v to destination d:

       A(v) == R(u) == A(d)

     Structural transport is not a conversion position in version one.

C12. Search and exact validation are read-only. IR mutation begins only after
     every component has produced a complete plan and the merged plan has
     produced a fully validated apply recipe. The applier follows that recipe
     exactly and performs no propagation or layout choice. Priority assignment
     is not an implicit fallback for cost-solver failure or budget exhaustion.

C13. One invocation selects exactly one engine: Frontier DP or DFS branch-and-
     bound. Both consume the same immutable problem, propagation transition,
     cost model, validation, ordering key, and budget contract.

C14. Exact means the selected engine exhausted or soundly bounded every plan
     that could beat its result. BestEffort is a complete legal incumbent found
     before a deterministic bound. NoCompletePlan means a bound was reached
     before a complete plan. Infeasible is returned only after a sound proof of
     no legal completion.

C15. Every layout-bearing operation port, SSA use, fixed layout, and structural
     equality must appear in the constructed problem. Every operation relation
     and normal mismatched use selected by a plan must be accepted by the shared
     support contract and have an explicit cost classification. There is no
     default-zero unsupported VMI relation operation. Recognized SCF, CFG,
     function, call, and return transport is represented by use and hard-
     equality factors rather than a fake VMI operation relation.
```

The objective is a deterministic static layout-conversion heuristic, not a
complete device-performance model.

## 3. Acceptance Cases

### 3.1 E2B f32 corridor

The regression is
`test/lit/vmi_new/opt/group_broadcast_load_e2b_layout_opt.pto`. For its concrete
`vreg<256xf32>`, eight-group, unit-stride input, E2B produces `d4`.

```text
Plan A: keep the corridor on d4

  group_broadcast_load d4        0
  dense load d4                  2 * vdintlv
  truncf d4 -> f8 c              3 * vor
  extf c -> f32 d4               0
  store d4 -> c                  4 * vintlv
  -------------------------------------------------
  PlanCost                       9

Plan B: put the conversion on the first use after the E2B producer

  group_broadcast_load d4        0
  ensure_layout d4 -> c          4 * vintlv
  dense load c                   0
  truncf c -> f8 ls4             0
  extf ls4 -> f32 c              0
  store c                        0
  -------------------------------------------------
  PlanCost                       4
```

The producer relation is `d4` in both plans. The solver decision is the layout
required by its uses and the resulting conversion position, not an alternative
`group_broadcast_load` relation. The solver must choose Plan B, and the `d4 -> c`
must be visible as `ensure_layout` in assignment IR.

### 3.2 ComputeY1 f16 -> f32 -> f8 corridor

The regression is
`test/lit/vmi_new/opt/compute_y1_to_fp8_fp16_vmi_opt.pto`. Its E2B producer emits
`d2`.

```text
Plan A: preserve the widening corridor

  E2B f16 d2                     0
  extf d2 -> f32 d4              0
  normal load f16 d2             0
  extf d2 -> f32 d4              0
  compute f32 d4                 0
  truncf f32 d4 -> f8 c          3 * vor
  store f8 c                     0
  -------------------------------------------------
  PlanCost                       3

Plan B: convert to the lane-stride corridor near the producer

  E2B f16 d2                     0
  ensure_layout d2 -> ls2        1 * vintlv + 4 * vzunpack
  normal load f16 ls2            0
  extf ls2 -> f32 c              0
  compute f32 c                  0
  truncf c -> f8 ls4             0
  store f8 ls4                   0
  -------------------------------------------------
  PlanCost                       5
```

The solver must choose Plan A. This is the opposite conversion-placement result
from Section 3.1 and prevents a producer- or consumer-priority policy from being
mistaken for the optimization.

### 3.3 Multiple uses

Every use owns an independent required-layout variable and conversion factor.
Converting one use does not implicitly convert another use. For the current
two-use E2B fixture the compared costs are equal:

```text
two explicit d4 -> c use conversions                       8
two hidden dense-load d4 costs plus one hidden store cost  8
```

The deterministic tie-break may select the second plan, but the test must not
claim that its PlanCost is lower. Another fixture must cover strictly unequal
multi-use costs.

## 4. Weighted Layout Constraint Graph

### 4.1 Variables and factors

The problem is a weighted constraint graph, equivalently a finite weighted CSP
after Section 5 constructs component candidate pools and variable domains.

```text
A(v)  assigned layout of layout-bearing SSA value v
R(u)  required layout of layout-bearing OpOperand u
```

Here layout-bearing means `VMIVRegType` or a concrete-granularity
`VMIMaskType`. Predicate-view masks carry no layout variable. Cost-solver mode
requires mask-granularity assignment to have established every concrete mask
layout port needed by an operation relation; an unexpected predicate-view port
that requires layout is malformed input, not an arbitrary layout choice.

The graph has four factor kinds:

```text
Operation factor
  scope: R(layout-bearing operands), A(layout-bearing results)
  legality: shared operation relation query
  cost: getRelationCost for the selected relation witness

Use factor
  scope: A(u.get()), R(u)
  normal use legality/cost: identity or registered explicit conversion
  structural use legality/cost: identity only, zero

Fixed factor
  scope: one value, use, or enclosing-IR port variable
  legality: variable == explicit verified layout
  cost: zero

Hard-equality factor
  scope: two structural variables
  legality: equal layouts
  cost: zero
```

This representation makes conversion placement explicit. Moving a conversion
means changing which use factor has unequal endpoints while keeping every
operation factor legal.

### 4.2 Stable port identity

Only layout-bearing ports participate. Their vector position must never be
confused with the raw MLIR operand/result number because VMI operations can
interleave pointers, indices, scalars, masks, and VRegs.

```cpp
using VMILayoutVariableId = uint32_t;

enum class VMILayoutPortKind { Operand, Result };

struct VMILayoutPortBinding {
  VMILayoutPortKind kind;
  unsigned opPortNumber; // Raw MLIR operand or result number.
  VMILayoutVariableId variable;
};

struct VMILayoutOpConstraint {
  Operation *op;
  SmallVector<VMILayoutPortBinding> ports;
};

struct VMILayoutUseConstraint {
  OpOperand *use;
  VMILayoutVariableId producer;
  VMILayoutVariableId requirement;
};

struct VMILayoutFixedConstraint {
  VMILayoutVariableId variable;
  VMILayoutAttr layout;
  VMILayoutFixedSource source;
};

struct VMILayoutHardEquality {
  VMILayoutVariableId lhs;
  VMILayoutVariableId rhs;
  VMILayoutHardEqualityKind kind;
};
```

Each layout-bearing `OpOperand` has one `R(u)` variable and one use constraint.
Multiple uses share `A(v)` but have distinct `R(u)` variables. Normal use
constraints remain uncontracted. Structural use constraints are identity-only;
their `A(source)`, `R(use)`, and destination variable are contracted by the
hard-equality closure before component discovery.

`RelationSchemaId` and stable support-row provenance belong to a returned
`VMILayoutRelationWitness`, not to `VMILayoutOpConstraint`: one concrete
operation constraint may admit witnesses from several table rows or rule-backed
schemas.

Structural transport uses the following exact mapping:

```text
source SSA value -> branch/yield/call/return operand
  A(source) == R(structural operand)

branch/yield destination block or region argument
  R(structural operand) == A(destination argument)

call operand and callee argument
  R(call operand) == functionInputPort == A(callee argument)

function input port and entry block argument
  functionInputPort == A(entry block argument)

return operand and function result port
  R(return operand) == functionResultPort

function result port and direct-call result
  functionResultPort == A(call result)
```

Consequently `A(source)`, `R(structural operand)`, and the corresponding
destination variable are one canonical variable after equality contraction.
The solver neither searches nor inserts a conversion on structural transport.
Adding such placement later requires a separate control-flow type-rewrite and
materialization design; it is not implicit in the normal use-factor rule.

Construction must reuse the current assignment's structural-edge discovery,
not infer edges from generic operand/result position. In particular it creates
the following hard-equality sets:

```text
scf.for
  init, region iter argument, corresponding yield operand, result

scf.while
  init, before-region argument, corresponding after-region yield operand
  condition forwarded value, after-region argument, corresponding result
  (these are two distinct carry sequences and are not united by index)

scf.if, scf.execute_region, scf.index_switch
  each result and every corresponding region yield operand

BranchOpInterface successor edge
  each successor operand and corresponding destination block argument

direct internal func.call
  each call operand, callee function-input port, and callee argument
  each call result and every corresponding callee return operand

func.return within one function
  corresponding return operands across all returns and the function-result port
```

Indirect calls, external declarations with VMI signatures, malformed successor
arity, and unsupported region/control-flow forms retain the current assignment
contract: problem construction rejects them rather than treating their ports as
independent layout choices.

### 4.3 Operation relations are constraints, not pre-expanded tuples

An operation relation is the set of legal complete assignments to one operation
factor's ports. The representation must preserve the semantics of the existing
support query:

- a finite table can return finite alternatives;
- a rule can inspect the concrete operation and operand facts;
- a same-layout rule can express one arbitrary `L` shared by all ports;
- a free-result rule can leave a result unrestricted;
- a cast or grouped operation can couple several ports and return several legal
  alternatives.

The provider is a stateless adapter over `VMILayoutSupport`. Every query receives
the concrete operation and the current finite port domains; it does not retain a
second copy of support relations. `SameLayout` and `FreeResult` are handled by
the same support query as table-backed operations, not by planner-side rules.

```cpp
struct VMILayoutPortDomains {
  ArrayRef<VMILayoutDomain> domains; // Indexed by VMILayoutPortBinding.
};

struct VMILayoutRelationPropagation {
  SmallVector<VMILayoutDomainReduction> reductions;
  bool infeasible;
};

struct VMILayoutRelationWitness {
  SmallVector<VMILayoutAttr> portLayouts; // Indexed by port binding.
  RelationSchemaId schema;
  uint32_t stableRowOrdinal;
  uint32_t preferenceRank;
  VMIRelationCostWitness costWitness;
};

using VMILayoutRelationVisitor =
    llvm::function_ref<LogicalResult(const VMILayoutRelationWitness &)>;

class VMILayoutFactVisitBudget {
public:
  LogicalResult consumeBeforeVisit();
  bool exhausted() const;
};

class VMILayoutRelationProvider {
public:
  FailureOr<uint64_t>
  visitDomainIndependentFacts(const VMILayoutOpConstraint &constraint,
                              VMILayoutFactVisitBudget &budget,
                              VMILayoutRelationVisitor visitor) const;

  FailureOr<VMILayoutRelationPropagation>
  propagate(const VMILayoutOpConstraint &constraint,
            const VMILayoutPortDomains &domains,
            VMILayoutFactVisitBudget &budget) const;

  FailureOr<VMILayoutRelationIterator>
  enumerate(const VMILayoutOpConstraint &constraint,
            const VMILayoutPortDomains &domains,
            VMILayoutFactVisitBudget &budget) const;

  FailureOr<VMILayoutRelationWitness>
  validate(const VMILayoutOpConstraint &constraint,
           ArrayRef<VMILayoutAttr> completePortLayouts,
           VMILayoutFactVisitBudget &budget) const;
};
```

`visitDomainIndependentFacts` visits the finite facts that the authoritative
support query can enumerate from the concrete operation, types, attributes, and
analyzable operand facts without candidate layout domains. The visitor permits
construction budgets to be checked before retaining each fact or pool entry. A
parameterized same/free relation visits zero facts and succeeds; it is not
unsupported and contributes no candidate layout. Once finite domains exist,
`propagate` and `enumerate` query every relation form uniformly. `propagate`
removes port values that occur in no legal completion and returns only
consequences common to all retained relations. `enumerate` lazily produces legal
concrete witnesses inside the domains and must not build a Cartesian product
first. `validate` is the exact singleton-assignment query used at completion and
commit.

Every method consumes the immutable operation constraint built in Section 4.2.
The provider uses its port bindings as the only raw-port-to-domain mapping; it
must not rediscover layout-bearing ports independently. Before examining each
finite fact or parameterized completion it calls `consumeBeforeVisit`; failure
stops the query before that fact is processed. The returned count is the number
of facts whose visitor was invoked. If the visitor fails, visitation stops
immediately and that failure is propagated; no returned count is observed on the
failed path. Construction and each search engine supply differently bounded
instances. Independent final validation supplies an explicitly unlimited
instance, so it uses the same query path without consuming search budget.
After a failed query, the caller tests `exhausted()` to distinguish a resource
bound from a support-contract or visitor failure. Construction maps exhaustion
to `ProblemResourceLimit`; an engine maps it to its bounded solve status.

The empty-success distinction is reported by `VMILayoutSupport` itself. The
provider must not recognize operation families or parse diagnostic strings to
manufacture it. A malformed operation, missing query implementation, or
non-parameterized operation with no legal concrete fact is a failed query, not
an empty successful result.

For one concrete operation, a complete ordered port-layout tuple is the search
identity of a relation. If multiple raw support/rule paths produce that tuple,
the provider compares their legality and complete `getRelationCost` result
before canonicalization. A disagreement is a support ambiguity and fails the
pass as a support-contract error, whether discovered during construction,
search, or exact validation; it is never interpreted as an infeasible state.
Equivalent duplicates retain the best deterministic preference and stable-row
metadata. Consequently a total legal port assignment has one cost-equivalent
canonical witness.

For a parameterized relation, all legality- and cost-symmetric instantiations
share one schema ID, preference rank, and generated-relation ordinal; the layout
parameter itself is represented only in `portLayouts` and the final stable layout
trace. A rule with layout-dependent preference, provenance, or cost is not
symmetric and must expose the corresponding finite behavior representatives.

The unified support result must carry the complete raw-port tuple, schema ID,
stable row ordinal, preference rank, and the symmetric facts needed by
`getRelationCost`. The provider maps raw ports through `VMILayoutPortBinding`
and canonicalizes duplicate tuples; it does not reconstruct discarded metadata,
classify operation families, or infer new legality.

## 5. Finite Active Layout Universe

### 5.1 Why a universe is needed

`VMILayoutAttr` is syntactically parameterized, and free/same-layout operations
can accept an arbitrary layout. Search must nevertheless operate on a finite set
without inventing a global list of all attributes.

For each constraint component, the planner constructs one candidate layout pool
from existing assets:

```text
1. layouts on fixed internal SSA values and enclosing-IR boundaries
2. layouts occurring in legal finite support facts of incident operations
3. the finite closure of registered explicit-conversion rows reachable from the
   layouts in rules 1-2, queried for every concrete VReg/mask type class in the
   component and instantiated only with parameters exposed by the component
4. one canonical contiguous layout if the component contains a parameterized
   relation or if the first three sets are empty
```

Problem construction calls `visitDomainIndependentFacts` for every incident
operation and adds every non-null layout in the returned tuples to the shared
component pool. Parameterized same/free relations contribute no layout in this
phase, but rule 4 supplies one representative for their otherwise unenumerated
symmetric layout parameter. This is required even when another port in the same
component already seeded the pool: for example, a free-result relation may have
a constrained `d4` input and an independent arbitrary-layout result. Because the
pool is component-wide, a layout introduced at an f32 port is available to an
f16, integer, or mask port connected through a polymorphic relation; its
applicability to that concrete type is checked when domains are built.

A parameterized relation may omit layouts only when all omitted instances are
legality- and cost-symmetric. If its legality or hidden cost distinguishes a
layout parameter, the provider must expose the finite distinguished layouts or
finite behavior representatives through rule 2. Otherwise the relation is not
conforming to cost-solver mode.

For rule 3, every layout currently in the pool is queried as both a source and a
result endpoint for every concrete type class in the component. Only registered
conversion facts may add the opposite endpoint. The query uses only group count,
slot count, and lane-stride parameters exposed by an incident operation, fixed
layout, or already registered endpoint. Successful endpoints are added to a
worklist until no new endpoint is found. The support registry must guarantee
finite closure for these concrete parameters; exceeding the construction bound
is `ProblemResourceLimit`.

### 5.2 Completeness argument

A layout outside the component pool can occur only inside a connected island of
parameterized free/same-layout relations:

- if it reaches a fixed boundary, it is included by rule 1;
- if it reaches a constrained operation relation, it is included by rule 2;
- if it crosses a non-identity use conversion, it is included by rule 3.

Therefore an outside layout that can affect legality or PlanCost would have
entered the component pool through one of those rules. Sharing the pool across
type classes preserves layouts carried through cross-type same-layout relations.
A remaining polymorphic layout parameter that is not anchored by rules 1-3 has
no layout-dependent relation legality or cost. Rule 4 represents its symmetric
class with canonical contiguous, defined as the best stable layout-trace
representative; identity uses also minimize explicit materializations. Choosing
it therefore cannot lose a plan that is better under any component of the
complete result ordering. The generated-relation metadata normalization in
Section 4.3 ensures that no earlier tie-break component varies with this
otherwise arbitrary layout parameter.

This proof is a required conformance property. Adding a support family whose
legal or cost-relevant layout is not exposed to rules 1-3 is a construction
error, not permission to fall back to priority assignment.

### 5.3 Domains

After the pool is built, each canonical variable starts with the pool members
accepted by `isApplicableVMILayout(type, layout)`. This pure helper rebuilds the
concrete VReg or mask type and applies exactly the type-level layout checks shared
with `VMIVRegType::verify` and `VMIMaskType::verify`; those verifiers must be
factored to call the same helper. In particular, it covers group-slot element
count and predicate-mask restrictions. It does not call an operation verifier,
test physical arity as a proxy for validity, or add operation-specific legality.
Fixed factors and operation relation queries then remove unsupported values.

```text
domain size 0  -> infeasible
domain size 1  -> logically forced
domain size >1 -> unresolved legal choice
```

No `Top` layout attribute exists in IR or in a committed plan.

## 6. Legality and Cost

### 6.1 Cost interfaces

```cpp
struct VMILayoutCost {
  int64_t instructions = 0;
};

enum class VMIRelationCostClass {
  ExplicitEdge,
  ZeroHidden,
  Hidden,
};

enum class VMIHiddenCostFamily {
  DenseLoad,
  DenseStore,
  GroupStoreStaging,
  MaskedStore,
  GeneratedMask,
  MaskGranularityLayout,
  NarrowCast,
  ChannelStaging,
};

struct VMIRelationCost {
  VMIRelationCostClass classification;
  std::optional<VMIHiddenCostFamily> hiddenFamily;
  VMILayoutCost cost;
};

FailureOr<VMILayoutCost>
getConversionCost(Type type, VMILayoutAttr sourceLayout,
                  VMILayoutAttr resultLayout);

FailureOr<VMIRelationCost>
getRelationCost(Operation *op,
                const VMILayoutRelationWitness &relation);
```

Use factors call `getConversionCost`. Operation factors call
`getRelationCost`. Fixed and equality factors have no cost. There is no implicit
`ZeroHidden`; every selected operation relation receives an explicit class.

The existing ensure tables also need one finite query view in
`VMILayoutSupport`:

```cpp
struct VMILayoutConversionDomains {
  std::optional<VMILayoutDomain> sources;
  std::optional<VMILayoutDomain> results;
};

struct VMILayoutConversionWitness {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  uint32_t stableRowOrdinal;
};

using VMILayoutConversionVisitor =
    llvm::function_ref<LogicalResult(const VMILayoutConversionWitness &)>;

FailureOr<uint64_t>
visitExplicitConversionFacts(Type logicalType,
                             const VMILayoutConversionDomains &domains,
                             VMILayoutFactVisitBudget &budget,
                             VMILayoutConversionVisitor visitor);
```

At least one endpoint domain must be present and finite. With one endpoint
present, the query visits the finite registered opposite endpoints and is used
for candidate-pool closure. With both present, it visits the supported pairs and
is used by use-factor propagation and evaluation. Singleton domains are the
exact validation query. Data and mask conversions share this interface but
remain distinct support-table rows. Identity is returned for every concrete
layout in the supplied endpoint domain; the query never attempts to enumerate
an unconstrained universe. As with operation relations, visitor failure is
propagated and `consumeBeforeVisit` is called before processing each fact.

This is an enumeration view of the authoritative ensure support tables, not a
second conversion registry. `getConversionCost` is called only for a canonical
witness accepted by this query and uses the same registered direct or composed
path. Singleton-query and existing `getEnsureLayoutFact`/
`getEnsureMaskLayoutFact` results must agree.

The ordered source/result pair is the conversion identity. Duplicate registered
paths for one pair must have the same legality and cost; disagreement is a
support-contract error, and equivalent duplicates retain the best stable row
ordinal. The solver does not choose between lowering paths.

`VMIRelationCostWitness` is an immutable typed payload at the support/cost
boundary. `VMILayoutSupport` owns and returns the selected lowering-relevant
facts, such as a direct/fallback path, staging endpoints, channel direction, or
the `q_i` narrowing partition. `VMILayoutCostModel` alone maps that payload to
`ExplicitEdge`, `ZeroHidden`, or a named hidden family and computes the scalar
cost. The payload is not an untyped integer vector and does not contain emitted
VPTO operations. `getRelationCost` must be a total checked function for every
witness returned by the relation query.

### 6.2 Explicit conversion table

Only registered conversions are legal and costable:

For a concrete conversion, let `S = getVMIPhysicalArity(sourceType)` and
`R = getVMIPhysicalArity(resultType)`, where the two types carry the concrete
source and result layouts. Division in the table is exact for the registered
row; a row whose divisibility precondition is not met is not legal.

```text
L -> L                         0
c <-> gs(1, 1, 1), one element 0

c -> d2                        R/2 * vdintlv
d2 -> c                        ceil(R/2) * vintlv
c -> d4                        4 * (R / 4) * vdintlv
d4 -> c                        (2*ceil(R/4) + ceil(R/2)) * vintlv

c -> ls2                       R * vzunpack
c -> ls4                       2R * vzunpack
ls2/ls4 -> c                   registered bounded pack tree

gs(g,8,l1) -> gs(g,8,l2)       S * abs(log2(l2) - log2(l1))

d2 <-> d4                      registered path through c
d2 <-> ls2                     registered path through c
ls2 <-> ls4                    registered path through c
```

The bounded pack-tree instruction count is defined by:

```text
PackTree(k, levels):
  cost = 0
  repeat levels times:
    cost += k                  // current carriers participate
    cost += floor(k / 2)       // pairwise merge instructions
    k = ceil(k / 2)
  return cost

ls2 -> c: PackTree(S, 1)
ls4 -> c: PackTree(S, 2)
```

Thus the registered groups produce `1, 3` instructions for `ls2` source counts
`1, 2`, and `2, 4, 7, 9` instructions for `ls4` source counts `1, 2, 3, 4`.

The cost lookup never invents an unregistered transitive path. A registered
composed path records its ordered intermediate layouts and its cost is the
checked sum of the constituent registered edge costs for the corresponding
intermediate types. It is one explicit use materialization for tie-breaking even
when its lowering emits several rearrangement instructions.

Here `gs(numGroups, slots, laneStride)` names all cost-relevant group-slot
parameters. The registered group-slot rows keep `numGroups` and `slots=8`
unchanged and vary only `laneStride` in `{1, 2, 4}`.

Mask conversions use the same arity formulas and bounded tree shape as their
data counterparts, with predicate instructions substituted:

```text
c -> d2                        R/2 * pdintlv
d2 -> c                        ceil(R/2) * pintlv
c -> d4                        4 * (R / 4) * pdintlv
d4 -> c                        (2*ceil(R/4) + ceil(R/2)) * pintlv
c -> ls2                       R * punpack
c -> ls4                       2R * punpack
ls2 -> c                       PackTree(S, 1) using ppack/por
ls4 -> c                       PackTree(S, 2) using ppack/por
c <-> bd2/bd4                  0, physical forwarding only
```

As for data conversions, division is exact for each registered mask row and
partial lane-stride groups use the bounded `PackTree` definition. The reverse
deinterleave formulas use `R` because dead tail outputs are removed; for full
groups they reduce to the former `S/2` and `S` counts.

### 6.3 Hidden-cost audit

The following list is exhaustive for version one:

```text
load, direct d4 row with post-load vdintlv rearrangement
  Hidden(DenseLoad): 2 * (R / 4)

load, fallback through contiguous
  Hidden(DenseLoad): ConversionCost(type, c, resultLayout)

store, staging value to contiguous
  Hidden(DenseStore): ConversionCost(type, valueLayout, c)

group_store, compact-small gs(g,8,l), l != 1, staged to gs(g,8,1)
  Hidden(GroupStoreStaging): ConversionCost(type, gs(g,8,l), gs(g,8,1))

masked_store, d2/d4 value and mask staged through contiguous
  Hidden(MaskedStore): value-to-c plus mask-to-c

create_group_mask, created as contiguous then staged
  Hidden(GeneratedMask): ConversionCost(maskType, c, resultLayout)

ensure_mask_granularity, carrier layout also changes
  Hidden(MaskGranularityLayout): registered predicate carrier conversion

truncf, trunci, narrowing fptosi/fptoui, converted partials are merged
  Hidden(NarrowCast): sum_i(q_i - 1)

channel_split, contiguous source staged to d2/d4
  Hidden(ChannelStaging): ConversionCost(sourceType, c, dN)

channel_merge, d2/d4 joined to contiguous result
  Hidden(ChannelStaging): ConversionCost(resultType, dN, c)
```

For `Hidden(NarrowCast)`, `q_i` is the number of converted source partials merged
into physical result part `i` by the selected relation witness. A merge tree of
`q_i` partials emits exactly `q_i - 1` layout-merge instructions. The witness
must expose this partition; the cost model does not reconstruct it from a
different lowering implementation.

Existing non-identity `ensure_layout` and `ensure_mask_layout` operations are
`ExplicitEdge`; identity ensures and ordinary relations with no relation-induced
rearrangement are `ZeroHidden`.

Semantic vselr/reduction trees, semantic vintlv/vdintlv, granularity-only mask
packing, memory-format packing independent of the selected layout relation, and
native instruction count differences such as `1 x vldsx2` versus `4 x vlds` do
not enter PlanCost by themselves.

Direct E2B/BRC `group_broadcast_load` and physical-noop bitcasts are zero. An
E2B-capable group broadcast load must avoid the low-performance fallback through
accurate support legality, not through an artificial cost penalty.

### 6.4 Cost conformance

For every support relation partition, tests must establish exactly one of:

```text
ExplicitEdge
ZeroHidden
Hidden(named family, scalar cost)
```

Every non-zero family must have a complete-backend fixture comparing the modeled
rearrangement count with emitted rearrangement instructions. Zero fixtures must
confirm that no relation-induced rearrangement is emitted. Operand facts used by
the lowering predicate belong to the concrete operation query, not to a global
layout key.

## 7. Problem Construction and Propagation

### 7.1 Problem construction

Problem construction is read-only and deterministic:

```text
1. Walk verified IR in stable preorder.
2. Create A(v) for every layout-bearing SSA value.
   Create a corresponding signature-port variable for every layout-bearing
   enclosing-IR boundary port.
3. Create R(u), a use factor, and a stable port binding for every layout-bearing
   operand use.
4. Create one operation constraint for every VMI relation operation. Missing
   relation-query support is an implementation error. Recognized SCF, CFG,
   direct-call, return, and signature transport contributes use constraints and
   structural hard equalities instead of an operation constraint.
5. Create fixed factors for every preassigned internal SSA layout and every
   verified enclosing-IR layout.
6. Create all structural hard equalities and contract their closure.
7. Discover connected components over operation, use, fixed, and equality
   factor incidence.
8. Build each component's finite candidate layout pool and per-variable domains.
9. Freeze the immutable solver problem, initial variable domains, and stable
   variable/factor order.
```

Construction does not run a distinct propagation phase. The root search state
is initialized from the constructed domains and passed through the same
`propagateAndEvaluate` transition as every child state. An empty root domain is
therefore handled by the ordinary state transition and reported as `Infeasible`.

Function arguments precede their body, function results follow the body, and
operation ports use raw operand/result number order. Pointer identity and hash
iteration never determine stable order.

Construction has deterministic cumulative limits for variables, factors,
support-fact visits, active layouts, domain entries, and immutable problem
bytes. Construction returns `FailureOr<VMILayoutProblem>`; malformed input,
missing relation-query support, and `ProblemResourceLimit` fail construction and
the pass before a solve status exists. They are not `Infeasible`.

```cpp
struct VMILayoutProblemBudget {
  uint64_t maxVariables;
  uint64_t maxFactors;
  uint64_t maxSupportFactVisits;
  uint64_t maxActiveLayouts;
  uint64_t maxDomainEntries;
  uint64_t maxProblemBytes;
};
```

`maxSupportFactVisits` covers only domain-independent facts visited while
constructing pools. Search-time relation completions are independently bounded
by `VMILayoutEngineBudget::maxRelationFactVisits`; the same fact is charged once
in each phase if both phases examine it. Each construction counter is checked
before the visitor callback work or container growth that it guards.

### 7.2 Hard propagation

Propagation is generalized arc consistency over current finite domains:

```text
worklist = factors incident to changed domains

while worklist is not empty:
  factor = pop in stable order
  remove each port layout that has no legal completion in factor
  if a domain becomes empty:
    reject this state
  enqueue factors incident to every reduced domain
```

For an operation factor, support relation queries determine legal completions.
For a use factor, identity and registered conversion rows determine legal pairs.
Fixed and equality factors have their direct meanings.

Propagation does not assign a layout merely because it is preferred, cheaper,
or encountered first. A singleton domain is committed in the search state only
when every other value has been proved illegal under the current decisions.
This property prevents consumer-backward propagation from prematurely fixing
the ComputeY1 corridor while still allowing an E2B producer constraint to force
its actual result layout.

Propagation is a common pruning subroutine used by both engines. It is not an IR
rewriter and does not call the priority assignment seed phases.

## 8. Shared Search Model

### 8.1 State and transition

A search state contains:

```text
current finite domain of every live canonical variable, including singletons
immutable reconstruction records for forgotten singleton variables and factors
per-live-factor evaluation cache derived from the current domains
exact settled cost for factors already forgotten by Frontier DP
admissible total lower bound and lower-bound materialization count
settled entries of the globally indexed preference/row/layout ordering traces
```

The root state and every child use the same transition:

```text
propagateAndEvaluate(state):
  propagate state to a fixed point
  if any domain is empty:
    reject

  reevaluate every factor incident to a changed domain
  if any factor has no legal completion:
    reject
  recompute the state's admissible lower bound from factor evaluations

  if every live variable is singleton and every forgotten variable is recorded:
    reconstruct and independently evaluate the complete plan
    return the validated complete plan

choose the first unresolved variable in the shared stable elimination order
for each layout in its current domain, ordered by:
  admissible cost estimate, preference metadata, stable layout ordinal
    child = state with variable fixed to layout
    propagateAndEvaluate(child)
    emit child if legal and not pruned by its lower bound
```

Branching on one canonical variable rather than pre-expanded operation tuples is
what makes parameterized and table-backed support relations uniform. Atomic
support-row coupling is preserved because propagation/validation rejects every
port combination that is not one complete legal relation witness.

Problem construction computes one stable elimination order from graph structure
using minimum fill, then highest incident-factor count, then stable variable
ordinal. It does not use search-state domains. Both engines use this same fixed
order and skip variables already reduced to singletons. This is a search
heuristic only; changing it cannot change an exhaustive result.

### 8.2 Factor evaluation and complete-plan cost

Factor cost is a pure function of the factor and the current domains:

```cpp
struct VMILayoutFactorEvaluation {
  VMILayoutCost lowerBound;
  std::optional<VMILayoutCost> exactCost;
  uint64_t materializationLowerBound;
  std::optional<uint64_t> exactMaterializationCount;
  std::optional<VMILayoutFactorWitness> exactWitness;
};
```

`VMILayoutFactorWitness` is the tagged operation-relation or use-pair witness;
fixed and equality factors need no payload. `lowerBound` is the minimum cost
among the factor's legal completions inside the current domains. Version one may
conservatively return zero for an unresolved factor. `exactCost` is present only
when every retained completion has the same cost; `exactWitness` additionally
requires one canonical retained completion. A factor with no legal completion
makes the state infeasible.

Materialization bounds follow the same rule. A non-identity use completion
counts as one explicit materialization even when its registered implementation
contains several rearrangement instructions. Operation, fixed, equality, and
identity-use factors contribute zero explicit materializations.

The partial-state lower bound is the checked sum of every live factor lower
bound plus exact cost already settled by Frontier DP. Factor scopes may overlap:
for every complete plan each factor cost is at least its own local minimum, so
the sum remains admissible. Overlap can make the bound weaker but cannot make it
unsafe.

Incremental recomputation is only a cache optimization. After propagation, only
factors incident to changed domains need reevaluation, but their cached values
must be reproducible from the domains and must not participate independently in
continuation identity.

A complete plan never trusts this cache. It is reconstructed and evaluated from
scratch in stable factor order:

```text
operation factor -> validate its unique canonical witness, then getRelationCost
use factor        -> validate identity or registered conversion, then
                     getConversionCost when non-identity
fixed/equality    -> validate and add zero
```

Checked summation produces `PlanCost` and the explicit materialization count.
The complete plan is rejected if any factor is not exact or disagrees with the
reconstructed assignment. This full evaluation is the authoritative cost used
for incumbent comparison; cached lower bounds are used only for ordering and
sound pruning.

### 8.3 Result ordering

Complete plans compare lexicographically by:

```text
PlanCost
explicit materialization count
normalized preference-rank trace
stable support/generated relation ordinal trace
stable layout trace
```

Preference never changes legality or PlanCost.

The stable layout ordinal is a structural attribute order: contiguous first,
then the remaining layout kinds in dialect enum order, followed lexicographically
by factor, block elements, slots, and lane stride. It never depends on attribute
storage addresses or discovery order. This definition makes canonical
contiguous the best representative used by the symmetry argument in Section 5.

Each trace entry has a stable global index fixed by problem construction. A
partial state stores known entries at those indices; it does not append values
in discovery order. Two states with the same continuation have identical
unsettled trace positions, so dominance may compare their settled sparse traces
at the first differing settled index. Calling this data a `prefix` is incorrect
when an earlier-indexed relation is still unresolved.

### 8.4 Acceptance-case execution

The E2B f32 component is processed as follows:

```text
support propagation forces A(group_broadcast_load.result) = d4
the first use factor retains every supported pair, including (d4,d4) and
  (d4,c); it does not equate A(value) with R(use)
downstream operation factors propagate their legal coupled port choices
search branches on the first unresolved corridor variable

branch R(first use) = c:
  the use factor becomes exact at d4 -> c with cost 4
  downstream relations settle on c/ls4/c and total cost becomes 4

branch R(first use) = d4:
  the use factor becomes exact identity with cost 0
  downstream d4 relations expose dense-load, narrowing, and store hidden costs
  and total cost becomes 9
```

The ComputeY1 component starts the same way with the producer fixed to `d2`:

```text
the widening-corridor branch keeps the producer use at d2, propagates the legal
  d2 -> d4 widening relations, and evaluates the final narrowing hidden cost 3

the lane-stride branch selects R(first use) = ls2, evaluates the registered
  composed d2 -> ls2 use conversion cost 5, and leaves later relations at zero
```

Thus neither case is decided by producer priority, consumer priority, or an
ambiguous `group_broadcast_load` query. The same graph and transition compare
the different legal conversion cuts and select opposite placements from their
costs.

## 9. Search Engines

### 9.1 Common result and budget

```cpp
enum class VMILayoutSolveStatus {
  Exact,
  BestEffort,
  NoCompletePlan,
  Infeasible,
};

enum class VMILayoutSolverKind { FrontierDP, DFSBranchAndBound };

struct VMILayoutEngineBudget {
  uint64_t maxCandidateAttempts;
  uint64_t maxTransitions;
  uint64_t maxRelationFactVisits;
  uint64_t maxUniqueStates;
  uint64_t maxRetainedEntries;
  uint64_t maxLiveDomainEntriesPerState;
  uint64_t maxRetainedBytes;
};
```

Every attempted layout branch counts as one candidate attempt. Every legal child
after propagation counts as one transition. Every finite support fact or
parameterized-relation completion examined during search propagation,
enumeration, or complete-leaf evaluation counts as one relation-fact visit.
Counters are checked before the work or container growth they guard. A bound
returns the best complete incumbent as `BestEffort`, or `NoCompletePlan` if no
complete plan has been reached. Independent pre-mutation validation is not
charged to the search budget; it performs one total-assignment query per factor.

Components are solved in stable order and retained without mutating IR. Module
status is `Infeasible` if any component is proven infeasible, otherwise
`NoCompletePlan` if any component has no incumbent at its bound, otherwise
`BestEffort` if any component lacks an optimality proof, otherwise `Exact`.
`Exact` and `BestEffort` both contain a complete legal module plan and proceed to
independent validation and application. `NoCompletePlan` and `Infeasible` fail
cost-solver mode without mutating IR.

### 9.2 DFS branch-and-bound

DFS applies the shared transition recursively. The first complete leaf becomes
the incumbent. A state is pruned when its cost lower bound is greater than the
incumbent cost, or when cost is equal and its explicit-materialization lower
bound is greater than the incumbent count. If both are equal, search continues;
version one does not prune from an estimated preference/row/layout suffix.

DFS does not forget variables or factors. Its optional memoization key is the
ordered domain of every canonical variable after fixed-point propagation. A
cache hit therefore denotes the same complete residual CSP, not merely the same
next decision position. The cache is only a performance optimization; version
one may omit it. If present, states with an identical key have identical factor
lower bounds and assignment traces, so a duplicate can be discarded. It must
not use a smaller key or compare states by a heuristic lower-bound estimate.

### 9.3 Frontier DP

Frontier DP applies the same transition layer by layer using the same stable
decision order. Its continuation key is:

```text
next stable decision position
ordered domains, including singleton domains, of every live variable
```

The key also identifies the layer, so states at different decision positions
are never merged. Factor evaluations are pure functions of the keyed domains.
States with the same key have identical legal and cost-relevant futures; only
the state with the lexicographically best settled cost, settled materialization
count, and globally indexed settled ordering traces is retained.

A factor may be forgotten only after all variables in its scope are singleton.
Its unique canonical witness, exact cost, materialization count, and indexed
ordering entries are copied to the immutable reconstruction record and settled
totals. A singleton variable may be forgotten only after every incident factor
has been forgotten; its selected layout is first copied to the reconstruction
record. This operational rule is the meaning of `last incidence`; the
implementation must not remove a variable merely because its position in the
decision order has passed. Forgotten factors no longer participate in live
propagation or in the live-factor lower-bound sum. These records do not belong
to the continuation key because forgotten factors have no future incidence;
they participate only in dominance and final plan reconstruction. Frontier
exhaustion at the final layer is `Exact`; an exhaustively empty frontier is
`Infeasible`.

The compiler user selects `frontier` or `dfs`. Comparing them requires separate
compiler invocations over byte-identical input and identical non-engine options;
there is no in-pass portfolio or winner selection. Exact results from both must
have the same complete ordering key.

## 10. Exact Validation and Plan Application

Search produces an explicit plan containing:

```text
A(v) for every layout-bearing SSA value
R(u) for every layout-bearing operand use
one complete relation witness for every operation factor
every fixed and hard-equality check
```

The component plans are merged in stable component order. Before mutation,
`VMILayoutPlanValidator` independently validates the resulting total assignment
through the same stateless relation provider and conversion support used during
search:

```text
every A(v) and R(u) is present and concrete
every operation tuple has one canonical legal relation witness
every normal mismatched use has one registered conversion
every structural use and hard equality is identity
every fixed boundary matches
fresh complete-plan cost equals the solver result
```

Validation produces an immutable `VMILayoutApplyRecipe` containing every value
type rewrite, every enclosing function input/result type update, and every
required data/mask ensure insertion and operand reconnection. Identity uses need
no mutation record: after their producer type is rewritten they remain connected
to that producer. Recipe construction resolves and validates all operation,
block, symbol, port, target-type, insertion-point, and conversion references and
may fail without changing IR.

`VMILayoutPlanApplier` consumes only a validated recipe. In stable order it:

```text
1. rewrites every OpResult and BlockArgument to its A(v) type;
2. rebuilds each affected func.func FunctionType from the rewritten entry block
   arguments and validated function-result-port types;
3. for each normal mismatched use, inserts the exact registered ensure_layout or
   ensure_mask_layout immediately before the owning operation and reconnects
   only that OpOperand;
4. verifies the assigned-layout IR contract.
```

Structural and identity uses are not reconnected through newly selected values;
their hard-equal endpoint types already agree. The applier never uses a global
`replaceAllUses`, which could move a conversion across independently planned use
cuts. It does not call `VMILayoutPropagator::request`, does not run propagation,
and cannot select a different layout or conversion position. After recipe
construction, steps 1-3 contain no recoverable legality failure; an internal
recipe/IR disagreement is an assertion-level implementation error. A failure of
the final verifier is an implementation bug and fails the pass. Ordinary
construction, solver, validation, and budget failures occur before any mutation,
which is the mutation guarantee required by C11; C11 does not promise rollback
after an internal applier bug.

The existing propagator's pure type-rewrite and ensure-insertion helpers should
be extracted for use by the applier. The priority assignment may continue to use
`VMILayoutPropagator` for decisions and then build the same apply recipe. The cost
solver never submits its completed plan back to the propagator.

## 11. Construction and Conformance Coverage

Here coverage means completeness of problem construction and conformance
fixtures, not source-code line coverage.

Runtime construction must account for:

```text
every layout-bearing operation port and operation factor
every layout-bearing operand use and use factor
every explicit internal/boundary fixed layout
every structural hard equality
every active layout contributed by applicable support/conversion facts
every selected operation relation cost class
every selected mismatched use conversion
```

Tests must cover each support-table row, rule-backed relation partition,
parameterized same/free schema, fixed-boundary extraction form, use-factor form,
hard-equality form, conversion row, and conversion-relevant lowering branch.
Parameterized rules are partitioned by finite behavior classes; tests do not
enumerate every concrete element count or arbitrary attribute value.

Required solver tests include:

```text
both acceptance corridors and strictly unequal multi-use placement
an all-polymorphic component canonicalized without search explosion
polymorphic chains connected to one and to two constrained endpoints
cross-type same-layout chains where a candidate is introduced on only one type
interleaved non-layout MLIR operands with stable port bindings
SCF, CFG, direct call, return, and boundary equality closure
propagation that removes impossible values but preserves multiple legal costs
DFS and Frontier agreement with exhaustive enumeration on small graphs
Frontier merge where singleton live domains differ, proving they are keyed
Frontier forgetting only after all incident factors are settled
equal-primary-cost Frontier histories distinguished by sparse global tie traces
every status and every deterministic bound
complete-plan validation rejection and exact plan application
multi-result ops, block arguments, multi-return functions, and direct-call
signature updates in exact plan application
one producer with identity and mismatched uses, proving only the planned use is
reconnected through an ensure
type/layout applicability agreement with VReg and mask type verifiers
one-ended conversion closure and singleton conversion-query conformance
fact-visit exhaustion before processing the fact that crosses each bound
```

Current relation-conformance ledger is intentionally explicit. The test-only
pass checks every relation returned for a marked operation, clones the
containing function once per relation, materializes that relation's operand
layouts, and compares each independently lowered instance with its predicted
rearrangement count. Marked fixtures therefore use concrete result layouts;
alternate operand relations are no longer cost-only evidence.

The materializer intentionally consumes IR whose relation result layouts are
concrete. A polymorphic result would require coordinated rewriting of the
function signature and return sites; that is a separate type-rewriting
operation and must not be reimplemented as a conformance-only special case.
Fixtures that need assignment before relation checking keep that pass ordering
explicit, while conformance remains responsible only for relation
materialization and lowering comparison.

| Relation family | Fixture | Cost/lowering status |
| --- | --- | --- |
| ensure data layout | `vmi_layout_cost_conformance_ensure_layout.pto`, `vmi_to_vpto_ensure_layout_deint4.pto` | all 26 constructible registered rows have distinct fixtures and exact cost/lowering conformance; six historical 64-bit rows were removed because `VMIVRegType` verifier only permits 8/16/32-bit logical elements. N=1 contiguous/group-slot identity is supplied by `VMIEnsureLayoutFact::forwardsPhysicalParts`; dense lane-stride recipes account for partial physical groups. |
| ensure mask layout | `vmi_layout_cost_conformance_ensure_mask_layout.pto`, `vmi_to_vpto_ensure_mask_layout.pto` | all 18 constructible registered rows have distinct fixtures and exact cost/lowering conformance. Six rows that exceeded predicate carrier capability (`b32` lane-stride 2/4 and `b16` lane-stride 4) were removed from Support; lane_stride remains a physical legality constraint, not a preference. Block-deinterleaved forwarding is exposed once by `VMIEnsureMaskLayoutFact` and consumed by both cost and lowering. |
| ensure mask granularity | `vmi_to_vpto_ensure_mask_granularity_direct.pto`, `...multistep.pto` | covered, exact |
| mask granularity casts | `vmi_layout_cost_conformance_mask_granularity.pto` | all 37 registered rows have distinct concrete fixtures and are cost/lowering-conformant; group-slot row-local widening uses one per-part pack/unpack and its cost is the physical arity |
| group-slot lane stride | `vmi_to_vpto_ensure_group_slot_layout.pto` | covered, exact |
| dense composed layout | `vmi_to_vpto_ensure_layout_dense_composed.pto` | covered, exact |
| channel split/merge | `vmi_layout_cost_conformance_channel.pto` | covered, exact; semantic split/merge is zero rearrangement cost |
| vintlv / vdintlv | `vmi_layout_cost_conformance_vintlv.pto` | all ten constructible registered rows (five per direction: contiguous, d2/d4 multi-chunk, and dense lane-stride 2/4) are instantiated and cost/lowering-conformant at zero additional rearrangement; direct semantic interleave instructions are excluded from the layout-rearrangement count |
| bitcast | `vmi_layout_cost_conformance_bitcast.pto` | all exposed equal-width relations costable; width-changing canonical relation exact |
| dense data casts | `vmi_layout_cost_conformance_cast.pto` | all eight cast op families covered across 2x/4x widening/narrowing, lane-stride, and same-width numeric relations; emitted partial merges are exact |
| group-slot data casts | `vmi_layout_cost_conformance_cast.pto`, `vmi_to_vpto_group_slot_integer_extension_matrix.pto` | every group-slot row is instantiated for integer widening/narrowing at `num_groups=2`; the four supported floating rows (`f32` to 16/8-bit, slots=1/8) are also cost/lowering-conformant; dense slots=2/4/8 widening records and verifies its per-part `vzunpack` cost, while direct slots=1 and packed lane-stride paths remain zero-rearrangement |
| cast operation-family boundary | `vmi_layout_gate_extf_group_slots_invalid.pto`, `vmi_layout_gate_numeric_group_slots_invalid.pto` | storage-width-compatible rows are filtered by the shared Support operation-family capability before planning, validation, and lowering; group-slot `extf`, `fptosi`, `fptoui`, and `sitofp` relations are rejected by individual negative gates because no corresponding VPTO recipes are registered |
| dense load/store | `vmi_layout_cost_conformance_load_store.pto` | all five dense load layouts and five distinct store layouts are instantiated, including all three exact preferred store shapes. Native contiguous/lane-stride/d2 memory recipes have zero extra rearrangement; d4 load/store staging is exact at 2/4. |
| masked load/store | `vmi_layout_cost_conformance_masked_load_store.pto` | the contiguous, d2/d4, lane-stride 2/4, and all three exact preferred masked-store shapes are instantiated. Costs are exact (`0/0`, `2/2`, `8/8`, lane-stride `0/0`); the sole masked-load row is also exact. |
| group/deinterleave memory | `vmi_layout_cost_conformance_group_memory.pto` | all eight group-load rows, all four group-slot memory layouts, and all three dense group-store rows are instantiated, alongside native deinterleave/interleave memory. Compact group-store staging is exact (`1/1`, `2/2`). |
| group broadcast/load | `vmi_layout_cost_conformance_group_broadcast.pto`, `...group_broadcast_op.pto` | all 11 group-broadcast-load rows and all 19 group-broadcast rows are instantiated and lowering-conformant at zero extra rearrangement. Direct availability is queried by `(shape, concrete result layout)`: a preferred E2B recipe for a shape cannot be reused for another layout, while the original E2B d4 assignment remains stable. |
| generated masks/constants | `vmi_layout_cost_conformance_generated.pto` | all four registered group-mask staging rows are instantiated (dynamic d2/d4/bd4 and constant bd4), alongside create/constant mask rows; costs and lowering rearrangement counts are exact. Block-d4 generation is direct in the target layout, while element-deinterleaved generation uses contiguous staging. |
| shuffle | `vmi_layout_cost_conformance_shuffle.pto` | forwarding and vselr shuffle paths are semantic/forwarding actions; cost is zero and lowering-conformant |
| vselr / histogram | `vmi_layout_cost_conformance_histogram_vselr.pto` | all six exact vselr shape rows (8-bit N=64/128/256, 16-bit N=64/128, 32-bit N=64) and both histogram op families are costable and lowering-conformant at zero additional rearrangement |
| group reduction partial slots | `vmi_layout_assignment_group_reduce_partial_slots8.pto` | d4 and block-d4 source relations are costable at zero; VCG/vadd and predicate/data split instructions remain semantic/layout realization, not hidden rearrangement cost |
| group reduction | `vmi_layout_cost_conformance_group_reduce.pto`, `...group_reduce_quarter.pto` | all ten registered block-shape rows and all six typed op families are instantiated and lowering-conformant at zero rearrangement cost; quarter-block lane-stride=4 is restricted to 8-bit elements because wider masks have no physical predicate granularity, and full d2/d4 rows share one factor-driven row-reduction lowering |
| plain reductions | `vmi_layout_cost_conformance_reduce.pto` | all six reduce op families, including multi-chunk contiguous add, are costable and lowering-conformant at zero layout-rearrangement cost |
| legacy grouped reductions | `vmi_layout_cost_conformance_legacy_reduce.pto`, `vmi_layout_cost_conformance_legacy_reduce_invalid.pto` | ungrouped `vcmax`/`vcmin` and grouped `vcadd` adapters reuse the shared contiguous/group-reduce facts and are lowering-conformant at zero extra rearrangement; partial physical chunks are rejected by both Support and lowering, so no second reduction cost path can advertise an invalid zero-cost relation |
| memory/compaction primitives | `vmi_layout_cost_conformance_memory_compaction.pto` | stride load/store, gather/scatter, compress, active-prefix, compress-store, and expand-load each have concrete contiguous relations accepted by the cost model and verified against lowering at zero extra rearrangement |
| producers and mask/data semantic ops | `vmi_layout_cost_conformance_producers.pto` | concrete constant, broadcast, iota, group-iota, pset/pge/plt, compare, select, and mask-binary relations are accepted and lowering-conformant at zero extra rearrangement. `group_iota` non-contiguous layouts are verifier-dead and are not counted as Support rows; unified pset/pge/plt are lowered through the existing unified-to-legacy paths before VPTO conversion. |
| generic same-layout elementwise ops | `vmi_layout_cost_conformance_elementwise.pto`, `vmi_layout_cost_conformance_producers.pto`, `vmi_layout_cost_conformance_unified.pto`, `vmi_layout_cost_conformance_same_layout_invalid.pto`, `vmi_layout_cost_conformance_vexpdif_invalid.pto`, `vmi_layout_cost_conformance_vmull_invalid.pto`, `vmi_layout_cost_conformance_unified_merge_invalid.pto`, `vmi_layout_cost_conformance_cmp_merge_invalid.pto`, `vmi_layout_cost_conformance_vsel_zero.pto` | shared binary, unary, vector-scalar, compare, select, mask, FMA, fused activation, carry, widening-multiply, and direct `vexpdif` schemas have concrete cost/lowering evidence at zero rearrangement. `pmode=merge` forms rejected by unified-to-legacy or direct lowering, plus `vmull` lane-stride forms rejected by lowering, are filtered by the shared candidate-aware Support capability; planner and cost model consume the same gate. Legacy integer arithmetic/bitwise variants include `subi/muli/negi/absi/ori/xori/shli/shrui/not`; scalar variants include `vmuls/vmaxs/vmins/vshls/vshrs`; unified unary `vln/vrelu` are covered with the same zero-mode relation path; mask-logic and zero-mode `vsel` paths are also concrete. Deinterleaved mask `mask_or/mask_xor/mask_not` are covered alongside `mask_and`. All currently registered same-layout op names have a concrete positive or negative fixture. |

| unified conversion phase boundary | `vmi_layout_cost_conformance_vcvt_phase_invalid.pto` | an uncanonicalized `vcvt` is rejected by relation conformance with an explicit phase diagnostic; after unified-to-legacy, the resulting canonical cast relation is covered by the dense/group-slot cast fixtures. This keeps `vcvt` semantics in one canonicalization owner and avoids a duplicate relation implementation. |

Other relation partitions remain audit targets unless the ledger explicitly
claims full table-row coverage. A business regression that happens to exercise
one row does not close one of these ledger entries; each entry needs a marked
relation fixture and, where the relation is concrete, a lowering instruction
count check.

For every concrete operation tuple, querying singleton port domains must return
the same canonical witness as `validate`. The current relation representation
does not yet carry schema IDs or row ordinals, so conformance compares every
identity-bearing field that is available: the complete port tuple,
`directProducer`, intrinsic rearrangement cost, and final relation cost. If
schema/row metadata is added later, it becomes part of this identity check.
Every finite support row and parameterized relation partition must be reachable
through the unified query without planner-side operation-family dispatch.

## 12. Implementation Mapping

```text
existing legality facts and rule queries  VMILayoutSupport
unified op relation adapter               VMILayoutRelationProvider
constraint graph, candidate pools/domains VMILayoutPlanner
hard domain filtering                     VMILayoutConstraintPropagation
conversion and hidden costs               VMILayoutCostModel
shared propagation/evaluation transition   VMILayoutConflictSolver
DFS engine                                VMILayoutDFSSolver
Frontier engine                           VMILayoutFrontierSolver
complete-plan validation                   VMILayoutPlanValidator
validated recipe application               VMILayoutPlanApplier
priority assignment decisions              VMILayoutPropagator
physical instruction emission              VMIToVPTO
```

The new planner, propagation, cost model, and engines live in independent files
under `lib/PTO/Transforms`. `vmi-layout-assignment` remains the pass entry and
selects priority, frontier, or DFS mode explicitly. Frontier and DFS share the
problem and transition implementation; neither owns support or cost rules.

## 13. Non-goals and Open Parameters

Version one does not model:

- semantic instruction count;
- native memory instruction-count differences;
- schedule, latency, overlap, or register pressure;
- loop-frequency weighting;
- conversion sharing across distinct SSA uses;
- arbitrary transitive conversion paths;
- external VMI function ABI layout synthesis;
- automatic fallback from cost mode to priority assignment.

Production values for construction/search budgets and the default selectable
engine remain measurement-driven parameters. They do not change legality, cost,
or the meaning of the four result statuses.
