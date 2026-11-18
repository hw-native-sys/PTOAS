# Deduplicating existing slot event protocols

The normal automatic-sync pipeline runs `deduplicateSlotEventFamilies` after
sync motion and the last access-based redundancy proof, before event allocation.
It removes duplicate event channels without rebuilding a slot-release protocol.
The optimization is enabled by default; `--pto-deduplicate-slot-events=false`
retains the previous pipeline for comparison.
The CLI forwards this setting through the `pto-insert-sync` pass option
`deduplicate-slot-events`; the protocol utility has no global option state.

## Equivalence and scope

Two eligible families must have exactly the same SET and WAIT attachment nodes,
loop identity, resource/core and pipe direction, per-endpoint slot SSA selectors,
lane counts, and boundary-token policy. Absent masks are distinct from explicit
masks, even if their bit patterns happen to be all ones. Matching is exact SSA
identity, not an assumed equality of modulo expressions or allocation roots.

Each family is an active, uncompensated, unallocated two-operation EVENT pair.
Both endpoints must occur exactly once, SET in `pipeAfter` and WAIT in
`pipeBefore`, strictly inside the same loop and in back-edge order. Both halves
must consistently describe one closed allocation root, including every
`depMemInfos` entry. Families that have been widened, reallocated, or assigned
other special bookkeeping are not eligible.

The SET frontier must contain only active EVENT SET operations, and the WAIT
frontier only active EVENT WAIT operations. In particular, an interleaved
blocking WAIT or barrier prevents deduplication: equal node indices alone do
not establish equal ordering relative to other synchronization commands.

The surviving family keeps its exact endpoints, slot selectors, and boundary
policy. All dependency roots and memory records are unioned into both surviving
halves. The duplicate SET and WAIT are removed as a whole and tombstoned before
the allocator sees them. Their original dependency-source indices need not
match: those identify the original reads, not the shared synchronization
frontier. The representative retains one original source index; no subsequent
access-pruning proof is allowed to interpret that single index as all sources.

## Why reader opcodes are unnecessary

Corresponding endpoints execute at the same dynamic positions and evaluate the
same per-side selector. Their per-lane SET/WAIT streams, including initial and
final tokens, are copies. Removing a duplicate channel preserves the existing
ordering obligations. This transformation inherits the original analysis's
coverage; it does not claim that matching roots, matching remainders, or merely
equal SET positions prove that a slot has been fully released.

No operation is moved, no new release frontier is selected, and no boundary mask
is introduced. Consequently there is no reader-opcode whitelist, architecture
check, slot-count-of-two check, or constant-loop-bound requirement in this
utility. This is not a proof for merging different frontiers, unrelated slot
phases, different roots, or arbitrary unknown memory effects. Those require a
separate access and control-flow proof.

The original #1727 input has four equivalent two-lane channels per operand after
sync motion. Exact dedup leaves one existing channel per operand: four IDs for
`MTE1 -> MTE2`, rather than sixteen requested IDs. Its computational IR and all
other pipeline dependencies are retained.

## Boundary and resource handling

Before removing any family, scan all stored pairs in the function. A live
loop-carried slot EVENT pair with different SET and WAIT selectors must have
explicit boundary masks on both endpoints; otherwise skip the entire function.
This scan is independent of dedup eligibility, allocation state, root, or loop
identity. Dead pairs and non-loop-carried pairs do not trigger this guard.

Legacy unmasked boundaries prime every allocated lane. In a shifted-prefetch
loop, slot 0 can already contain the preloaded input: its first overwrite must
wait for the first reader, rather than consume an entrance token. Even dedup of
an unrelated root can free IDs for this unproved rotation, replacing its safe
full-barrier fallback with unsafe priming. Exact equivalence of the removed
families does not prove that the resulting allocation of other families is safe.
The guard preserves the previous allocation input without deriving new masks;
supporting these rotations requires a separate boundary-token proof.

Unlike MR403's access-driven FREE-family construction, exact dedup does not
change legacy function-boundary prime/drain policy into masked loop-boundary
policy. The existing allocator may still move boundaries during its own
reallocation or spill to barriers when the event pool is exhausted. That
inherited behavior is not a lane-preservation guarantee made by this utility.
Resource-pressure tests therefore need to inspect the actual allocated
protocol, including fallback, rather than just a reduced family count.

## Regression coverage

The synthetic SyncIR test exercises the metadata that cannot all be expressed
independently in source IR: distinct endpoints, loop owners, per-side selectors,
lane counts, mask policies, resources/cores, roots, inconsistent pairs,
compensation, allocation/widening state, attachment multiplicity, and blocking
commands at a shared frontier. Positive cases check complete metadata union,
unchanged lifetime policy, and idempotence.
Guard cases include an unrelated unmasked rotation appended after the duplicate
candidates in another root and loop, an allocated rotation, and a missing mask
on either endpoint. Masked rotations, dead pairs, and forward pairs remain
compatible with dedup of unrelated eligible families.

The shifted-prefetch fixture preloads one ring and includes duplicate release
families for a second ring. For two, three, and eight iterations, explicit and
default dedup must leave all five families intact and emit identical C++ to the
disabled control. The ID, interference, and self-coverage gates run, and the
command-graph oracle checks Mat RAW/WAR ordering and token balance with the
required loop barrier fallback.

The pipeline fixtures use the original issue IR and legal full-shape Mat-to-L0
`tmov` reads. The latter is a protocol fixture, not a numerically equivalent
replacement for the issue's subtile extraction. The independent command-graph
oracle is derived from MR403's `check_mat_slot_protocol.py` at
`de5f1eb657871d6096dd8c1d72867a7301ee7012`, with Mat `tmov` read recognition.
It checks allocated IDs, Mat RAW/WAR happens-before, token reuse and prime/drain
balance, and has negative protocol mutants. The ID, interference, and sync
self-coverage gates also run on the pipeline cases, and the computational opcode
sequence must match the disabled control. Compilation and this fixture model
do not substitute for numerical device execution or hardware latency profiling.
