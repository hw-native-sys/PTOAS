# Continuous short stores with automatic memory barriers

VMI short stores may require a stateful `vstus` stream and a terminal
`vstas` flush. Reinitializing and flushing that stream on every loop iteration
adds work and obscures the memory dependence analysis.

`PTOStoreStreamAnalysis` certifies one restricted case shared by stream fusion
and automatic membar analysis. The certificate is computed from IR on every
analysis run; no user annotation or trusted attribute bypasses the proof.

The proof depends on the VPTO operation semantics, not a chip-name check.
It applies whenever the IR contains the supported `vstus`/`vstas` lifecycle,
independently of `pto.target_arch`. The 32-byte transaction envelope follows
the block unit already used by `VPTOAddressSemantics`. Target instruction
availability and lowering remain the responsibility of the target backend;
this analysis does not enable an unsupported instruction on a target.
If a future target needs different operation semantics, its lowering or
semantic model must express that difference before this proof is reused.

The admitted loop has a fixed positive trip count, a nonwrapping induction
variable, one post-update `vstus`, and a matching zero-offset flush. Each step
advances by a positive constant no larger than its vector payload. Typed
address evolution must prove that the next logical store starts exactly at
the preceding step's end. The complete address arithmetic and full loop
footprint must be bounded. Other effects are rejected except ordinary vector
loads proven disjoint from the entire stream across all iterations.

The certificate expands the stream footprint conservatively by 31 bytes at
each end to include partial 32-byte transactions. It therefore rejects an
otherwise logically disjoint load that could share a physical transaction.
Unknown roots, arithmetic, operations, nested regions, or accesses retain the
conservative behavior. The current implementation handles one store per
iteration; it does not certify arbitrary multiple streams or dynamic loops.

Fusion makes the align and base loop-carried, creates one initialization
before the loop, and one flush immediately after it. Zero-trip and dynamic
loops are not fused. The automatic membar analysis revalidates the complete
lifecycle and uses its conservative footprint for both store and flush.
Within this certified stream, consecutive store-pipeline state transitions
and the terminal flush need no separate write/write fence. Independent
accesses still participate in ordinary dependence analysis. In particular,
the final flush-to-load RAW requires `VST_VLD`; SSA dependencies alone do not
establish memory visibility.

No new command-line option is needed. The normal automatic mode remains
`--enable-vecscope-mem-bar=true`. Unsupported cases keep their original
flush and conservative barriers. This optimization does not establish the
complete compressor VF2 ten-to-two barrier target without its full input.

The loop-fusion proof is now mandatory, replacing an older check based only
on pointer-root identity. This conservatively keeps per-iteration flushes in
loops containing other stores, including the fused quant/dequant regression.
WAR value tracing also stops at loop-carried arguments and treats each loop
result separately, including its zero-trip initializer path.

Compiler regression tests exercise the proof and conservative fallbacks with
no explicit target, A3, and A5. These pass-level checks establish target
independence of the analysis; they do not claim that every backend can emit
or execute the same VPTO operations. The DSL runtime case explicitly selects
A5 because that is the available CA simulator environment. Its successful
result is runtime evidence for that environment only.
