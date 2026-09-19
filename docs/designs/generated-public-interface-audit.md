# Generated public interface audit

This audit addresses [PTOAS #1546](https://github.com/hw-native-sys/PTOAS/issues/1546)
against the split EmitC implementation in PTOAS 0.63. Generated C++ includes
`pto/pto-inst.hpp` and must use the public interfaces of its selected backend.
CCE-private entry points belong inside the compiler or backend implementation.

## Source and output coverage

The inventory covers `lib/`, `include/`, `tools/`, `ptodsl/`, `python/` and
`scripts/`. In addition to builtin spellings, the review follows opaque-call
builders, callee string concatenation, verbatim snippets, template expansion,
and final C++ postprocessing. The generated-source regression examines whole
translation units, including preambles, rather than just kernel bodies.

| Surface | Interface selection and disposition | Generated evidence |
| --- | --- | --- |
| EmitC arithmetic, tensor compute, reductions and conversion | `PTOToEmitC/{Arith,Tensor,Reduce}` select C++ scalar expressions, the general compiler builtins below, and public tile operations such as `TADD`, `TMAXS`, `TMOV`, `TMATMUL` and `TMATMUL_MX`. No additional CCE-private callee was found. | Scalar remainder/bitcast, scalar tile arithmetic, min/max, matmul, MX and quantization outputs |
| EmitC memory, tile allocation and views | `PTOToEmitC/{LoadStore,Memref,Tile}` and `PTOToEmitCCommon.cpp` select public tensor/tile interfaces, pointer expressions and macros. Concatenated `TALLOC`/`TFREE` template callees retain their public names. | Prefetch, load/store within tile and matmul cases, tensor metadata and pipe outputs |
| EmitC synchronization and communication | `PTOToEmitC/SyncComm` selects public `set_flag`, `wait_flag`, `pipe_barrier`, `dsb`, `dcci`, `SYNCALL`, pipe operations and `pto::comm::*`; repaired FFTS and intra-block calls are listed below. | Direct/named synchronization, event arrays, pipes, point-to-point and collective communication |
| Generated helpers | `PTOToEmitCPass.cpp` emits tensor-data access, event arrays, `TRANDOM`, auto-sync tails, cache maintenance and bitcast bodies. `Section.cpp` and function lowering emit public mask setup and architecture guards. | Random, event-array, bitcast and both cube/vector translation units |
| Driver rewrites and last-use markers | `tools/ptoas/ptoas_pipeline.cpp`, `ptoas_rewrite*.cpp` and `Lowering/CppPostprocess.cpp` rewrite pointer/tile accesses, metadata, event arrays, names and `[[pto::last_use]]`. The marker carries the original public tile callee; it does not construct a compiler builtin prefix. | Matrix checks run after `translateToCpp` and every final rewrite |
| PTODSL/TileLib and soft-library expansion | `ExpandTileOp.cpp` and `Lowering/PTOExpandSoftLib.cpp` create IR and unique helper symbols; they use the selected backend for final emission. | Existing frontend/expansion and backend regression suites |
| VPTO textual IR, LLVM IR and objects | `VPTOCANN900LLVMEmitter*`, `VPTO/*` and VMI lowering select `llvm.hivm.*` or LLVM intrinsics for the target compiler. These are compiler-internal interfaces, not C++ ISA-consumer calls; retain them. | Existing VPTO regression suite |
| VPTO host stubs and runtime wrappers | `VPTOHostStubEmission.cpp` emits entry signatures with empty bodies for host registration, not device synchronization implementations. `ObjectEmission.cpp` supplies CANN's compiler wrapper header; it does not emit CCE-private calls into ISA-facing source. | Stub source review and existing object/stub regressions |

EmitC supports A2, A3 and A5 and build levels 1, 2 and 3. Its repaired interface
selection depends on architecture, not on the discovered CANN version. CANN output
versions select the VPTO compiler ABI and object-emission details; those paths
must retain their target intrinsics. User-supplied external function names and
explicit opaque/verbatim input are not compiler-selected backend interfaces.

## Candidate inventory

Paths in the first five rows are relative to `lib/PTO/Transforms/PTOToEmitC/`.

| Origin | Exposure and coverage | Previous interface / arguments | Public contract and adaptation | Disposition and validation |
| --- | --- | --- | --- | --- |
| `Arith/ArithSupport.cpp`, `buildInterCoreSyncSetCallImpl`, reached from static and dynamic set helpers | Generated C++, all targets/levels; direct FFTS and named cross-block, plus A2/A3 named intra-block | `__builtin_cce_ffts_cross_core_sync(pipe, message)` | `ffts_cross_core_sync(pipe, message)`; preserve `getFFTSMsg`, `uint16_t`, mode, default base count and event encoding. A5 emits the numeric mode because its ISA header does not define `FFTS_MODE_VAL`. | Fixed; static/dynamic codegen and public-header consumer compilation |
| Same file, `buildInterCoreSyncWaitCall` | Generated C++, static event IDs, same target/operation coverage | `__builtin_cce_wait_flag_dev(event)` | A2/A3: `wait_flag_dev(event)`; A5: `wait_flag_dev(pipe, event)` using the IR pipe | Fixed; A5 requires the two-argument public overload, confirmed with CANN compilation |
| Same file, `buildInterCoreSyncWaitCallDyn` | Generated C++, dynamic index/integer events, same coverage | `__builtin_cce_wait_flag_dev(int32_event)` | Same target-specific overloads; preserve the existing explicit `int32_t` conversion | Fixed; dynamic operand/cast checks and consumer compilation |
| `SyncComm/Sync.cpp`, `PTONamedIntraSyncToEmitC::rewriteIntraBlock`, set branch | Generated C++, A5, static/dynamic integer event IDs | `__builtin_cce_set_intra_block(pipe, event)` | `set_intra_block(pipe, event)`; preserve pipe, event and dynamic conversion | Fixed; named static/dynamic A5 tests and CANN compile probes |
| Same function, wait branch | Generated C++, A5, static/dynamic integer event IDs | `__builtin_cce_wait_intra_block(pipe, event)` | `wait_intra_block(pipe, event)` with unchanged arguments | Fixed; named static/dynamic A5 tests and CANN compile probes |
| `Arith/ArithMisc.cpp`, `ArithRemFToEmitC` | Generated scalar code, all targets/levels; f16/f32/f64 | `__builtin_fmodf(lhs, rhs)` / `__builtin_fmod(lhs, rhs)` | General GCC/Clang builtins; f16 computes in float and casts back | Retained; not CCE-private. Scalar codegen and CANN f32 compile probe cover the distinction; do not replace with an unverified ISA spelling. |
| `PTOToEmitCPass.cpp`, bitcast helper | Generated optional helper, all targets/levels | `__builtin_memcpy(&to, &from, sizeof(To))` with equal-size assertion | General GCC/Clang memory-copy builtin for bit-preserving conversion | Retained; helper-output checks and CANN compile probe |
| `lib/PTO/Analysis/PTOValueEvolutionAnalysis.cpp`, checked arithmetic | Assembler host implementation only | `__builtin_add_overflow`, `__builtin_sub_overflow`, `__builtin_mul_overflow` | Host compiler checked-arithmetic contract | Retained; not emitted into consumer source; native build |
| `lib/PTO/IR/PTO.cpp` and `VPTOCANN900LLVMEmitterScalarPatterns.cpp`, block-index comments | Documentation of signed hardware query width inside compiler implementation | Mentions `__builtin_cce_get_block_idx`; does not emit that C++ call | Public EmitC block query / internal LLVM lowering | Retained; existing block-query tests |
| `include/PTO/IR/PTOOps.td`, intra-block descriptions | Operation documentation | Described private set/wait builtins | Describe semantics and supported public interfaces | Updated with the implementation |
| `include/PTO/IR/VPTOUbOps.td`, `ObjectEmission.cpp`, and `docs/designs/a2a3-*-builtins.md` | Compiler header references and low-level design inventories | CCE header filenames, runtime wrapper include and builtin/intrinsic mapping references | Compiler-owned backend contracts | Retained; these do not require ISA consumers to implement private names. |

The current split implementation has five private callee assignments because
static and dynamic FFTS set share one helper. Both forms are covered. Source
and generated-artifact inspection found no additional compiler-selected
CCE-private C++ calls in the operation families above.

## Public header requirements

- CANN 9.0.0-beta.1 / CCE Clang 15.0.5 supplies public FFTS and intra-block
  aliases in `cce_aicore_intrinsics.h` / `cce_aicore_intrinsics_3101.h`.
  A2/A3 accepts the one-argument FFTS wait. A5 requires its pipe and event
  overload; both the old private spelling and the public spelling fail on A5
  with only one argument. No event remapping or replacement with intra-block
  semantics is performed.
- The pinned A5 ISA header supplies `getFFTSMsg` but not the A2/A3
  `FFTS_MODE_VAL` macro. A5 passes the original mode numerically (including 2),
  preserving its encoding. A2/A3 retains its previous macro spelling. The
  declaration-only A5 fixture also omits the macro so this dependency cannot
  be hidden by the test header.
- The issue's pinned ISA revision
  `3b4faf67aebb3e0d41be7952c56908b3adba7a8f` exposes the A2/A3 CPU public
  `ffts_cross_core_sync(int, uint16_t)` and `wait_flag_dev(int)` wrappers in
  `include/pto/cpu/ffts.hpp`. Its supported protocol is mode 2, base count 1,
  events 0–15, one AIC and two AIV lanes. This fix does not expand CPU protocol
  support to mode 0/1 or A5 intra-block synchronization.
- No downstream pin update is required to use those existing public interfaces.
  Older ISA CPU headers without the public wrappers need an ISA update before
  consuming this output. Keep ISA's private-name compatibility wrappers for
  older generated artifacts; removing them is a separate downstream change.

## Regression and validation procedure

`test/lit/pto/public_sync_interfaces.pto` checks direct and named static/dynamic
events on all three targets, including message construction, wait conversion,
pipe arguments and ordering. Existing mode and control-flow tests also reject
private names throughout their output.

`test/lit/pto/public_generated_interface_audit.pto` runs 18 synchronization
translations (three architectures × three levels × two roles), compiles them
against declaration-only public headers, and scans 18 additional translations.
The fixture deliberately has no private-name declarations and supplies only
the correct target-specific wait overload. It is a compilation test, not a
simulator or replacement synchronization implementation.

For runtime evidence, use the pinned ISA `ffts` tests with matching AIC/AIV
participants; preserve their broadcast, two-lane join, repeated-event, device/
group isolation and data-visibility assertions. Never execute the standalone
codegen probes, which do not define a matched synchronization protocol.

Full dspark numerical acceptance additionally requires the pinned PyPTO,
simpler, compiler and frozen model inputs from the issue. Record model results
separately from these interface and synchronization checks; neither an
interface rename nor a passing compiler test establishes a fix for dspark's
reported numerical failure or timeout.

## Local validation results

Validation used Linux x86_64, Python 3.12.3 and the issue's pinned ISA, simpler,
PyPTO and pypto-lib revisions. These host details differ from the reported
Linux aarch64 / Python 3.10.19 environment. Supplementary model CPU compilation
used an isolated GCC 15.3.0 toolchain, rather than the reported GCC 15.2.1.
Logs and generated artifacts are retained under `build/issue1546/`.

| Check | Result / evidence |
| --- | --- |
| Exact minimal reproduction | Four public FFTS calls, zero private calls. A3 output is byte-for-byte identical to the previous output after substituting just the two callee names. `before.cpp`, `after.cpp` |
| Generated-interface matrix | All 18 synchronization translations compile with public declarations only; all 18 other generated translation units reject private CCE identifiers. Included in the lit suite. The same header rejects the old minimal reproduction. `public-header-negative-control.log` |
| Full lit suite | 1931 passed, one unsupported, zero failures. `full-lit-without-cann.log` |
| Full PTODSL suite | 45 passed, zero failures. `full-dsl-without-cann.log` |
| Real paired ISA CPU FFTS protocol | The generated dynamic signal/wait functions from `test/compile_cpp/public_ffts_runtime.pto` replace calls in the pinned ISA harness's three shared libraries. All 12 original tests pass after an explicit rebuild. Private FFTS identifiers are poisoned after including the ISA header and before the generated source. `isa-ffts-generated.log`, `isa-generated-adapter.diff` |
| Actual generated C++ with CANN and pinned ISA headers | A2/A3/A5, Cube/Vector, direct/named and static/dynamic events compile successfully; objects have no undefined synchronization functions. CANN 9.0.0-beta.1, CCE Clang 15.0.5. `generated-npu.log` |
| CANN alias equivalence probes | Public and private call variants, with identical target-specific arguments, produce identical instruction bytes on c220 Cube/Vector and c310 Cube/Vector. This includes dynamic FFTS and A5 intra-block events. These unpaired probes were not executed. `cce-equivalence.log` |
| General compiler builtins | f16/f32/f64 remainder and bitcast generated-source checks pass; complete scalar source compiles with CPU headers. CANN f32 remainder and bitcast probes compile on A3/A5 Vector. No CCE-private interface is introduced. |

The textual regression suites run without the shell's default CANN activation.
Activating the installed beta.1 makes 48 VPTO lit cases and two PTODSL cases
fail with the explicit requirement for CANN 9.0.0-beta.2 or newer official
lowering. The successful textual runs use the compiler's default ABI selection;
they do not establish device compilation on an installed beta.2 toolchain.
The six real CANN synchronization compilations above use beta.1 explicitly.

### Downstream integration remains open

The original pinned PyPTO/pypto-lib dspark regeneration was attempted without
changing model source or builtin spellings. It is incompatible with current
PTOAS 0.63: PyPTO emits `pto.load_scalar` / `pto.store_scalar`, while this
PTOAS revision recognizes `pto.load` / `pto.store`. The original diagnostics
are retained in `dspark-regenerate.log` and `dspark/report/codegen_errors.txt`.
This pre-existing IR-version mismatch must be coordinated before claiming
unchanged pinned end-to-end acceptance.

For supplementary interface evidence only, copies of the two failing PTO
files were migrated by changing those scalar operation syntax names alone.
`qk_pv` then generates both AIC and AIV functions with eight public FFTS calls
and zero private CCE calls. Both roles compile against pinned CPU ISA headers
with GCC 15. The plan kernel also generates without private calls.
`dspark-migrated-ir.log` and `dspark-cpu-{CUBE,VEC}.log` record this result;
it is not a passing original pinned model run.

The existing PyPTO A2A3 simulator syncall regressions were also attempted.
Three soft-sync cases report passing assertions, the hard-sync case reports
failure, and the test process segfaults during session cleanup. A second run
of just the soft cases also crashes during cleanup, so neither run is reported
as a clean integration pass. `pypto-syncall-sim.log` and
`pypto-syncall-soft-sim.log` retain the evidence. No claim is made that this
crash or the issue's dspark numerical failure/timeouts are caused or fixed by
the emitted-interface change. Model inputs and tolerances have not been
changed, and full numerical acceptance remains outstanding.
