// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptoas_internal.h"
#include "PTO/Transforms/VMILayoutPlanner.h"

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;
using namespace pto;

namespace {

/// LLVM 19's Func inliner interface accepts every call and callable, including
/// operations carrying the standard `no_inline` attribute. Keep the upstream
/// terminator handling while honoring the attribute used for PTO SIMT entry
/// functions.
struct PTOASFuncInlinerInterface final : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    (void)wouldBeCloned;
    return !call->hasAttr("no_inline") && !callable->hasAttr("no_inline");
  }

  bool isLegalToInline(Operation *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool,
                       IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, Block *newDest) const final {
    auto returnOp = dyn_cast<func::ReturnOp>(op);
    if (!returnOp) {
      return;
    }
    OpBuilder builder(op);
    builder.create<cf::BranchOp>(op->getLoc(), newDest,
                                 returnOp.getOperands());
    op->erase();
  }

  void handleTerminator(Operation *op, ValueRange valuesToRepl) const final {
    auto returnOp = cast<func::ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands())) {
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
    }
  }
};

static void registerPTOASFuncInlinerExtension(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, func::FuncDialect *dialect) {
    dialect->addInterfaces<PTOASFuncInlinerInterface>();
    ctx->getOrLoadDialect<cf::ControlFlowDialect>();
  });
}

} // namespace

int main(int argc, char **argv);

void mlir::pto::registerPTOASDialects(DialectRegistry &registry) {
  registerPTOASFuncInlinerExtension(registry);
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::math::MathDialect>();

  registry.insert<mlir::pto::PTODialect>();
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
}

void mlir::pto::registerPTOASPassesAndCLOptions() {
  mlir::registerConversionPasses();
  mlir::arith::registerArithPasses();
  mlir::func::registerFuncPasses();
  mlir::math::registerMathPasses();
  mlir::memref::registerMemRefPasses();
  mlir::registerSCFPasses();
  mlir::tensor::registerTensorPasses();
  mlir::registerTransformsPasses();

  mlir::pto::registerPTOPasses();
  mlir::pto::registerPTOInlineLibCall();
  mlir::pto::registerFoldTileBufIntrinsics();
  mlir::pto::registerLowerPTOToUBufOps();
  mlir::registerPassManagerCLOptions();
}

void mlir::pto::loadPTOASDialects(MLIRContext &context) {
  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}


llvm::cl::opt<VPTOSchedulerCLIMode> vptoSchedulerMode(
    "vpto-scheduler",
    llvm::cl::desc("VPTO scheduler mode"),
    llvm::cl::values(
        clEnumValN(VPTOSchedulerCLIMode::Off, "off", "Disable scheduling"),
        clEnumValN(VPTOSchedulerCLIMode::Analyze, "analyze",
                   "Report scheduling analysis without changing IR"),
        clEnumValN(VPTOSchedulerCLIMode::On, "on", "Run scheduler in on mode")),
    llvm::cl::init(VPTOSchedulerCLIMode::Off));

llvm::cl::opt<bool> vptoSchedulerTrace(
    "vpto-scheduler-trace",
    llvm::cl::desc("Print detailed VPTO on-mode scheduling results"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> vptoBridgePolicy(
    "vpto-bridge-policy",
    llvm::cl::desc("Path to a VPTO bridge route policy YAML file; selects "
                   "which families and operations use the C++ bridge. "
                   "Defaults to the built-in policy."),
    llvm::cl::init(""));

llvm::cl::opt<std::string> vptoPrintDTData(
    "vpto-print-dtdata",
    llvm::cl::desc("How a printing kernel receives its DebugTunnel pointer: "
                   "'implicit' takes the hidden argument the CCE launch "
                   "machinery fills (real NPU), 'explicit' takes the pointer "
                   "the host runner passes as the first kernel argument "
                   "(simulator)."),
    llvm::cl::init("implicit"));

llvm::cl::opt<bool> vptoSchedulerRemat(
    "vpto-scheduler-remat",
    llvm::cl::desc(
        "Enable bounded high-pressure VPTO rematerialization and rescheduling "
        "(defaults to on with the A5 scheduler; use =false to disable)"),
    llvm::cl::init(false));

llvm::cl::opt<bool> dumpSyncExtract("dump-sync-extract",
                                    llvm::cl::desc("Render the structured extraction of whatever sync the selected "
                                                   "mode emitted. Read-only"),
                                    llvm::cl::init(false));

llvm::cl::opt<bool> enableUnifiedSync("enable-unified-sync",
                                      llvm::cl::desc("Enable the unified intra-core sync allocator: event ids, buffer "
                                                     "ids and barriers as three resource classes over one "
                                                     "interval-colouring problem"),
                                      llvm::cl::init(false));

llvm::cl::opt<bool>
    enableUnifiedSyncDebug("enable-unified-sync-debug",
                           llvm::cl::desc("Print the unified allocator's model, routing, colouring and "
                                          "emission reports to stderr. Off by default; gate violations are "
                                          "reported either way"),
                           llvm::cl::init(false));

llvm::cl::opt<std::string>
    unifiedSyncForceMechanism("unified-sync-force-mechanism",
                              llvm::cl::desc("Test-only: route hazard group 0 off its TYPE-derived mechanism "
                                             "(bufid | bufid-spill), so the buffer-id path and the "
                                             "resource-exhaustion downgrade can be exercised on a kernel the "
                                             "router would otherwise leave alone"),
                              llvm::cl::init(""), llvm::cl::Hidden);

llvm::cl::opt<bool> checkAddrReuseWar("check-addr-reuse-war",
                                      llvm::cl::desc("List the synchronization orderings whose two endpoints are "
                                                     "distinct tile allocations placed at the same physical "
                                                     "address in one scope. Reports; never fails"),
                                      llvm::cl::init(false));

llvm::cl::opt<bool> dumpSyncCoverage("dump-sync-coverage",
                                     llvm::cl::desc("Write the happens-before edges this run's emitted sync induces, "
                                                    "as the reference profile for --check-sync-coverage"),
                                     llvm::cl::init(false));

llvm::cl::opt<std::string>
    checkSyncCoverage("check-sync-coverage",
                      llvm::cl::desc("Run oracle gate G1: assert this run establishes every ordering "
                                     "in the given file (a prior --dump-sync-coverage run). "
                                     "DIFFERENTIAL -- its strength is bounded by that reference"),
                      llvm::cl::init(""), llvm::cl::value_desc("profile-file"));

llvm::cl::opt<bool> dumpSyncCounts("dump-sync-counts",
                                   llvm::cl::desc("Write this run's kernel-body sync-op counts, one line per "
                                                  "function, as the reference for --check-sync-count-floor"),
                                   llvm::cl::init(false));

llvm::cl::opt<std::string>
    checkSyncCountFloor("check-sync-count-floor",
                        llvm::cl::desc("Run oracle gate G4: assert this run does not synchronize more "
                                       "than the given reference profile, per class (total, barriers, "
                                       "PIPE_ALL barriers). DIFFERENTIAL"),
                        llvm::cl::init(""), llvm::cl::value_desc("profile-file"));

llvm::cl::opt<bool> checkSyncIds("check-sync-ids",
                                 llvm::cl::desc("Run oracle gate G3: verify every static emitted sync id is "
                                                "legal for its direction, op and arch. Rotating set_flag_dyn / "
                                                "wait_flag_dyn ids resolve at runtime and are not checked "
                                                "(fails on violation)"),
                                 llvm::cl::init(false));

llvm::cl::opt<std::string>
    checkSyncIdsInjectFault("check-sync-ids-inject-fault",
                            llvm::cl::desc("Gate self-test: feed G3 a synthetic illegal id "
                                           "(event-oor | block-all-stomp | bufid-oor). Appended to the "
                                           "gate's own record list after extraction, never to the IR, "
                                           "because an out-of-range event id is unrepresentable in IR"),
                            llvm::cl::init(""), llvm::cl::Hidden);

llvm::cl::opt<std::string>
    checkSyncInterferenceInjectFault("check-sync-interference-inject-fault",
                                     llvm::cl::desc("Gate self-test: judge a synthetic pre-codegen shape "
                                                    "(comp-conflict | multi-id-conflict) on its own record list. "
                                                    "Never added to the IR, so it cannot reach codegen"),
                                     llvm::cl::init(""), llvm::cl::Hidden);

llvm::cl::opt<bool> checkSyncInterference("check-sync-interference",
                                          llvm::cl::desc("Run oracle gate G2: assert no two overlapping intervals "
                                                         "share a sync id (fails on nesting / leak)"),
                                          llvm::cl::init(false));

llvm::cl::opt<bool>
    checkSyncSelfCoverage("check-sync-self-coverage",
                          llvm::cl::desc("Run oracle gate G1-self: assert every dependency the "
                                         "front-end analysis (DepBetween) reports is ordered by the "
                                         "emitted sync, except pairs on mutually exclusive scf.if arms, "
                                         "PIPE_S->PIPE_S pairs on every arch, and, on A5, "
                                         "PIPE_V->PIPE_V pairs the target orders itself; all three "
                                         "exclusions are counted and reported. ABSOLUTE -- no "
                                         "reference profile and nothing inherited from the sync pass"),
                          llvm::cl::init(false));

llvm::cl::opt<bool> enableInsertSync("enable-insert-sync",
                                            llvm::cl::desc("Enable automatic synchronization insertion pass"),
                                            llvm::cl::init(false));

llvm::cl::opt<bool> enableVmiFastmath(
    "vmi-fastmath",
    llvm::cl::desc("Select the VMI conversion state for the whole pipeline: it "
                   "enables the conversion fast paths that may change NaN "
                   "results, including f32-to-bf16 narrowing, and lets layout "
                   "assignment offer the dense same-layout relation for those "
                   "casts. Enabled by default; pass --vmi-fastmath=false to "
                   "disable"),
    llvm::cl::init(true));

llvm::cl::opt<unsigned> vmiLayoutMaxFrontierEntries(
    "vmi-layout-max-frontier-entries",
    llvm::cl::desc("Maximum VMI layout search frontier entries per component "
                   "(must be positive)"),
    llvm::cl::init(VMILayoutPlannerOptions{}.maxFrontierEntriesPerComponent));

llvm::cl::opt<unsigned> vmiLayoutMaxTransitions(
    "vmi-layout-max-transitions",
    llvm::cl::desc("Maximum VMI layout search transitions per component "
                   "(must be positive)"),
    llvm::cl::init(VMILayoutPlannerOptions{}.maxTransitionsPerComponent));

llvm::cl::opt<bool> planMemoryOrderBySize(
    "plan-memory-order-by-size",
    llvm::cl::desc("Plan larger local buffers first inside one AddressSpace "
                   "before applying the basic SPEC_LEVEL_0 reuse strategy. "
                   "Defaults to false to match the previous default pipeline "
                   "(largest-first ordering is opt-in)"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enablePlanMemory(
    "enable-plan-memory",
    llvm::cl::desc("Enable automatic local memory planning (pto-plan-memory) "
                   "for pto.alloc_tile without an explicit addr. On by "
                   "default; disable with --enable-plan-memory=false. "
                   "--pto-level=level3 always skips planning because there the "
                   "user supplies every address explicitly."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableBufidSync(
    "enable-bufid_sync",
    llvm::cl::desc("Enable A5 buffer-id synchronization insertion pass"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableBufidSyncDebug(
    "enable-bufid-sync-debug",
    llvm::cl::desc("Enable verbose debug printing for --enable-bufid_sync"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableInjectBarrierAllSync(
    "enable-inject-barrier-all-sync",
    llvm::cl::desc("Enable conservative synchronization by inserting "
                   "pto.barrier PIPE_ALL before memory-effecting PTO pipe ops"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableVecScopeMemBar(
    "enable-vecscope-mem-bar", llvm::cl::desc("Insert memory barriers for VPTO vecscope hazards"),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableVecScopeMemBarAll(
    "enable-vecscope-mem-bar-all",
    llvm::cl::desc("Insert family-wide barriers before vecscope UB vector and scalar accesses"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableTileOpExpand(
    "enable-tile-op-expand",
    llvm::cl::desc(
        "Deprecated compatibility flag. TileOp expansion is controlled by "
        "--pto-backend=vpto."),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableVexpdifFusion(
    "enable-vexpdif-fusion",
    llvm::cl::desc("Enable vsub + vexp fusion into vexpdif"),
    llvm::cl::init(true));

llvm::cl::opt<llvm::cl::boolOrDefault> enableOpFusion(
    "enable-op-fusion",
    llvm::cl::desc("Control A5 tile fusion on level2/level3. Disabled by "
                   "default; pass --enable-op-fusion=true to opt in. EmitC "
                   "uses last-use annotation; VPTO uses fusion-region "
                   "lifecycle."),
    llvm::cl::init(llvm::cl::BOU_UNSET));

llvm::cl::opt<bool> enableUnrollAfterLoopFusion(
    "enable-unroll-after-loop-fusion",
    llvm::cl::desc("Partial-unroll the innermost scf.for in pto.fusion_region "
                   "by pto.fusion.row/col_unroll_factor. VPTO backend only; "
                   "requires --pto-arch=a5 and --enable-op-fusion."),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableShapeInference(
    "enable-shape-inference",
    llvm::cl::desc("Enable shape inference (ShapeConstraintSolver) for A5 tile "
                  "fusion. On by default: uses the ShapeConstraintSolver for "
                  "iteration-domain inference; pass --enable-shape-inference=false "
                  "to fall back to static/direct-bound inference."),
    llvm::cl::init(true));

llvm::cl::opt<bool> enableVfSimCostmodelOptimization(
    "enable-vfsim-costmodel-optimization",
    llvm::cl::desc("Enable optional VfSimulator costmodel-driven fusion "
                   "optimization. Requires the A5 tile-fusion pipeline. This "
                   "may annotate pto.fusion.row/col_unroll_factor; pass "
                   "--enable-unroll-after-loop-fusion to consume those "
                   "attributes in the VPTO backend."),
    llvm::cl::init(false));

llvm::cl::opt<bool> dumpVfSimUnrollTest(
    "dump-vfsim-unroll-test",
    llvm::cl::desc("Print VfSimulator unroll candidate timings for accepted "
                   "fusion groups. Debug dump only; does not enable or disable "
                   "the VfSimulator planner."),
    llvm::cl::init(false));

llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

llvm::cl::opt<bool> enableSoftPostUpdate(
    "enable-vpto-soft-postupdate",
    llvm::cl::desc("Enable VPTO soft post-update optimization (default: true)"),
    llvm::cl::init(true));

llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::emitMlirIR(
    "emit-pto-ir",
    llvm::cl::desc("Emit PTO IR after lowering instead of C++"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::ptoTargetArch(
    "pto-arch",
    llvm::cl::desc("Target Ascend architecture for codegen: a2, a3, or a5 (default: a3)"),
    llvm::cl::value_desc("a2|a3|a5"),
    llvm::cl::init("a3"));

llvm::cl::opt<std::string> ptoBuildLevel(
    "pto-level",
    llvm::cl::desc("Build level for pass pipeline: level1, level2, or level3 (default: level2)"),
    llvm::cl::value_desc("level1|level2|level3"),
    llvm::cl::init("level2"));

llvm::cl::opt<std::string> mlir::pto::ptoBackend(
    "pto-backend",
    llvm::cl::desc("Final PTOAS backend: emitc or vpto (default: emitc)"),
    llvm::cl::value_desc("emitc|vpto"), llvm::cl::init("emitc"));

llvm::cl::opt<bool> mlir::pto::emitVPTO(
    "emit-vpto",
    llvm::cl::desc("Write final post-pass VPTO IR to -o"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::emitVPTOLLVMDialect(
    "emit-vpto-llvm-ir",
    llvm::cl::desc("Write translated VPTO LLVM IR to -o"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::cannOutputVersion(
    "cann-output-version",
    llvm::cl::desc(
        "Override the CANN version the VPTO backend lowers for, instead of the "
        "toolchain-detected one (e.g. 9.0.0-beta.1); mainly for tests"),
    llvm::cl::value_desc("x.y.z[-beta.N]"), llvm::cl::init(""));

static llvm::cl::opt<bool> vptoPrintIR(
    "vpto-print-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> vptoLoweringStrategy(
    "vpto-lowering-strategy",
    llvm::cl::desc("VPTO vector lowering strategy: post-update or no-post-update"),
    llvm::cl::value_desc("post-update|no-post-update"),
    llvm::cl::init("post-update"));

llvm::cl::opt<bool> dumpVPTOIR(
    "dump-vpto-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

llvm::cl::opt<bool> mlir::pto::ptoPrintSeamIR(
    "pto-print-seam-ir",
    llvm::cl::desc("Print shared pre-backend seam IR to stderr"),
    llvm::cl::init(false));

llvm::cl::opt<std::string> mlir::pto::ptoSeamIRFile(
    "pto-seam-ir-file",
    llvm::cl::desc("Write shared pre-backend seam IR to a file"),
    llvm::cl::value_desc("path"),
    llvm::cl::init(""));

llvm::cl::opt<mlir::pto::VFSIMTSizeFixMode> mlir::pto::vptoFixVFSIMTSize(
    "vpto-fix-vfsimt-size",
    llvm::cl::desc("Validate or repair VF_SIMT code sizes in VPTO vector objects"),
    llvm::cl::value_desc("auto|off|verify"),
    llvm::cl::values(
        clEnumValN(mlir::pto::VFSIMTSizeFixMode::Auto, "auto",
                   "Repair the known invalid 0xffff size (default)"),
        clEnumValN(mlir::pto::VFSIMTSizeFixMode::Off, "off",
                   "Skip VF_SIMT size validation and repair"),
        clEnumValN(mlir::pto::VFSIMTSizeFixMode::Verify, "verify",
                   "Validate VF_SIMT sizes without repairing them")),
    llvm::cl::init(mlir::pto::VFSIMTSizeFixMode::Auto));

llvm::cl::opt<bool> mlir::pto::enableBishengVecMISched(
    "enable-bisheng-vec-misched",
    llvm::cl::desc(
        "Use Bisheng's default vector MI scheduler behavior for "
        "VPTO device compilation instead of explicitly disabling "
        "the scheduler"),
    llvm::cl::init(false));

llvm::cl::opt<mlir::pto::BishengSchedulerMode> mlir::pto::bishengSchedulerMode(
    "bisheng-vec-misched", llvm::cl::desc("Bisheng vector scheduling policy for device LLVM compilation"),
    llvm::cl::value_desc("auto|on|off"),
    llvm::cl::values(
        clEnumValN(
            BishengSchedulerMode::Auto, "auto",
            "Retry off on compilation failure or nonzero VF stack; prefer the smaller stack (default)"),
        clEnumValN(BishengSchedulerMode::On, "on", "Keep Bisheng default scheduling without retry"),
        clEnumValN(BishengSchedulerMode::Off, "off", "Disable Bisheng vector scheduling without retry")),
    llvm::cl::init(BishengSchedulerMode::Auto));
