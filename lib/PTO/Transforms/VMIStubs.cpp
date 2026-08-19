// Stub implementations for passes declared in Passes.td but whose
// full implementations reference rebuild-only IR ops not present in main.
#include "PTO/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace mlir::pto {

namespace {
struct StubPTOViewToMemrefPass : public PassWrapper<StubPTOViewToMemrefPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "pto-view-to-memref"; }
  StringRef getDescription() const final { return "Stub"; }
};
struct StubPTOMaterializeTileHandlesPass : public PassWrapper<StubPTOMaterializeTileHandlesPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "pto-materialize-tile-handles"; }
  StringRef getDescription() const final { return "Stub"; }
};
struct StubVPTONormalizeEquivalentVcvtPass : public PassWrapper<StubVPTONormalizeEquivalentVcvtPass, OperationPass<ModuleOp>> {
  void runOnOperation() override {}
  StringRef getArgument() const final { return "pto-vpto-normalize-equiv-vcvt"; }
  StringRef getDescription() const final { return "Stub"; }
};
} // namespace

std::unique_ptr<Pass> createPTOViewToMemrefPass() {
  return std::make_unique<StubPTOViewToMemrefPass>();
}
std::unique_ptr<Pass> createPTOMaterializeTileHandlesPass() {
  return std::make_unique<StubPTOMaterializeTileHandlesPass>();
}
std::unique_ptr<Pass> createVPTONormalizeEquivalentVcvtPass() {
  return std::make_unique<StubVPTONormalizeEquivalentVcvtPass>();
}

} // namespace mlir::pto
