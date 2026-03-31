// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "ptobc/ptobc_format.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>

#include <PTO/IR/PTO.h>

#include <iostream>
#include <optional>

namespace ptobc {
mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path);
PTOBCFile encodeFromMLIRModule(mlir::ModuleOp module);
void decodeFileToPTO(const std::string& inPath, const std::string& outPath);
}

static void usage() {
  std::cerr << "ptobc (v0)\n\n"
            << "Usage:\n"
            << "  ptobc encode <input.pto> -o <out.ptobc>\n"
            << "  ptobc decode <input.ptobc> -o <out.pto>\n";
}

struct CommandLineOptions {
  std::string cmd;
  std::string input;
  std::string output;
};

static std::optional<CommandLineOptions> parseCommandLine(int argc, char **argv) {
  if (argc < 2)
    return std::nullopt;

  CommandLineOptions options{argv[1], "", ""};
  if (options.cmd != "encode" && options.cmd != "decode")
    return options;
  if (argc < 5)
    return std::nullopt;

  options.input = argv[2];
  for (int i = 3; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc)
      options.output = argv[++i];
  }
  if (options.output.empty())
    return std::nullopt;
  return options;
}

static mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect, mlir::pto::PTODialect>();
  return registry;
}

static void preloadDialects(mlir::MLIRContext &ctx) {
  (void)ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  (void)ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  (void)ctx.getOrLoadDialect<mlir::affine::AffineDialect>();
  (void)ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
  (void)ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  (void)ctx.getOrLoadDialect<mlir::pto::PTODialect>();
}

static int runEncode(const CommandLineOptions &options) {
  mlir::MLIRContext ctx(buildRegistry());
  ctx.allowUnregisteredDialects(true);
  preloadDialects(ctx);

  auto module = ptobc::parsePTOFile(ctx, options.input);
  auto file = ptobc::encodeFromMLIRModule(*module);
  auto bytes = file.serialize();
  ptobc::writeFile(options.output, bytes);
  return 0;
}

static int runDecode(const CommandLineOptions &options) {
  ptobc::decodeFileToPTO(options.input, options.output);
  return 0;
}

int main(int argc, char **argv) {
  auto options = parseCommandLine(argc, argv);
  if (!options) {
    usage();
    return 2;
  }

  try {
    if (options->cmd == "encode")
      return runEncode(*options);
    if (options->cmd == "decode")
      return runDecode(*options);
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
