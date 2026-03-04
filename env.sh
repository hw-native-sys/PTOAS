#!/usr/bin/env bash
# PTOAS runtime environment (auto-generated)
export WORKSPACE_DIR=/home/zhangzhendong/ptoas-workspace
export LLVM_SOURCE_DIR=$WORKSPACE_DIR/llvm-project
export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
export PTO_SOURCE_DIR=$WORKSPACE_DIR/PTOAS
export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install

# tools from pip --user
export PATH=$HOME/.local/bin:$PATH

# Python path for MLIR core + PTO python package
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH

# Shared libraries for LLVM/PTO runtime
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH

# CLI tool
export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH
