# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'PTOIR'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.pto']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.ptoir_obj_root, 'test/lit')
config.ptoir_tools_dir = os.path.join(config.ptoir_obj_root, 'tools/ptoas')
config.ptoir_test_tools_dir = os.path.join(config.ptoir_obj_root,
                                           'tools/pto-test-opt')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

if getattr(config, 'pto_enable_vfsim_costmodel', False):
    config.available_features.add('vfsim-costmodel')

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP',
     'ASCEND_HOME_PATH', 'ASCEND_OPP_PATH', 'ASCEND_AICPU_PATH',
     'ASCEND_TOOLKIT_HOME', 'LD_LIBRARY_PATH', 'PYTHONPATH'])

# Keep build-tree lit tests self-contained. The PTOAS build stages the complete
# MLIR Python runtime together with generated PTO dialect modules and the PTO
# extension under one build-tree Python root.
if getattr(config, 'enable_bindings_python', False):
    llvm_config.with_environment(
        'PYTHONPATH', [config.ptoas_python_dir], append_path=True)

llvm_config.use_default_substitutions()

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [config.ptoir_tools_dir, config.ptoir_test_tools_dir,
             config.llvm_tools_dir]
tools = [
    'ptoas',
    'pto-test-opt',
    'pto-vpto-scheduler-tracker-test',
]

llvm_config.add_tool_substitutions(tools, tool_dirs)
