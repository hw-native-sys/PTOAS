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

option(PTOAS_LIT_TEST "Build pto lit tests." ON)
if(NOT PTOAS_LIT_TEST)
  return()
endif()
message(STATUS "Building pto lit tests")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(PTOIR_TEST_DEPENDS
  ptoas_runtime_deps
  pto-test-opt
  pto-vfsimt-size-patcher-test
  yaml2obj
)

# Installed LLVM packages do not export the build-tree targets for these test
# tools. Keep target dependencies when PTOAS uses an LLVM build tree without
# rejecting an installed LLVM prefix during configuration.
foreach(tool FileCheck count not)
  if(TARGET ${tool})
    list(APPEND PTOIR_TEST_DEPENDS ${tool})
  endif()
endforeach()

add_lit_testsuite(check-pto "Running the pto regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PTOIR_TEST_DEPENDS}
        )
set_target_properties(check-pto PROPERTIES FOLDER "Tests")
