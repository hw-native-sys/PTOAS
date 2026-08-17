# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ptobc tests (ctest)

find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(PTObc_TESTDATA_DIR "${CMAKE_CURRENT_LIST_DIR}/../testdata")

add_test(NAME ptobc_stage9_e2e
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOBC_ALLOW_GENERIC=1
    TESTDATA_DIRS=${PTObc_TESTDATA_DIR}:${CMAKE_SOURCE_DIR}/test/samples
    ${CMAKE_CURRENT_LIST_DIR}/stage9_e2e.sh
)

add_test(NAME ptobc_to_ptoas_smoke
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOBC_ALLOW_GENERIC=1
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ptobc_to_ptoas_smoke.sh
)

add_test(NAME ptobc_opcode_coverage_check
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_LIST_DIR}/opcode_coverage_check.py
    ${CMAKE_SOURCE_DIR}/include/PTO/IR/PTOOps.td
    ${CMAKE_SOURCE_DIR}/tools/ptobc/generated/ptobc_opcodes_v0.h
)

add_test(NAME ptobc_v0_fp_schema_compatibility_check
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_LIST_DIR}/v0_fp_schema_compatibility_check.py
    ${CMAKE_SOURCE_DIR}/tools/ptobc/generated/ptobc_opcodes_v0.h
)

add_test(NAME ptobc_tfillpad_legacy_v0_decode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tfillpad_legacy_v0_decode.sh
)

add_test(NAME ptobc_trowexpandsub_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/trowexpandsub_v0_encode.sh
)

add_test(NAME ptobc_tprint_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tprint_v0_encode.sh
)

add_test(NAME ptobc_tci_trowexpandadd_tmp_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tci_trowexpandadd_tmp_v0_encode.sh
)

add_test(NAME ptobc_tpartarg_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tpartarg_v0_encode.sh
)

add_test(NAME ptobc_func_call_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/func_call_v0_encode.sh
)

add_test(NAME ptobc_recent_ops_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/recent_ops_v0_encode.sh
)

add_test(NAME ptobc_frontend_pipe_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/frontend_pipe_v0_encode.sh
)

add_test(NAME ptobc_internal_pipe_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/internal_pipe_v0_encode.sh
)

add_test(NAME ptobc_aiv_subblockid_pipe_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/aiv_subblockid_pipe_v0_encode.sh
)

add_test(NAME ptobc_tconcat_set_validshape_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tconcat_set_validshape_v0_encode.sh
)

add_test(NAME ptobc_tstore_fp_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tstore_fp_v0_encode.sh
)

add_test(NAME ptobc_fp_operand_forms_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/fp_operand_forms_v0_encode.sh
)

add_test(NAME ptobc_mx_grp_axis_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    --unset=PTOBC_ALLOW_GENERIC
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/mx_grp_axis_v0_encode.sh
)

add_test(NAME ptobc_fp_extended_v0_compatibility
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/fp_extended_v0_compatibility.sh
)

add_test(NAME ptobc_comm_p2p_dynamic_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    PTOAS_BIN=${CMAKE_BINARY_DIR}/tools/ptoas/ptoas
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/comm_p2p_dynamic_v0_encode.sh
)

add_test(NAME ptobc_tdequant_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tdequant_v0_encode.sh
)

add_test(NAME ptobc_tscatter_maskpattern_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    PYTHON_EXECUTABLE=${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_LIST_DIR}/tscatter_maskpattern_v0_encode.sh
)

add_test(NAME ptobc_tcvt_satmode_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/tcvt_satmode_v0_encode.sh
)
