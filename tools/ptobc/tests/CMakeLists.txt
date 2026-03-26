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
    PTOAS_BIN=$<TARGET_FILE:pto-opt>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/ptobc_to_ptoas_smoke.sh
)

add_test(NAME ptobc_opcode_coverage_check
  COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_LIST_DIR}/opcode_coverage_check.py
    ${CMAKE_SOURCE_DIR}/include/PTO/IR/PTOOps.td
    ${CMAKE_SOURCE_DIR}/tools/ptobc/generated/ptobc_opcodes_v0.h
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

add_test(NAME ptobc_recent_ops_v0_encode
  COMMAND ${CMAKE_COMMAND} -E env
    PTOBC_BIN=$<TARGET_FILE:ptobc>
    TESTDATA_DIR=${PTObc_TESTDATA_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/recent_ops_v0_encode.sh
)
