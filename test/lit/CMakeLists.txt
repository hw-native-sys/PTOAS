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
  FileCheck count not
  pto-opt
)

# The native target may not be enabled, in this case we won't
# run tests that involves executing on the host: do not build
# useless binaries.
if(TARGET ${LLVM_NATIVE_ARCH})
  list(APPEND MLIR_TEST_DEPENDS
    mlir-cpu-runner
    mlir_c_runner_utils
    mlir_runner_utils
  )
endif()

add_lit_testsuite(check-pto "Running the pto regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PTOIR_TEST_DEPENDS}
        )
set_target_properties(check-pto PROPERTIES FOLDER "Tests")

add_lit_testsuites(PTO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${PTOIR_TEST_DEPENDS})
