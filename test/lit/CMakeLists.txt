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

add_lit_testsuite(check-pto "Running the pto regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${PTOIR_TEST_DEPENDS}
        )
set_target_properties(check-pto PROPERTIES FOLDER "Tests")
