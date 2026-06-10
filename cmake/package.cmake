# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}")

set(PTOAS_ROOT_DIR "${CMAKE_SOURCE_DIR}")

set(SCRIPTS_FILES
    ${CANN_CMAKE_DIR}/scripts/install/check_version_required.awk
    ${CANN_CMAKE_DIR}/scripts/install/common_func.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.csh
    ${CANN_CMAKE_DIR}/scripts/install/common_interface.fish
    ${CANN_CMAKE_DIR}/scripts/install/version_compatiable.inc
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/cleanup.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/help.info
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/install.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/pto_common.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/pto_custom_install.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/pto_custom_uninstall.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/pto_install.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/pto_uninstall.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/uninstall.sh
    ${PTOAS_ROOT_DIR}/scripts/package/pto_as/scripts/ver_check.sh
)

install(FILES ${SCRIPTS_FILES}
    DESTINATION share/info/pto_as/script
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
    PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE
)

set(COMMON_FILES
    ${CANN_CMAKE_DIR}/scripts/install/install_common_parser.sh
    ${CANN_CMAKE_DIR}/scripts/install/common_func_v2.inc
    ${CANN_CMAKE_DIR}/scripts/install/common_installer.inc
    ${CANN_CMAKE_DIR}/scripts/install/script_operator.inc
    ${CANN_CMAKE_DIR}/scripts/install/version_cfg.inc
)

set(PACKAGE_FILES
    ${COMMON_FILES}
    ${CANN_CMAKE_DIR}/scripts/install/multi_version.inc
)

set(CONF_FILES
    ${CANN_CMAKE_DIR}/scripts/package/cfg/path.cfg
)

install(FILES ${CMAKE_BINARY_DIR}/version.pto-as.info
    DESTINATION share/info/pto_as
    RENAME version.info
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
)

install(FILES ${CONF_FILES}
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/conf
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
)

install(FILES ${PACKAGE_FILES}
    DESTINATION share/info/pto_as/script
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
)

install(DIRECTORY ${PTOAS_ROOT_DIR}/include/
    DESTINATION ${CMAKE_SYSTEM_PROCESSOR}-linux/include
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE
    PATTERN CMakeLists.txt EXCLUDE
    PATTERN pto-c EXCLUDE
    PATTERN PTO EXCLUDE
)

if(CMAKE_BUILD_TYPE)
    string(TOLOWER "${CMAKE_BUILD_TYPE}" PTOAS_TARGETS_CONFIG_SUFFIX)
else()
    set(PTOAS_TARGETS_CONFIG_SUFFIX "noconfig")
endif()

install(FILES
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/PTOAS/PTOASTargets.cmake
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/PTOAS/PTOASTargets-${PTOAS_TARGETS_CONFIG_SUFFIX}.cmake
    ${CMAKE_INSTALL_PREFIX}/lib/cmake/PTOAS/PTOASConfig.cmake
    DESTINATION lib/cmake/PTOAS
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
)

install(FILES
    ${PTOAS_ROOT_DIR}/python/pto/dialects/pto.py
    ${CMAKE_BINARY_DIR}/lib/Bindings/Python/_pto_ops_gen.py
    DESTINATION mlir/dialects
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
)

set(PTOAS_PACKAGE_STAGE_DIR
    ${CMAKE_BINARY_DIR}/package_runtime/tools/ptoas)

install(FILES ${PTOAS_PACKAGE_STAGE_DIR}/bin/ptoas
    DESTINATION tools/ptoas/bin
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
    PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
)

install(FILES ${PTOAS_PACKAGE_STAGE_DIR}/bin/ptoas.real
    DESTINATION tools/ptoas/bin
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
    PERMISSIONS OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
)

install(DIRECTORY ${PTOAS_PACKAGE_STAGE_DIR}/lib/
    DESTINATION tools/ptoas/lib
    ${INSTALL_OPTIONAL}
    COMPONENT pto-as
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_EXECUTE
    FILES_MATCHING PATTERN "*.so*"
)
