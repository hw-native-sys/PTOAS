# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# -----------------------------------------------------------------------------------------------------------
#
# Appended to the auto-generated RPM pre-uninstall / DEB prerm by cann-cmake.
# The package files are still present at this lifecycle stage, so reuse the
# shared helper before cann-cmake removes the component files and symlinks.
# It removes tools/ptoas/python and the .ptoas-python.path interpreter record.

# RPM passes 1 while replacing an old package, and DEB passes upgrade. In both
# cases the newly installed package must retain the runtime created by its
# post-install hook. Only a real removal cleans the private runtime.
case "${1:-}" in
    1|upgrade)
        exit 0
        ;;
esac

PTOAS_COMMON="${INSTALL_PATH}/share/info/pto_as/script/pto_common.sh"
if [ -r "${PTOAS_COMMON}" ]; then
    . "${PTOAS_COMMON}"
    pto_uninstall_wheel "${INSTALL_PATH}" "${INSTALL_PATH}/share/info/pto_as"
else
    rm -rf "${INSTALL_PATH}/tools/ptoas/python"
    rm -f "${INSTALL_PATH}/tools/ptoas/.ptoas-python.path"
fi
