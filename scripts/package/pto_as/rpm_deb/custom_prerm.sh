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
# It removes PTOAS's full payload from the shared python/site-packages (all
# top-level names plus PTOAS's own console script in the shared bin/), the
# <version>/bin/ptoas symlink, and the .ptoas-python.path interpreter record,
# leaving nothing behind in the shared directories. Sibling components that
# share those directories are never touched.

# RPM passes 1 while replacing an old package, and DEB passes upgrade. In both
# cases the newly installed package must retain the runtime created by its
# post-install hook. Only a real removal cleans the installed runtime.
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
    # Last resort: pto_common.sh is missing or unreadable, which the package
    # layout rules out (the files are still present here). Repeat the payload
    # names inline so removal can never fail to clean the runtime, and keep this
    # list in sync with PTOAS_SITE_PACKAGES_TOPLEVEL / PTOAS_LEGACY_PRIVATE_PYTHON
    # in pto_common.sh, which is the authoritative definition.
    rm -rf "${INSTALL_PATH}/python/site-packages/ptoas" \
           "${INSTALL_PATH}/python/site-packages/ptodsl" \
           "${INSTALL_PATH}/python/site-packages/TileOps" \
           "${INSTALL_PATH}/python/site-packages/SoftOps" \
           "${INSTALL_PATH}/python/site-packages/ptoas.libs"
    rm -rf "${INSTALL_PATH}/python/site-packages/"ptoas-*.dist-info
    rm -f "${INSTALL_PATH}/python/site-packages/bin/ptoas"
    # Best effort: these stay when a sibling component still uses the directory.
    rmdir "${INSTALL_PATH}/python/site-packages/bin" 2>/dev/null || true
    rmdir "${INSTALL_PATH}/python/site-packages" 2>/dev/null || true
    rmdir "${INSTALL_PATH}/python" 2>/dev/null || true
    [ -L "${INSTALL_PATH}/bin/ptoas" ] && rm -f "${INSTALL_PATH}/bin/ptoas"
    rm -f "${INSTALL_PATH}/tools/ptoas/.ptoas-python.path"
    # The pre-site-packages runtime and any staging/backup tree left behind by
    # an interrupted install belong to PTOAS alone, so they can be removed
    # wholesale rather than by name.
    rm -rf "${INSTALL_PATH}/tools/ptoas/python" \
           "${INSTALL_PATH}/tools/ptoas/.ptoas-wheel-staging" \
           "${INSTALL_PATH}/tools/ptoas/.ptoas-wheel-backup"
fi
