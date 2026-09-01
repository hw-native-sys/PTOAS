# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Appended to the auto-generated postinst by cann-cmake gen_postinst_prerm.py for the rpm/deb package.
#
# IMPORTANT: cann-cmake inlines this whole file into the RPM post section via
# CPACK_RPM_POST_INSTALL_SCRIPT_FILE, so RPM spec macro processor will expand any literal percent.
# The DEB backend uses the same text verbatim. To stay safe on both backends, this script MUST NOT
# contain any literal percent sign (no format strings, no date format, no heredocs with it). Plain shell
# globbing and find -exec are fine.
#
# pto-as installs: ptoas binary + runtime shared libs, headers, and install scripts. The
# run package's pto_install.sh removes lib/ (cmake config) and mlir/ (python dialects)
# after copy -- they are dev-only artifacts not needed at runtime. The custom postinst
# does the same removal so rpm/deb install matches the run package's final layout.
# The top-level symlinks (bin/lib64/include/...) and the package database
# (var/ascend_package_db.info) are already handled by the cann-cmake-generated postinst
# body driven by the EngineeringCommon block in pto_as.xml. RPM/DEB users uninstall via
# rpm -e / dpkg -r, so no extra cann_uninstall.sh entry or ascend_install.info record is
# created here.
#
# What this script does: install the private PTOAS wheel runtime and align installed file/dir permissions with the makeself run package's
# "install for all" mode (IS_FOR_ALL=y, root install), so the files are readable by every user.
# The run package's pto_install.sh does this dynamically; rpm/deb have fixed permissions baked in at
# pack time, so we replay the same mode bits here.
#
#   builtin files (headers/scripts/executables/libs): 555
#   version/scene info files:                        444
#   package db file:                                 644
#   directories:                                     755 (top-level + custom) or 555 (builtin subtrees)
#
# Permission cleanup is best-effort.  The CI placeholder package intentionally
# has no wheel; pto_install_wheel treats that as a successful no-op.  A supplied
# wheel is still validated and installed strictly.

sourcedir="${INSTALL_PATH}"
PTO_PLATFORM_DIR="pto_as"
INSTALL_ROOT="${sourcedir}"
ARCH_DIR="$(uname -m)-linux"

# Only run when the install root actually exists.
if [ -d "${INSTALL_ROOT}" ]; then

    # Install a supplied wheel into the same private runtime used by the run
    # package.  The helper skips cleanly when the placeholder package contains
    # no wheel; when present, it selects exactly one wheel, installs it into
    # tools/ptoas/python with --no-deps --target, and records the interpreter
    # consumed by tools/ptoas/bin/ptoas.
    PTOAS_COMMON="${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/script/pto_common.sh"
    if [ ! -r "${PTOAS_COMMON}" ]; then
        echo "[pto-as] missing wheel runtime helper: ${PTOAS_COMMON}" >&2
        exit 1
    fi
    . "${PTOAS_COMMON}"
    if ! pto_install_wheel "${INSTALL_ROOT}" "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}"; then
        echo "[pto-as] failed to install PTOAS wheel runtime" >&2
        exit 1
    fi

    # 1. Builtin headers and scripts: align to 555.
    find "${INSTALL_ROOT}/${ARCH_DIR}/include" -type f ! -name "pto_as_version.h" 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
    find "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/script" -type f 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
    # fall back to any *-linux arch dir if uname-based path is absent
    if [ ! -d "${INSTALL_ROOT}/${ARCH_DIR}" ]; then
        for arch_dir in "${INSTALL_ROOT}"/*-linux; do
            [ -d "${arch_dir}" ] || continue
            find "${arch_dir}/include" -type f ! -name "pto_as_version.h" 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
        done
    fi

    # 2. Executables and runtime shared libs: align to 555.
    find "${INSTALL_ROOT}/tools/ptoas/bin" -type f 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
    find "${INSTALL_ROOT}/tools/ptoas/lib" -type f -name "*.so*" 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true

    # 3. Cmake config files and python dialects are NOT installed by the run
    # package (pto_install.sh removes lib/ and mlir/ after copy). Remove them
    # here too so rpm/deb install matches the run package's final layout.
    if [ -d "${INSTALL_ROOT}/lib" ]; then
        rm -rf "${INSTALL_ROOT}/lib"
    fi
    if [ -d "${INSTALL_ROOT}/mlir" ]; then
        rm -rf "${INSTALL_ROOT}/mlir"
    fi

    # 5. Read-only info files: version header, scene.info, version.info (444,
    # ONLYREAD_PERM under IS_FOR_ALL=y, matching run package behavior).
    for f in \
        "${INSTALL_ROOT}/${ARCH_DIR}/include/version/pto_as_version.h" \
        "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/scene.info" \
        "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}/version.info"; do
        [ -f "$f" ] && chmod 444 "$f" 2>/dev/null || true
    done
    # fall back to any *-linux arch dir for version header
    if [ ! -d "${INSTALL_ROOT}/${ARCH_DIR}" ]; then
        for arch_dir in "${INSTALL_ROOT}"/*-linux; do
            [ -d "${arch_dir}" ] || continue
            for f in "${arch_dir}/include/version/pto_as_version.h"; do
                [ -f "$f" ] && chmod 444 "$f" 2>/dev/null || true
            done
        done
    fi

    # 6. Package database file: align to 644 (owner writable, others read-only).
    [ -f "${INSTALL_ROOT}/var/ascend_package_db.info" ] && chmod 644 "${INSTALL_ROOT}/var/ascend_package_db.info" 2>/dev/null || true

    # 7. Directories: builtin subtrees -> 555, custom/top-level dirs -> 755.
    find "${INSTALL_ROOT}/${ARCH_DIR}" -type d 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
    find "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}" -type d 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
    # fall back to any *-linux arch dir
    if [ ! -d "${INSTALL_ROOT}/${ARCH_DIR}" ]; then
        for arch_dir in "${INSTALL_ROOT}"/*-linux; do
            [ -d "${arch_dir}" ] || continue
            find "${arch_dir}" -type d 2>/dev/null -exec chmod 555 {} + 2>/dev/null || true
        done
    fi

    # custom/top-level dirs -> 755 (matches run package CUSTOM_PERM under IS_FOR_ALL=y)
    for d in \
        "${INSTALL_ROOT}" \
        "${INSTALL_ROOT}/share" \
        "${INSTALL_ROOT}/share/info" \
        "${INSTALL_ROOT}/share/info/${PTO_PLATFORM_DIR}" \
        "${INSTALL_ROOT}/tools" \
        "${INSTALL_ROOT}/tools/ptoas" \
        "${INSTALL_ROOT}/tools/ptoas/bin" \
        "${INSTALL_ROOT}/tools/ptoas/lib" \
        "${INSTALL_ROOT}/var" \
        "${INSTALL_ROOT}/${ARCH_DIR}/include/pto"; do
        [ -d "$d" ] && chmod 755 "$d" 2>/dev/null || true
    done
    # fall back to any *-linux arch dir for top-level include dirs
    if [ ! -d "${INSTALL_ROOT}/${ARCH_DIR}" ]; then
        for arch_dir in "${INSTALL_ROOT}"/*-linux; do
            [ -d "${arch_dir}" ] || continue
            for d in "${arch_dir}/include/pto"; do
                [ -d "$d" ] && chmod 755 "$d" 2>/dev/null || true
            done
        done
    fi

fi

exit 0
