#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

_CURR_PATH=$(dirname $(readlink -f $0))

# The component's shared helpers, for the record name of the directories the
# installer created. Guarded so a stray layout still runs the removal below.
PTO_COMMON_FILE="${_CURR_PATH}/pto_common.sh"
[ -r "${PTO_COMMON_FILE}" ] && . "${PTO_COMMON_FILE}"

# error number and description
OPERATE_FAILED="0x0001"
FILE_NOT_EXIST="0x0080"
PERM_DENIED="0x0093"
PERM_DENIED_DES="Permission denied."
# log functions
getdate() {
    _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
    echo "${_cur_date}"
}

logandprint() {
    is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
    if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
        echo "[pto_as] [$(getdate)] ""$1"
    fi
    echo "[pto_as] [$(getdate)] ""$1" >> "${_INSTALL_LOG_FILE}"
}

if [ "$(id -u)" != "0" ]; then
    _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
else
    _LOG_PATH="/var/log/ascend_seclog"
    _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
fi

# init install cmd status, set default as n
is_quiet=n
quiet_parameter=""
if [ "$#" != "0" ]; then
    if [ "$1" = "--quiet" ] && [ "$#" = "1" ]; then
        is_quiet=y
        quiet_parameter="--quiet"
    else
        logandprint "Please use correct parameters, only support input nothing or only --quiet parameter."
        exit 1
    fi
fi

install_shell="${_CURR_PATH}/install.sh"

# shell exist check
if [ ! -f "${install_shell}" ]; then
    logandprint "[ERROR]: ERR_NO:${FILE_NOT_EXIST};pto_as module is not installed or some pto_as source files are lost.\
If there are any residual files, please manually remove those files."
    exit 1
fi

# shell execute perm check
if [ ! -x "${install_shell}" ]; then
    logandprint "[ERROR]: ERR_NO:${PERM_DENIED};ERR_DES:The user do \
not have the permission to execute this file, please reset the file \
to a right permission."
    exit 1
fi

# The version root is four levels up: script -> pto_as -> info -> share -> <version>.
# It is passed to install.sh as the install path on purpose. install.sh
# reconstructs the version directory from the install path and only trusts the
# name it finds there when that path already looks like a version directory
# (is_version_dirpath: it contains share/info). Handing it the *parent* instead
# makes it fall back to the hardcoded "cann" name, so any install whose version
# directory is called something else -- the multi-version layout
# <prefix>/cann-9.2.0/ascend-toolkit, for example -- is looked up at
# <parent>/cann, found missing, and left completely uninstalled. Because that
# branch also exits 0, the removal reports success while the payload stays on
# disk.
version_root="$(cd "${_CURR_PATH}/../../../.."; pwd)"

# Read the installer's record of the directories it created before the removal
# deletes it along with the rest of the component metadata. Without it the
# cleanup below can only remove the version directory, and an install prefix the
# installer created would survive as an empty skeleton.
created_dirs_record="${version_root}/share/info/pto_as/${PTOAS_CREATED_DIRS_RECORD}"
created_dirs=""
if [ -r "${created_dirs_record}" ]; then
    created_dirs="$(cat "${created_dirs_record}" 2> /dev/null)"
fi

cd ~
sh "${install_shell}" "--aa" "--aa" "--uninstall" "--install-path=${version_root}" "${quiet_parameter}"
ret_status="$?"
if [ "${ret_status}" != "0" ]; then
    exit "${ret_status}"
fi

# Verify the component really went away rather than trusting the exit status: a
# mis-derived install path, or a removal that failed partway, is reported as
# success by the layer below, which leaves the user believing the package is
# uninstalled while files stay on disk. Every path listed here belongs to PTOAS
# alone (see the component filelist in pto_as.xml), so a sibling component
# sharing the version directory cannot make this fire.
pto_first_residual() {
    local entry
    # The component's metadata and scripts tree, the component-owned tools tree
    # (wheels plus launcher plus the interpreter record) and the arch header.
    [ -e "${version_root}/share/info/pto_as" ] && { echo "${version_root}/share/info/pto_as"; return 0; }
    [ -e "${version_root}/tools/ptoas" ] && { echo "${version_root}/tools/ptoas"; return 0; }
    for entry in "${version_root}"/*-linux/include/version/pto_as_version.h; do
        [ -e "${entry}" ] && { echo "${entry}"; return 0; }
    done
    # The shared site-packages payload and the command symlink.
    for entry in ptoas ptodsl TileOps SoftOps ptoas.libs; do
        [ -e "${version_root}/python/site-packages/${entry}" ] &&
            { echo "${version_root}/python/site-packages/${entry}"; return 0; }
    done
    for entry in "${version_root}"/python/site-packages/ptoas-*.dist-info; do
        [ -e "${entry}" ] && { echo "${entry}"; return 0; }
    done
    for entry in "${version_root}/python/site-packages/bin/ptoas" \
        "${version_root}/bin/ptoas" "${version_root}"/*-linux/bin/ptoas; do
        # Match either form: the launcher is normally a symlink, but a console
        # script copied in as a regular file is just as much a leftover, and a
        # symlink test alone would miss it. The per-architecture path is checked
        # separately because the shared <version>/bin symlink may already be gone
        # by the time the component is removed, which would hide a dangling
        # command link in the arch directory.
        { [ -e "${entry}" ] || [ -L "${entry}" ]; } && { echo "${entry}"; return 0; }
    done
    return 1
}

residual="$(pto_first_residual)"
if [ -n "${residual}" ]; then
    logandprint "[ERROR]: ERR_NO:${OPERATE_FAILED};ERR_DES:Residual pto_as files are still present\
 in (${version_root}) after uninstallation, for example (${residual}). Please remove them manually."
    exit 1
fi

# Drop the emptied skeleton, but only the directories this component is entitled
# to remove: the version directory itself, plus the install prefix when the
# installer had to create it. The installer records the latter, because nothing
# at removal time can tell an empty directory it created from an empty directory
# the user already had. Removal stops there -- there is deliberately no walk up
# the tree, so an unrelated empty directory such as /data/team/Ascend is never a
# candidate even when it happens to be empty.
#
# Every removal is an rmdir: a directory that still holds anything -- a sibling
# component, or the user's own files -- is left alone. The recorded list is swept
# repeatedly rather than once in file order, so an ancestor recorded above its
# descendant goes on a later pass instead of surviving as an empty shell.
rmdir "${version_root}" 2> /dev/null
pto_rmdir_created_dirs "${created_dirs}"
exit 0

