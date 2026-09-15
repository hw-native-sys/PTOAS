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

if [ "$(id -u)" != "0" ]; then
  _LOG_PATH=$(echo "${HOME}")"/var/log/ascend_seclog"
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
else
  _LOG_PATH="/var/log/ascend_seclog"
  _INSTALL_LOG_FILE="${_LOG_PATH}/ascend_install.log"
fi

# log functions
getdate() {
  _cur_date=$(date +"%Y-%m-%d %H:%M:%S")
  echo "${_cur_date}"
}

logandprint() {
  is_error_level=$(echo $1 | grep -E 'ERROR|WARN|INFO')
  if [ "${is_quiet}" != "y" ] || [ "${is_error_level}" != "" ]; then
    echo "[pto-as] [$(getdate)] ""$1"
  fi
  echo "[pto-as] [$(getdate)] ""$1" >>"${_INSTALL_LOG_FILE}"
}

# create opapi soft link
createrelativelysoftlink() {
  local src_path_="$1"
  local dst_path_="$2"
  local dst_parent_path_=$(dirname ${dst_path_})
  # echo "dst_parent_path_: ${dst_parent_path_}"
  local relative_path_=$(realpath --relative-to="$dst_parent_path_" "$src_path_")
  # echo "relative_path_: ${relative_path_}"
  if [ -L "$2" ]; then
    return 0
  fi
  ln -s "${relative_path_}" "${dst_path_}" 2>/dev/null
  if [ "$?" != "0" ]; then
    return 1
  else
    return 0
  fi
}

createOpapiLatestSoftlink() {
  targetPkg=$2
  if [ "${targetPkg}x" = "x" ]; then
    #CHANGED
    targetPkg=pto_as
  fi

  osName=""
  if [ -f "$1/$targetPkg/scene.info" ]; then
    . $1/$targetPkg/scene.info
    osName=${os}
  fi
  opapi_lib_path="$1/pto_as/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
  opapi_include_level1_path="$1/pto_as/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
  opapi_include_level2_path="${opapi_include_level1_path}/level2"
  if [ ! -d ${opapi_lib_path} ] || [ ! -d ${opapi_include_level1_path} ] || [ ! -d ${opapi_include_level2_path} ]; then
    return 3
  fi
  if [ -d $(dirname $1)/cann/${architectureDir}/lib64 ]; then
    for file_so in $(ls -1 $1/${architectureDir}/lib64 | grep -E "libaclnn_|libopapi.so"); do
      latest_arch_lib64_src_path="$1/${architectureDir}/lib64/${file_so}"
      latest_arch_lib64_dst_path="$(dirname $1)/cann/${architectureDir}/lib64/${file_so}"
      if [ -f $latest_arch_lib64_dst_path ] || [ -L $latest_arch_lib64_dst_path ]; then
        rm -fr "$latest_arch_lib64_dst_path"
      fi
      createrelativelysoftlink ${latest_arch_lib64_src_path} ${latest_arch_lib64_dst_path}
    done
  fi

  # second the headfiles with 1 and 2 level
  if [ -d $1/${architectureDir}/include/aclnnop ]; then
    for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//'); do
      latest_arch_include_src_path="${opapi_include_level1_path}/${file_level1}"
      latest_arch_include_dst_path="$(dirname $1)/cann/${architectureDir}/include/aclnnop/${file_level1}"
      if [ -f $latest_arch_include_dst_path ] || [ -L $latest_arch_include_dst_path ]; then
        rm -fr "$latest_arch_include_dst_path"
      fi
      createrelativelysoftlink ${latest_arch_include_src_path} ${latest_arch_include_dst_path}
    done
  fi

  if [ -d $1/${architectureDir}/include/aclnnop/level2 ]; then
    for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//'); do
      latest_arch_include_src_path="${opapi_include_level2_path}/${file_level2}"
      latest_arch_include_dst_path="$(dirname $1)/cann/${architectureDir}/include/aclnnop/level2/${file_level2}"
      if [ -f $latest_arch_include_dst_path ] || [ -L $latest_arch_include_dst_path ]; then
        rm -fr "$latest_arch_include_dst_path"
      fi
      createrelativelysoftlink ${latest_arch_include_src_path} ${latest_arch_include_dst_path}
    done
  fi
}

createOpapiSoftlink() {
  osName=""
  if [ -f "$1/pto_as/scene.info" ]; then
    . $1/pto_as/scene.info
    osName=${os}
  fi
  opapi_lib_path="$1/pto_as/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
  opapi_include_level1_path="$1/pto_as/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
  opapi_include_level2_path="${opapi_include_level1_path}/level2"

  if [ ! -d ${opapi_lib_path} ] || [ ! -d ${opapi_include_level1_path} ] || [ ! -d ${opapi_include_level2_path} ]; then
    return 3
  fi
  # first the libopapi.so
  if [ -d $1/${architectureDir}/lib64 ]; then
    for file_so in $(ls -1 ${opapi_lib_path} | grep "so"$); do
      arch_lib64_src_path="${opapi_lib_path}/${file_so}"
      arch_lib64_dst_path="$1/${architectureDir}/lib64/${file_so}"
      if [ -f $arch_lib64_dst_path ] || [ -L $arch_lib64_dst_path ]; then
        rm -fr "$arch_lib64_dst_path"
      fi
      createrelativelysoftlink ${arch_lib64_src_path} ${arch_lib64_dst_path}
    done
  fi

  if [ -d $1/pto_as/lib64 ]; then
    for file_so in $(ls -1 $1/${architectureDir}/lib64 | grep -E "libaclnn_|libopapi.so"); do
      pto_lib64_src_path="$1/${architectureDir}/lib64/${file_so}"
      pto_lib64_dst_path="$1/pto_as/lib64/${file_so}"
      if [ -f $pto_lib64_dst_path ] || [ -L $pto_lib64_dst_path ]; then
        rm -fr "$pto_lib64_dst_path"
      fi
      createrelativelysoftlink ${pto_lib64_src_path} ${pto_lib64_dst_path}
    done
  fi

  # second the headfiles with 1 and 2 level
  if [ -d $1/${architectureDir}/include/aclnnop ]; then
    for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//'); do
      arch_include_src_path="${opapi_include_level1_path}/${file_level1}"
      arch_include_dst_path="$1/${architectureDir}/include/aclnnop/${file_level1}"
      if [ -f $arch_include_dst_path ] || [ -L $arch_include_dst_path ]; then
        rm -fr "$arch_include_dst_path"
      fi
      createrelativelysoftlink ${arch_include_src_path} ${arch_include_dst_path}

      pto_include_src_path="${arch_include_dst_path}"
      pto_include_dst_path="$1/pto_as/include/aclnnop/${file_level1}"
      if [ -f $pto_include_dst_path ] || [ -L $pto_include_dst_path ]; then
        rm -fr "$pto_include_dst_path"
      fi
      createrelativelysoftlink ${pto_include_src_path} ${pto_include_dst_path}
    done
  fi

  if [ -d $1/${architectureDir}/include/aclnnop/level2 ]; then
    for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//'); do
      arch_include_src_path="${opapi_include_level2_path}/${file_level2}"
      arch_include_dst_path="$1/${architectureDir}/include/aclnnop/level2/${file_level2}"
      if [ -f $arch_include_dst_path ] || [ -L $arch_include_dst_path ]; then
        rm -fr "$arch_include_dst_path"
      fi
      createrelativelysoftlink ${arch_include_src_path} ${arch_include_dst_path}

      pto_include_src_path="${arch_include_dst_path}"
      pto_include_dst_path="$1/pto_as/include/aclnnop/level2/${file_level2}"
      if [ -f $pto_include_dst_path ] || [ -L $pto_include_dst_path ]; then
        rm -fr "$pto_include_dst_path"
      fi
      createrelativelysoftlink ${pto_include_src_path} ${pto_include_dst_path}
    done
  fi
}

# remove opapi soft link
removeopapisoftlink() {
  local path="$1"
  if [ -L "$1" ]; then
    rm -fr ${path}
    return 0
  else
    return 1
  fi
}

latestSoftlinksRemove() {
  targetdir=$1
  osName=""
  if [ -f "$targetdir/pto_as/scene.info" ]; then
    . $targetdir/pto_as/scene.info
    osName=${os}
  fi
  opapi_lib_path="$targetdir/pto_as/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
  opapi_include_level1_path="$1/pto_as/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
  opapi_include_level2_path="${opapi_include_level1_path}/level2"

  if [ -d $(dirname $targetdir)/cann/${architectureDir}/lib64 ]; then
    for file_so in $(ls -l "$(dirname $targetdir)/cann/${architectureDir}/lib64/" | grep -E "libaclnn_|libopapi.so"); do
      latest_arch_lib64_path="$(dirname $targetdir)/cann/${architectureDir}/lib64/${file_so}"
      removeopapisoftlink ${latest_arch_lib64_path}
    done
  fi

  # second the headfiles with 1 and 2 level
  if [ -d $(dirname $targetdir)/cann/${architectureDir}/include/aclnnop ]; then
    for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//'); do
      latest_arch_include_path="$(dirname $targetdir)/cann/${architectureDir}/include/aclnnop/${file_level1}"
      removeopapisoftlink ${latest_arch_include_path}
    done
  fi

  if [ -d $(dirname $targetdir)/cann/${architectureDir}/include/aclnnop/level2 ]; then
    for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//'); do
      latest_arch_include_path="$(dirname $targetdir)/cann/${architectureDir}/include/aclnnop/level2/${file_level2}"
      removeopapisoftlink ${latest_arch_include_path}
    done
  fi
}

softlinksRemove() {
  targetdir=$1
  osName=""
  if [ -f "$targetdir/pto_as/scene.info" ]; then
    . $targetdir/pto_as/scene.info
    osName=${os}
  fi
  opapi_lib_path="$targetdir/pto_as/built-in/op_impl/ai_core/tbe/op_api/lib/${osName}/${architecture}"
  opapi_include_level1_path="$targetdir/pto_as/built-in/op_impl/ai_core/tbe/op_api/include/aclnnop"
  opapi_include_level2_path="${opapi_include_level1_path}/level2"

  # first the libopapi.so
  if [ -d $targetdir/${architectureDir}/lib64 ]; then
    for file_so in $(ls -1 $targetdir/${architectureDir}/lib64 | grep -E "libaclnn_|libopapi.so"); do
      arch_lib64_path="$targetdir/${architectureDir}/lib64/${file_so}"
      removeopapisoftlink ${arch_lib64_path}
    done
  fi

  if [ -d $targetdir/pto_as/lib64 ]; then
    for file_so in $(ls -l $targetdir/pto_as/lib64 | grep -E "libaclnn_|libopapi.so"); do
      pto_lib64_path="$targetdir/pto_as/lib64/${file_so}"
      removeopapisoftlink ${pto_lib64_path}
    done
  fi

  # second the headfiles with 1 and 2 level
  if [ -d $targetdir/${architectureDir}/include/aclnnop ]; then
    for file_level1 in $(ls -1 -F ${opapi_include_level1_path} | grep -v [/$] | sed 's/\*$//'); do
      arch_include_path="$targetdir/${architectureDir}/include/aclnnop/${file_level1}"
      removeopapisoftlink ${arch_include_path}

      pto_include_path="$targetdir/pto_as/include/aclnnop/${file_level1}"
      removeopapisoftlink ${pto_include_path}
    done
  fi

  if [ -d $targetdir/${architectureDir}/include/aclnnop/level2 ]; then
    for file_level2 in $(ls -1 -F ${opapi_include_level2_path} | grep -v [/$] | sed 's/\*$//'); do
      arch_include_path="$targetdir/${architectureDir}/include/aclnnop/level2/${file_level2}"
      removeopapisoftlink ${arch_include_path}

      pto_include_path="$targetdir/pto_as/include/aclnnop/level2/${file_level2}"
      removeopapisoftlink ${pto_include_path}
    done
  fi
}

# Keep wheel helpers POSIX-sh compatible: the CANN installer may source this
# file from a /bin/sh process even though direct execution uses Bash.
pto_find_wheel() {
  local wheel_dir="$1"
  local wheel_list wheel_count
  wheel_list=$(find "${wheel_dir}" -maxdepth 1 -type f -name 'ptoas*.whl' -print 2>/dev/null)
  wheel_count=$(printf '%s\n' "${wheel_list}" | sed '/^[[:space:]]*$/d' | wc -l)
  if [ "${wheel_count}" -ne 1 ]; then
    echo "[pto-as] expected exactly one PTOAS wheel in ${wheel_dir} (found ${wheel_count})" >&2
    return 1
  fi
  printf '%s\n' "${wheel_list}"
}

# Top-level names the PTOAS wheel installs into the shared site-packages. Kept
# explicit (as the pypto component does) so uninstall removes exactly PTOAS's
# payload from the shared directory without disturbing sibling components. The
# distribution is named "ptoas"; the wheel's RECORD lists these top-level
# entries: ptoas, ptodsl, TileOps, SoftOps, ptoas.libs, and ptoas-<version>.dist-info.
PTOAS_SITE_PACKAGES_TOPLEVEL="ptoas ptodsl TileOps SoftOps ptoas.libs"

# Console scripts the wheel's [project.scripts] makes pip generate in
# <site-packages>/bin. That directory is shared with sibling components, so
# these are handled file by file and never by replacing the directory.
PTOAS_SITE_PACKAGES_SCRIPTS="ptoas"

# Layout history: wheels used to be unpacked into this private tree. Newer
# packages install into the shared site-packages instead; without this cleanup,
# upgrading or uninstalling over a legacy install strands the old payload.
PTOAS_LEGACY_PRIVATE_PYTHON="tools/ptoas/python"

# Component-owned scratch trees. pip unpacks into the staging tree, and the
# payload already in place is parked in the backup tree while a new one is put
# in. Both live under the component so a crash cannot strand them elsewhere in
# the install root, and both are dot-prefixed so Python never imports them.
PTOAS_WHEEL_STAGING="tools/ptoas/.ptoas-wheel-staging"
PTOAS_WHEEL_BACKUP="tools/ptoas/.ptoas-wheel-backup"

# Remove only PTOAS's own payload from a shared site-packages directory. Never
# removes the shared directory itself: sibling components may own other entries.
# Afterwards, drop the python tree too when PTOAS was its last occupant.
pto_remove_site_payload() {
  local site_packages="$1"
  local pkg script
  [ -d "${site_packages}" ] || return 0
  for pkg in ${PTOAS_SITE_PACKAGES_TOPLEVEL}; do
    rm -rf "${site_packages}/${pkg}"
  done
  rm -rf "${site_packages}"/ptoas-*.dist-info
  for script in ${PTOAS_SITE_PACKAGES_SCRIPTS}; do
    rm -f "${site_packages}/bin/${script}"
  done
  # Best effort: these stay non-empty whenever a sibling component shares the
  # directory. Guarded so a caller running under `set -e` is not aborted by a
  # perfectly normal "directory still in use" outcome.
  rmdir "${site_packages}/bin" 2>/dev/null || true
  rmdir "${site_packages}" 2>/dev/null || true
  rmdir "$(dirname -- "${site_packages}")" 2>/dev/null || true
}

# The CANN per-architecture directories (and the <version>/bin symlink that
# points at one of them) are shared by every component and are created with the
# filelist mode 550, so their owner cannot add entries to them without first
# restoring write permission. These two helpers bracket such an operation and
# put the original mode back, leaving the shared directory exactly as found.
#
# <version>/bin is normally a symlink to <version>/<arch>/bin, so the mode must
# be read through the link: a plain `stat` would report the link's own 777 and
# the restore step would then chmod the real shared directory to 777.
pto_relax_dir_write() {
  local dir="$1" mode
  [ -d "${dir}" ] || return 0
  if [ -w "${dir}" ]; then
    return 0
  fi
  mode=$(stat -L -c %a "${dir}" 2>/dev/null) || return 0
  chmod u+w "${dir}" 2>/dev/null || return 0
  printf '%s\n' "${mode}"
}

pto_restore_dir_write() {
  local dir="$1" mode="$2"
  [ -n "${mode}" ] || return 0
  [ -d "${dir}" ] || return 0
  chmod "${mode}" "${dir}" 2>/dev/null || true
}

# Park PTOAS's installed payload outside the shared site-packages. Replacing a
# payload that spans several top-level entries cannot be done in one atomic
# step, so the version already in place is moved aside first: if anything later
# in the install fails, it is put back instead of the user being left with
# neither the old nor a working new payload. Only PTOAS's own names are touched.
pto_backup_site_payload() {
  local site_packages="$1" backup="$2"
  local pkg entry script
  rm -rf "${backup}"
  mkdir -p "${backup}/scripts" || return 1
  for pkg in ${PTOAS_SITE_PACKAGES_TOPLEVEL}; do
    [ -e "${site_packages}/${pkg}" ] || continue
    mv "${site_packages}/${pkg}" "${backup}/${pkg}" || return 1
  done
  for entry in "${site_packages}"/ptoas-*.dist-info; do
    [ -e "${entry}" ] || continue
    mv "${entry}" "${backup}/$(basename -- "${entry}")" || return 1
  done
  for script in ${PTOAS_SITE_PACKAGES_SCRIPTS}; do
    [ -e "${site_packages}/bin/${script}" ] || continue
    mv "${site_packages}/bin/${script}" "${backup}/scripts/${script}" || return 1
  done
  return 0
}

# Put the parked payload back, touching only the names the backup actually
# holds. A backup that failed partway therefore restores exactly what it moved
# and leaves the entries it never reached alone.
#
# The shared directories are only created when something is actually restored:
# an empty backup (the fresh-install case) must not leave new directories behind.
#
# When a move back fails the backup is deliberately kept: it is the only
# remaining copy of the previous install, so discarding it would turn a
# recoverable failure into data loss. The caller reports the path instead.
pto_restore_site_payload() {
  local site_packages="$1" backup="$2"
  local pkg entry script rc=0
  [ -d "${backup}" ] || return 0
  for pkg in ${PTOAS_SITE_PACKAGES_TOPLEVEL}; do
    [ -e "${backup}/${pkg}" ] || continue
    mkdir -p "${site_packages}" 2>/dev/null || rc=1
    rm -rf "${site_packages}/${pkg}" || rc=1
    mv "${backup}/${pkg}" "${site_packages}/${pkg}" || rc=1
  done
  for entry in "${backup}"/ptoas-*.dist-info; do
    [ -e "${entry}" ] || continue
    mkdir -p "${site_packages}" 2>/dev/null || rc=1
    rm -rf "${site_packages}/$(basename -- "${entry}")" || rc=1
    mv "${entry}" "${site_packages}/" || rc=1
  done
  for script in ${PTOAS_SITE_PACKAGES_SCRIPTS}; do
    [ -e "${backup}/scripts/${script}" ] || continue
    mkdir -p "${site_packages}/bin" 2>/dev/null || rc=1
    mv "${backup}/scripts/${script}" "${site_packages}/bin/${script}" || rc=1
  done
  if [ "${rc}" -eq 0 ]; then
    rm -rf "${backup}"
  else
    echo "[pto-as] could not move the previous PTOAS payload back into ${site_packages}; the only remaining copy is preserved in ${backup}" >&2
  fi
  return "${rc}"
}

# Undo a failed install: drop what this attempt placed and put the parked
# payload back, leaving the shared directory as it was before the attempt. On a
# fresh install the backup is empty, so this reduces to removing what was just
# placed. Reports failure only when the previous payload could not be put back,
# which the caller surfaces along with the preserved backup path.
pto_rollback_site_payload() {
  local site_packages="$1" backup="$2"
  local pkg script
  for pkg in ${PTOAS_SITE_PACKAGES_TOPLEVEL}; do
    rm -rf "${site_packages}/${pkg}"
  done
  rm -rf "${site_packages}"/ptoas-*.dist-info
  for script in ${PTOAS_SITE_PACKAGES_SCRIPTS}; do
    rm -f "${site_packages}/bin/${script}"
  done
  rmdir "${site_packages}/bin" 2>/dev/null || true
  rmdir "${site_packages}" 2>/dev/null || true
  rmdir "$(dirname -- "${site_packages}")" 2>/dev/null || true
  pto_restore_site_payload "${site_packages}" "${backup}"
}

# Move the staged wheel payload into the shared site-packages. Foreign files
# (sibling components) are left alone; the entries are renamed rather than
# copied, so this cannot fail on a full disk.
pto_migrate_site_payload() {
  local staging="$1" site_packages="$2"
  local entry rc
  [ -d "${staging}" ] || return 1
  rc=0
  while IFS= read -r entry; do
    [ -n "${entry}" ] || continue
    if [ "${entry}" = "bin" ]; then
      pto_migrate_site_scripts "${staging}/bin" "${site_packages}/bin" || rc=1
      continue
    fi
    mkdir -p "${site_packages}" || { rc=1; continue; }
    rm -rf "${site_packages}/${entry}"
    mv "${staging}/${entry}" "${site_packages}/${entry}" || rc=1
  done <<EOF
$(cd "${staging}" && ls -A)
EOF
  return "${rc}"
}

# Migrate only PTOAS's own console scripts. The shared bin directory keeps every
# other component's commands.
pto_migrate_site_scripts() {
  local staging_bin="$1" target_bin="$2"
  local script
  [ -d "${staging_bin}" ] || return 0
  mkdir -p "${target_bin}" || return 1
  for script in ${PTOAS_SITE_PACKAGES_SCRIPTS}; do
    [ -e "${staging_bin}/${script}" ] || continue
    mv "${staging_bin}/${script}" "${target_bin}/${script}" || return 1
  done
  return 0
}

# Install the PTOAS wheel into the CANN shared site-packages rather than a
# private tree. The toolkit set_env.sh already prepends
# <version>/python/site-packages to PYTHONPATH and <version>/bin to PATH, so
# both `import ptoas`/`import ptodsl` and the `ptoas` command become available
# after the standard `source set_env.sh` with no PTOAS-specific environment
# script. The launcher stays under the component-owned tools/ptoas/bin tree and
# is exposed through <version>/bin via a relative symlink (mirroring the opapi
# softlink convention used elsewhere in this file).
#
# Every failure fails the install, and every failure after the payload was
# parked puts the previous payload back. Two consequences are deliberate:
# `ptoas` is never reported as installed while the command is missing, and a
# failed upgrade leaves the previously installed version working.
#
# pip unpacks into the component's staging tree first. `pip install --upgrade
# --target` deletes and rebuilds its target's generated-script directory, so
# pointing pip at the shared site-packages would take sibling components'
# commands with it; staging plus a selective migration keeps them intact.
pto_install_wheel() {
  local version_root="$1" share_info_dir="$2"
  local wheel_dir="${version_root}/tools/ptoas/wheels"
  local site_packages="${version_root}/python/site-packages"
  local record="${version_root}/tools/ptoas/.ptoas-python.path"
  local launcher="${version_root}/tools/ptoas/bin/ptoas"
  local staging="${version_root}/${PTOAS_WHEEL_STAGING}"
  local backup="${version_root}/${PTOAS_WHEEL_BACKUP}"
  local bin_dir="${version_root}/bin"
  local bin_link="${bin_dir}/ptoas"
  local python_bin wheel desired_link saved_mode previous_link
  python_bin=$(command -v "${PTOAS_PYTHON:-python3}" 2>/dev/null || true)
  if [ -z "${python_bin}" ]; then
    echo "[pto-as] Python interpreter is unavailable" >&2
    return 1
  fi
  wheel=$(pto_find_wheel "${wheel_dir}") || return 1

  # Nothing has been touched yet, so a failure here leaves the installed version
  # exactly as it was and needs no rollback.
  rm -rf "${staging}"
  mkdir -p "${staging}" || return 1
  if ! "${python_bin}" -m pip install --no-deps --upgrade --target "${staging}" "${wheel}"; then
    rm -rf "${staging}"
    return 1
  fi

  mkdir -p "${site_packages}" "${share_info_dir}" || return 1
  if ! pto_backup_site_payload "${site_packages}" "${backup}"; then
    pto_restore_site_payload "${site_packages}" "${backup}"
    rm -rf "${staging}"
    echo "[pto-as] cannot move the installed PTOAS payload aside in ${site_packages}" >&2
    return 1
  fi
  if ! pto_migrate_site_payload "${staging}" "${site_packages}"; then
    pto_rollback_site_payload "${site_packages}" "${backup}"
    rm -rf "${staging}"
    echo "[pto-as] cannot install the PTOAS payload into ${site_packages}" >&2
    return 1
  fi
  rm -rf "${staging}"

  # Publish the command in the shared <version>/bin, which is a symlink to the
  # per-architecture bin created with mode 550: relax the mode for the link
  # operation and restore it afterwards. The directory is created by the
  # packaging filelist before this helper runs and is never created here, so a
  # missing one is a broken layout rather than something to paper over. These
  # checks fail the install rather than degrading to a warning: a package whose
  # launcher cannot be run is not a valid installation, and "installed but the
  # command is missing" is exactly the failure this change set out to remove.
  if [ ! -f "${launcher}" ]; then
    pto_rollback_site_payload "${site_packages}" "${backup}"
    echo "[pto-as] missing launcher ${launcher}; the ptoas command was not installed" >&2
    return 1
  fi
  if [ ! -d "${bin_dir}" ]; then
    pto_rollback_site_payload "${site_packages}" "${backup}"
    echo "[pto-as] missing shared command directory ${bin_dir}; the ptoas command was not installed" >&2
    return 1
  fi
  desired_link=$(realpath --relative-to="${bin_dir}" "${launcher}" 2>/dev/null || true)
  if [ -z "${desired_link}" ]; then
    pto_rollback_site_payload "${site_packages}" "${backup}"
    echo "[pto-as] cannot resolve ${launcher} relative to ${bin_dir}" >&2
    return 1
  fi
  if [ -L "${bin_link}" ] &&
    [ "$(readlink "${bin_link}" 2>/dev/null || true)" = "${desired_link}" ]; then
    # Already published and already correct: leave it untouched, so nothing
    # below can take the previous install's command away.
    :
  else
    # Remember any pre-existing command entry so a failed publish can put it
    # back. Upgrading from the pre-site-packages layout has none, but a stale
    # or absolute link from an earlier package does.
    previous_link=""
    if [ -L "${bin_link}" ]; then
      previous_link=$(readlink "${bin_link}" 2>/dev/null || true)
    fi
    saved_mode=$(pto_relax_dir_write "${bin_dir}")
    removeopapisoftlink "${bin_link}" || true
    if ! createrelativelysoftlink "${launcher}" "${bin_link}" ||
      [ "$(readlink "${bin_link}" 2>/dev/null || true)" != "${desired_link}" ]; then
      removeopapisoftlink "${bin_link}" || true
      if [ -n "${previous_link}" ]; then
        ln -s -- "${previous_link}" "${bin_link}" 2>/dev/null || true
      fi
      pto_restore_dir_write "${bin_dir}" "${saved_mode}"
      pto_rollback_site_payload "${site_packages}" "${backup}"
      echo "[pto-as] cannot expose the ptoas command as ${bin_link}" >&2
      return 1
    fi
    pto_restore_dir_write "${bin_dir}" "${saved_mode}"
  fi

  # Committed. Record the interpreter and drop the parked payload, plus the
  # pre-site-packages layout an upgrade may have come from. The record is only
  # written here so a failed install cannot leave a launcher pointing at a
  # runtime that is not installed.
  printf '%s\n' "${python_bin}" >"${record}"
  rm -rf "${backup}" "${staging}" "${version_root}/${PTOAS_LEGACY_PRIVATE_PYTHON}"
}

pto_uninstall_wheel() {
  local version_root="$1" share_info_dir="$2"
  local site_packages="${version_root}/python/site-packages"
  local bin_dir="${version_root}/bin"
  local bin_link="${bin_dir}/ptoas"
  local saved_mode
  pto_remove_site_payload "${site_packages}"
  rm -rf "${version_root}/${PTOAS_LEGACY_PRIVATE_PYTHON}" \
         "${version_root}/${PTOAS_WHEEL_STAGING}" \
         "${version_root}/${PTOAS_WHEEL_BACKUP}"
  if [ -L "${bin_link}" ]; then
    saved_mode=$(pto_relax_dir_write "${bin_dir}")
    removeopapisoftlink "${bin_link}" || true
    pto_restore_dir_write "${bin_dir}" "${saved_mode}"
  fi
  rm -f "${version_root}/tools/ptoas/.ptoas-python.path"
}
