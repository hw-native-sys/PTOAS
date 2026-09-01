#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

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

pto_install_wheel() {
  local version_root="$1" share_info_dir="$2"
  local wheel_dir="${version_root}/tools/ptoas/wheels"
  local python_dir="${version_root}/tools/ptoas/python"
  local record="${version_root}/tools/ptoas/.ptoas-python.path"
  # Keep this function POSIX-sh compatible: the CANN installer may source this
  # file from a /bin/sh process even though direct execution uses Bash.
  local wheel_list wheel_count wheel
  if [ ! -d "${wheel_dir}" ]; then
    rm -rf "${python_dir}"
    rm -f "${record}"
    echo "[pto-as] no PTOAS wheel directory; skipping optional wheel runtime"
    return 0
  fi
  wheel_list=$(find "${wheel_dir}" -maxdepth 1 -type f -name 'ptoas*.whl' -print 2>/dev/null)
  wheel_count=$(printf '%s\n' "${wheel_list}" | sed '/^[[:space:]]*$/d' | wc -l)
  if [ "${wheel_count}" -eq 0 ]; then
    rm -rf "${python_dir}"
    rm -f "${record}"
    echo "[pto-as] no PTOAS wheel found; skipping optional wheel runtime"
    return 0
  fi
  if [ "${wheel_count}" -ne 1 ]; then
    echo "[pto-as] expected exactly one PTOAS wheel in ${wheel_dir}" >&2
    rm -rf "${python_dir}"
    return 1
  fi
  local python_bin
  python_bin=$(command -v "${PTOAS_PYTHON:-python3}" 2>/dev/null || true)
  if [ -z "${python_bin}" ]; then
    echo "[pto-as] Python interpreter is unavailable" >&2
    return 1
  fi
  wheel=${wheel_list}
  rm -rf "${python_dir}"
  mkdir -p "${python_dir}" "${share_info_dir}"
  if ! "${python_bin}" -m pip install --no-deps --upgrade --target "${python_dir}" "${wheel}"; then
    rm -rf "${python_dir}" "${record}"
    return 1
  fi
  printf '%s\n' "${python_bin}" >"${record}"
}

pto_uninstall_wheel() {
  local version_root="$1" share_info_dir="$2"
  rm -rf "${version_root}/tools/ptoas/python"
  rm -f "${version_root}/tools/ptoas/.ptoas-python.path"
}
