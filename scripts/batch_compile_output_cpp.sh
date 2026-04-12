#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${HOME}/cann}"

DEFAULT_SOURCE_DIR="${PTO_SOURCE_DIR:-${REPO_ROOT}}"
SRC_ROOT="${PTOAS_OUT_DIR:-${DEFAULT_SOURCE_DIR}/build/output}"
BUILD_ROOT="${DEFAULT_SOURCE_DIR}/build/output_asm"
LOG_DIR="${DEFAULT_SOURCE_DIR}/build/output_log"

COMPILER="${COMPILER:-}"
PTO_ISA_PATH="${PTO_ISA_PATH:-${PTO_ISA_ROOT:-}}"
EXTRA_ARGS=()

JOBS="${JOBS:-$(nproc)}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
MEM_BASE_DEFINE="${MEM_BASE_DEFINE:-REGISTER_BASE}"
ENABLE_DEFAULT_ARGS=1

print_usage() {
	cat <<'EOF'
批量编译 output 目录下所有 .cpp 文件为 .S，并汇总结果。

用法:
  scripts/batch_compile_output_cpp.sh \
    [--compiler <编译器路径>] \
    [--pto-isa-path <PTO-ISA路径>] \
    [--compile-arg <单个参数>]... \
    [--jobs <并行数>] \
    [--aicore-arch <arch>] \
    [--mem-base-define <宏名>] \
    [--src-root <源码目录>] \
    [--build-root <产物目录>] \
    [--log-dir <日志目录>]

参数说明:
  --compiler, -c         编译器路径。默认优先使用环境变量 COMPILER，
                         其次使用 PATH 中的 bisheng 或
                         ${ASCEND_HOME_PATH}/bin/bisheng
  --pto-isa-path, -p     PTO-ISA 根路径。默认优先使用环境变量
                         PTO_ISA_PATH / PTO_ISA_ROOT。脚本会自动检测 include 目录:
                         1) <PTO-ISA>/include
                         2) <PTO-ISA>/tests/common (存在时自动加入)
                         3) <PTO-ISA>
  --compile-arg          额外编译参数，可重复传入
  --jobs, -j             并行编译任务数，默认: nproc
  --aicore-arch          默认: dav-c220-vec
  --mem-base-define      默认: MEMORY_BASE (可改为 REGISTER_BASE)
  --no-default-args      不使用脚本内置默认参数（仅使用 --compile-arg）
  --src-root             要扫描的 .cpp 根目录，默认: $PTOAS_OUT_DIR
                         或 $PTO_SOURCE_DIR/build/output
  --build-root           .S 产物目录，默认: $PTO_SOURCE_DIR/build/output_asm
  --log-dir              编译日志目录，默认: <build-root>/logs
  --help, -h             显示帮助

推荐先执行:
  source scripts/ptoas_env.sh

默认编译参数来源:
  由 test/npu_validation/scripts/generate_testcase.py 中
  CMAKE_CCE_COMPILE_OPTIONS + target_compile_options(<kernel>) 提取：
  -xcce -fenable-matrix --cce-aicore-enable-tl -fPIC -Xhost-start -Xhost-end
  -mllvm -cce-aicore-function-stack-size=0x8000
  -mllvm -cce-aicore-record-overflow=true
  -mllvm -cce-aicore-addr-transform
  -mllvm -cce-aicore-dcci-insert-for-scalar=false
  --cce-aicore-arch=<arch> -D<MEM_BASE_DEFINE> -std=c++17
EOF
}

die() {
	echo "[ERROR] $*" >&2
	exit 1
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--compiler | -c)
		[[ $# -ge 2 ]] || die "--compiler 缺少参数"
		COMPILER="$2"
		shift 2
		;;
	--pto-isa-path | -p)
		[[ $# -ge 2 ]] || die "--pto-isa-path 缺少参数"
		PTO_ISA_PATH="$2"
		shift 2
		;;
	--compile-arg)
		[[ $# -ge 2 ]] || die "--compile-arg 缺少参数"
		EXTRA_ARGS+=("$2")
		shift 2
		;;
	--jobs | -j)
		[[ $# -ge 2 ]] || die "--jobs 缺少参数"
		JOBS="$2"
		shift 2
		;;
	--aicore-arch)
		[[ $# -ge 2 ]] || die "--aicore-arch 缺少参数"
		AICORE_ARCH="$2"
		shift 2
		;;
	--mem-base-define)
		[[ $# -ge 2 ]] || die "--mem-base-define 缺少参数"
		MEM_BASE_DEFINE="$2"
		shift 2
		;;
	--no-default-args)
		ENABLE_DEFAULT_ARGS=0
		shift
		;;
	--src-root)
		[[ $# -ge 2 ]] || die "--src-root 缺少参数"
		SRC_ROOT="$2"
		shift 2
		;;
	--build-root)
		[[ $# -ge 2 ]] || die "--build-root 缺少参数"
		BUILD_ROOT="$2"
		shift 2
		;;
	--log-dir)
		[[ $# -ge 2 ]] || die "--log-dir 缺少参数"
		LOG_DIR="$2"
		shift 2
		;;
	--help | -h)
		print_usage
		exit 0
		;;
	*)
		die "未知参数: $1 (使用 --help 查看用法)"
		;;
	esac
done

if [[ -z "${COMPILER}" ]]; then
	if command -v bisheng >/dev/null 2>&1; then
		COMPILER="$(command -v bisheng)"
	elif [[ -n "${ASCEND_HOME_PATH:-}" && -x "${ASCEND_HOME_PATH}/bin/bisheng" ]]; then
		COMPILER="${ASCEND_HOME_PATH}/bin/bisheng"
	fi
elif [[ "${COMPILER}" != */* ]] && command -v "${COMPILER}" >/dev/null 2>&1; then
	COMPILER="$(command -v "${COMPILER}")"
fi

[[ -n "${COMPILER}" ]] || die "未找到编译器，请先 source scripts/ptoas_env.sh，或通过 --compiler/COMPILER 指定 bisheng 路径"
[[ -n "${PTO_ISA_PATH}" ]] || die "未找到 PTO-ISA 路径，请通过 --pto-isa-path、PTO_ISA_PATH 或 PTO_ISA_ROOT 指定"
[[ -x "${COMPILER}" ]] || die "编译器不可执行: ${COMPILER}"
[[ -d "${SRC_ROOT}" ]] || die "源码目录不存在: ${SRC_ROOT}"
[[ -d "${PTO_ISA_PATH}" ]] || die "PTO-ISA 路径不存在: ${PTO_ISA_PATH}"
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs 必须为正整数"

if [[ -z "${LOG_DIR}" ]]; then
	LOG_DIR="${BUILD_ROOT}/logs"
fi

mkdir -p "${BUILD_ROOT}" "${LOG_DIR}" || die "创建目录失败"

INCLUDE_DIRS=()
if [[ -f "${PTO_ISA_PATH}/include/pto/pto-inst.hpp" ]]; then
	INCLUDE_DIRS+=("${PTO_ISA_PATH}/include")
fi
if [[ -d "${PTO_ISA_PATH}/tests/common" ]]; then
	INCLUDE_DIRS+=("${PTO_ISA_PATH}/tests/common")
fi
if [[ -f "${PTO_ISA_PATH}/pto/pto-inst.hpp" ]]; then
	INCLUDE_DIRS+=("${PTO_ISA_PATH}")
fi
[[ ${#INCLUDE_DIRS[@]} -gt 0 ]] || die "未找到 pto/pto-inst.hpp，请检查 --pto-isa-path"

if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}/include" ]]; then
	INCLUDE_DIRS+=("${ASCEND_HOME_PATH}/include")
fi
ASCEND_DRIVER_PATH="${ASCEND_DRIVER_PATH:-/usr/local/Ascend/driver}"
if [[ -d "${ASCEND_DRIVER_PATH}/kernel/inc" ]]; then
	INCLUDE_DIRS+=("${ASCEND_DRIVER_PATH}/kernel/inc")
fi

DEFAULT_ARGS=()
if [[ ${ENABLE_DEFAULT_ARGS} -eq 1 ]]; then
	DEFAULT_ARGS=(
		"-xcce"
		"-fenable-matrix"
		"--cce-aicore-enable-tl"
		"--cce-aicore-only"
		"-fPIC"
		"-Xhost-start"
		"-Xhost-end"
		"-mllvm" "-cce-aicore-stack-size=0x8000"
		"-mllvm" "-cce-aicore-function-stack-size=0x8000"
		"-mllvm" "-cce-aicore-record-overflow=true"
		"-mllvm" "-cce-aicore-addr-transform"
		"-mllvm" "-cce-aicore-dcci-insert-for-scalar=false"
		"--cce-aicore-arch=${AICORE_ARCH}"
		"-D${MEM_BASE_DEFINE}"
		"-std=c++17"
	)
	if [[ "${AICORE_ARCH}" == dav-l310* || "${AICORE_ARCH}" == dav-l311* ]]; then
		FILTERED_DEFAULT_ARGS=()
		i=0
		while [[ ${i} -lt ${#DEFAULT_ARGS[@]} ]]; do
			if [[ "${DEFAULT_ARGS[${i}]}" == "-mllvm" ]] && [[ $((i + 1)) -lt ${#DEFAULT_ARGS[@]} ]] &&
				[[ "${DEFAULT_ARGS[$((i + 1))]}" == "-cce-aicore-stack-size=0x8000" ]]; then
				i=$((i + 2))
				continue
			fi
			FILTERED_DEFAULT_ARGS+=("${DEFAULT_ARGS[${i}]}")
			i=$((i + 1))
		done
		DEFAULT_ARGS=("${FILTERED_DEFAULT_ARGS[@]}")
	fi
fi

declare -a CPP_FILES=()
while IFS= read -r -d '' file; do
	CPP_FILES+=("${file}")
done < <(find "${SRC_ROOT}" -type f -name "*.cpp" -print0 | sort -z)

TOTAL_COUNT=${#CPP_FILES[@]}
[[ ${TOTAL_COUNT} -gt 0 ]] || die "未在 ${SRC_ROOT} 下找到 .cpp 文件"

STATUS_FILE="$(mktemp "${BUILD_ROOT}/compile_status.XXXXXX")" || die "创建状态文件失败"
trap 'rm -f "${STATUS_FILE}"' EXIT

record_compile_status() {
	local status="$1"
	local rel_path="$2"
	printf '%s\t%s\n' "${status}" "${rel_path}" >>"${STATUS_FILE}"
}

cleanup_work_dir() {
	local work_dir="$1"
	[[ -n "${work_dir}" ]] && rm -rf -- "${work_dir}"
}

get_log_failure_reason() {
	local log_path="$1"
	local excerpt

	excerpt="$(grep -E -i 'error:|fatal:|undefined reference|undefined symbol|undeclared identifier|exception|traceback|failed' "${log_path}" | tail -n 5 || true)"
	if [[ -z "${excerpt}" ]]; then
		excerpt="$(tail -n 10 "${log_path}" 2>/dev/null || true)"
	fi
	printf '%s' "${excerpt}"
}

find_generated_output() {
	local work_dir="$1"
	local src_stem="$2"
	local candidate

	for candidate in \
		"${work_dir}/${src_stem}.o" \
		"${work_dir}/${src_stem}.S" \
		"${work_dir}/${src_stem}.s"; do
		if [[ -f "${candidate}" ]]; then
			printf '%s\n' "${candidate}"
			return 0
		fi
	done

	find "${work_dir}" -maxdepth 1 -type f \( -name "*.o" -o -name "*.S" -o -name "*.s" \) | head -n 1
}

write_rebuild_cmd() {
	local cmd_path="$1"
	local asm_path="$2"
	local src_stem="$3"
	shift 3
	local -a cmd=("$@")
	local cmd_text=""
	local arg

	for arg in "${cmd[@]}"; do
		printf -v cmd_text '%s %q' "${cmd_text}" "${arg}"
	done
	cmd_text="${cmd_text# }"

	{
		echo "#!/usr/bin/env bash"
		echo
		echo "set -euo pipefail"
		echo
		printf 'ASM_PATH=%q\n' "${asm_path}"
		printf 'SRC_STEM=%q\n' "${src_stem}"
		printf 'WORK_ROOT=%q\n' "${BUILD_ROOT}"
		echo
		echo 'WORK_DIR="$(mktemp -d "${WORK_ROOT}/tmp_rebuild.XXXXXX")"'
		echo 'trap '\''rm -rf -- "${WORK_DIR}"'\'' EXIT'
		echo
		echo 'cd "${WORK_DIR}"'
		echo "${cmd_text}"
		echo
		echo 'GENERATED_FILE=""'
		echo 'for candidate in "${WORK_DIR}/${SRC_STEM}.o" "${WORK_DIR}/${SRC_STEM}.S" "${WORK_DIR}/${SRC_STEM}.s"; do'
		echo '  if [[ -f "${candidate}" ]]; then'
		echo '    GENERATED_FILE="${candidate}"'
		echo '    break'
		echo '  fi'
		echo 'done'
		echo
		echo 'if [[ -z "${GENERATED_FILE}" ]]; then'
		echo '  GENERATED_FILE="$(find "${WORK_DIR}" -maxdepth 1 -type f \( -name "*.o" -o -name "*.S" -o -name "*.s" \) | head -n 1)"'
		echo 'fi'
		echo
		echo 'if [[ -z "${GENERATED_FILE}" || ! -f "${GENERATED_FILE}" ]]; then'
		echo '  echo "[ERROR] 编译成功但未找到输出文件，期望类型: .o/.S/.s" >&2'
		echo '  exit 1'
		echo 'fi'
		echo
		echo 'mkdir -p "$(dirname -- "${ASM_PATH}")"'
		echo 'mv -f -- "${GENERATED_FILE}" "${ASM_PATH}"'
		printf 'echo "已更新: %s"\n' "${asm_path}"
	} >"${cmd_path}" || return 1

	chmod +x "${cmd_path}"
}

compile_one() {
	local src="$1"
	local rel_path asm_path log_path cmd_path src_base src_stem work_dir generated_file
	local -a cmd=()

	rel_path="${src#"${SRC_ROOT}/"}"
	asm_path="${BUILD_ROOT}/${rel_path%.cpp}.S"
	log_path="${LOG_DIR}/${rel_path%.cpp}.log"
	cmd_path="$(dirname -- "${log_path}")/cmd.sh"
	src_base="$(basename -- "${src}")"
	src_stem="${src_base%.cpp}"

	mkdir -p "$(dirname -- "${asm_path}")" "$(dirname -- "${log_path}")" || {
		record_compile_status "FAIL" "${rel_path}"
		return 0
	}

	cmd=("${COMPILER}")
	if [[ ${#DEFAULT_ARGS[@]} -gt 0 ]]; then
		cmd+=("${DEFAULT_ARGS[@]}")
	fi
	if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
		cmd+=("${EXTRA_ARGS[@]}")
	fi
	local inc
	for inc in "${INCLUDE_DIRS[@]}"; do
		cmd+=("-I${inc}")
	done
	cmd+=("-c" "${src}")

	if ! write_rebuild_cmd "${cmd_path}" "${asm_path}" "${src_stem}" "${cmd[@]}"; then
		record_compile_status "FAIL" "${rel_path}"
		return 0
	fi

	echo "[BUILD] ${rel_path}"
	work_dir="$(mktemp -d "${BUILD_ROOT}/tmp_compile.XXXXXX")" || {
		record_compile_status "FAIL" "${rel_path}"
		return 0
	}

	if ! (cd "${work_dir}" && "${cmd[@]}") >"${log_path}" 2>&1; then
		cleanup_work_dir "${work_dir}"
		record_compile_status "FAIL" "${rel_path}"
		return 0
	fi

	generated_file="$(find_generated_output "${work_dir}" "${src_stem}")"

	if [[ -z "${generated_file}" || ! -f "${generated_file}" ]]; then
		{
			echo
			echo "[ERROR] 编译成功但未找到输出文件，期望类型: .o/.S/.s"
			echo "[ERROR] 临时目录: ${work_dir}"
		} >>"${log_path}"
		cleanup_work_dir "${work_dir}"
		record_compile_status "FAIL" "${rel_path}"
		return 0
	fi

	if mv -f -- "${generated_file}" "${asm_path}"; then
		cleanup_work_dir "${work_dir}"
		record_compile_status "OK" "${rel_path}"
	else
		{
			echo
			echo "[ERROR] 输出重命名失败: ${generated_file} -> ${asm_path}"
		} >>"${log_path}"
		cleanup_work_dir "${work_dir}"
		record_compile_status "FAIL" "${rel_path}"
	fi
}

START_TIME="$(date +%s)"

echo "[INFO] 编译器: ${COMPILER}"
echo "[INFO] 源目录: ${SRC_ROOT}"
echo "[INFO] 产物目录(.S): ${BUILD_ROOT}"
echo "[INFO] 日志目录: ${LOG_DIR}"
echo "[INFO] PTO-ISA: ${PTO_ISA_PATH}"
echo "[INFO] 并行度: ${JOBS}"
echo "[INFO] include: ${INCLUDE_DIRS[*]}"
if [[ ${ENABLE_DEFAULT_ARGS} -eq 1 ]]; then
	echo "[INFO] 默认参数(来自 generate_testcase.py): ${DEFAULT_ARGS[*]}"
else
	echo "[INFO] 默认参数: 已禁用 (--no-default-args)"
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
	echo "[INFO] 额外参数: ${EXTRA_ARGS[*]}"
fi
echo "[INFO] 文件总数: ${TOTAL_COUNT}"
echo

running_jobs=0
for src in "${CPP_FILES[@]}"; do
	compile_one "${src}" &
	running_jobs=$((running_jobs + 1))
	if [[ ${running_jobs} -ge ${JOBS} ]]; then
		wait -n
		running_jobs=$((running_jobs - 1))
	fi
done

wait

SUCCESS_COUNT="$(awk -F'\t' '$1=="OK"{c++} END{print c+0}' "${STATUS_FILE}")"
FAIL_COUNT="$(awk -F'\t' '$1=="FAIL"{c++} END{print c+0}' "${STATUS_FILE}")"

declare -a FAILED_FILES=()
while IFS= read -r failed; do
	[[ -n "${failed}" ]] && FAILED_FILES+=("${failed}")
done < <(awk -F'\t' '$1=="FAIL"{print $2}' "${STATUS_FILE}")

END_TIME="$(date +%s)"
ELAPSED="$((END_TIME - START_TIME))"

echo
echo "========== 编译汇总 =========="
echo "总文件数 : ${TOTAL_COUNT}"
echo "成功数   : ${SUCCESS_COUNT}"
echo "失败数   : ${FAIL_COUNT}"
echo "耗时(秒) : ${ELAPSED}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
	failure_reason=""
	echo
	echo "失败文件列表:"
	for f in "${FAILED_FILES[@]}"; do
		echo "  - ${f} (log: ${LOG_DIR}/${f%.cpp}.log)"
		failure_reason="$(get_log_failure_reason "${LOG_DIR}/${f%.cpp}.log")"
		if [[ -n "${failure_reason}" ]]; then
			while IFS= read -r line; do
				[[ -n "${line}" ]] || continue
				echo "    reason: ${line}"
			done <<<"${failure_reason}"
		fi
	done
	exit 1
fi

echo "[INFO] 全部编译成功"
exit 0
