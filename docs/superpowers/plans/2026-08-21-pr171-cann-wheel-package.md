# PR171 CANN Wheel 安装包恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在保留 PR171 wheel 运行时的前提下，恢复与 master 一致的 CANN run/RPM/DEB 打包、安装、升级和卸载契约。

**架构：** `build.sh` 继续负责构建和 auditwheel repair，但将修复后的 wheel 交给 CMake 安装规则，最终统一调用 CANN CPack `package` target。CANN 安装脚本把 wheel 安装到 `<CANN版本根>/tools/ptoas/python`，随包启动器通过该私有目录运行 `ptoas._cli`，不写入系统或用户 site-packages。

**技术栈：** Bash、Python unittest、CMake/CPack、cann-cmake、makeself、pip/auditwheel。

## 全局约束

- 基线提交固定为 PR171 head `fc455e690ab68878634b9819c2f9c6046c795433`。
- 保留 `--pkg-type=run|rpm|deb|all`，命名、依赖元数据及输出目录与 master 一致。
- payload 中必须只有一个 `ptoas*.whl`。
- wheel 安装目标固定为 `<CANN版本根>/tools/ptoas/python`。
- 不恢复 `ptoas.real`，不在外层复制 LLVM/MLIR DSO。
- 不向全局或用户 Python site-packages 安装 PTOAS。

---

### 任务 1：建立打包契约测试

**文件：**
- 新建：`test/package/test_cann_package_contract.py`

**接口：**
- 输入：仓库根目录下的 `build.sh`、`CMakeLists.txt`、`cmake/package.cmake`、`version.cmake` 和 `scripts/package/`。
- 输出：可由 `python -m unittest test.package.test_cann_package_contract -v` 独立运行的静态契约测试。

- [ ] **步骤 1：编写失败测试**

  测试检查标准 CANN 资源存在、`build.sh` 通过 CMake package target 打包、CMake 安装 wheel/启动器/元数据、package XML 声明 wheel 和 SDK 文件、RPM/DEB 配置与 master 对齐，并禁止活动链路调用 `make_ptoas_run`。

- [ ] **步骤 2：确认 RED**

  运行：`python -m unittest test.package.test_cann_package_contract -v`

  预期：因 PR171 缺少 `scripts/package/package.py`、`pto_as.xml`、安装脚本、`version.cmake`，且仍调用 `make_ptoas_run` 而失败。

- [ ] **步骤 3：提交测试**

  ```bash
  git add test/package/test_cann_package_contract.py
  git commit -m "test: define CANN wheel package contract"
  ```

### 任务 2：恢复 CANN package 元数据与 CPack 入口

**文件：**
- 修改：`CMakeLists.txt`
- 修改：`cmake/package.cmake`
- 新建：`version.cmake`
- 恢复并修改：`scripts/package/package.py`
- 恢复：`scripts/package/module/ascend/EngineeringCommon.xml`
- 恢复：`scripts/package/module/ascend/EngineeringFiles.xml`
- 新建：`scripts/package/pto_as/pto_as.xml`
- 恢复：`scripts/package/pto_as/rpm_deb/custom_postinst.sh`

**接口：**
- 输入：`PTOAS_WHEEL_FILE`（修复后 wheel 的绝对路径）、`PACKAGE_TYPE`。
- 输出：CMake `package` target；run/RPM/DEB/all CPack 产物。

- [ ] **步骤 1：恢复最小 CANN 打包入口**

  在 `CMakeLists.txt` 中加载 package helpers、`version.cmake` 和 `pack_built_in()`；在 `cmake/package.cmake` 中校验 `PTOAS_WHEEL_FILE` 并用 `install(FILES)` 安装 wheel、标准脚本、版本信息、CMake 配置及启动器。

- [ ] **步骤 2：定义 package XML**

  `pto_as.xml` 声明标准 `package_info`、版本头、scene.info、脚本、wheel、启动器、CMake 文件和目录；不声明 `ptoas.real` 或外层 DSO 目录。

- [ ] **步骤 3：运行契约测试至对应断言通过**

  运行：`python -m unittest test.package.test_cann_package_contract -v`

- [ ] **步骤 4：提交**

  ```bash
  git add CMakeLists.txt cmake/package.cmake version.cmake scripts/package
  git commit -m "build: restore CANN CPack metadata for wheel payload"
  ```

### 任务 3：实现私有 wheel 安装与启动器

**文件：**
- 新建：`scripts/package/pto_as/bin/ptoas`
- 恢复并修改：`scripts/package/pto_as/scripts/install.sh`
- 恢复并修改：`scripts/package/pto_as/scripts/pto_install.sh`
- 恢复并修改：`scripts/package/pto_as/scripts/pto_uninstall.sh`
- 恢复：`scripts/package/pto_as/scripts/cleanup.sh`
- 恢复：`scripts/package/pto_as/scripts/help.info`
- 恢复：`scripts/package/pto_as/scripts/pto_common.sh`
- 恢复：`scripts/package/pto_as/scripts/pto_custom_install.sh`
- 恢复：`scripts/package/pto_as/scripts/pto_custom_uninstall.sh`
- 恢复：`scripts/package/pto_as/scripts/uninstall.sh`
- 恢复：`scripts/package/pto_as/scripts/ver_check.sh`
- 新建：`test/package/test_wheel_runtime_scripts.py`

**接口：**
- `pto_install_wheel <version-root> <share-info-dir>`：安装唯一 wheel 到 `<version-root>/tools/ptoas/python` 并记录 Python 路径。
- `pto_uninstall_wheel <version-root> <share-info-dir>`：删除私有 wheel 目录和解释器记录。
- `tools/ptoas/bin/ptoas [args...]`：从相邻私有目录加载 `ptoas._cli`。

- [ ] **步骤 1：先写脚本级失败测试并确认 RED**

  使用临时目录、合成 wheel 和记录参数的假 Python，验证唯一 wheel、`--target` 参数、失败清理、启动器缺失环境错误和卸载清理。

- [ ] **步骤 2：实现最小安装/卸载函数与启动器**

  安装使用 `python -m pip install --no-deps --upgrade --target`，禁止 `--user`；安装失败删除私有目标。启动器读取解释器记录，设置 `PYTHONPATH`，执行 `-m ptoas._cli`。

- [ ] **步骤 3：确认 GREEN**

  运行：`python -m unittest test.package.test_wheel_runtime_scripts -v`

- [ ] **步骤 4：提交**

  ```bash
  git add scripts/package/pto_as test/package/test_wheel_runtime_scripts.py
  git commit -m "feat: install PTOAS wheel in CANN private runtime"
  ```

### 任务 4：切换 build.sh 到统一 CPack 链路

**文件：**
- 修改：`build.sh`
- 修改：`cmake/superbuild/CMakeLists.txt`（仅在构建入口仍使用 superbuild 时传递 wheel 路径）

**接口：**
- `stage_ptoas_wheel`：输出唯一修复 wheel 的绝对路径。
- `package`：以 `PTOAS_WHEEL_FILE` 和 `PACKAGE_TYPE` 重新配置后执行 CMake `package` target。

- [ ] **步骤 1：修改 build.sh**

  wheel 仍构建到 `build/wheelhouse`，但不再生成自定义 Shell archive；将绝对 wheel 路径传入 CMake，执行 `cmake --build build --target package`，最后列出 `build_out`。

- [ ] **步骤 2：运行静态契约测试确认 GREEN**

  运行：`python -m unittest discover -s test/package -v`

- [ ] **步骤 3：运行 Shell/CMake 语法检查**

  ```bash
  bash -n build.sh scripts/package/pto_as/bin/ptoas scripts/package/pto_as/scripts/*.sh
  cmake -P scripts/package/check_package_contract.cmake
  ```

  若无需独立 CMake 检查脚本，则使用契约测试对 CMake 文本和 1211 configure 进行验证。

- [ ] **步骤 4：提交**

  ```bash
  git add build.sh cmake/superbuild/CMakeLists.txt
  git commit -m "build: package repaired wheel through CANN CPack"
  ```

### 任务 5：1211 端到端验证

**文件：**
- 不修改生产文件；使用远端临时工作区和日志。

**接口：**
- 输入：当前分支补丁。
- 输出：run/RPM/DEB 产物、SHA256、解包清单和生命周期验证日志。

- [ ] **步骤 1：同步补丁并运行快速测试**

  在 `/home/zoujiangjiang/pr171_cann_fix_20260821` 应用当前分支补丁，运行 package unittest 和 `bash -n`。

- [ ] **步骤 2：构建 run 包**

  ```bash
  ./build.sh --pkg --pkg-type=run
  ```

  预期：退出码 0，生成标准 `cann-pto-as_9.2.0_linux-aarch64.run`。

- [ ] **步骤 3：验证内容和 help**

  检查标准 CANN 参数、脚本、version/scene、版本头、CMake 配置、启动器和唯一 wheel；确认外层没有 `libLLVM*`/`libMLIR*`。

- [ ] **步骤 4：验证安装、升级、卸载**

  安装到临时前缀，运行安装后的 `ptoas --version`，执行 upgrade 和 uninstall，确认私有 Python 目录和安装记录清理。

- [ ] **步骤 5：验证 RPM/DEB**

  分别执行 `--pkg-type=rpm`、`--pkg-type=deb`（或 `all`），检查产物命名和依赖元数据与 master 规则一致。若 1211 缺少系统打包工具，记录精确依赖错误，并至少通过 CPack 配置和生成器检查。

- [ ] **步骤 6：运行全部快速测试并提交最终修正**

  ```bash
  python -m unittest discover -s test/package -v
  git diff --check
  ```
