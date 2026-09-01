# PR171 CANN Wheel 安装包恢复设计

## 背景

PR171 将原有的 CANN CPack/makeself 安装包替换成了自定义 Shell 自解压包，其 payload 只有一个经过修复的 PTOAS wheel：`tools/ptoas/wheels/ptoas*.whl`。

该 wheel 内包含 Python 包、PTOAS/MLIR 原生扩展、TileOps/SoftOps 资源，以及 auditwheel 打入的 LLVM/MLIR 动态库。但是，由此生成的 `.run` 不再具备标准 CANN 安装包所要求的安装接口、元数据、SDK 文件和生命周期管理能力。

本次修复的目标是：继续以 wheel 作为 PTOAS 唯一的运行时实现，同时恢复 master 分支和 PyPTO 所使用的 CANN 安装包交付契约。

## 目标

- 使用 CANN CMake/CPack 和 makeself 生成 PTOAS `.run`。
- 保留 PR171 的 wheel 构建、payload 校验和 auditwheel 修复流程。
- 在 CANN payload 中只放置一个修复后的 PTOAS wheel。
- 恢复标准 CANN 安装、检查、升级、卸载、多版本和元数据管理能力。
- 恢复 PTOAS 版本头文件以及安装后的 CMake package 文件。
- 提供由 wheel 驱动的 `tools/ptoas/bin/ptoas` 命令。
- 不在外层 payload 中重复放置 wheel 已经包含的 LLVM/MLIR/PTOAS 原生库，避免重复文件和依赖解析不一致。

## 非目标

- 不恢复旧的独立原生程序 `ptoas.real`。
- 不从构建缓存复制 LLVM/MLIR 动态库到 `tools/ptoas/lib`。
- 不修改 PTOAS 编译器行为或 wheel 内容。
- 不重新设计通用 CANN 安装框架。
- 不新增超出 master 现有行为范围的 RPM/DEB 特性；修复后的 PR171 必须保留并对齐
  master 当前已有的 `--pkg-type=run|rpm|deb|all` 打包能力、依赖元数据转换、包名和
  产物命名规则。

## 选定方案

### 构建与打包流程

`build.sh --pkg` 执行以下步骤：

1. 配置并编译 PTOAS。
2. 构建并校验 PTOAS wheel。
3. 使用 auditwheel 修复 wheel，并再次校验。
4. 将修复后的 wheel 放到构建目录中的稳定位置，供 CMake 打包使用。
5. 将 wheel 路径和 CANN package type 传给 CMake 打包配置。
6. 执行 CMake `package` target，通过 CANN CPack/makeself 在 `build_out` 下生成 `cann-pto-as_<version>_linux-<arch>.run`。

PR171 的自定义 `make_ptoas_run()` 不再进入实际打包链路，也不能作为失败后的 fallback，否则会再次静默丢失 CANN 安装包契约。

当 `--pkg-type` 为 `rpm`、`deb` 或 `all` 时，仍由 CANN CPack 生成对应的 RPM/DEB
产物；这些产物需要与 master 保持一致，包括运行依赖的版本表达、组件包名、文件名和
输出目录。此次修复只是把 PR171 的 wheel 纳入同一套 CPack 安装树，不改变 master
已有的 RPM/DEB 语义。

### CANN 外层 payload

CMake 的 `pto_as` 安装组件包含：

- `tools/ptoas/wheels/ptoas*.whl`：一个修复后的 wheel。
- `tools/ptoas/bin/ptoas`：一个小型启动器，用于定位已安装的 CANN 根目录，并从受管理的 wheel 运行环境中执行 `ptoas._cli`。
- `aarch64-linux/include/version/pto_as_version.h`：由 CANN package 描述生成；其他架构使用相应的架构目录。
- `lib*/cmake/PTOAS/PTOASConfig.cmake` 及导出的 targets 文件。
- `share/info/pto_as/version.info` 和 `scene.info`。
- PTOAS 专用安装脚本及 cann-cmake 公共脚本，包括由打包工具生成的 `filelist.csv`。

外层 payload 不再包含 `tools/ptoas/lib/libLLVM*`、`libMLIR*` 或原生 `ptoas.real`，因为这些运行实现已经由修复后的 wheel 提供。

### wheel 安装环境

CANN 安装脚本使用选定的 Python 解释器，将 wheel 安装到：

```text
<CANN 版本根目录>/tools/ptoas/python
```

该目录只由 PTOAS 组件管理。安装命令为：

```text
python -m pip install --no-deps --upgrade --target <目录> <wheel>
```

使用组件私有目录可以避免修改系统或用户的 site-packages，同时保证卸载过程可控、可复现。随包提供的启动器会把该目录加入 `PYTHONPATH`，然后执行 `ptoas._cli`，从而保留 `ptoas --version` 和正常的编译器命令行行为。

安装时使用的 Python 解释器路径记录在 `share/info/pto_as` 下。启动器默认使用该解释器；如果解释器或私有 wheel 环境不存在，则输出明确错误，不会误用系统中的其他 `ptoas`。

### CANN 生命周期集成

恢复后的 `pto_as.xml` 声明 wheel、启动器、元数据、安装脚本、版本头文件、CMake SDK 文件以及所需目录。标准安装入口仍为 `share/info/pto_as/script/install.sh`。

安装和升级流程如下：

1. 由标准 CANN parser 复制安装包文件，并维护 filelist 和版本元数据。
2. 将 wheel 安装到组件私有目录。
3. 通过 `ptoas --version` 校验安装后的启动器。
4. 如果 Python、pip、wheel 或安装后的命令入口不可用，则操作失败，不得报告部分安装成功。

卸载时删除组件私有 wheel 目录和 Python 解释器记录，然后由标准 CANN 流程删除其余文件。升级使用标准 CANN 的卸载再安装流程，避免旧版本 wheel 文件残留。

`--run`、`--full` 和 `--devel` 均安装相同的运行时 wheel；`--devel` 额外选择 SDK 文件。外层 makeself/CANN 框架继续支持：

- `--check`
- `--pre-check`
- `--upgrade`
- `--install-for-all`
- `--docker-root`
- `--setenv`
- 包内容查看和解压参数

## 错误处理

- 如果修复后的 PTOAS wheel 数量不是一个，打包直接失败。
- 如果 CANN package 资源或生成的版本元数据缺失，打包直接失败。
- 如果没有带 pip 的可用 Python 解释器，安装在输出成功信息前失败。
- 如果 wheel 与目标架构或 Python 版本不兼容，由 pip 的兼容性校验拒绝安装。
- 如果 wheel 安装失败，删除本次新建的私有 wheel 目录，避免留下看似可用的部分安装。
- 如果解释器或私有 wheel 环境缺失，启动器输出明确错误，不回退到系统中的其他 `ptoas`。

## 测试方案

### 快速契约测试

仓库内增加测试，检查：

- 所有必需的 package 资源均存在。
- CMake 打包规则会安装修复后的 wheel、标准元数据和安装脚本。
- package 描述包含 wheel、启动器、版本头文件、CMake SDK 文件和标准 CANN 安装入口。
- `build.sh --pkg` 调用 CMake `package` target，不再调用自定义自解压打包函数。
- `--pkg-type=run|rpm|deb|all` 的行为、包名、文件名和依赖元数据与 master 保持一致。
- 安装和卸载脚本使用组件私有 wheel 目录，不向系统或用户 site-packages 安装。

该测试必须先编写，并在未修改的 PR171 上以预期原因失败。

### 脚本级测试

使用合成 wheel 和临时安装根目录验证：

- wheel 数量必须严格等于一个。
- pip 参数指向组件私有 `--target` 目录。
- 启动器能正确解析运行环境，并在运行环境缺失时明确失败。
- 卸载能够清理私有 wheel 目录和 Python 解释器记录。

### 1211 端到端验收

在 1211 aarch64 机器上执行：

1. 从干净的打包构建目录运行 `./build.sh --pkg`。
2. 确认构建退出码为 0，并记录 `.run` SHA256。
3. 检查 `.run --help` 是否提供标准 CANN 参数。
4. 解包 `.run`，检查预期的 payload 路径。
5. 确认只有一个 wheel，且外层不存在重复的 LLVM/MLIR DSO 目录。
6. 使用 `--full --install-path=...` 安装到临时目录。
7. 执行安装后的 `tools/ptoas/bin/ptoas --version` 和 CLI help 冒烟测试。
8. 执行 `--upgrade`，然后执行 `--uninstall`，确认私有 Python 目录、启动器、元数据和安装记录按预期清理。

## 验收标准

- 生成物是标准 CANN makeself `.run`，不是 PR171 的自定义归档。
- `--pkg-type=run|rpm|deb|all` 均可用；RPM/DEB 产物与 master 的命名、依赖和输出位置一致。
- 外层 payload 包含标准安装脚本、`version.info`、`scene.info`、版本头文件、CMake package 文件、启动器和一个修复后的 wheel。
- 外层 payload 不重复包含 wheel 已有的 LLVM/MLIR 动态库。
- 安装过程不向系统或用户 Python site-packages 写入 PTOAS。
- 安装后的启动器能够在 1211 上成功执行 `ptoas --version`。
- 标准检查、安装、升级和卸载流程能够在临时安装目录中成功完成。
