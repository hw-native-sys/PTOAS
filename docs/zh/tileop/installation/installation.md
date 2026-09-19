# 安装指南

## 安装说明

本文介绍通过CANN PTO-AS `run`安装包安装PTO-AS组件并部署PTOAS工具的方法。

PTOAS随PTO-AS组件一起安装。安装PTO-AS组件时，安装程序会部署PTOAS启动程序、运行时依赖和PTOAS wheel，并将PTOAS wheel安装到CANN版本根目录下的共享`python/site-packages`中，无需在当前Python环境中单独执行`pip install`。

## 安装前准备

安装前，请确认当前环境已安装Python 3.8及以上版本，并获取与当前系统架构匹配的PTO-AS `run`安装包。

## 执行安装

进入PTO-AS安装包所在目录，执行安装命令。

以下示例以9.2.0版本、x86_64架构的安装包为例：

```bash
bash cann-pto-as_9.2.0_linux-x86_64.run --full --install-path=xxx
```

其中：

- `cann-pto-as_9.2.0_linux-x86_64.run`为PTO-AS安装包名称，实际使用时请以获取到的安装包名称为准。
- `--full`表示使用`full`模式安装。
- `--install-path=xxx`用于指定安装根目录，请将`xxx`替换为实际安装路径。

安装脚本支持`run`、`full`和`devel`三种安装模式，一次安装只能指定其中一种。

### 说明

本文以 `full` 模式为例，不展开说明 `run`、`full` 和 `devel` 三种安装模式的内容差异。

## 配置环境变量

### 安装时使用 --setenv

安装时可以使用`--setenv`配置运行所需的环境变量：

```bash
bash cann-pto-as_9.2.0_linux-x86_64.run --full --install-path=xxx --setenv
```

使用 `--setenv` 安装之后新打开的登录/交互式 shell 会自动加载 PTO-AS 运行所需的环境；当前已打开的 shell 以及脚本、CI 等非交互式场景不受影响，仍需按下文手动加载环境。

### 安装时未使用 --setenv

如果安装时未使用`--setenv`，在使用PTO-AS前需要手动加载环境变量：

```bash
source <version_root>/set_env.sh
```

`<version_root>` 表示 CANN 版本根目录。

### 默认安装路径

如果安装时未指定`--install-path`，默认安装根目录如下：

- 普通用户：`${HOME}/Ascend`
- root用户：`/usr/local/Ascend`

## 其他安装选项

PTO-AS `run`安装包还支持以下常用选项：

- `--install-for-all`：为所有用户安装。
- `--quiet`：静默安装，跳过交互确认。
- `--pre-check`：执行安装前检查。

例如，执行安装前检查：

```bash
bash cann-pto-as_9.2.0_linux-x86_64.run --pre-check
```

## 查看PTOAS运行文件

安装PTO-AS组件时，安装程序会自动安装PTOAS wheel，并部署PTOAS相关运行文件。

以下 `<version_root>` 表示 CANN 版本根目录。默认布局中 `<version_root>` 为 `<install_path>/cann`，`<install_path>` 为 `--install-path` 指定的安装根目录；安装时也可以直接指定版本目录，其实际位置以安装结果及环境脚本为准。

```text
<version_root>/
├── bin/
│   └── ptoas -> ../tools/ptoas/bin/ptoas
├── python/site-packages/
│   ├── ptoas/
│   ├── ptodsl/
│   └── ...
└── tools/ptoas/
    ├── bin/ptoas
    └── wheels/
```

`tools/ptoas/bin/ptoas` 为启动程序，`tools/ptoas/wheels` 保存随安装包部署的 wheel；Python 包及其运行依赖部署在共享的 `python/site-packages`，而不是 PTOAS 专属的 Python 子目录。安装器自动安装 wheel，无需另行安装到用户当前 Python 环境。

## 检查安装结果

加载环境脚本后，确认命令和 Python 包可以使用：

```bash
command -v ptoas
ptoas --version
python3 -c 'import ptoas; import ptodsl'
```

Python 导入检查应使用与安装 wheel 兼容的解释器。继续使用 [vec_add.pto](../examples/vec_add.pto) 验证实际编译，在该示例所在目录执行：

```bash
ptoas vec_add.pto --pto-arch=a3 --pto-backend=emitc \
  --enable-insert-sync -o vec_add_kernel.cpp
```

上述检查分别验证环境、包导入和 PTOAS 生成代码功能；设备运行流程见 [构建与运行示例](../examples/build_and_run_overview.md)。

## 卸载PTO-AS

使用对应的`run`安装包执行卸载：

```bash
bash cann-pto-as_9.2.0_linux-x86_64.run --uninstall --install-path=xxx
```

其中，`xxx`应与安装时指定的安装根目录保持一致。
