# 安装指南

## 安装说明

本文介绍通过CANN PTO-AS `run`安装包安装PTO-AS组件并部署PTOAS工具的方法。

PTOAS随PTO-AS组件一起安装。安装PTO-AS组件时，安装程序会部署PTOAS启动程序、运行时依赖和PTOAS wheel，并将PTOAS wheel安装到组件使用的私有Python目录中，无需在当前Python环境中单独执行`pip install`。

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

> [!NOTE]说明
> 本文以`full`模式为例，不展开说明`run`、`full`和`devel`三种安装模式的内容差异。

## 配置环境变量

如果安装时未使用`--setenv`，在使用PTO-AS前需要配置`ASCEND_PTO_AS_PATH`：

```bash
export ASCEND_PTO_AS_PATH=<install_path>/cann/pto_as
```

其中，`<install_path>`为`--install-path`指定的安装根目录。

如果安装时未指定`--install-path`，默认安装根目录如下：

- 普通用户：`${HOME}/Ascend`
- root用户：`/usr/local/Ascend`

安装时也可以使用`--setenv`配置运行所需的环境变量：

```bash
bash cann-pto-as_9.2.0_linux-x86_64.run --full --install-path=xxx --setenv
```

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

相关文件位于：

```text
<install_path>/cann/tools/ptoas/
```

主要目录如下：

```text
tools/ptoas/
├── bin/
├── lib/
├── python/
└── wheels/
```

目录说明如下：

| 目录 | 说明 |
| --- | --- |
| `bin` | PTOAS启动程序 |
| `lib` | PTOAS运行时共享库 |
| `python` | PTOAS私有Python包目录 |
| `wheels` | PTOAS wheel文件 |

PTOAS wheel由安装程序自动安装，无需在当前Python环境中单独执行`pip install`。

## 检查安装结果

安装完成后，可以检查PTOAS启动程序是否已经部署：

```bash
ls <install_path>/cann/tools/ptoas/bin/ptoas
```

该检查用于确认PTOAS启动程序已经安装，不代表PTOAS编译功能已完成验证。

## 卸载PTO-AS

使用对应的`run`安装包执行卸载：

```bash
bash cann-pto-as_9.2.0_linux-x86_64.run --uninstall --install-path=xxx
```

其中，`xxx`应与安装时指定的安装根目录保持一致。
