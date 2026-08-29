# PTODSL ST Guide

`test/dsl-st/` 放的是基于 PTODSL 的 simulator/ST 用例。

这类测试适合验证：

- PTODSL surface 是否能正确生成目标 IR
- `ptoas --pto-backend=vpto` 后端是否能稳定接受这类新形态
- kernel 在 simulator / runtime 路径上的真实执行结果

如果你要验证的是 parser、verifier、pass dump、IR rewrite 之类“只看编译输出”的行为，优先放到 `test/lit`，不要放这里。

## 环境前提

`test/dsl-st/` 现在假设 `ptodsl` 已经通过支持的 `ptoas` 安装合同进入当前
Python 环境，而不是在运行时自己修 `PYTHONPATH` 或 `sys.path`。

推荐准备方式：

```bash
cd $PTOAS_REPO_ROOT
pip install -e . --no-build-isolation
```

如果你在验证 release/CI wheel，也可以直接：

```bash
pip install /path/to/ptoas*.whl
```

快速自检：

```bash
python3 -c "import ptodsl; from ptodsl import pto; print(ptodsl.__file__)"
```

注意：

- `ptoas-bin-*.tar.gz` 只是 compiler-only artifact，不能替代上面的 Python 安装。
- 如果 `import ptodsl` 失败，应先修复安装环境，而不是给测试脚本补 `sys.path`。

## 推荐写法

当前目录下推荐使用 [common.py](/home/zhangzhendong/ptoas-workspace/PTOAS/test/dsl-st/common.py) 里的两个 helper：

- `golden_output_case(...)`
- `auto_main(globals())`

目录级运行时，还可以直接使用：

- `python3 test/dsl-st`
- `scripts/sim_dsl.sh test/dsl-st`

对大多数单输出测试，开发者只需要写：

1. kernel
2. 输入构造
3. golden 计算
4. `CASES = [...]`
5. 文件末尾一行 `auto_main(globals())`

## 最小模板

```python
#!/usr/bin/env python3

import numpy as np

from common import auto_main, golden_output_case
from ptodsl import pto


@pto.jit(name="my_kernel", kernel_kind="vector", target="a5", mode="explicit")
def my_kernel(
    inp_ptr: pto.ptr(pto.f32, "gm"),
    out_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
):
    # write kernel here
    ...


def make_inputs():
    return [np.ones((4, 64), dtype=np.float32)]


def make_expected(inp):
    # compute golden from host inputs
    return inp + 1.0


CASES = [
    golden_output_case(
        "my_kernel_basic",
        my_kernel,
        inputs=make_inputs,
        expected=make_expected,
        rtol=0.0,
        atol=0.0,
    ),
]


auto_main(globals())
```

## `golden_output_case(...)` 约定

`golden_output_case(...)` 默认适合“若干输入 + 一个输出”的测试。

它会自动做这些事情：

- 把 `inputs` 转成 host numpy 数组
- 根据 `expected` 的 shape / dtype 自动分配一个全零输出
- 把这个输出作为最后一个参数传给 kernel
- 默认把最后一个 device tensor 拿出来和 golden 比较

常用参数：

- `inputs`
  - 可以是函数，返回 `list[np.ndarray]`
  - 也可以直接传一个 `list[np.ndarray]`
- `expected`
  - 可以是函数，签名是 `expected(*host_inputs)`
  - 也可以直接传 numpy 数组
- `output_shape`
  - 如果输出 shape 不能直接从 golden 推出来，可以显式指定
- `output_dtype`
  - 如果输出 dtype 需要和 golden 分开控制，可以显式指定
- `output_index`
  - 默认比较最后一个 tensor；如果不是最后一个，改这里
- `rtol` / `atol`
  - 浮点结果建议显式写；位级结果一般用 `0.0`

## 什么时候需要自定义 case

如果测试不是“单输出比 golden”这一路径，就不要硬塞进 `golden_output_case(...)`。

常见例子：

- 需要比较多个输出
- 需要读取中间 buffer
- 需要自定义断言信息
- 需要根据运行时结果做结构化检查而不是 `allclose`

这时可以直接继续使用 `run_cases(...)` 的底层接口，自定义：

- `make_case()`
- `check(device_inputs, expected)`

## 当前参考用例

可以直接参考：

- [predicate_pack.py](/home/zhangzhendong/ptoas-workspace/PTOAS/test/dsl-st/predicate_pack.py)
- [cube_matrix_pipeline.py](/home/zhangzhendong/ptoas-workspace/PTOAS/test/dsl-st/cube_matrix_pipeline.py)

它演示了：

- PTODSL kernel authoring
- raw predicate image 的 host golden 写法
- `golden_output_case(...)` 的标准接入方式
- cube matrix pipeline 的端到端 simulator/ST 写法

## 运行方式

单文件打印生成的 MLIR：

```bash
python3 test/dsl-st/predicate_pack.py --emit-mlir
```

单文件走 simulator ST：

```bash
scripts/sim_dsl.sh test/dsl-st/predicate_pack.py
```

自动发现整个目录下的测例并列出 case name：

```bash
python3 test/dsl-st --list
```

自动发现整个目录下的测例并打印合并 MLIR：

```bash
python3 test/dsl-st --emit-mlir
```

自动发现整个目录下的测例并走 simulator ST：

```bash
scripts/sim_dsl.sh test/dsl-st
```

如果只是先做编译链检查，也可以先跑：

```bash
python3 ptodsl/tests/test_jit_compile.py
```

## 编写建议

- 优先让 golden 直接表达语义，不要把 expected 写成难懂的魔数堆。
- 尽量让一个测试只保护一个回归点；如果要覆盖一组紧密相关的形态，可以像 `predicate_pack.py` 一样在同一个 kernel 里并排 materialize。
- 对 predicate / bit-level 结果，优先用 `psts` 这类“直接 materialize raw state”的方式观测，不要绕远路通过别的算子副作用来猜结果。
- 能用 Python 原生字面量的地方就直接用，减少不必要的 `pto.const(...)` 噪音。
- 如果某个写法依赖当前 backend / raw image 约定，最好在 golden 附近留一小段注释，解释为什么 expected 长这样。

## 自动发现约定

`test/dsl-st/` 目录级 runner 会自动加载当前目录下所有顶层 `.py` 用例文件，但会跳过：

- `common.py`
- `__main__.py`
- 以下划线开头的辅助模块

每个被发现的模块都需要定义非空 `CASES` 列表，且所有 case name 在目录内必须唯一。
