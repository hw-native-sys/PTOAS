# 5.4 Reserved Buffer

## 1. 概述

`Reserved Buffer` 可以把它理解成：

- 先给一块本地缓冲区起一个名字
- 之后在需要的位置，通过这个名字引用同一块缓冲区

它不是普通的 `tile`。  
普通 `tile` 更像“某次计算临时用一下的本地对象”；而 `reserved buffer` 更像“我要提前预留一块固定用途的本地区域，后面多处都要按这个约定来使用它”。

## 2. 它是用来解决什么问题的

最常见的问题是：

- 某些cv pipe 或本地 FIFO 需要一块稳定的本地缓冲区
- 这块缓冲区不只是某一个 op 临时使用，而是要作为一条通信路径的一部分长期存在
- producer 和 consumer 两侧都要知道“说的是同一块缓冲区”

如果没有 `reserved buffer`，你就很难在 IR 里清楚地表达：

- “这里需要预留一块本地区域”
- “另一侧引用的是同一块区域”

所以它的核心作用就是：

- 给本地缓冲区建立一个稳定的名字
- 让多处代码通过这个名字对齐到同一块地址

## 3. 什么时候需要用它

如果你只是写普通的：

- `tload`
- `tmov`
- `tmatmul`
- `tstore`

这类常规 kernel，通常不需要 `reserved buffer`。

更可能需要它的场景是：

- cv pipe
- 本地 slot buffer
- producer / consumer 配对的本地通信路径
- 一块本地缓冲区要被多处按约定共同使用

所以，`reserved buffer` 不是“每个 kernel 都要写”的对象，而是“在cv pipe或本地保留区域场景下才需要写”的对象。

## 4. 两个核心操作

### 4.1 `pto.reserve_buffer`

这个操作的意思是：

- 在当前函数里声明一块“预留本地缓冲区”

例如：

```mlir
%buf = pto.reserve_buffer {
  name = "c2v_fifo",
  size = 8192,
  location = #pto.address_space<vec>,
  auto = false,
  base = 0
} -> i32
```

这里每个字段可以这样理解：

- `name`
  这块预留缓冲区的名字。

- `size`
  这块缓冲区要预留多少字节。

- `location`
  这块缓冲区放在哪类本地地址空间里。

- `auto`
  这块缓冲区的基址是否自动分配。

- `base`
  当不使用自动分配时，显式指定这块缓冲区的基址。

结果值 `-> i32` 可以理解成：

- 这块预留缓冲区最后对应的本地地址

### 4.2 `pto.import_reserved_buffer`

这个操作的意思是：

- 在另一处引用已经声明过的那块预留缓冲区

例如：

```mlir
%peer_buf = pto.import_reserved_buffer {
  name = "c2v_fifo",
  peer_func = @producer
} -> i32
```

它表达的是：

- 我自己这里不重新声明一块新缓冲区
- 我引用 `@producer` 里那块名为 `"c2v_fifo"` 的预留缓冲区

## 5. 最直观的理解方式

可以把它类比成：

- `pto.reserve_buffer`：先“占一个车位”，并给车位贴上名字
- `pto.import_reserved_buffer`：在另一处根据名字找到这个车位

重点不是“再新建一个车位”，而是“确认双方说的是同一个车位”。

## 6. 一个最小例子

下面用一个简化场景说明。

假设有两个函数：

- `@producer`：生产数据
- `@consumer`：消费数据

它们之间约定要使用一块本地 `vec` 缓冲区，名字叫 `"c2v_fifo"`。

### 6.1 producer 侧先声明

```mlir
func.func @producer() {
  %buf = pto.reserve_buffer {
    name = "c2v_fifo",
    size = 8192,
    location = #pto.address_space<vec>,
    auto = false,
    base = 0
  } -> i32

  // 后续某些本地对象初始化使用 %buf
  return
}
```

这里的意思是：

- 在 `producer` 侧预留一块 `vec` 本地缓冲区
- 名字叫 `"c2v_fifo"`
- 大小是 `8192` 字节
- 基址显式指定为 `0`

### 6.2 consumer 侧通过名字导入

```mlir
func.func @consumer() {
  %buf = pto.import_reserved_buffer {
    name = "c2v_fifo",
    peer_func = @producer
  } -> i32

  // 后续某些本地 pipe / slot buffer 初始化使用 %buf
  return
}
```

这里的意思是：

- `consumer` 侧不自己重新保留一块新缓冲区
- 它直接引用 `@producer` 里那块 `"c2v_fifo"`

### 6.3 这个例子真正说明了什么

这个例子说明的是：

- 两边共享的是“同一块预留缓冲区”的语义
- 不是 producer 一块、consumer 再单独来一块
- 名字和 peer 关系，是两边对齐到同一块缓冲区的关键

## 7. 显式基址写法

如果你需要把 reserved buffer 绑定到一个确定的本地地址，可以显式给出基址：

```mlir
%buf = pto.reserve_buffer {
  name = "c2v_fifo",
  size = 8192,
  location = #pto.address_space<vec>,
  auto = false,
  base = 0
} -> i32
```

它的含义是：

- 这块缓冲区放在哪里，由用户自己指定
- `base` 表示这块预留缓冲区的起始地址
- `auto = false` 表示这里不使用自动分配

## 8. 用户最该记住的约束

最重要的几条可以直接记成：

- `name` 在同一函数里应唯一。
- `location` 必须写对地址空间。
- `auto = false` 时，必须同时提供 `base`。
- `import_reserved_buffer` 必须能在 `peer_func` 里找到同名 `reserve_buffer`。

## 9. 什么时候用，什么时候不用

建议使用的情况：

- 你在写CV Pipe
- 你在写本地 slot buffer
- 你需要 producer / consumer 双方共享同一块本地缓冲区语义

不建议主动引入的情况：

- 你只是写普通算子 kernel
- 你没有本地 FIFO / pipe / peer buffer 需求
- 普通 `alloc_tile` 就足够表达你的意图

## 10. 实际写法建议

- 先把 `name`、`size`、`location` 写清楚。
- 需要固定本地地址时，再显式写出 `base`。
- 如果另一侧只是“引用同一块缓冲区”，就用 `pto.import_reserved_buffer`，不要再重复声明一块新的 `reserve_buffer`。
