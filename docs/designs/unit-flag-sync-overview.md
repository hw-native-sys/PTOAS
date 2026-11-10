# Unit Flag 与自动同步
---

## 一、Unit Flag 是什么

### 1.1 它替代什么

Cube 上一次矩阵计算跨两条流水线：`PIPE_M` 上的 MAD（Multiply-ADd）做 `C = A×B`，结果落在 L0C/ACC；`PIPE_FIX` 上的 fixpipe 把结果搬出去。两者异步，所以需要同步。常规做法是一对显式事件：

```cpp
TMATMUL(acc, lhs, rhs);
set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
TSTORE(out, acc);
```

UF 是同一件事的**硬件内建版本**：MAD 和 fixpipe 各带一个 2 bit 控制字段，由硬件维护一个 flag 完成交接，不再需要那对事件，也不占用事件 ID。

在 PTO IR 中，这两个字段就是 matmul 的 `accPhase` 与 ACC 搬出的 `stPhase`：

```mlir
pto.tmatmul ins(%lhs, %rhs : !left, !right) outs(%acc : !acc)
  {accPhase = #pto<acc_phase final>}
pto.tstore ins(%acc : !acc) outs(%out : !out)
  {stPhase = #pto<st_phase final>}
```

它们一路直达硬件字段，中间没有编译器翻译。

### 1.2 flag 状态机

UF 是**每个核一个**的 flag，不是每块 tile 一个状态。这一点决定了后面所有的证明义务。

| 指令与相位 | 编码 | 对 flag 的动作 |
|---|---|---|
| MAD `AccPhase::Partial` | `0x2` | 不动 |
| MAD `AccPhase::Final` | `0x3` | **阻塞等 flag 清零 → 置位**（check_and_set） |
| fixpipe `STPhase::Partial` | `0x2` | 阻塞等 flag 置位 → 搬出 → **不清位** |
| fixpipe `STPhase::Final` | `0x3` | 阻塞等 flag 置位 → 搬出 → **搬完后清位** |

一轮完整握手是 `0 → MAD(Final) 置 1 → fixpipe(Final) 清 0 → 回到 0`。

两处细节要记住：MAD 侧是 **check_and_set**，flag 仍置位时会停顿；fixpipe 的清位发生在**搬出完成之后**，不是指令发射时。

权威语义见 `pto-isa/include/pto/common/type.hpp` 中 `AccPhase` / `STPhase` 的注释。

### 1.3 两侧相位不对称

两个属性名字相同、编码相同，但问的是不同问题：

- `accPhase` 的 Partial/Final —— **是不是 K 方向累加的最后一拍？** 只有最后一拍置位。
- `stPhase` 的 Partial/Final —— **是不是这块 L0C 的最后一次搬出？** 只有最后一次清位。

一块 L0C 有时要分多次搬出，这几次都要等同一个"数据就绪"flag；若第一次就清位，后续搬出会等一个不会再来的 flag。所以 store 侧需要"不释放"的 Partial：

```
MAD(Partial)      K 第 1 拍，不置位
MAD(Final)        K 最后一拍 → 置位
fixpipe(Partial)  搬第 1 块，不清位
fixpipe(Final)    搬最后一块 → 清位
```

左侧的节奏由 K 维切分决定，右侧由搬出次数决定，两者没有对应关系。

还有一层前置依赖：`STPhase::Final` 要求产出该 L0C 的 matmul 已经是 `AccPhase::Final`。因此 `STPhase::Partial` 配 `AccPhase::Partial` 是**禁忌组合**——MAD 从不置位，fixpipe 却在阻塞等置位，会挂死。

### 1.4 覆盖两个方向

ACC 复用场景有两条方向相反的依赖，UF 两条都覆盖：

| 方向 | 含义 | UF 中的实现 |
|---|---|---|
| `PIPE_M → PIPE_FIX` | matmul 写完 ACC，store 才能读（RAW） | fixpipe 等 flag **置位** |
| `PIPE_FIX → PIPE_M` | store 读完 ACC，下一个 matmul 才能覆写（WAR） | MAD 的 check_and_set 等 flag **清零** |

合起来是一个**深度 1 的有界缓冲握手**：生产者等槽位腾空才能填，消费者等槽位有货才能取。所以一个 UF 握手可以顶掉**两对**事件，而非一对。

---

## 二、UF 分析在同步流程中的位置

### 2.1 它是一个过滤器，不是同步生成器

自动同步的主体仍是 `InsertSync`：枚举依赖对、做内存别名分析、分配事件 ID、发射同步。

UF 分析只在其中承担一个角色：对一条**已经被别名分析认定存在**的依赖，判断"硬件是否已经保证了这个顺序"。成立则把这条依赖丢掉，不再为它生成事件。

由此有三条性质：

- **只做减法。** UF 不参与别名分析，不会让一条本不存在的依赖凭空出现，也不会放宽别名判断。
- **粒度是依赖对，不是指令。** 一个 matmul 可能同时访问 bias、scratch，这些依赖**原样保留**并照常得到事件同步。UF 不是"这条指令不需要同步"的开关。
- **证明不成立就保守保留。** 所有判定 fail-closed。

### 2.2 判定顺序

一对（当前指令，前驱指令）要不要插同步，按顺序过筛，任一步认定"不需要"即结束：

```
① 同步记录里已登记过这对        → 跳过
② 别名分析认定无依赖            → 跳过
③ 多缓冲槽位在每轮都互不相交    → 丢弃该对
④ 两次迭代访问的字节区间不相交  → 丢弃该对
⑤ UF 已保证顺序                 ←── UF 在这里
⑥ 同管线 barrier 可剪裁        → 跳过
⑦ 事件 ID 已复用               → 跳过
⑧ 插入同步并登记记录
```

③④ 与 UF 正交互补：前者说"两次访问其实没碰到同一块字节"，UF 说"碰到了同一块字节，但硬件已排好序"。一条依赖被任一方消除即可。

### 2.3 判定条件

UF 覆盖一条依赖，需同时满足：

1. **角色与相位配对** —— 两端分别是发布侧与消费侧，管线分别是 `PIPE_M` 与 `PIPE_FIX`，相位组合有效；
2. **同一执行路径** —— 两端处于同一层控制流嵌套与同一分支臂；
3. **确实是彼此的对端** —— 见下；
4. **ACC 区域一致** —— 同一分配源头、范围有界、别名精确。UF 不替一次模糊的别名判断背书。

第 3 条是最本质的一条。因为 flag **每核只有一个**，"两端都说 UF 协议"并不等于"这个 MAD 发布的正是这个 store 消费的那次"：两个 `Final` MAD 发布同一枚 flag，紧随其后的 store 可能在第一次发布上就通过检查，而第二个 MAD 还在写 ACC。

事件 ID 是显式配对的，UF 的 flag 是隐式共享的全局状态——配对关系只能靠编译期分析证明。方案采用**结构化的 token 状态模型**：为每个区域计算"进入要求 / 离开产生"的摘要，分支处取两臂汇合（不一致则判为未知），循环要求体摘要净中性。顺序、分支、`scf.for`、`scf.while` 都是同一模型的实例。

对回边方向还有一条额外要求：**循环体的 token 收支必须平衡**（恰好一次发布、一次消费，发布在前，且都无条件执行）。这条保障的是**活性**而非仅数据顺序——由于 MAD 是 check_and_set，收支不平衡的循环体不会表现为竞态，而会直接死锁。

---

## 三、支持范围

### 3.1 参与的算子

**发布侧**（带 `accPhase`，`Final` 时置位）：

`tmatmul`、`tmatmul.acc`、`tmatmul.bias`、`tmatmul.mx`、`tmatmul.mx.acc`、`tmatmul.mx.bias`、`tgemv`、`tgemv.acc`、`tgemv.bias`、`tgemv.mx`、`tgemv.mx.acc`、`tgemv.mx.bias`

**消费侧**（带 `stPhase`，`Final` 时清位）：

`tstore`、`tmov` —— 两者都**仅在源操作数位于 ACC 时**参与。`tmov` 也用于 L1→L0A/L0B 等搬运，那类走 MTE 流水，与 UF 无关。

相位为 `Unspecified` 或不带相位属性的算子，硬件上 `unitFlag=0`，对 flag 透明。

> 可见与可优化是两个集合：所有能携带相位的算子都**必须**被分析看见（否则一个不可见的发布会让分析误判配对），但允许作为 UF 证明一端、从而真正删除同步的集合可以更小、逐步放开。

### 3.2 支持矩阵

| # | 场景 | M→FIX（RAW） | FIX→M（WAR） |
|---|---|---|---|
| 1 | 直线代码，相位配对 | 消除 | 消除 |
| 2 | 循环体内，同一迭代 | 消除 | 不适用 |
| 3 | 跨迭代（回边），循环体收支平衡 | 不适用 | 消除 |
| 4 | 两端在同一执行路径上，且分支两臂 token 效果一致 | 消除 | 消除 |
| 5 | 循环体收支不平衡 | 保守保留 | 保守保留 |
| 6 | 两臂 token 效果不一致 | 保守保留（并告警） | 保守保留（并告警） |
| 7 | 两端不在同一执行路径上（一进一出 / 互斥两臂） | 保守保留 | 保守保留 |
| 8 | 相位为 `Unspecified` | 保守保留 | 保守保留 |
| 9 | 一个消费前有多次发布 | 保守保留 | 保守保留 |
| 10 | ACC 别名不可证明 / 范围未知 | 保守保留 | 保守保留 |
| 11 | 依赖对中的非 ACC 部分（bias、scratch 等） | 始终保留 | 始终保留 |
| 12 | 循环体内消费在前、发布在后 | 消除 | 消除 |

回边方向被消除时，为该事件生成的**循环边界预置与排空**（循环前的 `set_flag`、循环后的 `wait_flag`，及其零次迭代守卫）也随之消失。

**适用边界**：本方案覆盖 `--enable-insert-sync`。`--enable-unified-sync` 不在范围内，选用该后端时相位只影响生成代码的模板参数，不影响同步决策。架构上 PTOAS 支持的 a2/a3/a5 对 UF 行为一致，分析无需按架构区分。

### 3.3 各场景的最小示例

以下示例省略 tile 类型、`alloc_tile` 与操作数列表。matmul 一律 `accPhase = final`（**发布**），store 一律 `stPhase = final`（**消费**），另有标注的除外。

**1 — 直线代码**

```mlir
pto.tmatmul outs(%acc)      // 发布
pto.tstore  ins(%acc)       // 消费
```

两个方向都消除：M→FIX 由 fixpipe 等置位覆盖，FIX→M 由下一个发布者的 check_and_set 覆盖。

**2、3 — 循环同迭代与回边**

```mlir
scf.for %i = 0 to 128 {
  pto.tmatmul outs(%acc)    // 发布
  pto.tstore  ins(%acc)     // 消费
}
```

同一个形态同时覆盖两行：体内的 M→FIX（第 2 行）与跨迭代的 FIX→M（第 3 行）一并消除，循环前的预置与循环后的排空也随之消失。体摘要为 `Clear → Clear`，净中性。

**4 — 两端在同一执行路径上，分支两臂效果一致**

条件是两端**在同一条执行路径上**。最常见的满足方式是两端同处一条臂：

```mlir
scf.if %cond {
  pto.tmatmul outs(%acc0)   // 发布
  pto.tstore  ins(%acc0)    // 消费
} else {
  pto.tmatmul outs(%acc1)   // 发布
  pto.tstore  ins(%acc1)    // 消费
}
```

两臂摘要均为 `{Clear→Clear}`，取交集非空，两臂内的两对都能证明。**结论不依赖对侧臂**——各臂独立计算后求交，不会引用另一条臂里不会执行的指令。

缺失 `else` 的情形同样成立：空区域按恒等处理，`{Clear→Clear} ∩ 恒等 = {Clear→Clear}`。

但"同臂"不是必要条件。另一种满足方式是**一端是叶子、另一端是整个分支区域**：

```mlir
scf.if %cond {
  pto.tmatmul outs(%acc)    // 发布
} else {
  pto.tmatmul outs(%acc)    // 发布（另一条臂）
}
pto.tstore ins(%acc)        // 消费，在 if 之后
```

汇合效果 `{Clear→Set} ∩ {Clear→Set} = {Clear→Set}`，即**整个 `scf.if` 就是一个发布者**。无论走哪条臂都恰好发生一次发布，随后的 store 消费它，两对的 M→FIX 都消除。

反之，两端分处**互斥的两臂**（一臂发布、另一臂消费）则不在此列——那两条指令永不同时执行，根本不构成交接，见第 7 行。

**5 — 循环体收支不平衡**

```mlir
scf.for %i = 0 to 4 {
  pto.tmatmul outs(%acc)    // 1 次发布
  pto.tstore  ins(%acc)     // 第 1 次消费
  pto.tstore  ins(%acc)     // 第 2 次消费 → 不平衡
}
```

第二次消费会等一个已被清掉的 flag。两次发布配一次消费同理（下一轮的发布撞上仍置位的 flag）。体摘要非中性，两个方向都保留。

**6 — 两臂 token 效果不一致**

```mlir
scf.if %cond {
  pto.tmatmul outs(%acc)    // 只发布
} else {
  pto.tmatmul outs(%acc)    // 发布
  pto.tstore  ins(%acc)     // 并消费
}
```

`{Clear→Set} ∩ {Clear→Clear} = ∅`。离开 `scf.if` 时 flag 处于哪个状态取决于走了哪条臂，无法证明，保留并告警。

**7 — 两端不在同一执行路径上**

一进一出：

```mlir
pto.tmatmul outs(%acc)      // 发布在分支外
scf.if %cond {
  pto.tstore ins(%acc)      // 消费在分支内
}
```

注意这个形态本身就会死锁（见 §4.1）——`%cond` 为假时 flag 滞留置位，后续发布者永久停顿。

分处互斥两臂：

```mlir
scf.if %cond {
  pto.tmatmul outs(%acc)    // 发布
} else {
  pto.tstore  ins(%acc)     // 消费
}
```

这两条指令**永不同时执行**，不构成交接，UF 无从覆盖。这条依赖确实可以不同步，但**依据是互斥而非硬件排序**——与"字节区间不相交"同类，且与相位无关（两端都不带相位时同样成立），因此不由 UF 处理。

**8 — 相位为 `Unspecified`**

```mlir
pto.tmatmul outs(%acc)      // 无 accPhase
pto.tstore  ins(%acc)       // 无 stPhase
```

两条指令的 `unitFlag` 都是 0，硬件不做任何握手，必须用事件同步。

**9 — 一个消费前有多次发布**

```mlir
pto.tmatmul outs(%acc)      // 发布
pto.tgemv   outs(%acc)      // 又一次发布
pto.tstore  ins(%acc)       // 消费的是哪一次？
```

flag 每核只有一个，store 可能在第一次发布上就通过检查，而第二个发布者还在写 ACC。无法证明配对关系，保留。

**10 — ACC 别名不可证明**

两端访问的 ACC 经动态地址或整数转指针得到，别名分析无法给出确定的根缓冲与字节范围时，UF 不参与判定。这是通用的 fail-closed 条件，不针对某种写法——UF 不替一次模糊的别名判断背书。

**11 — 依赖对中的非 ACC 部分**

```mlir
pto.tmatmul.bias ins(%lhs, %rhs, %bias) outs(%acc)   // 发布，同时读 bias
pto.tstore       ins(%acc)                            // 消费
```

ACC 那条依赖被消除，但 `%bias` 的依赖（若有前驱写它）**照常生成事件**。UF 的粒度是依赖对，不是指令。

**12 — 循环体内消费在前、发布在后**

与第 2、3 行是同一组指令、顺序相反，这是软流水的稳态形态，需要序言与尾声：

```mlir
pto.tmatmul outs(%acc)      // 序言：先发布
scf.for %i = 0 to 128 {
  pto.tstore  ins(%acc)     // 消费第 i-1 轮的结果
  pto.tmatmul outs(%acc)    // 发布第 i 轮
}
pto.tstore ins(%acc)        // 尾声：消费最后一轮
```

两个方向都消除，但**与第 2、3 行的分工正好互换**：

| | 第 2、3 行（发布在前） | 第 12 行（消费在前） |
|---|---|---|
| 同迭代 | M→FIX | FIX→M |
| 跨回边 | FIX→M | M→FIX |

体摘要是 `{Set→Set}`——**要求进入循环时 flag 已置位**，这正是序言那条发布的作用。若缺少序言，首轮消费会等一个从未发出的置位而挂死（见 §4.1），分析会保守保留并告警。

这一行说明支持范围不是按"直线 / 循环 / 分支"划分的，也不取决于指令顺序，而是取决于每一对的证明是否成立。



### 3.4 Partial / Final 链

K 方向分多拍累加时，前若干拍用 `AccPhase::Partial`、最后一拍用 `Final`。`Partial` 的 MAD 不发布 flag，因此不作为 UF 证明的一端。

一块 L0C 分多次搬出时，前若干次用 `STPhase::Partial`、最后一次用 `Final`。Partial 的搬出既不发布也不消费，对 token 透明。

注意 §1.3 的禁忌：`STPhase::Partial` 不可配 `AccPhase::Partial`。

---

## 四、使用注意

### 4.1 相位写错会死锁，而不是算错

因为 MAD 是 check_and_set，**相位写错的后果不是数值错误，而是挂死**：

| 形态 | 后果 |
|---|---|
| `STPhase::Partial` 配 `AccPhase::Partial` | fixpipe 等一个永不到来的 flag |
| 条件化的消费（如 `if { store{Final} }` 无 else） | flag 滞留置位 → 后续 MAD 永久停顿 |
| 循环中某轮跳过 `AccPhase::Final` 的 matmul | fixpipe 等不到置位 |

第二行值得注意：它**不是编译器删同步导致的**，无论是否启用 UF 优化都会挂。方案中的 token 模型能在编译期识别这类写法并告警。

手写相位的规则：一块 L0C 分多次搬出，必须且只能让最后一次用 `Final`；且 `Final` 的搬出不应放在可能被跳过的分支里。

### 4.2 UF 不缩短关键路径

UF 消除的是**指令条数与事件 ID 占用**，不是数据依赖造成的串行等待。

这一点最容易被误判，因为"同步都删掉了"听起来像并行度提高了。实际上 check_and_set 并没有**消除**依赖，只是把依赖的实施方式从显式事件换成了硬件停顿：第 i+1 轮的 MAD 仍要等第 i 轮搬出完成才能发射，关键路径完全不变。

以 ACC 单块复用的循环为例：即便两个方向的事件全部消除、循环体在 ACC 依赖上变为无事件，MAD 与 fixpipe 仍严格交替，实测延迟可能几乎没有变化。这种场景的优化方向是**ACC 多缓冲**（不同迭代使用不同 ACC 地址）：相邻迭代访问不同地址，既不再需要回边事件，**也不会触发 check_and_set 停顿**，两条流水才能真正重叠。那条路走的是 §2.2 第 ④ 步的不相交判定，与 UF 正交。

### 4.3 相位与同步是一体的

UF 证明的前提是相位配对在最终生成代码时仍然成立。不要在同步插入之后手工改写生成代码中的相位或同步——删掉一个 `Final` 搬出会让 flag 滞留，后果是挂死而非报错。

---

## 参考

- `pto-isa/include/pto/common/type.hpp` —— `AccPhase` / `STPhase` 的权威语义
- `test/lit/pto/uf_aware_sync*.pto` —— 支持与不支持场景的回归用例
- [issue1694-uf-aware-sync-loop-design.md](issue1694-uf-aware-sync-loop-design.md) —— 实现方案与分阶段计划
