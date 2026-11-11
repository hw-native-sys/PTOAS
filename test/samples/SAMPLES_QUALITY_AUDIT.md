# test/samples tile op 覆盖与质量审计

> 目的：为 **PyPTO 所用 tile op** 建立测试基线。PyPTO 直接发射 `pto.*` tile op 到 `.pto`（不经过 ptodsl），
> 与 `test/samples` 的裸 op 编写形态一一对应，故以 samples 为统计对象。
> 基线 `37f5f25e3` + 本 PR；内容级扫描生成（builder `.py` 逐目录解析，噪声关键词已剔除）。

## 总览

| 统计项 | 数量 |
|---|---:|
| op 总数（含 4 个卷积专属 op） | 146 |
| 其中 TileOp（142）+ 卷积专属 op（4） | 146 |
| 有同名样例目录 | 137 |
| 有 golden（含 ⏳ 待确认 0 个） | **135** |
| **其中板上数值验证通过 ✅** | **135** |
| 仅 builder（只编译执行、无比对） | 2 |
| 无样例目录 | 5 |

板上通过 ✅ 的 135 个的参数轴分布：单格式（0 轴）86、单轴 41、双轴 7、三轴及以上 0。
参数轴口径：目录内出现**多于一个取值**才算遍历（dtype / 形状 / 布局 row+col / valid 域 / op 级参数多值）。

本 PR 的样例编写约定：**每 op 一个 builder `.py`，单个 kernel 内混合多种 dtype（如 f32+i32）验证**，
不做逐格式文件遍历。已按此补齐：`Tmov/`、`Tsetval/`、`Tgetval/`（单文件多 dtype）、`Add/`、
`Random/`（trandom，Philox 参考）、`Fmod/`、`Fmods/`、`Tpow/`、`Tpows/`、`Rowprod/`、`Colprod/` 的 golden，
以及 `Histogram/` 全套（thistogram：byte 双变体，**累积直方图语义**——dst 为桶前缀和，参照 pto-isa 参考实现）。
**逐元素与行/列归约两个家族已 golden 全覆盖**。全部新样例在 238 板（Ascend950PR/A5）通过数值验证。
tmatmul/.acc/.bias 曾被 hold：builder 把结果经 vec tile 中转存出（acc→vec tmov + tstore），产出混合段 kernel，
在混合编译模式（dav-c310-vec + 强制 __DAV 宏）下 set_flag 管线白名单不含 MTE1/MTE2 而被 bisheng 拒绝。
已**单段化**修复（tstore 直接从 acc tile 存出，B mat tile 改 col_major、bias 改 none_box），238 板（Ascend950DT/A5）independent-golden 通过。
AIV/AIC 生命周期 8 op（talloc/tpush/tpop/tfree × aiv/aic）以 `Sync/` 下 4 个跨核样例补齐 golden
（tpipe_identity=tile 路径 C2V、tpipe_v2c=tile 路径 V2C、tpipe_gm_c2v/v2c=GM-slot 路径含 talloc）。
要点：① cross-core kernel 须以纯 `dav-c310` 混合模式编译（npu_validation 已补检测：TPUSH/TPOP/TFREE/TALLOC 宏），
否则 vec-arch+强制 DAV 宏路径的 set_flag 白名单拒绝 MTE1/MTE2；② split=1（上下拆分）语义下单 pop 收到错乱数据，样例统一 split=0；
③ cube 段内不可做 GM→ubuf DMA（copy_gm_to_ubuf_align_v2 不支持该 target feature），数据走 mat/cbuf 路径。
全部 4 case 在 238 板（Ascend950DT/A5）independent-golden 通过。

## 明细（按家族分组，共 142 项）

pr+ 列：✅ = 本 PR 在 test/samples/ 下新增的 golden 验证（50 个 op）。
golden 列：✅ 有数值比对（板上通过）；⏳ golden 已写、正确性待板上确认；⚠️ 仅执行冒烟（有 builder 无 golden）；— 无样例目录。
处理意见列：「待建目录/待补 golden」为常规缺口；其余为无法在当前环境完成 golden 的原因
（环境挂了暂无法验证 / A2/A3 专属 / 需要 vpto 后端 / 无独立数值语义建议豁免等）。

| # | 家族 | op | 样例目录 | pr+ | golden | dtype | 形状数 | 布局 | valid | 参数遍历 | kernel数 | 轴数 | 评级 | 处理意见 |
|---:|---|---|---|:-:|---|---:|---|:-:|---|---:|---:|---|---|
| 1 | DMA 数据搬运 | `pto.tload` | — |  | — | — | — | — | — | — | — | — | 无样例 | 非独立计算 op（数据搬运），由 TPushTPop/TPipe/整网样例覆盖 |
| 2 | DMA 数据搬运 | `pto.tstore` | — |  | — | — | — | — | — | — | — | — | 无样例 | 非独立计算 op（数据搬运），由 TPushTPop/TPipe/整网样例覆盖 |
| 3 | DMA 数据搬运 | `pto.tprefetch` | `TPrefetch` |  | ⚠️ | — | 1 | row | — | — | 1 | — | 无golden | 纯预取（无计算输出），现有冒烟样例 + lit 覆盖 |
| 4 | DMA 数据搬运 | `pto.ttrans` | `Trans` |  | ⚠️ | — | 1 | row | — | — | 1 | — | 无golden | 仅 builder 冒烟；转换语义建议经 ptodsl/TileOps 模板路径验证 |
| 5 | DMA 数据搬运 | `pto.tmov` | `Tmov` | ✅ | ✅ | — | 0 | row | — | — | 1 | 0 | 单格式 |  |
| 6 | MatMul / GEMV | `pto.tmatmul` | `MatMul` | ✅ | ✅ | f32 | 1 | col+row | — | — | 1 | 0 | 单格式 |  |
| 7 | MatMul / GEMV | `pto.tmatmul.acc` | `MatMul` | ✅ | ✅ | f32 | 1 | col+row | — | — | 1 | 0 | 单格式 |  |
| 8 | MatMul / GEMV | `pto.tmatmul.bias` | `MatMul` | ✅ | ✅ | f32 | 1 | col+row | — | — | 1 | 0 | 单格式 |  |
| 9 | MatMul / GEMV | `pto.tmatmul.mx` | `MatmulMxLowPrecision` |  | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 10 | MatMul / GEMV | `pto.tmatmul.mx.acc` | `MatmulMxLowPrecision` |  | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 11 | MatMul / GEMV | `pto.tmatmul.mx.bias` | `MatmulMxLowPrecision` |  | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 12 | MatMul / GEMV | `pto.tgemv` | `Gemv` | ✅ | ✅ | — | 0 | col+row | — | — | 3 | 1 | 单轴 |  |
| 13 | MatMul / GEMV | `pto.tgemv.acc` | `Gemv` | ✅ | ✅ | — | 0 | col+row | — | — | 3 | 1 | 单轴 |  |
| 14 | MatMul / GEMV | `pto.tgemv.bias` | `Gemv` | ✅ | ✅ | — | 0 | col+row | — | — | 3 | 1 | 单轴 |  |
| 15 | MatMul / GEMV | `pto.tgemv.mx` | `Gemvmx` |  | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 16 | MatMul / GEMV | `pto.tgemv.mx.acc` | `Gemvmx` |  | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 17 | MatMul / GEMV | `pto.tgemv.mx.bias` | `Gemvmx` |  | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 18 | AIV/AIC 生命周期与同步 | `pto.talloc_to_aiv` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 19 | AIV/AIC 生命周期与同步 | `pto.talloc_to_aic` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 20 | AIV/AIC 生命周期与同步 | `pto.tpush_to_aiv` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 21 | AIV/AIC 生命周期与同步 | `pto.tpush_to_aic` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 22 | AIV/AIC 生命周期与同步 | `pto.tpop_from_aic` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 23 | AIV/AIC 生命周期与同步 | `pto.tpop_from_aiv` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 24 | AIV/AIC 生命周期与同步 | `pto.tfree_from_aic` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 25 | AIV/AIC 生命周期与同步 | `pto.tfree_from_aiv` | `Sync` | ✅ | ✅ | f32 | — | — | — | — | 0 | 单格式 |  |
| 26 | AIV/AIC 生命周期与同步 | `pto.talloc` | — |  | — | — | — | — | — | — | — | — | 无样例 | 生命周期伪 op（auto 模式隐式覆盖），行为验证归 lit |
| 27 | AIV/AIC 生命周期与同步 | `pto.tpush` | — |  | — | — | — | — | — | — | — | — | 无样例 | 生命周期伪 op（A5 展开为 *_to_aiv/_from_aic），行为验证归 lit |
| 28 | AIV/AIC 生命周期与同步 | `pto.tpop` | — |  | — | — | — | — | — | — | — | — | 无样例 | 生命周期伪 op（A5 展开为 *_to_aiv/_from_aic），行为验证归 lit |
| 29 | AIV/AIC 生命周期与同步 | `pto.tfree` | — |  | — | — | — | — | — | — | — | — | 无样例 | 生命周期伪 op（A5 展开为 *_to_aiv/_from_aic），行为验证归 lit |
| 30 | AIV/AIC 生命周期与同步 | `pto.tsync` | `Sync` |  | ✅ | — | 0 | — | — | — | 0 | None | 双轴 |  |
| 31 | 逐元素一元 | `pto.tabs` | `Abs` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 32 | 逐元素一元 | `pto.tneg` | `Neg` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 33 | 逐元素一元 | `pto.tnot` | `Not` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 34 | 逐元素一元 | `pto.texp` | `Exp` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 35 | 逐元素一元 | `pto.tlog` | `Log` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 36 | 逐元素一元 | `pto.tsqrt` | `Sqrt` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 37 | 逐元素一元 | `pto.trsqrt` | `Rsqrt` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 38 | 逐元素一元 | `pto.trecip` | `Recip` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 39 | 逐元素一元 | `pto.trelu` | `Relu` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 40 | 逐元素一元 | `pto.tlrelu` | `Lrelu` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 41 | 逐元素一元 | `pto.trandom` | `Random` | ✅ | ✅ | — | 0 | row | — | — | 1 | 0 | 单格式 |  |
| 42 | 逐元素二元 | `pto.tadd` | `Add` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 43 | 逐元素二元 | `pto.tsub` | `Sub` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 44 | 逐元素二元 | `pto.tmul` | `Mul` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 45 | 逐元素二元 | `pto.tdiv` | `Div` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 46 | 逐元素二元 | `pto.tmax` | `Max` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 47 | 逐元素二元 | `pto.tmin` | `Min` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 48 | 逐元素二元 | `pto.tfmod` | `Fmod` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 49 | 逐元素二元 | `pto.trem` | `Rem` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 50 | 逐元素二元 | `pto.tpow` | `Tpow` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 51 | 逐元素二元 | `pto.tand` | `And` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 52 | 逐元素二元 | `pto.tor` | `Or` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 53 | 逐元素二元 | `pto.txor` | `Xor` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 54 | 标量操作数变体 | `pto.tadds` | `Adds` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 55 | 标量操作数变体 | `pto.tsubs` | `Subs` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 56 | 标量操作数变体 | `pto.tmuls` | `Muls` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 57 | 标量操作数变体 | `pto.tdivs` | `Divs` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 58 | 标量操作数变体 | `pto.tmaxs` | `Maxs` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 59 | 标量操作数变体 | `pto.tmins` | `Mins` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 60 | 标量操作数变体 | `pto.tfmods` | `Fmods` | ✅ | ✅ | — | 1 | row | — | — | 2 | 0 | 单格式 |  |
| 61 | 标量操作数变体 | `pto.trems` | `Rems` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 62 | 标量操作数变体 | `pto.tpows` | `Tpows` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 63 | 标量操作数变体 | `pto.tands` | `Ands` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 64 | 标量操作数变体 | `pto.tors` | `Ors` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 65 | 标量操作数变体 | `pto.txors` | `Xors` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 66 | 标量操作数变体 | `pto.taddsc` | `Addsc` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 67 | 标量操作数变体 | `pto.tsubsc` | `Subsc` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 68 | 标量操作数变体 | `pto.taddc` | `Addc` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 69 | 标量操作数变体 | `pto.tsubc` | `Subc` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 70 | 标量操作数变体 | `pto.taxpy` | `TAxpy` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 71 | 融合激活 | `pto.taddrelu` | — |  | — | — | — | — | — | — | — | — | 无样例 | A2/A3 专属（A5 verifier 拒绝），需 A2/A3 目标环境 |
| 72 | 融合激活 | `pto.tprelu` | `Prelu` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 73 | 移位 | `pto.tshl` | `Shl` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 74 | 移位 | `pto.tshr` | `Shr` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 75 | 移位 | `pto.tshls` | `Shls` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 76 | 移位 | `pto.tshrs` | `Shrs` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 77 | 行/列归约与统计 | `pto.trowsum` | `Rowsum` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 78 | 行/列归约与统计 | `pto.trowprod` | `Rowprod` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 79 | 行/列归约与统计 | `pto.trowmax` | `Rowmax` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 80 | 行/列归约与统计 | `pto.trowmin` | `Rowmin` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 81 | 行/列归约与统计 | `pto.trowargmax` | `TRowArgMax` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 82 | 行/列归约与统计 | `pto.trowargmin` | `TRowArgMin` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 83 | 行/列归约与统计 | `pto.tcolsum` | `Colsum` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 84 | 行/列归约与统计 | `pto.tcolprod` | `Colprod` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 85 | 行/列归约与统计 | `pto.tcolmax` | `Colmax` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 86 | 行/列归约与统计 | `pto.tcolmin` | `Colmin` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 87 | 行/列归约与统计 | `pto.tcolargmax` | `TColArgMax` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 88 | 行/列归约与统计 | `pto.tcolargmin` | `TColArgMin` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 89 | 行/列归约与统计 | `pto.thistogram` | `Histogram` | ✅ | ✅ | — | 0 | col+row | — | — | 1 | 1 | 单轴 |  |
| 90 | 部分元素 | `pto.tpartadd` | `Partadd` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 91 | 部分元素 | `pto.tpartmul` | `Partmul` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 92 | 部分元素 | `pto.tpartmax` | `Partmax` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 93 | 部分元素 | `pto.tpartmin` | `Partmin` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 94 | 部分元素 | `pto.tpartargmax` | `Partarg` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 95 | 部分元素 | `pto.tpartargmin` | `Partarg` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 96 | 广播 / 扩展 | `pto.texpands` | `Expands` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 97 | 广播 / 扩展 | `pto.trowexpand` | `Rowexpand` |  | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 98 | 广播 / 扩展 | `pto.trowexpandadd` | `Rowexpandadd` | ✅ | ✅ | — | 2 | col+row | — | — | 2 | 2 | 双轴 |  |
| 99 | 广播 / 扩展 | `pto.trowexpandsub` | `Rowexpandsub` |  | ✅ | — | 2 | col+row | — | — | 1 | 2 | 双轴 |  |
| 100 | 广播 / 扩展 | `pto.trowexpandmul` | `Rowexpandmul` |  | ✅ | — | 2 | col+row | — | — | 1 | 2 | 双轴 |  |
| 101 | 广播 / 扩展 | `pto.trowexpanddiv` | `Rowexpanddiv` |  | ✅ | — | 2 | col+row | — | — | 1 | 2 | 双轴 |  |
| 102 | 广播 / 扩展 | `pto.trowexpandexpdif` | `Rowexpandexpdif` | ✅ | ✅ | — | 2 | row | — | — | 2 | 1 | 单轴 |  |
| 103 | 广播 / 扩展 | `pto.trowexpandmax` | `Rowexpandmax` | ✅ | ✅ | — | 4 | col+row | — | — | 2 | 2 | 双轴 |  |
| 104 | 广播 / 扩展 | `pto.trowexpandmin` | `Rowexpandmin` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 105 | 广播 / 扩展 | `pto.tcolexpand` | `Colexpand` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 106 | 广播 / 扩展 | `pto.tcolexpandadd` | `Colexpandadd` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 107 | 广播 / 扩展 | `pto.tcolexpandsub` | `Colexpandsub` | ✅ | ✅ | — | 2 | col+row | — | — | 2 | 2 | 双轴 |  |
| 108 | 广播 / 扩展 | `pto.tcolexpandmul` | `Colexpandmul` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 109 | 广播 / 扩展 | `pto.tcolexpanddiv` | `Colexpanddiv` | ✅ | ✅ | — | 3 | row | — | — | 2 | 1 | 单轴 |  |
| 110 | 广播 / 扩展 | `pto.tcolexpandexpdif` | `Colexpandexpdif` | ✅ | ✅ | — | 2 | row | — | — | 2 | 1 | 单轴 |  |
| 111 | 广播 / 扩展 | `pto.tcolexpandmax` | `Colexpandmax` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 112 | 广播 / 扩展 | `pto.tcolexpandmin` | `Colexpandmin` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 113 | 比较 / 选择 | `pto.tcmp` | `Cmp` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 114 | 比较 / 选择 | `pto.tcmps` | `Cmps` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 115 | 比较 / 选择 | `pto.tsel` | `Sel` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 116 | 比较 / 选择 | `pto.tsels` | `Sels` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 117 | 比较 / 选择 | `pto.tci` | `Ci` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 118 | 比较 / 选择 | `pto.ttri` | `TTri` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 119 | 类型转换 / 量化 | `pto.tcvt` | `Cvt` |  | ✅ | — | 1 | row | — | — | 3 | 0 | 单格式 |  |
| 120 | 类型转换 / 量化 | `pto.tquant` | `Quant` |  | ✅ | — | 0 | col+row | — | — | 2 | 1 | 单轴 |  |
| 121 | 类型转换 / 量化 | `pto.tquant.mx` | `TquantMx` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 122 | 类型转换 / 量化 | `pto.tdequant` | `Dequant` |  | ✅ | — | 0 | row | — | — | 2 | 0 | 单格式 |  |
| 123 | 类型转换 / 量化 | `pto.tget_scale_addr` | — |  | — | — | — | — | — | — | — | — | 无样例 | 内部 MX 地址绑定 op，无独立数值输出；由 Gemvmx 样例端到端覆盖，建议豁免 |
| 124 | 填充 / 补齐 | `pto.tfillpad` | `Fillpad` |  | ✅ | — | 2 | row | — | — | 5 | 1 | 单轴 |  |
| 125 | 卷积（ConvTile 路径） | `pto.timg2col` | `Img2col` |  | ✅ | — | 2 | col+row | — | — | 1 | 2 | 双轴 |  |
| 126 | 卷积（ConvTile 路径） | `pto.set_fmatrix` | `ConvPipeline` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 127 | 卷积（ConvTile 路径） | `pto.set_img2col_rpt` | `ConvPipeline` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 128 | 卷积（ConvTile 路径） | `pto.set_img2col_padding` | `ConvPipeline` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 129 | 数据重排 / 索引访存 | `pto.tgather` | `Gather` | ✅ | ✅ | — | 1 | row | — | — | 2 | 0 | 单格式 |  |
| 130 | 数据重排 / 索引访存 | `pto.tgatherb` | `Gatherb` |  | ✅ | — | 0 | row | — | — | 1 | 0 | 单格式 |  |
| 131 | 数据重排 / 索引访存 | `pto.tscatter` | `Scatter` |  | ✅ | — | 0 | row | — | — | 1 | 0 | 单格式 |  |
| 132 | 数据重排 / 索引访存 | `pto.mgather` | `Mgather` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 133 | 数据重排 / 索引访存 | `pto.mscatter` | `Mscatter` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 134 | 数据重排 / 索引访存 | `pto.tconcat` | `Concat` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 135 | 数据重排 / 索引访存 | `pto.tconcatidx` | `TConcatIdx` |  | ✅ | — | 0 | — | — | — | 0 | 0 | 单格式 |  |
| 136 | 数据重排 / 索引访存 | `pto.tinterleave` | `Interleave` |  | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 137 | 数据重排 / 索引访存 | `pto.tdeinterleave` | `DeInterleave` |  | ✅ | — | 2 | row | — | — | 2 | 1 | 单轴 |  |
| 138 | 数据重排 / 索引访存 | `pto.tmrgsort` | `Mrgsort` | ✅ | ✅ | — | 6 | row | — | — | 3 | 1 | 单轴 |  |
| 139 | 数据重排 / 索引访存 | `pto.tsort32` | `Sort32` | ✅ | ✅ | — | 1 | row | — | — | 1 | 0 | 单格式 |  |
| 140 | 数据重排 / 索引访存 | `pto.textract` | `Extract` | ✅ | ✅ | — | 1 | col+row | — | — | 2 | 1 | 单轴 |  |
| 141 | 数据重排 / 索引访存 | `pto.tinsert` | `TInsert` | ✅ | ✅ | — | 1 | col+row | — | — | 2 | 1 | 单轴 |  |
| 142 | 数据重排 / 索引访存 | `pto.treshape` | `Reshape` | ✅ | ✅ | — | 1 | col+row | — | — | 4 | 1 | 单轴 |  |
| 143 | 杂项 / 调试 | `pto.tsetval` | `Tsetval` | ✅ | ✅ | — | 0 | row | — | — | 1 | 0 | 单格式 |  |
| 144 | 杂项 / 调试 | `pto.tgetval` | `Tgetval` | ✅ | ✅ | — | 2 | row | — | — | 1 | 1 | 单轴 |  |
| 145 | 杂项 / 调试 | `pto.tprint` | `Print` | ✅ | ✅ | — | 1 | row | — | — | 3 | 0 | 单格式 |  |
| 146 | 杂项 / 调试 | `pto.tdump` | — |  | — | — | — | — | — | — | — | — | 无样例 | 需要 vpto 后端，超出 samples 层作用域 |

> 注：本表统计"与 op 同名的样例目录"。部分 op 在非同名目录/其它层仍有覆盖（如 tload/tstore 见 `TPushTPop`、
> `TPipe`、整网样例；跨层视图见 `OP_INVENTORY.md`）。

