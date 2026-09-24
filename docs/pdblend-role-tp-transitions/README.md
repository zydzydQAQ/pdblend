# Role change 与 TP 变换图

两图使用同一套实例、GPU 和资源计数规则，分别展示固定 TP 的角色/副本变换，以及改变实例 TP 的拓扑变换。GPU 编号是说明性示例；本次未执行 GPU 变换实验，也没有给出虚构的耗时或性能测量。

| 图 | PNG 预览 | SVG 矢量图 | PDF |
|---|---|---|---|
| TP1 role change | [role-change-tp1.png](role-change-tp1.png) | [role-change-tp1.svg](role-change-tp1.svg) | [role-change-tp1.pdf](role-change-tp1.pdf) |
| TP change | [tp-change.png](tp-change.png) | [tp-change.svg](tp-change.svg) | [tp-change.pdf](tp-change.pdf) |

合并文档：[transitions.pdf](transitions.pdf)。

## 统一读图规则

- `P`：prefill 实例；`D`：decode 实例；`M`：mixed 实例，在同一个实例内完成 prefill 和 decode。
- 一个实例是一个 vLLM engine / GPU group。本文均为 `PP=1`，因此一个 `TP=t` 实例恰好占 `t` 张 GPU；TP1 的每个小实例框只有一张 GPU。
- `PD` 大框是逻辑服务单元，内部至少有一个 P 实例和一个 D 实例。P、D 各自持有完整模型的分片集合，不能理解为各存一半模型层。
- P→D 箭头是请求的 KV handoff，不是模型权重的拆分。兼容池内可以有多个 P 和多个 D，路由不要求固定一对一配对。
- `S` 统称停驻资源：`idle`、`L1`、`off`。示例中的“增加/删除 GPU”指活跃分配增减，物理 GPU 总数仍守恒。
- 图二的 `F` 表示旧引擎的独占资源已释放、可重新分配的 GPU；它与图一可能仍驻留进程和权重的 `S` 不同。

一般式：令 `p,d,m,s` 分别为 P、D、M、停驻实例数量，在同构 TP 池中

```text
N_instance = p + d + m + s
G_active   = t × (p + d + m)
G_assigned = t × (p + d + m + s)       （PP = 1）
```

TP1 时，实例数就是分配的 GPU 数。多个异构池的资源量按各自 TP 求和。合法服务布局须有 `m>0` 或同时 `p>0,d>0`；孤立的 P 或 D 不能构成完整 PD 路径。实际候选还必须满足模型显存、profile 覆盖、SLO、M floor 与 backlog 约束。

源码依据：[launcher.py 第 19、43 行](/home/pdblend4/src/pdblend/engine/launcher.py:19)，[pool.py 第 392–413 行](/home/pdblend4/src/pdblend/planner/pool.py:392)，[router.py 第 205–237 行](/home/pdblend4/src/pdblend/online/router.py:205)。

## 图一：固定 TP1 的 role change

### 12 个有向角色原语

四个状态 `P、D、M、S` 之间的六条双向边覆盖所有 12 个有向原语。`S` 的具体停驻深度另行展开。

| 原语 | 逆向原语 | 当前执行方式 |
|---|---|---|
| P→D | D→P | 保留实例及 GPU，修改角色与目标频率；旧请求仍绑定原实例 |
| P→M | M→P | 同上；改变后续请求的服务方式 |
| D→M | M→D | 同上；改变后续请求的服务方式 |
| P→S | S→P | 排空并停驻 / 唤醒为 P |
| D→S | S→D | 排空并停驻 / 唤醒为 D |
| M→S | S→M | 排空并停驻 / 唤醒为 M |

**Active→active 不统一排空。** 当前 `_activate()` 改频后更新角色表、打开准入，明确记录 `existing_requests_pinned=True`。已有请求继续使用原来记录的 prefill/decode 实例和 KV；角色变更不迁移它们。因此过渡时某个新标为 P 的实例仍可能完成旧 decode 工作。这是当前实现行为，不应画成每次角色切换都重启引擎。

**Active→S 才关闭准入并排空。** 顺序为关闭准入→proxy drain→native drain（配置后）→停驻动作。proxy 同时检查 `inflight_seqs=0` 和 `inflight_prefill_tokens=0`，不能只看 decode 序列数。PDblend benchmark 为其 controller 配置 native control；native 回执进一步验证各 rank、waiting/running、保留 KV、pending transfers 和 KV allocation 已释放。

| 停驻状态 | 排空后保留什么 | 唤醒步骤 |
|---|---|---|
| idle | 进程及权重保留，重置时钟 | 设频、native resume、发布角色与准入 |
| L1 | 进程及权重保留，SM/memory 时钟降至停驻档 | unpark 后执行唤醒步骤 |
| off | 引擎进程停止 | start→ready 后执行唤醒步骤 |

停驻内部还有 `idle↔L1`、`idle↔off`、`L1↔off` 六个方向，属于改变停驻深度。`idle/L1` 不是可以直接供另一个引擎占用的空 GPU；复用其显存必须先释放原引擎资源。

依据：[controller.py 第 141–190 行](/home/pdblend4/src/pdblend/online/controller.py:141)、[第 241–267 行](/home/pdblend4/src/pdblend/online/controller.py:241)，[native_control.py 第 15–49 行](/home/pdblend4/src/pdblend/online/native_control.py:15)，[bench/run.py 第 158 行](/home/pdblend4/src/pdblend/bench/run.py:158)。注意 `pool.py` 第 590 行的旧注释仍泛称所有角色切换先 drain，本文以实际 `_activate()` 实现为准。

### Add / delete GPU 完整覆盖

下表每一条 `↔` 同时表示增加及其逆向删除，括号内为该实例拥有的 GPU。

| 场景 | Before ↔ After | 约束或含义 |
|---|---|---|
| Mixed pool ±1 GPU | `M(G0)+S(G1) ↔ M(G0)+M(G1)` | 增删 M replica，原 M 的 TP 不变 |
| PD 增删 P | `P(G0)+D(G1)+S(G2) ↔ P(G0)+P(G2)+D(G1)` | 缩减后仍保留 P |
| PD 增删 D | `P(G0)+D(G1)+S(G2) ↔ P(G0)+D(G1)+D(G2)` | 缩减后仍保留 D |
| Mixed 与 P 互借 | `P(G0)+D(G1)+M(G2) ↔ P(G0)+D(G1)+P(G2)` | 总活跃 GPU 数不变 |
| Mixed 与 D 互借 | `P(G0)+D(G1)+M(G2) ↔ P(G0)+D(G1)+D(G2)` | 总活跃 GPU 数不变 |
| P/D 内部重平衡 | `P(G0)+P(G1)+D(G2) ↔ P(G0)+D(G1)+D(G2)` | 一张卡由 P 改 D 或逆向 |
| 单卡 M 与双卡 PD | `M(G0)+S(G1) ↔ P(G0)+D(G1)` | 增加一张活跃卡形成 PD；逆向缩为 M |
| 保留另一张卡的对称情况 | `S(G0)+M(G1) ↔ P(G0)+D(G1)` | 逆向保留 G1，释放 G0 |
| 整个服务单元启停 | `S(G0)↔M(G0)`；`S(G0)+S(G1)↔P(G0)+D(G1)` | 删除后若仍需服务，须由其他完整路径接纳新请求 |

一般增删式为 `p→p±k`、`d→d±k`、`m→m±k`，并在停驻池或其他角色池中反向调整相同数量的实例。当前 controller 操作已有 fleet；不是热插物理 GPU 或动态追加任意 `InstanceSpec` 的接口。

**最后一个 P 或 D 的删除要特别处理：** 不能留下 `p=0,d>0` 或 `p>0,d=0` 作为 PD 稳态。必须替换该角色、把剩余实例转为 M，或完整撤销该 PD 单元。图中的全部局部变换是覆盖集合，并不表示任意负载下都满足 planner 的全局约束。依据：[pool.py 第 468–492 行](/home/pdblend4/src/pdblend/planner/pool.py:468)、[controller.py 第 196–218 行](/home/pdblend4/src/pdblend/online/controller.py:196)。

### Merge / split 的含义和条件

| 类型 | 双向例子 | 应如何理解 |
|---|---|---|
| 两个 M 与一个 PD 单元 | `M(G0)+M(G1) ↔ [P(G0)→D(G1)]` | 角色专门化 / 改回 Mixed；两边始终是两个实例 |
| PD 吸收 / 释放 M | `[P(G0)→D(G1)]+M(G2) ↔ [{P(G0),P(G2)}→D(G1)]` | 也可令 G2 加入 D；split 后原 PD 仍须完整 |
| M pool 合并 / 拆分 | `{M0,M1}+{M2,M3} ↔ {M0,M1,M2,M3}` | 改变逻辑路由集合，四个实例均保留 |
| PD pool 合并 / 拆分 | `{P0,D1}+{P2,D3} ↔ {P0,P2;D1,D3}` | 每个独立 PD 子池必须各有 P 和 D |

当前没有独立 `merge()` / `split()` 操作。前两行可组合已有角色原语；后两行是逻辑 pool 分组示意。单个 Router 默认枚举兼容 P×D，画出两个 PD 子框不等于已经实现路由隔离。若分组涉及修改 `pool_id`、generation 或拓扑身份，须配置对应路由；`set_instance_metadata()` 在仍有 inflight 时会拒绝身份修改。

**物理实例的 merge/split 属于 TP 图：** `M_TP1(G0)+M_TP1(G1)` 合成占两张卡的一个实例，就不再是 TP1；一个 TP1 实例也不能凭空拆出第二张 GPU。依据：[router.py 第 100–111 行](/home/pdblend4/src/pdblend/online/router.py:100)、[第 205–212 行](/home/pdblend4/src/pdblend/online/router.py:205)。

## 图二：TP 增加与减少

### 四类变换及资源守恒

下列四类均保持角色，改变每个实例的 tensor parallel 宽度。`t→u` 时不能仅改标签，必须形成满足目标 TP 的 GPU group 和权重布局。

| 类型 | 保持实例数的例子 | 活跃 GPU 变化 |
|---|---|---:|
| PD TP 增加 | `P1(G0)+D1(G2)+F(G1)+F(G3) → P2(G0,G1)+D2(G2,G3)` | 2→4 |
| PD TP 减少 | `P2(G0,G1)+D2(G2,G3) → P1(G0)+D1(G2)+F(G1)+F(G3)` | 4→2 |
| Mixed TP 增加 | `M1(G0)+F(G1) → M2(G0,G1)` | 1→2 |
| Mixed TP 减少 | `M2(G0,G1) → M1(G0)+F(G1)` | 2→1 |

保持实例数时的一般式：

```text
Mixed：ΔG = m × (u − t)
PD：   ΔG = (p + d) × (u − t)        （P、D 目标均为 TP=u）
单 P/D 对：G = 2t，ΔG = 2(u − t)
```

TP 增加所需 GPU 可来自真正可复用的空闲卡，或排空并撤销的 donor 实例；TP 减少释放的卡可以成为停驻资源，或用于建立更多 replica。不能将仍驻留的 donor 与新目标实例重叠占用同一 GPU。

**固定 GPU 预算时，副本数可以同时变化：**

```text
Mixed merge：M1(G0) + M1(G1) ↔ M2(G0,G1)                 2 个实例 ↔ 1 个实例
PD merge：   P1(G0)+P1(G1)+D1(G2)+D1(G3)
             ↔ P2(G0,G1)+D2(G2,G3)                      4 个实例 ↔ 2 个实例
```

逆向分别是 Mixed 和 PD 同角色实例的 split。一般而言，`u=k×t` 时，`k` 个同角色 TP=t 实例可在**排空和重建的概念流程**中合成一个 TP=u 实例；逆向可拆成 `k` 个 TP=t replica。原副本的活跃 KV 不自动合并或分裂。

图二的 `W` 仅示意可按 TP 分片的权重部分；P、D 每个实例均须具有完整模型的分片集合。TP2 拆为两个 TP1 实例时，每个新 TP1 都须重新形成完整模型权重，不能把原来的半份权重直接当成完整模型。实际显存还包含复制张量、KV、workspace、通信缓冲和重建期间的临时峰值。

图例用 TP1↔TP2 展开；计数规则同样覆盖 TP2↔TP4、TP1↔TP4。当前候选 TP 为 `{1,2,4}`、PP1，并非每个模型都支持每个 TP；必须通过模型显存资格和对应 topology 的独立 profile 校验。P/D 对还须有 `2t` 的最低 GPU 预算。依据：[topology.py 第 48–64 行](/home/pdblend4/src/pdblend/planner/topology.py:48)、[tp_runtime.py 第 36–48 行](/home/pdblend4/src/pdblend/online/tp_runtime.py:36)。

### PD 单侧 TP 变换的兼容约束

当前 P2P connector 没有跨 TP 的 KV shape remap。同一个新 PD 请求要求 P、D 的 `model_id / TP / PP / pool_id / generation` 一致，且 PP1。因此：

- 不能把 `P1→D2` 或 `P2→D1` 画成可直接服务的 PD 路径。
- 若只升级一个 P 实例，必须已有同 TP、同身份的兼容 D 可配对；否则这个新 P 在补齐 D 前不能承接完整 PD 路径。只升级 D 完全对称。
- 图中的成对变换表示目标 P、D 都匹配后才发布目标 PD 路由。它不要求所有重建动作在物理上同一时刻完成，但过渡期间不得派发不兼容路径。
- 混合保留旧 TP 对和新 TP 对时，可以分别承接新请求；已有请求始终绑定自己的实例和 generation。这属于多个兼容路径并存，不是跨 TP handoff。

依据：[router.py 第 205–212 行](/home/pdblend4/src/pdblend/online/router.py:205)、[tp_modes.py 第 127–134 行](/home/pdblend4/src/pdblend/online/tp_modes.py:127)。

### TP 重建流程与当前实现边界

图二是布局变化及所需事务流程示意。当前运行入口在 `slow_reshard_tp` 模式直接抛出 `UnsupportedTPMode`，要求先提供经 GPU 资格验证的 native transaction backend。已有回执检查协调框架，不能将其当作可执行、已验证的在线 TP 重建实现。

概念执行顺序为：

1. 准备资源与目标身份；源/目标 GPU 重叠时，排空前只做 metadata prepare。
2. 关闭受影响源实例的新准入，完成旧请求，逐 rank 确认 inflight、live KV、pending transfers 清零。
3. 对需要复用的 GPU 释放旧引擎资源，建立目标 TP workers / communicator，从 checkpoint、host cache 或已验证的权重源加载目标分片，建立空 KV 池。
4. 验证权重身份、输出、KV、取消清理及逐 rank 事务回执；目标 P/D 必须兼容。
5. 目标就绪后发布路由与新 generation，开放新请求；完成源实例退役。未受影响实例可继续服务，但承载能力仍受自身容量限制。
6. 失败则恢复并验证源布局；无法证明恢复完成时隔离相关 GPU 并保留资源所有权。

协调框架的阶段为 `prepare→drain→transfer→verify→activate→retire`，并有 rollback / quarantine 分支；其中 `transfer` 明确要求 `live_kv_transferred=False`。具体 teardown、加载和 communicator 操作由尚需落地的 native backend 承担，不能把阶段名称理解为已实现零中断迁移。

`fixed_tp` 固定拓扑；`offline_tp` 在启动前选拓扑；`resident_hetero_tp` 使用预先存在、GPU 不重叠的不同 TP 池给新请求选路。它们都不能证明一个运行中实例已经改变 TP。已有的 resident 路由能力也不提供活跃 KV 跨 TP 迁移。

依据：[tp_runtime.py 第 67–86 行](/home/pdblend4/src/pdblend/online/tp_runtime.py:67)、[第 107–163 行](/home/pdblend4/src/pdblend/online/tp_runtime.py:107)，[reshard.py 第 153–227 行](/home/pdblend4/src/pdblend/online/reshard.py:153)，[tp_modes.py 第 191–207 行](/home/pdblend4/src/pdblend/online/tp_modes.py:191)。

固定 TP 角色控制本身也不是多实例原子事务：`Controller.execute()` 并发执行各实例动作，失败时关闭受影响实例准入；图中组合箭头表示目标布局，不额外承诺原子发布或无损 SLO。依据：[controller.py 第 196–239 行](/home/pdblend4/src/pdblend/online/controller.py:196)。

## 编辑、复现与检查

图由 [build_figures.py](build_figures.py) 以确定坐标绘制，可编辑 SVG 文字、GPU 块和箭头。PDF 嵌入中文字体，PNG 为高分辨率预览。依赖见 [requirements.txt](requirements.txt)，字体使用相邻 `pdblend-method/assets/` 中附带许可证的 Noto Sans CJK。

在独立文档环境安装依赖后，运行：

```bash
python /home/pdblend4/docs/pdblend-role-tp-transitions/build_figures.py
```

[validation.json](validation.json) 记录画布边界、PDF 页数和可检索字符数。另由三个 subagent 分别核对 role change、TP 机制及图示覆盖；生成后复查 GPU 守恒、实例边界、权重完整性、当前实现边界和文字可读性。所有生成与检查仅使用 CPU。
