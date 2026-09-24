# 当前 PDblend Planner：决策、调度与可复算例子

本文按 2026-09-24 工作区源码解释通用 `PoolPlanner` 和正式 `pdblend` 策略，源码哈希见 [example.json](example.json)。不把旧实验冻结源码、实验策略或离线布局能量变体当作默认在线实现。

完整图：[PNG](planner-current.png) · [SVG](planner-current.svg) · [PDF](planner-current.pdf)。复算：[reproduce.py](reproduce.py)。全部数字来自仓库 synthetic profile，是当前代码的 CPU 计算例子，不是 GPU 实测结果。未修改调度实现。

**1. 先分清三个层次。**

Planner 为一个池决定部署配置：P（只处理 prefill）、D（接收 KV 后 decode）、M（同一实例同时承担 prefill 和 decode），以及 idle/L1/off 状态、各角色频率和输入长度分流阈值 τ。Controller 将计划落到具体实例；Router 在请求到达时选择实例并记账；引擎才负责实例内部的队列、continuous batching、chunked prefill 和 token 执行。Planner 没有逐 token 中央 FIFO，也不逐条决定引擎 batch。

这里的一个 slot 是一个模型实例，不一定是一张 GPU。TP1 时 8 张卡对应 8 slots；TP2 时对应 4 slots；TP4 时对应 2 slots。P 和 D 分离需要两个独立实例，实例之间必须满足当前 KV 传输的拓扑约束。

旧入口 `src/pdblend/control/planner.py` 只有兼容转发；真正实现是 [pool.py](../../src/pdblend/planner/pool.py)。同理，在线实现已移至 `src/pdblend/online/`。

**2. 当前默认策略与通用类默认值不同。**

| 参数 | 正式 `pdblend` 策略 / 常规运行入口 | 用处 |
|---|---|---|
| 规划周期 | 10 s，可由运行参数覆盖 | 周期性重新预测与选方案 |
| Shield 检查 | 1 s | 观察风险；级别改变也可触发重规划 |
| 能量比较窗口 H | 60 s | 比较运行能量和转换成本，不是每 60 s 才决策 |
| safety | 0.85 | TTFT、TPOT 预测线设为 SLO 的 85% |
| margin | 8% | 当前可行时，新方案必须严格节省超过 8% 才切换 |
| 最小计划保持 | 30 s | 一般计划变更先过保持期 |
| 缩容/降频确认 | 2 个独立规划窗口 | Shield 的 1 s tick 不算额外投票 |
| M 实例下限 | min(4, slots) | 正式策略保留容量下限；可由匹配资格文件显式调整 |
| home_margin | 1% | 初始方案重新可行且比候选功率低超过 1% 时，可作为返回候选，再走控制门控 |
| P 频率候选 | profile 中存在的 2100、2520 MHz；均不存在则用最高档 | 缩小 prefill 搜索空间 |
| D/M 频率候选 | profile 支持的频率集合 | 逐角色独立枚举 |
| τ 候选 | 0、1024、4096 tokens | 仅 P/D 与 M 共存时有分流意义 |
| 停驻候选 | L1、off | L1 保留进程/权重并压低时钟；off 停进程，仍有静态功耗 |
| ρP/ρdecode | 0.7 / 0.92 | prefill 占用与 decode 吞吐容量保护 |
| mixed tail_target | 0.9 | 预测 M 请求平均 TPOT 因 prefill 干扰超标的比例 ≤10% |
| KV 上限 | 模型容量的 90% | 留出余量 |
| max_num_seqs | 从实际实例配置同步 | 通用类默认 256，不应代替实际引擎配置 |

通用 `PlannerConfig` 自身的 margin=3%、min_M=0；`Controller` 自身 hold=0、votes=1。上表是 `POLICIES['pdblend']` 及 `_make_controller` 组合后的行为，不能仅看 dataclass 默认值。例子显式采用 max_num_seqs=256、peak_batch_cap=160；实际频率与容量随 profile/实例配置变化。

**3. Planner 从哪里获得信息。**

Forecaster 从代理请求记录得到到达时刻、输入长度、完成输出数及请求 ID。默认每秒形成到达计数，更新约 30 s 和 120 s 两条 EWMA；近期长度样本窗为 120 s。设短 EWMA 为 s、原始近期到达率为 r，最终到达率为 `max(s, (s+r)/2)`；s=0 时直接采用 r。`trend_rps=s−long` 会记录，但普通 PoolPlanner 并未拿它另加一个趋势惩罚项。

启动可用离线轨迹统计作为先验，随首个流量到达后的一个样本窗渐退。当前 benchmark 的 warm start 确实使用离线 trace，不应描述为完全没有先验的纯在线预测。样本不足时，普通路径等待至少 30 个输入样本且观察到流量约 20 s；正式 pdblend 保持初始计划，其他无初始方案路径可全 M 最高频保守启动。

输出条件统计依赖同一 request ID 对应的 `(input, output)`，不会把独立到达与完成队列直接 zip。分支流量占比来自到达输入长度，防止短请求较快完成造成比例偏差。没有该分支的配对样本时，输出均值回退为全局估计。

此外，Controller 每 tick 从 `router.active` 建立完整 backlog，不因请求超过统计窗口而丢掉：

- waiting_prefill_tokens：还没观察到首 token 的请求输入总量；不是引擎实际剩余 prefill token 的精确测量。
- remaining_decode_tokens：`max(0, max_tokens − tokens_so_far)` 的总量，属于基于请求输出上限的保守估计。
- occupied_kv_tokens：输入 token 加已生成 token 的估计；不是逐 rank KV allocator 实测。
- 请求原有分支和 pool ownership：改 τ 只影响新请求；已经派发的请求不会按新阈值迁移。
- uncertain/cancel-pending 请求在原生确认清理前继续保留 ownership 和资源预留。

**4. 枚举什么，为什么是“联合决策”。**

候选表示为 `x=(nP,nD,nM,nL1,noff,fP,fD,fM,τ)`，实例数之和为 slots。P 和 D 必须同时存在；至少存在 M 或完整 P/D；正式策略的 M 下限和现有 backlog 分支都必须保留。L1/off 可以同时存在。

当 M 与 P/D 共存时，以 `input ≥ τ` 分给 PD，其余给 M。Planner 为两个分支分别重算输入均值、输入 p95、输出均值和绑定 backlog，然后在各自的负载下评估 P、D、M。这意味着改阈值同时改变三类池的压力，不能独立挑 τ 后再任意缩容。

τ=0 通常将所有新流量交给 PD；若 M 没有未来流量也没有已绑定 backlog，这种 PD+M 空分支候选会被剔除。纯 M/纯 PD 的 τ=0 只作占位。普通正式策略 min_M>0 时枚举不会产生纯 PD；实验 pressure 策略才有专门的纯 PD 逃生分支。

为避免短输入付出过高的 PD 固定成本，默认要求全局输入 p95 及 PD 分支平均输入满足 min_pd_input_tokens=256。模型查询缺少 coverage/组件时，角色候选可被淘汰；不保证所有缺 profile 情况都能靠 fallback 恢复。

**5. 如何预测能不能满足 SLO。**

下面 H=60 s；I/O 为分支平均输入/输出；Qp/Qd 为分支待 prefill/剩余输出 token。

`λP = λ + Qp/(H·I)`，`λD = λ + Qd/(H·O)`。

这是把存量工作摊到规划窗口，叠加未来到达。上下文估计取 `max(I+O/2, 已占KV/在途数)`；平均 batch 还至少为 `在途请求数/实例数`，因此“近期没有新请求”不等于可以把仍在工作的池关掉。

P 池使用 profile 的完整 prefill 时间 s_full 和增量 token 时间 s_m。每实例 `u=λP·s_m/nP`，达到 0.7 即拒绝；通过队列近似、批处理周期、p95 输入计算 TTFT，再加 KV 传输时间和 backlog 等待。批周期近似为 `max(s_full−s_m,0)/(1−u)`；等待是 M/D/1 近似和周期两者较小值。模型包含有限阶近似，并非请求级离散事件模拟。

D 池解 Little 定律固定点：`B = λD·O·step(B,ctx,fD)/nD`，最多 32 次迭代。检查请求输出 token/s 不超过峰值吞吐的 92%，检查模型覆盖、batch cap 和 KV 容量；TPOT 取 decode step 时间。峰值吞吐在 B=8,16,…,min(max_num_seqs,peak_batch_cap) 的模型网格上搜索。该均值 B 不是某一时刻原生实际 batch。

M 池同时处理两阶段。prefill 时间占比 `uP=λP·s_m/nM`；其解为 `B=λD·O·step(B,ctx,fM)/(nM·(1−uP))`。TPOT 也被 `1/(1−uP)` 放大，代表 prefill 挤占 decode 时间；TTFT 为等待 + p95 prefill + TPOT。还会对长度样本做分位取样和 Poisson stall 估计，检查“被其他 prefill 打断导致本请求平均 TPOT 超标”的预测比例是否 ≤10%。有配对样本时最多取 36 个配对分位点；否则采用 6×6 输入/输出组合。

有边界的 profile 计算非空 decode step 时使用至少 B=1；低于 1 的平均占用通过 idle duty cycle 估计功率。M 有对应实测 mixed-window 功率查询时优先采用；否则通用规划器可采用 `uP·Pprefill+(1−uP)·Pdecode` 的组合估计，标记 `composed_unqualified`。这正是本合成例子的状态。

最终要求每条启用分支 TTFT ≤0.85×SLO_TTFT、TPOT ≤0.85×SLO_TPOT，同时通过容量和 M tail 约束。总 Plan 的 TTFT/TPOT 分别取分支最大值。这些是预测代理量，不是客户端 p99 或 joint attainment 的数学保证。

**6. 如何从可行候选中选最优。**

先产生完整可行候选集合。每轮对相同 `(role, branch forecast, 实例数, 频率)` 缓存结果，避免重复角色查询；缓存每轮重置，不改变候选集合。

没有当前方案时，按稳态预测功率选择；功率相同优先更多 active。存在当前方案时，对每一个候选计算：

`J(x | current) = H × P(x) + E_switch(current → x)`。

运行功率包含 P/D/M 和所有 L1/off 静态功率。切换成本优先采用通过资格验证的转换目录；启用 qualified_only 且某转换没证据时，成本为无穷并跳过该转换。没有合格数值、且允许 fallback 时，采用旧估计：唤醒耗时×最高频 active_idle 功率，以及频率三元组改变时 `freq_switch_s×active_idle功率×新active数`。

旧估计没有完整计算停机、重路由、drain、KV 增量能量等所有转换开销；`E_switch=0` 仅表示当前估计没有计入额外项，不是操作真实无能耗。其频率成本会对新方案所有 active 实例收费，并不只算实际改频的实例。

当前计划需要在新负载下重评，不能使用旧的功率/可行性。若当前仍可行，只有 `Jbest < 0.92×H×Pcurrent` 才采纳；等于边界也保持。若当前不可行，则选最低成本可行候选，不要求节能。Planner 的这一决定仍要通过 Controller 的保持期/门控；紧急 Shield 在门控后可覆盖。

若可行候选集合为空，fallback 尽量全开最高频。没有绑定 PD backlog 时通常 M 全开；有 PD/M backlog 时先保留现有路径。fallback 并不证明满足 SLO；特定缺能量覆盖情况还会抛出错误。

**7. 数值例子：稳态最低功率为什么没赢。**

明确条件：8 张 GPU、TP1/PP1、8 slots；采用正式 pdblend 的 min_M=4、margin=8%、H=60 s；SLO 为 TTFT 1 s、TPOT 20 ms，所以筛选线为 850 ms/17 ms；没有 backlog。synthetic profile 频率为 900/1200/1500/1800/2100/2520 MHz。当前 A 是全 M8、2520 MHz，是为了演示有当前方案时的决策，非宣称每次默认启动都是 A。

到达率 18 req/s。输入 256、512、2048、4096 各占 25%，输入均值 1728，p95=4096；每个输出先为 64。τ=1024 时，M 接收前两类，共9 req/s，平均输入384；PD 接收后两类，共9 req/s，平均输入3072。

| 候选 | 角色配置 | P/D/M MHz | τ | 功率 W | TTFT ms | TPOT ms | 切换 J | 60s总成本 kJ | 可行 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| A 当前 | M8 | —/—/2520 | 0 | 1016.039 | 212.575 | 14.976 | 0 | 60.962 | 是 |
| B | M6+off2 | —/—/2520 | 0 | 894.612 | 216.895 | 16.326 | 0 | 53.677 | 是 |
| C 稳态最低 | P2+D1+M4+off1 | 2520/2520/2100 | 1024 | 875.673 | 281.765 | 15.629 | 47.6 | 52.588 | 是 |
| D 最终选择 | P2+D1+M4+off1 | 2520/2520/2520 | 1024 | 875.714 | 281.765 | 14.454 | 0 | 52.543 | 是 |
| E | P2+D1+M4+off1 | 2100/2520/2520 | 4096 | 893.409 | 317.377 | 14.620 | 47.6 | 53.652 | 是 |
| F | M5+off3 | —/—/2520 | 0 | 834.150 | 220.853 | 17.596 | 0 | 50.049 | 否，TPOT>17 |

共40个可行候选，表里仅挑代表，不是只枚举6种。F 的功率虽低，但不能进最优化集合。C 每秒只比 D 少0.041167 W，60s仅省2.470 J；C 的改频估计多47.6 J，因此 D 总成本反而少45.130 J。D 相比 A 的模型成本收益为13.8109%，超过8%，Planner采纳 D。

D 的总功率可展开为 P池327.065 + D池111.649 + M池403.000 + off34 =875.714 W。D池 batch≈8.325、TPOT≈14.454ms；M池每实例 batch≈1.849、prefill占比≈3.49%、TPOT≈12.842ms；取较大者形成计划TPOT。PD的TTFT≈267.311+14.454=281.765ms。这些分解均来自实际调用当前 planner，不是手填的示意数值。

**8. 计划如何变成真实请求路径。**

`assign_roles` 尽量保留已有角色，再按当前状态深度和代理负载选择可改实例。按 G0…G7 的当前字典顺序、起始全 M、无额外负载，D 被映射为：G0–G3=M，G4/G5=P，G6=D，G7=off。保留同角色的第一轮按迭代顺序，不是全局最优的装箱/迁移求解。

取此时代理负载：M的reserved sequences分别为 G0=2、G1=0、G2=1、G3=3；P待首token输入总量 G4=4096、G5=1024；D的G6预留序列为8。

- 新短请求 Rs=(256输入,64输出)：256<1024，选 M。Router先比较在途序列，再以待prefill输入打破平局，所以选G1。G1序列0→1；首token后解除prefill记账；请求完成后序列1→0。
- 新长请求 Rl=(2048,64)：2048≥1024，选PD。比较P端pending tokens，再比较D端序列和pending，选G5→G6。G5的pending输入1024→3072；G6序列在P开始前就8→9。首token到达时G5回1024；请求完成后G6回8。
- PD首个真实输出token由P产生并先转发给客户端。D拿到原prompt的KV，以带首token的2049-token prompt继续，剩余输出预算63。2048-token KV在合成模型为112MiB，transfer模型为28.488ms；这不是新的GPU传输实测。max_tokens=1时有P_ONLY特例，无远程KV续接。

普通 Router 使用阈值和上述最小负载规则；`inflight_seqs` 包括尚在P执行但已预留D的请求，不等于原生decode batch。PD配对要求P/D不同实例、PP1且TP/PP/pool/generation/model一致。普通Router本身没有每请求预测能耗排名；异构ResidentRouter才有额外容量、上下文和模型覆盖筛选，以及可选的增量能量排名。

执行方面：活跃P/D/M的角色更改更新新请求路由，已派发请求保持绑定；并非所有角色变化都先drain。停车先禁新准入，等待代理prefill/序列排空，接入native_control时再做原生drain，然后L1压频或off停进程。超时保持禁入并报错，不会直接杀仍在途工作。唤醒先start/ready或unpark，再设频、原生resume，最后开放路由；动作失败隔离受影响实例。

**9. 负载变化后怎么重新决策。**

保持到达率和输入分布不变，将配对输出统计从64变为128，仍以空backlog的规划截面说明（不是假设实际请求瞬间全部完成）。旧D方案在新负载下为916.438 W、TPOT18.281ms，超过17ms安全线。此时有17个可行候选。

Planner选择原来的E布局：P2+D1+M4+off1不变，τ从1024升到4096，P频率从2520降到2100；D/M仍2520。2048-token新请求转到M，PD到达率9→4.5，M到达率9→13.5，减轻唯一D实例的输出压力。新计划TPOT15.576ms、TTFT318.918ms、功率927.670 W；60s含47.6J改频成本为55.708kJ，高于旧D在新负载下54.986kJ，但当前D不可行，所以仍选E。

改τ只影响新请求，旧2048-token PD请求继续在原P/D上；有真实backlog时必须重新计算，结果不一定等于上述空backlog例子。P降频属于downshift，因此线上执行还要过30s保持期、两次同一候选确认和Shield保护，不能把planner返回时间画成实际执行时间。

若进一步把输出改为256，本合成域严格可行候选为0；在无绑定PD backlog的截面，fallback为M8最高频，TPOT17.408ms，仍超过17ms安全线。它小于20ms原SLO，但仍不能由此保证真实客户端达标。fallback的切换能量不参与“可行候选排名”；example.json中后算的2094.4J只作审计。

另一个小例子：在途同分支Qp=61,440token、I=1024，H=60，则prefill等效到达率增加1req/s；Qd=15,360、O=128，则decode等效率增加2req/s。到达率即使下降，也要为这些未完成工作保留资源。

**10. Shield为什么会覆盖Planner。**

Shield默认每1s观察：最近5s的TTFT/TPOT p90、全部仍在等待首token的请求、未完成请求最近一次token距今多久。超过0.8×SLO即有压力，级别最多每2s升一级。level1提到最高频，level≥2在可用停车实例范围内增加active；已有PD时按prefill/decode压力补P或D，纯M部署补M。没有空闲实例时无法凭空增加GPU。

普通计划先过保持/缩容门控，Shield随后应用，所以它能在普通30s保持期内提频或唤醒。新增容量形成floor_active；持续安静后每30s下降一个保护级别，再通过安静探测逐实例释放容量；失败则回退并延长探测窗，最高8倍。普通pdblend的额外protect_s为0，实验dominance策略才配置60s保护租约和45s转换冷却。

因此“每10s找到最低功耗方案”并不等于“每10s一定改部署”，60s目标窗口也不等于30s保持期；三者各自解决预测成本、执行抖动与即时风险。

**11. TP与其他扩展的实际范围。**

- `TopologyPlanner`：在有模型、内存和profile支持的TP1/2/4、PP1上执行内层PoolPlanner，典型为离线选TP；不是每10s任意拆合GPU。8卡TP2有4slots，默认M下限夹为4；TP4有2slots，下限夹为2，所以通常仅剩全M的频率选择。
- `ResidentAllocationPlanner`：显式启用双驻留TP池时，外层枚举0、0.1、…、1流量比例并加入当前比例/实发反馈；内层各自选角色/频率。所有驻留实例及未使用静态功率都计费。默认要求有界timing、decode power和实测mixed能量；缺失时拒绝联合方案，由调用侧保持服务配置。
- `ResidentRouter`：满足TTFT、TPOT、KV、并发和coverage的候选才可排名；可选经验证的incremental J证据完整时按J排序，否则该批候选整体回到时间评分。目标份额通过weighted deficit实现，原目标池不可行时可转其他池。
- `pdblend_dominance`：显式实验策略，增加压力门控（通常0.75进入/0.55退出）、小M区域验证、保持/稳定窗口和纯PD逃生路径；默认pdblend不自动启用。
- `NativeLayoutPlanner`：独立且显式的32B TP2/M4/native32布局能量修订，直接查询整8卡layout功率；当前要求空backlog和限定频率/范围，主要用于离线组件重放，不是通用 `_make_controller` 默认创建的PoolPlanner。
- `slow_reshard_tp`：原生TP重分片仍有独立验收门槛，不能把角色重写或驻留双池分流描述成在线TP重分片已全面支持。

**12. 如何复核。**

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pdblend/.venv/bin/python \
  /home/pdblend4/docs/planner-current-2026-09-24/reproduce.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/pdblend4/src \
  /home/pdblend/.venv/bin/python -m pytest -q \
  /home/pdblend4/tests/pdblend/test_planner.py \
  /home/pdblend4/tests/pdblend/test_planner_progressive.py
```

复算脚本验证全枚举选择、加入转换成本后的第二名胜出、输出增长后的τ变更、无解fallback、实际Router选择和计数释放。本轮相关测试26/26通过。运行源码、算例JSON与静态图生成脚本都保留在本目录。合成功率、队列近似和旧转换估计仅用于解释决策机制，不能据此宣称实测节能13.81%。

配套交互图完成360px/736px、亮色/暗色、三种输出统计的12组合浏览器检查；切换更新正常，没有脚本错误、横向溢出或SVG标签重叠/越界。

| 要核对的逻辑 | 当前源码位置 |
|---|---|
| 正式策略8% / min_M4 / hold30 / votes2 | [policies.py](../../src/pdblend/online/policies.py#L62) |
| 输出条件分支与已有work ownership | [forecast.py](../../src/pdblend/planner/forecast.py#L72) |
| P / D / M模型 | [pool.py](../../src/pdblend/planner/pool.py#L251) |
| 布局可行性 | [pool.py](../../src/pdblend/planner/pool.py#L396) |
| 枚举与缓存 | [pool.py](../../src/pdblend/planner/pool.py#L473) |
| 切换能量 | [pool.py](../../src/pdblend/planner/pool.py#L533) |
| 全候选总成本排序与迟滞 | [pool.py](../../src/pdblend/planner/pool.py#L578) |
| 实例角色映射 | [pool.py](../../src/pdblend/planner/pool.py#L658) |
| 普通逐请求选路和记账 | [router.py](../../src/pdblend/online/router.py#L219) |
| 控制门控与Shield覆盖顺序 | [controller.py](../../src/pdblend/online/controller.py#L410) |
| 快保护规则 | [shield.py](../../src/pdblend/online/shield.py#L33) |
