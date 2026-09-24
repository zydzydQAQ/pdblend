# 当前在线控制层审计（2026-09-24 工作树）

审计对象为 `/home/pdblend4/src/pdblend` 当前工作树；未修改业务代码。以下“默认”特指 homogeneous `bench/run.py::_make_controller` 以 `policy=pdblend` 接线的路径，不能把 `Router()`/`Controller()` 类自身的兼容默认值当成 benchmark 当前配置。

## 1. 三种时间尺度，三个不同职责

1. **逐请求 Router**：使用已经发布的角色表与阈值 τ，选 `(path, prefill_instance, decode_instance)`；当前 runner 默认再加模型支持的 SLO admission / PD→M spillover。它不改变角色数量、GPU 时钟、停车状态。
2. **慢环 PoolPlanner / Controller**：默认每 10 s，根据 profile、SLO、短期需求预测及已绑定的 backlog，输出 `counts={P,D,M,idle,L1,off}`、`f_P/f_D/f_M`、τ。Controller 执行角色映射和硬件动作。
3. **快环 Shield**：Controller 每 1 s 观察首 token 和输出流，必要时提升 Shield level 并触发重规划；无需等到下一个 10 s 周期。第一档提满频，第二档及以后逐步唤醒更多容量。角色分配和请求分配是两个不同的决策。

来源：

- `/home/pdblend4/src/pdblend/online/controller.py:28`（Controller 字段，`period_s=10.0`, `tick_s=1.0`）。
- `/home/pdblend4/src/pdblend/online/controller.py:509`（控制循环）；`:527`（Shield observe/update）；`:541`（scheduled / level changed / strategy changed / floor restore 触发 replan）。
- `/home/pdblend4/src/pdblend/bench/run.py:121`（实际装配 Controller、Shield、Forecaster）；`:146`（开启 SLO routing）。
- `/home/pdblend4/src/pdblend/online/router.py:435`（逐请求 dispatch）。

## 2. 当前默认接线与实验开关

| 条目 | 当前 homogeneous `pdblend` runner | 来源 |
|---|---|---|
| SLO request routing | 默认开启；safety=0.85 | `bench/pdblend_runtime_options.py:10`, `bench/run.py:146` |
| Shield mode | `budget_aware` | `bench/pdblend_runtime_options.py:11`, `bench/run.py:122` |
| 过载无可行解时保留已有容量 | 默认开启 | `bench/pdblend_runtime_options.py:11`, `bench/run.py:85` |
| 安全扩容绕过普通 hold/votes | 默认开启 `safety_recovery=True` | `bench/pdblend_runtime_options.py:12`, `online/controller.py:418` |
| M 最少实例数 | `min_m_instances=4`，实例总数不足 4 时取 N | `online/policies.py:62`, `planner/pool.py:564` |
| warm start / bootstrap forecast | 开启；由传入的 selection trace 离线统计给初始布局与预测 prior | `online/policies.py:62`, `bench/run.py:109` |
| 普通布局驻留时间 | 30 s | `online/policies.py:63`, `online/controller.py:480` |
| 缩容 / 降频确认 | 2 个慢环 forecast windows | `online/policies.py:63`, `online/controller.py:494` |
| 节能切换门槛 | margin=0.08 | `online/policies.py:62`, `planner/pool.py:692` |
| home anchor | 初始布局可行且比候选更便宜 1% 时可回到它，仍经过门控 | `online/policies.py:63`, `online/controller.py:328` |
| pressure gate / dynamic M floor | **默认关闭**；`pdblend_dominance` 才显式开启 | `online/policies.py:30`, `online/policies.py:80`, `bench/run.py:91` |
| capacity-floor artifact / transition catalog | 需要显式绑定文件；不是随便选“最新” | `bench/run.py:93`, `bench/pdblend_runtime_options.py:84` |
| resident hetero-TP / incremental-energy selection | 另一分支，非 homogeneous 主图 | `bench/run.py:281`, `bench/pdblend_runtime_options.py:61` |

特别注意：裸 `Router()` 的 `slo_routing_enabled=False`、裸 `Shield()` 的 `mode='legacy'` 是向后兼容默认。当前 PDblend benchmark runner 主动启用上述新功能。反过来，不能把 `pdblend_dominance` 的实验 pressure gate 画成标准 `pdblend` 必经条件。

## 3. 谁决定角色、可选布局、怎样发布

PoolPlanner 枚举总活跃数、M/P/D 数量、停车级别和 per-role 时钟，在 latency / capacity 可行集合内考虑稳态功率与转换能耗。预测目标使用 `0.85 × TTFT/TPOT SLO` 留余量。带当前布局的目标为 `60s × predicted_power + transition_energy`，还要满足节能 margin。

正常标准 PDblend 枚举要求 M≥min(4,N)，允许纯 M、M+P+D；不接受孤立 P 或孤立 D。纯 P+D 是通用 planner 支持的布局，但标准 M floor 下不会被正常枚举；显式配置取消 floor / 特定 experimental pressure policy 才能搜索纯 PD。`evaluate()` 也拒绝移除仍拥有 backlog 的 M/PD 分支。

τ 的标准候选为 `{0,1024,4096}`，但若 P+D 与 M 都存在，按 τ 分割后某分支既无未来流量又无 backlog，会被丢弃，因此不能仅由候选集合推断 τ=0 混合布局一定可行。P/D 或纯 M 单一路径时 τ=0。

`Controller.execute(plan)` → `assign_roles(current,counts,inflight)` → 逐实例执行：

- 尽量保留原角色，将数量映射到具体实例；空闲实例的深度与已绑定序列数作为次序参考。
- active→active：设置时钟，再发布新角色；旧请求绑定的 p/d 仍保持，改变的是新请求路由。
- active→parked：关闭 admission / 发布 parked，等待 proxy prefill 与 decode accounting 清空，必要时 native drain，再 reset/park 或 stop。
- parked→active：start/ready 或 unpark，设置频率，native resume，然后发布 accepting role。
- 所有角色动作完成后发布计划的 τ；停车三种状态在 Router 中统一表示 `parked`。

来源：

- `/home/pdblend4/src/pdblend/planner/pool.py:19`（active/park roles、τ 集合）；`:465`（evaluate）；`:552`（数量枚举）；`:588`（时钟/τ枚举）；`:654`（目标函数与迟滞）；`:789`（角色到实例映射）。
- `/home/pdblend4/src/pdblend/online/controller.py:25`（parked role 映射）；`:159`（drain）；`:168`（park）；`:193`（wake）；`:212`（execute）；`:267`（active role change，existing requests pinned）。
- `/home/pdblend4/src/pdblend/online/router.py:359`（角色表与阈值发布）。

## 4. Forecast 与 Shield 观测对象

Forecaster 从 Router 的 arrival/finish listeners 获得请求输入输出长度，保留显式成对的 input/output 样本；backlog 是每 tick 完整重新采样，包含 `waiting_prefill_tokens`、`remaining_output_tokens`、`kv_tokens`、当前 branch/pool。切换阈值不会重新归类旧请求的 ownership。

默认 Forecaster 有 30 s/120 s 双 EWMA，长度样本窗 120 s。未充分观测时需要至少 30 arrival samples 且首流量后 20 s；标准 PDblend warm start 保留 initial plan，非 warm start 则 fail-open 全 M 满频。

Shield 使用 `threshold=0.8`, `window_s=5`, `cooldown_s=30`：

- Prefill risk：任一尚无首 token 的请求等待超过 `0.8 × TTFT_SLO`，或近 5 s TTFT p90 超过该阈值。
- Decode risk（budget_aware）：近窗 TPOT p90 超过 `0.8 × TPOT_SLO`；或任一持续 1 s token stall；或已用时间即使剩余 token 瞬时生成仍不能回到允许的平均 TPOT；或两 token 输出的唯一 token interval 已超限；或至少 2 个请求且≥25% 活跃 decode 请求存在超限 gap。
- 对请求路径区分 M/PD，记录 `prefill_paths/decode_paths`，供安全容量恢复判断瓶颈归属。
- level 最多每 2 s 增加 1，level 1 将所有 active role 满频，level k≥2 在提满频基础上要求额外 k−1 个 active instances。若已有 P+D，则 prefill-only risk 加 P、decode-only risk 加 D；两者都有则给 P/D 较小池；mixed-only 则加 M。从 idle、L1、off 中依次唤醒。
- 静稳 30 s 降一级；归零后仍保留曾需要的 active floor，通过 quiet probe 窗逐个释放，失败恢复 floor 并加倍探测窗，至多 8 倍。

来源：

- `/home/pdblend4/src/pdblend/planner/forecast.py:72`（条件 forecast 但 backlog 保持 owner）；`:119`（Forecaster）。
- `/home/pdblend4/src/pdblend/online/controller.py:310`（informed 判定）；`:512`（backlog snapshot）。
- `/home/pdblend4/src/pdblend/online/shield.py:42`（默认参数）；`:74`（观测）；`:143`（level/floor 更新）；`:183`（动作覆盖）。

## 5. 实验 pressure gate，建议只放侧栏或脚注

`pdblend_dominance` 开启 `dynamic_m_floor=True` 并令 PlannerConfig.pressure_controls=True。M-pool 归一化压力是 prefill busy/rho_max、decode offered-load/rho_decode、KV 利用率、短输出 stall tail risk、归一化 TTFT/TPOT 的最大值。输入包括完整 offered load 与 backlog；模型不可行返回 pressure=2。

当 input_p95≥1024，预测 M 压力≥0.75，或连续两个计划窗观测 SLO 风险，或 Shield active，可进入 PD pressure mode。退出需压力≤0.55、保持至少30 s且2个稳定计划窗。**pressure mode 不直接覆写逐请求分流**：它改变 planner 候选空间并触发计划，Router 使用实际已发布的角色与 τ，防止 planner/router 的流量假设分裂。

压力模式下若可行则优先含 P+D 候选，M+PD 的 τ 固定1024；纯 PD 还需要 input_p95≥2048。动态 M floor 目标低至2，但每次减1之前需稳定窗、30s持有、对缩后池做25%额外负载验证；压力恢复则迅速回到 canonical floor4。此实验另有 Shield protection 60s、transition cooldown45s。

来源：

- `/home/pdblend4/src/pdblend/online/policies.py:80`。
- `/home/pdblend4/src/pdblend/planner/pool.py:275`（压力定义）；`:481`（pressure候选限制）；`:603`（PD优先）。
- `/home/pdblend4/src/pdblend/online/controller.py:353`（dynamic floor）；`:397`（strategy update）。
- `/home/pdblend4/src/pdblend/online/router.py:297`（pressure配置）；`:313`（模式进入退出）；`:403`（按已发布阈值choose）。

## 6. 对主图布局的建议

顶部画控制层：`Profile + SLO` / `Request telemetry + backlog` → `Forecast` → `PoolPlanner (10s)`；Shield (1s) 回馈 Controller 并覆盖时钟/容量；Controller 输出两路虚线：`role table + τ` 到 Request Router，`DVFS / wake / park` 到实例池。

中间画逐请求数据面：`request(input tokens, output budget)` → `Router + SLO admission` → M 或 P→KV→D。M实例内部是一条绿色 prefill 后接蓝色 decode 的时间线；P/D分开两条线、KV为灰色转移箭头，黄色标 token 输出。不要画成 Prefill 完成后再随意挑 D，因为 Router 在 dispatch 时即为请求选定并占用 P/D pair。

右侧画灰色 Parked 池，通过 Controller wake/park 虚线连接。下方画 metrics反馈：first token、token gap、pending prefill、decode remaining、KV occupancy。pressure gate 与 resident routing 放可选/实验框，避免误导为标准算法主流程。

`control` 与 `proxy` 并非另一套旧实现：

- `/home/pdblend4/src/pdblend/control/controller.py:1` 是 `online.controller` 兼容别名。
- `/home/pdblend4/src/pdblend/control/planner.py:1` 是 `planner.pool` 兼容别名。
- `/home/pdblend4/src/pdblend/proxy/router.py:1` 是 `online.router` 兼容别名。
- `bench/run.py` 仍用这些兼容 import，实际执行相同类。
