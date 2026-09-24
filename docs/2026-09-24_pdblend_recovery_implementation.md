# PDblend 恢复修复与单次 GPU A/B

实施范围是满足原 SLO 下减少八卡服务窗加完整排空尾部能耗。历史测量、SLO 和 baseline 算法不变。用户随后将本次针对性 A/B 改为每条件一次；单次结果仅用于诊断，不支持统计节能结论或容量排名。

## 实现

- `planner/pool.py`：可选的容量保留 fallback 接收当前布局；无可信预测时保持角色和分流阈值并升频，有预测时检查保持原角色容量的增容候选。
- `online/controller.py`：安全增容和有模型依据的角色救援可越过节能保持期；Shield 升级不覆盖救援候选；周期规划和紧急检查分别计时。未知时钟状态必须实际 reset。
- `online/router.py`：仅对新请求预测 P 排队、P/D 交接、首次 D token 及 M 干扰风险；可向预测满足 SLO 的 M 分流。已派发请求保留 ownership 和 KV 生命周期。缺 profile 预测不会被当成已证明安全。
- `online/shield.py`：可选预算判据结合累计 TPOT、剩余总预算、短输出、持续停顿及共同停顿比例；已收完 token 但等待 terminal 不再算 decode stall。
- `bench/pdblend_runtime_options.py`：显式配置与 artifact SHA 绑定，回执区分请求启用、实际启用、触发次数和回退原因。阈值仍是 development 参数，尚未通过独立 tuning 资格化。
- 时钟审计区分依赖阻塞、缺失操作、顺序错误和真实频率不符；保留历史资格，不追认旧窗口。

普通 PDblend 入口启用新的恢复与风险判据；类本身保持 legacy 默认，baseline 不接受 PD override。异构 resident selector 尚未集成本次 SLO 路由，显式请求会在启动前报错；同构比较不允许把 joint/incremental resident 配置记成已启用。容量下限和转换成本须传入已绑定、通过现有 guard 的 artifact。

工作区后续增加两 token 请求保护：旧 `transfer_seconds` 不能证明从 P 首 token 到 D 首 token 的完整间隔。缺少独立 `pd_first_gap_seconds` 预测时，仅在 M 容量及 SLO 预测安全时转 M；没有安全 M 则保留原可接纳路径并记录未证明。已派发请求不会迁移。显式 first-gap 包含首个 D step，不重复累加。

合格容量下限接线要求直接 artifact 与原验收 SHA 绑定，每个 plan 记录完整 Forecast 并由审计重放适用域。离开域时先唤醒保留槽位恢复 M，保持现有 P/D 数量与阈值；此动作越过保持期。comparison 低 M 仅允许 profile 最高频率，并要求原验收 `fixed_plan.f_M` 明确等于该频率；缺失或低频验收不能推广。此接线尚未进入下面的 v4/guard GPU 快照，也没有本轮 GPU 节能结果。

## A/B 实验与后续队列

后续用户要求“测过的不重测、尽快结束 profile”，因此下面 v4 profile 与 combined v1 的原调度已经修订。当前安排见 [profile 去重与调度修订](2026-09-24_pdblend_profile_deduplication.md)：32B 保留完整 timing 后停止可选 layout；7B/14B 改为各 188 个单次点；组合 A/B 使用 v2 调度，实验本身不变。

使用 `results/2026-09-24/pdblend-recovery-ab-v4/campaign.json`，v1–v3 为未入队准备版本。

| 模型 / 轨迹 | 条件 |
|---|---|
| 7B LongBench ×1 | 旧冻结源码、固定全 M 满频、恢复修复 |
| 14B LongBench ×1 | 旧冻结源码、固定全 M 满频、交接风险路由 |
| 7B ShareGPT ×0.25 | 旧冻结源码、预算 Shield |
| 32B Alpaca ×1 | 旧冻结源码、预算 Shield |

原 A/B 共 10 个 150 秒服务窗，六个 resident job。每窗独立 reset 和完整 drain，相同请求、输出预算、seed701；所有条件都计入八卡。control 使用真实历史源码，candidate SHA 为 `7134dc01315e008d8c95a60670cd247f6648080233da2074cf47e71a6a7882c9`。引擎与数值计量方法 SHA 一致。

`scheduled-jobs.json` 是实际队列 envelope：独立 model/source group 按优先级和八卡独占租约串行运行，不要求前一个模型成功，避免无关失败阻止后续任务。已有队列 worker 执行，未另起争抢 GPU 的进程。

`results/2026-09-24/pdblend-profile-recovery-v4/jobs.json` 的三模型补测排在全部 A/B 终态之后，A/B 完成后 32B 已自动开始，7B、14B 接续。它们补 timing 和 32B 有界布局组件，不能自动资格化完整 canonical profile。具体缺失域与重放命令在该目录 `dependencies.json` 和 `readiness.json`。

另有 `results/2026-09-24/pdblend-first-gap-ab-v1/campaign.json`：复用上述已排队/完成的对照，只追加一个 14B `first_gap_guard` 窗口。其源码是 v4 加 `online/router.py` 单文件修正，SHA `d0b954840eb14c7996b458f8bbf655daea81aa13f7717161fde447993bb03ae4`；新 job 为 `comparison-14b-d6bba393e2ed520b`，在原 14B A/B 后、32B 前执行。新窗口已通过真实镜像 CPU preflight 及冻结源码 33 项路由/Shield 检查。

以上 11 个窗口 / 7 个 job 均已执行完成；每个条件一次，没有将内部采样或窗口拆分算作独立重复。最终绑定报告为 `results/2026-09-24/pdblend-first-gap-ab-v1/final-report.json`，`all_jobs_terminal=true`。GPU 测量以八卡独占租约串行进行；三个 subagent 并行实现、独立复算和 CPU 验证。

已追加组合验证 `results/2026-09-24/pdblend-combined-recovery-ab-v1/campaign.json`：四个场景各一次，同时启用恢复、SLO 路由、两 token first-gap 保护和预算 Shield，参数保持不变；容量下限、转换成本、异构与增量能量机制均未启用。冻结完整源码 SHA 为 `8bb8cdb8c9b5f986717ae747c0fc31cffe8ac274aef779b5f486e396091bd276`。三组真实镜像 CPU preflight 均通过，只将 `new-jobs.json` 的三个新 job 入队，按八卡独占租约接在三模型 profile 补测后。入队回执为该目录 `enqueue-receipt.json`。复用前面四个原策略对照，不重复执行；对照与候选分时运行，不能排除时间混杂。此处仅记录已入队，尚无组合 GPU 结果。

这些窗口隔离了恢复、交接路由或预算 Shield；完整默认组合、新合格容量下限与新 native profile 尚未一起完成 GPU 全矩阵验收。单个 arm 的结果不能代表所有优化同时启用的最终配置。

## 已完成 GPU 观测（单次，非正式资格）

下表来自 `results/2026-09-24/pdblend-first-gap-ab-v1/final-report.json`。能量覆盖同一批请求的八卡服务窗与完整排空尾部；goodput 使用该批请求从服务开始到请求完成的时间。所有行的原始八卡功耗和客户端 canonical metrics 均通过对应检查；7B/14B 未通过完整测量资格，32B 两行通过测量审计。全部仍为单次诊断且 profile 未完全资格化，不能宣布统计节能、完整 profile 合格或最晚饱和。能耗取独立 comparison-metering/receipt 数值，不混用 native-result 内部积分。

| 模型 / 数据 | 条件 | 联合 SLO | TTFT P99 / s | TPOT P99 / ms | 总能量 / kJ | J/达标 token | goodput / token/s |
|---|---|---:|---:|---:|---:|---:|---:|
| 7B LongBench ×1 | 原策略 | 76.28% | 55.590 | 51.132 | 290.940 | 16.726 | 83.782 |
| 7B LongBench ×1 | 固定全 M | 100% | 1.209 | 43.372 | 272.111 | 11.618 | 150.063 |
| 7B LongBench ×1 | 恢复修复 | 100% | 3.249 | 48.645 | 250.732 | 10.705 | 148.926 |
| 7B ShareGPT ×0.25 | 原策略 | 100% | 0.640 | 26.460 | 242.770 | 3.113 | 485.569 |
| 7B ShareGPT ×0.25 | 预算 Shield | 100% | 0.673 | 29.700 | 190.366 | 2.441 | 486.690 |
| 14B LongBench ×1 | 原策略 | 100% | 3.899 | 118.762 | 173.978 | 29.108 | 39.487 |
| 14B LongBench ×1 | 固定全 M | 100% | 2.469 | 43.029 | 190.704 | 31.906 | 39.505 |
| 14B LongBench ×1 | 交接风险路由 | 100% | 4.091 | 101.729 | 161.758 | 27.063 | 39.485 |
| 14B LongBench ×1 | 缺 first-gap 保护 | 100% | 3.890 | 124.388 | 164.492 | 27.521 | 39.487 |
| 32B Alpaca ×1 | 原策略 | 100% | 0.167 | 65.622 | 320.094 | 2.022 | 946.613 |
| 32B Alpaca ×1 | 预算 Shield | 100% | 0.194 | 70.670 | 318.799 | 2.014 | 948.688 |

- 7B LongBench 原策略的 409 个请求全部完成，97 个 TTFT 违约；不是 97 个失败或超时请求。本次 fallback 将 3P＋1D＋4M 改为 1P＋7D，随后 122 个请求进入唯一 P。恢复修复保持 3P＋1D＋4M，避免重现这一故障链；比固定全 M 的含尾部能耗观测低 7.86%。运行回执中 `safety_recoveries=0`，路由无 M spillover，不能把效果归于未触发机制。
- 7B ShareGPT 预算 Shield 的能耗观测比原策略低 21.59%，最终保留 M6/off2；共同停顿保护仍触发，尚未完成独立 tuning，也没有证明胜过同点最省电 baseline。
- 14B 原策略本次已通过 SLO，未复现历史两 token 违约。交接风险路由将 5 条请求转 M，能耗观测比原策略低 7.02%；固定全 M 反而比原策略高 9.61%。额外 first-gap guard 因缺失预测覆盖将 6 条请求转 M，能耗比原策略低 5.45%，但比 v4 交接路由高 1.69%；单次结果不能确定这一差异超出重复波动。
- 32B 的 1,189 个请求两边均达标；总能耗观测仅低 0.40%，服务窗本身反而略高，不能认定已解决 32B 节能不足。
- 7B/14B 候选的完整审计均在物理频率处失败；例如固定全 M 申请 2520 MHz 时存在明显低于容差的实际频率，主要发生于有请求且高利用率的时段。没有同期完整温度/功率限幅原因日志，不能把历史偏差单独归因于热或功率限制。旧 control 还保留旧控制审计/级联异常，未追认其资格。32B 两窗的实际频率检查通过。

## 验证与复算

最终核心 CPU 整合回归：342 passed，记录于 `results/2026-09-24/recovery-core-final-cpu-validation.xml`，覆盖 planner/controller/router/Shield、请求 carry/ownership、floor artifact 和审计、计量依赖、冻结 A/B 报告与 profile/handoff 准备。之前的 244 项记录保留于 `results/2026-09-24/recovery-cpu-validation.xml`，两批有重叠，不相加。六组原 A/B 新旧冻结源码、额外 guard 及三组 profile job 均通过真实运行镜像 CPU preflight；它不等于 GPU 验证。

随后新增 floor 大于 canonical reserve 的启动前拒绝，相关 55 项通过（`recovery-floor-boundary-cpu-validation.xml`）；容量准备、receipt、倍增/二分及跨五系统区间比较先通过 131 项，再补继承证据重放通过 138 项（`recovery-capacity-inherited-cpu-validation.xml`），最后两处来源链收紧通过 9 项专项检查（`recovery-capacity-provenance-cpu-validation.xml`）。五份最终 JUnit 按 testcase 去重共 485 项，零失败，重复检查不累计。

组合准备工具另外通过 8 项检查（`recovery-combined-preparation-cpu-validation.xml`），包括原对照引用不变、完整源码冻结、配置与调度绑定；六份 JUnit 去重合计 **493 项通过**。

```bash
PYTHONPATH=src /home/pdblend/.venv/bin/python -B \
  scripts/2026-09-24_pdblend_recovery_campaign.py report \
  --campaign results/2026-09-24/pdblend-first-gap-ab-v1/campaign.json \
  --queue results/2026-09-22/three-model/queue.json \
  --out /tmp/pdblend-recovery-report-new.json
```

报告检查原始 artifact、源码、profile、轨迹、session 与租约绑定，再计算服务窗＋尾部能耗及 J/达标 token；缺失或无效测量不进入均值。不能使用这批 evaluation A/B 选择容量点或调参数。

容量判定位于 `bench/slo_capacity.py`：仅接受 calibration/tuning，默认至少三次一致重复和每次至少 100 请求，倍增/二分至通过和失败区间宽度不超过 5%。缺上界、重复分歧或不单调时不宣布容量。`capacity_workloads.py` 冻结独立 corpus/anchor、三个新 seed 与所有系统共用的服务时长，按预先声明最低 rate 延长窗口；`slo_capacity_receipts.py` 重放每个请求，绑定 family 与八卡身份。`capacity_comparison.py` 仅在五系统区间均收敛、且 PDblend 下界严格高于四个 baseline 上界时报告观测边界领先，完整正式资格另行保留。

实际容量准备包为 `results/2026-09-24/pdblend-capacity-workloads-v2/manifest.json`：9 个 model/dataset family、27 条 x1 轨迹、135 个五系统 assignment，全部重放通过。三个独立 seed 为 8801/8802/8803；预声明最低倍率 .25，所有系统共用同点请求及同 family 服务时长。7B Alpaca/ShareGPT/LongBench 分别为 150/150/300s，14B 为 150/150/600s，32B 为 150/300/2400s。此三 seed 是未来容量搜索准备，未执行；本次 GPU A/B 仍每条件一次。

32B 复用此前已成功的 `longbench-anchor-recovery-v1`。Alpaca/ShareGPT 只沿显式 `prior_inputs.inherited` 追溯原请求并重放，旧 failed 父任务只提供来源，未被晋级为成功；因此无需重复进行已有的 anchor GPU 恢复。v1 最初因传入旧 partial anchor 而产生的阻塞诊断保留，v2 显式绑定成功恢复包。若未来运行整个 x1 容量准备矩阵，纯服务窗为 18.125 小时（不含加载/排空），并未入队。

新的独立容量执行 scope 仍待接到各系统 runner 和 audit：现有比较入口固定 evaluation、seed701、150s（部分入口允许300s），因此不能把准备好的新轨迹强行提交或将旧 A/B 重标为 tuning。完整 native profile 域、设备交接组件与独立 holdout、最终组合和四 baseline 全矩阵、容量 GPU 搜索及计量不确定性仍是未完成工作。
