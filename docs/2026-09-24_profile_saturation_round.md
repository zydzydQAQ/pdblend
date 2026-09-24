# 新 Profile 全量复测与单次观测边界

当前运行包：`results/2026-09-24/profile-saturation-repaired-v1`。唯一作图入口为
`results/compare.csv`，同时保留历史 revision/attempt 和原始回执。

用户随后明确选择“当前窗口完成后交接，关闭未取得完整证书的 low-M，冻结已验证修复并继续矩阵”。
额外 low-M 资格任务已停止，最终运行源为 `d4fe2c6ac5f5b753e710d56f8c072a276eb6a9032e298a12e567ad4feac222fb`，
36 个原点和 9 个 ×1.25 扩点的实际镜像 CPU 预检通过；low-M、切换成本和增量能量
三个缺输入 artifact 均关闭。正式接受绑定见当前包的 `orchestration/accepted-round-v2.json`。
五对修复测量仍属于 46e3 源码，不能当成 d4fe 的测量；本轮使用同一 d4fe 版本重新运行。
7B PD 常驻组已完成原 12 点及边界搜索，正在完成本模型边界 baseline。

首组完成 8 个有效窗口后，LongBench ×0.75 在重置阶段启动 pd4 失败，未进入服务。
21:38:21 主 agent 为更新 CPU 服务配置执行过 `systemctl daemon-reload`；21:40:21
旧容器的新 vLLM 进程无法识别 GPU。此时序与 NVIDIA 记录的 legacy 容器设备访问
丢失问题相符，但旧容器已清理，未再直接探测它。新 DistServe 容器的八卡可见性正常。
GPU 租约期间禁止再次执行 daemon-reload；后续 CPU 分析源通过绑定的
`host-analysis/active-source.json` 在进程启动时选择，仅重启 CPU 调度器。
参考：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/troubleshooting.html

恢复使用 `orchestration/pd-recovery-7b-v1`：保留原 12 点、相同 d4fe 源及相同 policy，
只新增租约身份。8 个完整窗口的目录只读链接到原证据，边界从原 `manifest-000008`
继续；`000009` 唯一新增的服务前失败保留在历史及 CSV，但退出活跃边界分支。
正式 runtime 会重新核验全部原始哈希后跳过这 8 点；只重试未进入服务的点和剩余点。
已开始的 DistServe 两点能耗补测先正常完成，之后接 PD 恢复；后续同系统缺口与
新边界端点仍合组。旧 selection/campaign 保留，恢复选择使用新的版本路径。

恢复首轮已跳过原 8 点，完成剩余 4 点；7B 原 12 点均通过本轮 SLO。
7B 另完成 11 个边界扩点，新版通过/失败倍率分别为 Alpaca 2.44140625/3.0517578125、
ShareGPT 1.953125/2.44140625、LongBench 1.25/1.5625。LongBench 上端因 TPOT p99
失败，另两组上端因 TTFT p99 失败；旧版本的通过点不拼入新版边界。
原 12 点中 ShareGPT ×0.25 的服务能耗高于已冻结 EcoServe 约 10.79%，因此不能宣称
PDblend 所有条件最优。7B 原矩阵的 11 个 baseline 能耗缺口现已补齐，原 12 点均有
完整五系统比较：同一新版 PD 的 12 点全满足 SLO，其中 11 点能耗第一、上述 1 点
第二。领先点的服务节能为 21.2083%～51.8998%，含尾部为 20.6525%～50.5547%，
未发现含尾部反转。完整汇总绑定在
`profile-saturation-repaired-v1-independent-metrics-review/7b-original-twelve-complete-five-system-review.json`。
这些结论目前仅覆盖 7B 原 12 点，不能外推到尚待完成的 14B/32B 或边界两侧。
DistServe 的 Alpaca ×0.75、×1 两个能耗补测均完整记录，虽 SLO 失败仍按用户要求冻结。
CPU 汇总源为独立 `8d1438f6…`，GPU 执行源仍是 d4fe；恢复 publication 仅从活跃
边界选择中排除唯一服务前失败分支，不删除其 CSV 行或任何原始证据。
`host-analysis/execution-recovery-deployment.json` 绑定恢复补丁，45 项提案测试及
9 项部署测试通过。恢复后的 CSV 已保留全部旧测量并追加新结果。

CPU writer 第一次恢复时的启动至首发布区间 22:10:56.884245–22:11:16.356297
位于 LongBench ×1 的正式服务窗中，共 19.472052 秒。该区间不是实测 CPU 忙时，
不据此推断结果因果，也不丢弃或补跑原数值。独立诊断见
`profile-saturation-repaired-v1-independent-metrics-review/longbench-x1-cold-writer-overlap.json`。
后续 CPU 自动冷启动默认等待活动 GPU 租约结束再进行历史冷导出；
运行中保持增量汇总缓存，避免重复读取历史大文件。

7B Mixed 四个新端点已完成，均完整记录能耗并保留实际 SLO 失败；与两条匹配的历史
Mixed 冻结回执一起覆盖新版六端点。Alpaca ×2.44140625、ShareGPT ×2.44140625
两窗尾部结束至最终回执分别相隔约 184、224 秒，文件时间与冻结源码表明主要位于
两遍 CPU 请求日志归约/验收处理，不是 GPU 排空或请求超时。当前冻结执行源码不改。
详情见 `profile-saturation-round-v3-independent-baseline-preflight-v1/`
`pd-execution-recovery-proposal-v1/mixed-post-tail-overhead-review-v1.json`。
四窗原生请求摘要已在下一次装载期间单次读取；三个高负载窗的失败均为
`no accepting Mixed replica`。冻结策略的八副本各有 32 个在途请求接纳上限，合计
256；客户端 pre-dispatch outstanding 的 257 峰值包含路由前即被拒绝的请求。
这不证明 HTTP 发送队列或硬件饱和，实际网络发送队列仍未知。

DistServe 新边界组完成两个 LongBench 窗后，ShareGPT ×1.953125 的原生执行已生成
completion，但后续 `/baseline/drain` 返回 HTTP 500，session 隔离并完成干净清理。
已执行失败点保留原始证据，不重测。三个尚未创建窗口的端点以原源码、部署、Profile、
完整 point dict 和 trace 排入新租约 `comparison-7b-dd9742f537d25cc3`，在 EcoServe
当前组之后、DynamoLLM 之前运行。恢复回执绑定在
`orchestration/distserve-unstarted-recovery-v1-activation.json`；无需修改 campaign 或重启
CPU writer，现有队列发现机制会发布这个额外租约的结果。

9 月 25 日 00:27（本地时间），EcoServe 六端点全部完成并干净释放租约后，已部署
仅主机调度使用的未启动点恢复逻辑。它逐次核验失败组及清理证据，只为从未创建窗口的
原 point 子集建立后续租约；完成、服务后失败或重置失败的窗口均不自动重复。
新调度器已采用上述手工 DistServe 恢复任务，未重复排队。首次冷 CSV 导出全在 GPU
空闲期间完成，258 条既有绑定记录的 59 项指标逐字段不变；GPU 执行源码仍冻结。
部署回执见 `host-analysis/unstarted-recovery-deployment-v1/activated.json`。
EcoServe 六个端点均有完整服务和尾部能耗，均未满足 SLO，且每窗尾部能耗均高于
服务能耗；这些结果完整保留，不修复算法或重测以选择较好结果。

中断的 DistServe ShareGPT ×1.953125 原生结果和整组功率采样已在新 DistServe
装载期间读取一次：2349 个请求均有终态，812 成功、1537 deadline 失败，联合 SLO
达标 147 个；150 秒服务能耗可核为 221623.48 J，八卡覆盖完整。四个 decode 的
release cleanup 未通过，尾部终点未知。原 failed 回执与 canonical 空值保持不变；
这些请求与服务能耗将用明确的 recovered 补充列写入同一 CSV，不进入能耗排名。
随后 ShareGPT ×2.44140625 也在排空阶段失败，2942 请求中 770 成功、2172 deadline
失败，联合达标 137 个；服务能耗可恢复为 220879.69 J，尾部仍未知。两窗只在下一组
装载时读取各自原生证据一次，采用独立身份绑定。调度器自动生成仅包含未启动 Alpaca
×3.0517578125 的 `comparison-7b-ffdc154e50004397`；已取得租约的 DynamoLLM
15 窗正常完成后再执行此单点，不中断或重复已执行窗口。
最终附加列的顺序及绑定见 `orchestration/final-annotation-plan-v3.json`；恢复列包含
逐卡 UUID、时间加权利用率、峰值及覆盖率，window token goodput 明确标为基于
native finished_s 的保守下界，不冒充末 token 计时口径。应用前必须
确认唯一 CSV writer 已停止、所有 GPU 租约已释放。

新版 Alpaca ×3.0517578125 的请求/SLO 观察有效，八卡服务采样覆盖约 99.23%，
但最长 1.15 秒缺口超过原 1 秒门槛，正式服务/尾部能耗仍为空。跨缺口线性插值的
280418.67/51984.52 J 仅作为显式估算附加列，不替代实测，也不参与能耗排名。
同一解释器内的两条采样线程都出现相近停顿；尚不能区分 GIL、OS 调度或其他工作。
独立进程辅助采样已在 14B 装载阶段启动，保持本轮 GPU 源码、参数及正式原始计量证据不变。
辅助范围已经在 14B/32B 测量前冻结为两模型全部未来 PD 窗（原点、扩点及续租），
不按主采样是否缺失选择使用。13 项 CPU 测试通过；14B 的 ready 时间为
1790272047.245，当时没有正式窗口。八卡 UUID 与冻结组逐项一致，独立子进程正常。
其 `auxiliary-meter/14b-attempt-0001-af09b7cc9fa64b8db39fa6bda9736ad0/root-activation.json`
绑定 observer、采样子进程与仅在租约边界生效的自有 stop marker；32B 尚待启动。
参见 `orchestration/auxiliary-meter-plan-v1.json` 及
`orchestration/auxiliary-meter-worker-boundary-procedure-v1.json`：只在确认的目标 PD
租约运行中设置有身份绑定的临时 worker.stop；现有 worker 完成整个租约才停止领取
新任务，CPU writer 继续逐窗发布。待租约释放后导出辅助证据，移除自己创建的 marker
并重建相同 worker，整个流程不调用 daemon-reload，也不暂停 GPU 控制器。

队列 worker 仅取得当前 run_id 的任务；共享队列的 `active-execution-scope.json`
防止旧调度器抢回 GPU。矩阵调度器与 GPU worker 分离，发布和完成审计的更新不重启引擎。
外部完成审计要求 36 点、9 序列、可确定端点及原 20 个能耗缺口都有实测或绑定失败证据；
仅有队列终态不算完成。调度、范围隔离和完成审计的 24 个轻量 CPU 测试通过。

2026-09-24 后续执行顺序已由用户明确调整为“先完成修复与验证，再冻结新版继续矩阵”。
旧轮的调度器已停止；`orchestration/priority-coordination.json` 保存交接和恢复状态，
调度器在启动、提交及等待时检查该记录，避免误启动旧轮。新版必须新建冻结运行包，
重新测其 36 个 PD 点并重新找边界，不能把旧版 7B 结果拼入新版矩阵。
旧轮已测 26 窗（PD 原矩阵 12、扩展 12、Mixed 端点 2）保留；其中 25 窗服务能耗完整。
两窗 Mixed 均为实际 SLO 失败，继续冻结。20 个能耗缺口尚未由旧轮补测。
停止后误恢复的 Mixed 组在装载期间、任何窗口开始前退出，清理通过，没有新增测量。
当前完成度与原始回执清单见
`results/2026-09-24/profile-saturation-round-v3-independent-metrics-review/handoff-metrics-snapshot.json`。

修复候选 46e3 与预先选定的五个新对照已经全部完成：十窗均满足 SLO、八卡服务与
尾部能耗完整。候选服务能耗合计减少 8.8134%，含尾部减少 9.1167%；两个 Alpaca
点的服务能耗分别增加 3.6809% 和 2.1877%。这是跨 campaign 顺序执行的单次配对，
不是随机同期 A/B，也不证明所有点或五系统全面最优。固定配对清单与统计保存在
`profile-saturation-round-v3-independent-metrics-review/repair-v2-fixed-five-summary.json`。
新矩阵启动前 CSV 有 220 个唯一实测回执，原 215 个回执的所有 METRIC_FIELDS 未变。

独立 low-M 的 2M/2 req/s 和 3M/4 req/s、2100 MHz、seed8801 初测均通过；
后续资格筛选已按用户指令停止。它们不算 seed701 正式矩阵，不等于完整多域资格。
矩阵集成源 `pdblend-matrix-integration-candidate-v1` 基于 e537
只改动态扩点频率上限和旧 startup contract 的处理，现已正式接受。
36 点独立 tuning 重算与真实 controller startup CPU 预演已通过。
startup 验证放在外部脚本；没有最终源精确匹配的完整 low-M 证书时使用原容量下限，
不会为缺输入证书重新采集 Profile。11 对实际源的计量兼容审阅已通过。

新版调度器原子发布 CSV，并在替换前核对每个历史 receipt 的所有 METRIC_FIELDS；
旧扩点 manifest 只用于导出，不能喂给新一轮边界搜索。
`scripts/2026-09-24_audit_saturation_completion.py` 按准确点身份和 hash-bound 回执
审计 36 点、新边界及端点 baseline；排队任务终态不当成窗口完成，reset 失败回执
计可查阻碍，SLO 失败和缺能耗的完整窗口仍保留观察。完成审计的 baseline verifier
已与新版 builder 成套部署。

以下为已停止旧轮的执行设计。该轮冻结 36 个 PDblend 点、20 个 baseline 服务能耗补测点，以及三个模型各三个
数据集的自适应倍率规则。每窗 150 秒、seed 701、每倍率一次；倍率扩展重新生成
完整 evaluation 轨迹，初始计划仅使用独立 tuning。PD 源码 SHA 为
`498e446c2fc72e3ccd3ea925fcea05746408dd67e60129a1f1aa7577e43a0fab`，
三模型 Profile 绑定在 `pdblend-development-composite-v2/manifest.json`。

Profile 融合保留训练/holdout 分离、timing 与功率各自适用域、域内插值与域外继承。
实测 endpoint first-gap 包含首个 decode step，当前仅覆盖输出 16、batch 1、请求
2520 MHz 及实测输入域；其他条件保留原风险保护。查询来源和实际优化触发写入每窗
native 回执及 `profile-query-provenance.json`，估计不标为实测。

为兼容长上下文下的实测 KV 容量，PD 规划器的 decode 峰值枚举补齐小 batch：
原来只枚举 8、16、24、32，14B/32B 某些长上下文只能容纳更小 batch，会误判零容量。
新枚举保留所有物理容量拒绝。未知预测使用明确的 null/不可用注释保存，仅观测模式
可解码，正式资格路径不放宽。

调度由 `scripts/2026-09-24_run_saturation_round.py` 单进程管理队列及 CSV：
7B → 14B → 32B；每模型 PD 原 12 点与边界扩展共用租约，随后
Mixed → DistServe → EcoServe → DynamoLLM。八卡始终独占，PD 的 off/park/唤醒照常。
每窗独立 reset、预热、服务和排空；租约结束核验清理。规则与版本不随结果调整。

基线算法、Profile、控制周期和旧证据不变。补测使用已验证的独立进程采样；
旧冻结源码包只更新租约层的“记录已完成窗口后继续”行为，以保留资格审查失败的实测。
124 个完整能耗逻辑点的比较回执在测量前已确定，20 个缺口追加一次 attempt。
147 条历史 baseline 回执的能耗、goodput、p99、成功率和利用率数值在导出预检中零变化。

`single_observation_slo_boundary/v1` 与严格重复容量测试分开。它按 ×1.25 扩展，
必要时下探或收窄到上/下界比不超过 1.25。缺能耗不抹掉请求及 SLO 结果；执行、
排空或请求计时不完整标明阻碍，不能冒充饱和。低样本 p99 保留标记，不延长窗口。
非单调序列保留完整观察，并可对最高通过/上方首次失败配对测 baseline，不声称唯一容量。
租约预算耗尽时用相同 policy 与源码继续，已完成窗口凭回执跳过。

运行状态位于包内 `orchestration/status.json`，事件位于 `orchestration/events.jsonl`。
调度器检查点保留已完成模型、当前 campaign 和动态 manifest；baseline 逐组提交，
避免前组 `blocked` 后依赖链停住。准备完成标记绑定 campaign、jobs 与边界选择，
不完整的准备目录保留，恢复时在新的固定路径重建。
首轮固定服务时间为 140 分钟，加载、重置、预热、尾部及扩展点另计。

作图时先选 `status=measured`、`measurement_usable=True`：PD 行再选本轮
`run_id=profile-saturation-repaired-v1`，baseline 行选 `frozen_baseline_selected=True`。
不要对全表只筛 run_id，否则会漏掉复用的旧冻结 baseline。能耗图要求
`analysis_energy_usable=True`；SLO 失败行可以展示，能耗排名只使用达标行。

完整五系统比较以本轮 PD 行的 `comparison_complete_five_systems=True` 为准，
并用 `comparison_participant_receipts` 内的 SHA 匹配四条冻结 baseline。
原矩阵点的 `experiment_phase` 不是 `slo_boundary_extension`；扩展点只有最终
`boundary_status=bracketed` 且 `boundary_role=L/U` 的两端参与五系统结论。
PD 独测中间点不能称为五系统最优。历史 baseline 对多个 PD revision 有不同排名时，
使用 `energy_rank_by_revision` 中本轮 PD revision 对应的值。

2026-09-25 的 14B 首租约完成了 12 个原始点及 Alpaca ×1.25，13 窗均满足 SLO。
LongBench ×1.25 已开始执行，随后出现 P2P 监听线程 KeyError 和自然 TimeoutError；
缺少完整请求、排空及服务计量，保留为执行阻碍，不作为饱和上界，不重跑此点。
租约最终清理通过。外部动态失败恢复保留全部 14 个 observation 和两个 decision，
引用并跳过 13 个完整窗口，只继续 Alpaca、ShareGPT；冻结 GPU 源码 d4fe 不变。
恢复包、12 项测试和独立导出守恒检查绑定在
`profile-saturation-repaired-v1/host-analysis/dynamic-observation-recovery-deployment-v1/deployed.json`。

独立辅助采样覆盖首租约的 13 个完整窗口；失败窗口因无可信服务时界标为不可用。
辅助采样只追加审计列，不能替换正式计量或参与排名。导出在无活动租约时完成，
耗时约 25 秒；服务能耗与正式计量的最大差异为 0.03718%，尾部为 0.26020%。
最终附列计划更新为 `orchestration/final-annotation-plan-v4.json`，等待后续租约的
辅助回执合并后再统一执行。
