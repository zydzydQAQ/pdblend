# Profile 去重与调度修订

用户要求三模型已测内容不再重测，尽快结束本轮 profile，并保持八卡有序接续。

## 保留与新增

| 模型 | 保留的测量 | 本轮剩余工作 |
|---|---|---|
| 7B | 已有 legacy 曲线、1500/2520 MHz 小 batch 与 Mixed 功耗、已有 runtime 原始组件 | 188 个缺失的 native CUDA 耗时点各一次，其中 24 个独立 holdout |
| 14B | 同上，按本模型的身份与适用域保留 | 同上 |
| 32B | 旧 runtime 与 1500/2520 MHz 功耗；本次完整 runtime、48 个干扰观测、564 个耗时采样项 | 完成 CPU 重放，不再启动新的 GPU profile |

32B 的 564 项中，384 项有实际耗时测量，180 项是在提交请求前确认容量不支持的记录。四个 TP2 owner 各有 141 项，库存完整，不能把 180 项计作测得的耗时。运行中冻结采集器没有动态跳点接口，因此保留已接近完成的整个 timing stage，再在已保存阶段之后停止，避免重载后回测。

32B 的 12 个布局能耗训练窗和 24 个布局独立验证窗未启动，移出本轮快速补测，明确标为延期；它们不是已完成项目。仅服务时间便节省 72 分钟，未把加载、排空和重复观测时间算入这一数值。

7B/14B 不再收集 runtime、power-pilot、request-cycle 或 layout；保留原 188 个唯一形状和独立 holdout，减少每点重复。原每模型 564 个主窗口加 96 个隔离/并发对照窗口，改为 188 个单次并行窗口，两模型合计从 1320 降至 376，减少 71.5%。没有将旧 token 时间戳或 scheduler 墙钟记录转换为 CUDA 内部耗时。

## 资格和停止证据

新单次采样使用独立 development schema。八张卡并行处理不同点，但本轮没有重新采集隔离/并发干扰对照，也没有重复性证据。原始窗口、容量排除、拟合与独立 holdout 仍重放；`parallel_qualified`、`component_qualified` 和完整正式 profile 资格均保持 false，不能冒充旧三次重复协议通过。已有组件的原资格不变。

32B 完整 timing-stage 已落盘，单次 SIGINT 停止请求及实际投递分别记录在 `results/2026-09-24/pdblend-profile-deduplicated-v1/32b-stop/`。父任务保持真实 `failed` 状态（worker exit 2）；八卡物理清理通过。信号发生在 timing-stage 保存后、collector 尚未更新自身 snapshot 阶段状态的边界，因此复用须同时重放 stage、操作请求、真实信号回执和终态清理，不能只修改父任务状态。

该严格终态重放现已通过，subagent 和 root 分别复算：`component_qualified=true`、`timing_component_reusable=true`，564 项和 48 个并发对照原始记录均重放，物理清理与终态租约通过，父任务仍为 failed。证据为上述目录 `terminal-timing-operator-evidence-v1.json`，独立复算为 `root-independent-replay.json`。

## 实际队列

- 原 7B/14B 三次重复任务已阻止启动，attempts 均为 0，原定义保留。
- 新任务 `pdblend-native-timing-7b-10f4013850a5ef9f`、`pdblend-native-timing-14b-72bfbc7f70afa446` 已入队，priority 1000，均为八卡独占、最多一次执行，只依赖已终态的 32B。
- 组合 A/B 调度修订为 `results/2026-09-24/pdblend-combined-recovery-ab-v2/`，取消对可选 profile 扩展的等待，priority 900；实验点、源码、已有对照、请求和计量口径不变。新 profile 准备时，队列已接上 7B 组合 A/B；该任务结束后优先执行新 profile，再继续其余 A/B。
- 不启动第二个争抢 GPU 的 worker，不修改正在运行的冻结源码，不自动重跑失败任务或已采集的前缀。

新补测包：`results/2026-09-24/pdblend-single-pass-timing-v1/manifest.json`。实际入队回执：同目录 `enqueue-receipt.json`。两个真实运行镜像 CPU preflight 均通过；独立 subagent 检查全部 389 个冻结源码文件、点清单、资格标记与调度依赖。旧采集器相关 159 项回归通过；新协议另通过单次、capacity、builder、replay 检查。

历史库存和 SHA：`results/2026-09-24/pdblend-32b-dedup-audit-v1/inventory-terminal.json`。这里记录执行安排；新单次 GPU 任务的成功、失败和覆盖范围以终态回执及独立原始数据重放为准。

终态若只完成部分点，使用 `python -m pdblend.profile.collection.native_timing_partial --attempt ATTEMPT --queue results/2026-09-22/three-model/queue.json` 盘点。该只读工具逐点区分已测、容量不支持、失败、无效与缺失，并绑定原始身份和 SHA；它不提交队列，禁止自动整任务重试，已完成前缀不得重采。

## 17:03 终态更新

两项 single-pass GPU 任务均因 `observed frequency coverage differs` 停止，清理通过，没有重试。7B 为 76 个有效已测、7 个连带取消、1 个实频无效、104 个缺失；14B 为 40 个有效已测、15 个容量不支持、7 个连带取消、1 个实频无效、125 个缺失。各项盘点在 `results/2026-09-24/pdblend-single-pass-partial-v1/{7b,14b}/inventory.json`。这两份 profile 尚未完成，任务终止不能算 profile 验收通过。

组合 A/B 四个新窗口均完成，全部请求成功且联合 SLO 为 100%。八卡服务窗＋完整尾部的能耗依次为：7B LongBench 232108.895 J（相对原 PDblend -20.22%）、7B ShareGPT 181789.412 J（-25.12%）、14B LongBench 154755.759 J（-11.05%）、32B Alpaca 316673.794 J（-1.07%）。原始 artifact/canonical/能耗检查通过，但完整测量资格均未通过，仍是单次诊断，不作正式节能或最晚饱和声明。完整结果为 `results/2026-09-24/pdblend-combined-recovery-ab-v2/final-report.json`。

此时八卡全部清理且空闲，当前批次无运行中的 GPU 任务。后续若补测，只能保留上述有效点并针对缺失或已查明原因的失败点生成新计划。
