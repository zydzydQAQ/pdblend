# PDblend 当前入口

GitHub 上的本次更新是代码、测试与文档快照；新增的原始测量、prepared 语料、模型和运行队列
保留在实验机，不随代码提交。下文 `results/2026-09-22/three-model` 与 `results/2026-09-23`
链接指向实验机上的证据。发布范围和未完成门槛见[复现进度说明](docs/REPRODUCTION-STATUS-2026-09-23.md)。

当前三模型 seed 701 工作流从[实时状态快照](results/2026-09-23/status/current.md)进入。
本轮优先执行 PDBlend 三模型功能与 300 秒 A/B，再执行 7B 八卡自动 P/D；
阶段凭据见[优先冒烟状态](results/2026-09-23/native-ab-status/current.md)。
使用 `python3 scripts/campaign/native_ab_status.py --prepared results/2026-09-23/native-ab-smoke-prepared-v3 --out results/2026-09-23/native-ab-status`
可刷新该快照。A/B 使用相同初始计划和 Shield，比较固定计划与每 10 秒重规划，
不用于正式三数据集排名或总能耗结论；Dynamo 长周期验收及无关补测已后移。
9 月 23 日这轮四个任务已完成：三模型功能及六个 A/B 窗口均通过，周期规划在两实例配置中
没有产生额外配置变化，见[三模型独立报告](results/2026-09-23/native-ab-independent-audit-v3/report.md)。
7B 八卡对照两臂 SLO 均为 100%，TTFT p99 从 0.860 降至 0.603 秒；B 同时触发了提频与唤醒，
因此该延迟结果不代表节能收益。详细路径和控制动作见
[八卡独立报告](results/2026-09-23/native-ab-independent-audit-v3/report-pd8.md)。
运行 `python3 scripts/2026-09-23_live_status.py` 可按实际队列和验收凭据刷新；快照时间写在报告内。
实验指标和画图从 [`results/runs.csv`](results/runs.csv) 进入；profile 测量点另列为
`results/profile_points.csv`。读取、比较键、压缩日志与历史保留规则见
[结果使用说明](docs/RESULTS.md)，源码职责与兼容入口见[目录说明](docs/CODE-LAYOUT.md)。
[本次文件与查询维护结案](results/maintenance/2026-09-23-closeout/report.md)记录实际释放空间、
O(1) 标量查询基准、CSV 导出、日志压缩及删除后验证；不代表 GPU 正式比较已完成。
历史实现状态保留在
[`results/2026-09-22/three-model/IMPLEMENTATION-STATUS.md`](results/2026-09-22/three-model/IMPLEMENTATION-STATUS.md)。

已复核的 TP4 校准组件见[版本报告](results/2026-09-23/calibration-versions-v1/report.md)。
使用 [`load_version`](src/pdblend/profile/query/versions.py) 显式选择版本，返回模型、覆盖域和可写入
结果 manifest 的 `profile_key`；[消费接口凭据](results/2026-09-23/calibration-version-consumer-v1/receipt.json)
记录了实际测试。组件资格允许指定范围的 development 查询，完整正式资格仍单独验收。

最新[增量测量收尾审计](results/2026-09-23/incremental-wave-closeout-v1/report.md)记录
7B TP1、32B TP2 长输入和 14B TP4 局部修复通过的精确覆盖域；14B TP1 的长输入仅完成
训练，24 个独立 holdout 已准备并入队；不重复采集其 36 个训练点。
新补测证据不自动扩大旧版本的可查询范围。

本轮[DistServe/EcoServe 六窗口审计](results/2026-09-23/dist-eco-completed-audit-v1/report.json)
记录真实请求、输出、排空和复用同一对引擎的证据。清理记录见 `results/archive/cleanup-2026-09-23*.json`，
删除清单和保留原因均可追溯。本轮历史清理改为保留指标与身份 CSV；当前依赖闭包内的原始
测量继续保留。标记 `raw_pruned` 的历史记录不可复核，也不参与正式排名。

[本轮恢复与清理报告](results/2026-09-23/maintenance-closeout/report.md)汇总任务结果和剩余门槛。
32B Dynamo 修复后的 [17/17 复跑审计](results/2026-09-23/dynamo-reroute-completed-audit-v1/audit.json)
保留原轨迹和 18 个 profile 点；[7B 异构池审计](results/2026-09-23/pdblend-tp-resident-completed-audit-v1/report.json)
保留早期三实例启动及 6/6 请求完成的证据。后续三卡双池资格任务已完成
195/195 请求，TP1 路由 57 次、TP2 路由 138 次；其作用是证明真实双池分流，
仍不等同于整机能耗比较或 KV/output golden 验收。

当前 seed701、300 秒评价轨迹入口为
[`first-five-system-batch-v3`](results/2026-09-23/first-five-system-batch-v3/report.md)。
其中 8/9 条已冻结；32B LongBench 尚缺通过 tail SLO 的独立 tuning，因此对应五点仍为
`inconclusive`。7B 复用 v2 原字节。三个模型分别使用自己的 calibration/tuning
倍率，不共用 7B 的 token 长度或容量。当前执行阶段、补测进度和正式通过数以实时状态页及
原始 receipt 为准；不把采样完成、功能通过或局部机制通过写成正式比较完成。

`results/2026-09-22/three-model/queue.json` 是可变的调度状态，会被 worker heartbeat、lease
和完成 receipt 更新，不能当作 immutable 证据。原始 samples、模型/tokenizer manifest、
predictor、verification receipt 和冻结源码位于各自 results/archive 或 attempt 路径，清理时
必须依据带 hash 的 tombstone 逐项核对。

当前环境固定为 vLLM 0.10.1.1、Torch 2.7.1、CUDA 12.8.1 和 8×L20；当前功能工作负载统一
使用 seed701，并要求 Qwen2.5-7B、14B、32B 三个模型分别通过 model/tokenizer identity
校验。镜像为 `pdblend:l20-cu128-vllm-v1`。

CPU 检查使用仓库 venv：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python -m pytest -q \
  -m 'not gpu and not historical' tests/pdblend tests/independent_baselines
```

GPU worker 必须使用 queue lease 提供的 GPU UUID、本地索引、端口和 concurrency environment；
不要手写物理 GPU，不要删除未完成 attempt，也不要把 development smoke 或 CPU replay 认定为
formal 或 energy qualification。迁移细节见 [`MIGRATION.md`](MIGRATION.md)，恢复手册见
[`RESTART.md`](RESTART.md)。
