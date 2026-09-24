# 实验结果与原始证据

日常查看和作图使用 `results/runs.csv`，一行对应一个有独立身份的实验 attempt 或该 attempt
内明确命名的服务窗口。失败重试保留不同 run_id；倍率标定的 aggregate receipt 不重复计为
额外实验。profile 采样点放入 `results/profile_points.csv`，不与服务窗口行混合。

CSV 包含模型、系统、TP/PP 布局、TP mode、数据集、rate、seed、时长、请求与 token 数、
联合 SLO、TTFT/TPOT 分位数、各阶段和总能量、J/token、J/good-token、状态、资格与证据身份。
原始记录没有提供的指标留空，不能把未知能量写成零。CSV 是便于分析的索引，不授予实验资格。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python -B -m pdblend.results.catalog --root results
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python -B -m pdblend.results.profile_points --root results
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python -B -m pdblend.results.plot --csv results/runs.csv --out results/plots
```

`purpose` 区分 historical、development、calibration、profiling 和正式比较。正式配对必须同时
匹配模型、数据集、冻结 trace、实际 rate、seed、时长、SLO、镜像及公共计量/运行时协议身份。
各系统自身的源码与 profile 必须分别有效，不能要求它们共用同一个 profile hash。`formal_eligible`
为 false、身份不全、raw_pruned 的行均不参与正式排名。单 seed 701 结果只支持单 seed 结论。

PDBlend 本轮合成 A/B 使用 `development_functional`、`development_ab`、
`development_attempt` 区分功能、服务臂和总 attempt；`arm` 为 A（固定初始计划）或 B（周期规划）。
`queue_status`/`execution_complete` 表示执行完成情况，`audit_status`、`latency_comparable`
和 `efficacy_classification` 表示可比性及效果，不能把任务执行成功当作策略改善。
规划 p95、吞吐、实际动作数与 P/D 请求数来自独立审计；两臂共享 trace、初始计划及 Shield。
并行干扰失败的窗口标记 `parallel_unqualified`，保留原值，不借用后续串行复测指标。
若执行后发现 Plan 身份缺陷，`audit-plan-identity.json` 保存绑定同一 source/trace 的修订审计；
原 `audit.json` 和 completion 保留原字节，CSV 显式指向修订审计的路径与 SHA。
本轮 A/B 始终不具备正式三数据集或总能耗比较资格。

## 新日志

`pdblend.results.journal` 同时读取历史 JSONL 和新版 gzip journal。新版 payload 按事件 ID
只定义一次；累计文本用精确前缀编辑和增量保存，保留 detokenizer 修正 Unicode 尾部的语义。
EcoServe native/client 两次观测共享 payload，但分别保存时间、hold/flush 与控制动作证据。
外部引擎 SSE 协议不变，outcome 和 completion 仅存指标、计数及证据引用。

```python
from pdblend.results.journal import iter_journal
for event in iter_journal("events.jsonl.gz"):
    pass  # 恢复旧事件形状，包括原时间、token IDs 与文本
```

截断的 gzip 或缺失事件引用会报错，不把不完整日志当作完整凭据。正在运行及排队任务继续使用
各自冻结源码，因此本轮不能声称所有现有日志已采用新格式。已有被引用 raw 不就地压缩或改写。

## 历史清理与保留

有效 profile、校准、predictor、trace、机制资格以及运行、排队和待恢复任务组成当前依赖闭包。
保留其原路径及 checksum。退役实验先导出指标 CSV 和逐文件身份 CSV，再逐项检查 SHA、文件
状态、依赖及打开句柄；变化或新出现引用的文件跳过。历史报告中指向已删除 raw 的路径由身份
CSV 解释，不修改绑定 hash 的原报告。

删除过原始数据的历史行标记 `raw_pruned`，保留指标但明确不可复核。结果目录仍需保存当前原始
证据；一个聚合 CSV 无法替代功率积分、请求尾延迟、KV、generation、取消、迁移与回滚的审计。
清理记录位于 `results/maintenance/`，其中 Git 临时垃圾与研究数据采用不同的指纹要求。

Git 临时文件只删除经稳定性与进程/锁检查的 tmp_pack_/tmp_obj_；正式对象、分支和历史保留。
研究数据记录完整 SHA。predictor 资产即使建立新目录或兼容硬链接，也不计作磁盘空间释放。

## 计量、资产和压缩验证

新 Mixed rate anchor、resident campaign 与 EcoServe automatic 任务的 `power.json` 是计量
manifest，绑定 `power.samples.jsonl.gz`。GPU/source/field identity 归入 source epochs；功率、
频率、利用率、NVML 时间及返回码保留原精度。使用
`pdblend.results.power_archive.read_power_archive(path)` 可读取新旧两种格式并还原原数组。
总能耗图仅接受明确的全生命周期能量及一致 scope；未知总能耗不会由服务窗口能量补全。

绘图只使用具备完整共同身份的正式行；缺任一系统或同组有多个 attempt 时不自动挑选一次。
没有合格配对时仅生成 `results/plots/index.json`，不制造排名图。

供新任务冻结的源码中，compact journal 已接 Mixed、EcoServe native/automatic、DistServe
native/deployment、Dynamo native/transition probe。冻结任务按原版本继续运行。
`scripts/results/benchmark_compact_journal.py` 检验压缩前的存储复杂度及逐事件精确重放；
128/256/512/1024 token 合成双观测流的新原始量为 64,676/130,029/260,789/520,346 字节，
旧累计格式为 262,638/918,942/3,411,174/13,112,370 字节。该结果是指定合成流的验证，
不是所有实际实验的固定压缩率承诺。

`artifacts/predictors/index.json` 将三个 predictor 归入资产索引。其目录与旧 results 路径
共享硬链接，原容器挂载及 checksum 不变；禁止原地编辑这些不可变资产。
