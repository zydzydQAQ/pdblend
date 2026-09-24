# 150 秒驻留比较

作图入口为 `results/compare.csv`。当前分析显式启用
`--analysis-policy all_recorded_windows/v1`：`status=measured`、
`measurement_usable=true` 表示绑定 result/receipt 中有实际完成的 150 秒服务窗口、
实测能耗和请求/延迟统计；频率、profile 和原生状态的新鲜度审计失败不丢弃实测数据。
`original_status`、`qualification_*`、`original_result_*` 和 `original_receipt_*`
保留原资格结果，原 `evidence_valid`、`formal_eligible`、`baseline_frozen` 不改为通过。
`analysis_baseline_frozen=true` 表示该实际 baseline 窗口在分析中采用并保留，
包括原审计未通过或 SLO 失败的窗口；它不会重新授权执行已测点。
`analysis_slo_pass` 按成功率 100%、联合 SLO 比例至少 90%、TTFT/TPOT p99 均达标
重新判断排名条件；原 `slo_pass` 和所有原始数值保持不变。请求失败的窗口仍是可用数据。
无数据的启动失败或未执行点保持 failed/blocked/prepared；空值不当作零。
执行清单按不可变版本追加，包含 180 个逻辑点；
恢复产生的无效历史 attempt 另留一行，因此 CSV 行数可以超过 180。
各行的 `campaign_id`、`revision`、receipt 和 session 身份用于定位实际证据。
当前 watcher 的命令保存在对应 campaign 目录的 `csv-watch*.json`。

此分析模式的主 `rank_eligible`/`energy_rank` 比较身份匹配且满足上述 SLO 的
实际测量，不以前置 profile 资格或目标频率合规作为条件。模型、tokenizer、
轨迹、数据集、倍率、seed、GPU、镜像、计量协议和 SLO 阈值必须匹配。
`comparison_scope=available_recorded_system_variants_single_observation` 明确为当前可用子集；
参与系统数、是否齐全五系统及逐版本 receipt 单列。每个 PD revision 独立比较，
不同 baseline revision 保留各自变体，同系统同 revision 重复 attempt 不挑最低值。
baseline 的多个 PD 对比排名见 `energy_rank_by_revision`，具体参与证据见
`comparison_participant_receipts_by_revision`；单系统第一名不代表优于其他系统。
`strict_*` 保存旧严格排名，`observed_*` 仍保存旧“已验收数据”的描述层。
这些旧字段可能为空，即使新主排名已有结果。`tail_reverses_saving` 仍单列尾部抵消。
不启用参数时，导出保持原有严格规则。以下涉及旧验收、无效窗口与严格排名的描述，
均说明原始执行/资格协议，不能当作当前显式分析模式的数据剔除规则。

同一八卡租约内，一个兼容引擎组只装载一次，按 x0.5、x0.25、x0.75、x1.0
运行数据集窗口。每窗有独立预热、排空、重新开放接纳、generation 和策略状态。
每个 baseline 的第一次有效观察立即冻结，包括 SLO 失败；重试只处理证据无效
或未执行点。`ResidentGroupSession --previous` 支持验证旧证据并跳过已完成点。
恢复时只对本次待执行点做当前源码的启动验收；旧冻结点保持原来的源码与证据身份。
Eco7/14 恢复脚本的 `--unattempted-from <session>` 进一步限定为该终态 session 的
`planned_points − windows − skipped`。它核验已释放租约、worker 终态、清理、
原 group 和逐点 receipt 哈希，并保存独立选择凭据。此模式不重试历史无效窗口，
也不会因引擎签名相同把这些窗口误带进新任务组；未指定时仍沿用原恢复语义。

队列默认优先满足已就绪的整机预留，普通分区任务不能仅靠提高 priority 绕过它。
必要的短功能补测可在新的不可变任务中显式设置
`precedes_host_reservations=true`，且其 priority 必须严格高于下一整机任务；
此例外仍遵守任务依赖、活跃 sampling cohort、主机锁和 GPU 租约。
发布前须确认它的 GPU 数在当前设备范围内，避免不可执行的前置任务阻塞队列。
更换 worker 时，旧 worker 的 stop 文件仅在任务之间生效；新 worker 使用同一
队列与主机锁等待，当前租约正常结束，不能借重排中止正在测量的窗口。

EcoServe 可在新点清单显式启用
`observation_failure_policy=continue_after_verified_frequency_rejection`。
仅当唯一失败门为 `eco.observed_active_frequency`，全部原始文件、绑定、
配置、请求、协议、计量和最终排空门均通过时，session 才会保存该无效窗口并
继续下一次完整 reset。频率标准仍为原来的 2520±30 MHz 与最长 1 秒采样间隔；
无效点不会冻结或参与排名。其他失败以及 reset、execute、drain 异常仍隔离 session。
此时 session 的 `complete=true` 只表示排程已执行完，
`all_observations_valid` 和 `invalid_observations` 单列测量有效性。
其中仅实际频率失败的窗口，可能仍有完整请求与功率原始样本；
`invalid_measurement` 表示它不满足当前固定频率比较协议，不能解释为所有
原始计量都失真。2026-09-24 的 Eco7 ShareGPT ×0.25 同窗诊断中，
活跃区间的低频样本伴随 350 W 上限下的 `SwPowerCap` 标记，证据见
`results/2026-09-24/eco7-v6-sharegpt-x025-clock-diagnostic-v1/correlation.json`。
该 1 Hz 只读诊断不能代替正式 100 ms 采样，也不能追认此前无效点。
固定频率资格可能排除部分高负载窗口，故剩余有效子集不能证明全域最优。

`energy_service_j` 是固定 150 秒内八卡总 GPU 板卡能耗，停车卡也包括在内。
`energy_tail_j` 单列排空尾部，`energy_service_tail_j` 包含两者。
冷启动和预热不进入服务能耗。TTFT 从计划到达算起，TPOT 使用客户端首末
token 时间。`goodput_request_s` 和 `goodput_token_s` 只计窗口结束前已完成且
同时满足 TTFT/TPOT 的请求；`cohort_goodput_*` 使用整个到达批次排空后的时长。

排名仅接受相同模型、轨迹、GPU、镜像、公共引擎、公共计量实现和协议的有效
SLO 合格观察。每个 PDblend revision 单独对同四个冻结 baseline 比较，
不会跨版本拼最优点。`tail_reverses_saving` 标记尾部能耗抵消服务窗口优势。
一个 seed、一次观察不能证明统计显著性或全局最优。

共同的实际频率证据另外由 `common_clock_evidence` 标为 `pass/fail/unknown`。
冻结 Mixed 观察仅验证请求的初始 2520 MHz，原始服务窗没有频率采样，
所以该字段是 `unknown`；其已有 `evidence_valid`、能耗、SLO 和冻结状态不变。
EcoServe 与 PDblend 的实际频率门通过后，才可能取得对应共同证据。
该 scope 是各系统是否跟随自己的请求频率，并不证明各系统实际频率相同。
原 `rank_eligible`/`energy_rank` 必须全组通过共同频率证据，原子集比较也只接受通过的 baseline；
`clock_unqualified_baseline_systems` 明列因此排除的系统，不能把它们隐去后
声称全域最优。补充字段是对已绑定验收源码和 receipt 的独立解释，
不为旧结果补造频率，不修改冻结文件，也不自动重跑冻结 baseline。

五系统尚未齐全时，`available_*` 和 `pdblend_saving_vs_best_available_baseline`
提供已完成且身份匹配的冻结 baseline 子集比较；字段同时列出参与系统及证据。
它不填充完整比较的 `energy_rank`。一个可行 baseline 已足以指出 PDblend 在
当前观察中落后；子集第一名不能解释成五系统最优。各 revision 仍单独比较，
重复有效 attempt 不选较低能耗，尾部抵消另由 `available_tail_reverses_saving` 标记。

同一 CSV 的 `observed_*` 是单独的实测系统描述层，
`observed_comparison_scope=system_as_executed_single_observation/v1`。
它与上述严格比较复用相同身份、原证据有效性、SLO、baseline 冻结、重复 attempt、
revision 和尾部规则，只是不把额外的共同频率证据作为描述实测 J/SLO 的前提。
因此 Mixed 频率仍为 unknown 时，其已验收能耗仍可参与此层；
仅频率失败的 invalid Eco 点仍不纳入，也不追认或冻结。
`observed_energy_rank`/`observed_rank_eligible` 要求完整四个 baseline 与一个 PD revision；
未齐全时看 `observed_available_*`、`observed_best_available_feasible_baseline` 和
`observed_pdblend_saving_vs_best_available_baseline`，参与系统及 receipt 均列出。
`observed_tail_reverses_saving` 与 `observed_available_tail_reverses_saving` 仍单列尾部抵消。
这些字段表达当前绑定观察的能耗顺序，不能证明同频机制因果优势、统计显著性或全域最优。
原 `common_clock_evidence` 与严格排名继续并列，不改旧测量值或资格。

session 成本在 CSV 中随窗口重复展示，作用域是 `session_cost_scope_id`，
汇总时必须按此列去重，不能把 `session_engine_load_s` 等字段按行相加。
会话结束前，尚未知的清理耗时与总耗时留空。原始日志、功率、请求与不可变
receipt 保留用于审计，完整代码快照和 SHA 位于 campaign 的 `sources/`。
`session_engine_loads` 在会话结束后采用实际累计引擎启动次数，包含 PDblend
算法主动关闭后的重启及下一窗口恢复库存所需的装载；初始装载另有字段。
编排层不会为保持“常驻”而取消 PDblend 的 off、park 或唤醒行为。
公共计量身份包含 metrics、metering、client 和 measure 实现；系统专属
调度及验收代码另由完整源码 SHA 和实际启动参数绑定。

PD 原生补测也在单租约内复用 Fleet。新 14B 补测将完整观测到的频率偏差
与协议、采样、恢复失败分开：前者只保存无效功率样本，必须通过独立恢复后
才能继续无关采样；频率缺测、ACK 或代际损坏等仍停止任务。旧 attempt 不追认。
新 32B 补测顺序为 runtime、CUDA timing、整组能耗的训练和独立 holdout，
不再附带不能直接解锁该模型的旧 power pilot 与 60 秒 discovery cycles。
`resident-timing-stage.json` 只证明仍存活且已排空的 Fleet 上完成了时延采样，
不代表队列成功或 GPU 已释放。最终使用该组件仍需绑定实际 worker 完成、
队列终态和八卡清理凭据。32B 的全 M4/TP2 静态布局组件不等于在线 PD 策略资格；
在线积压、切频和滞回的覆盖未完成时，正式 PD 比较继续保留阻碍。

后续 v2 原生补测可显式使用 `--timing-first`：runtime（若指定）、CUDA timing、
不可变 timing stage、独立能耗补测。每段保存实际开始、结束和失败阶段。
新 `pdblend-native-terminal-timing-stage-evidence/v1` 消费者允许独立的后续补测
失败时单独重放已完成的 timing；它仍要求原始时延、独立 holdout、源码身份、
实际队列终态和八卡清理全部合格。整个失败任务保持失败，不据此取得完整策略资格。
旧整任务凭据的成功要求不变，不能把过去根本没有执行的 timing 补造成已有组件。

未来 request-cycle 采集把完整的实际频率偏离与采样、请求或协议失败分开。
只有其余原始验收及全部 rank 的初始时钟、测量、排空 ACK 均通过时，才保存
无效窗口、执行全 Fleet 恢复并验证库存，然后继续下一独立窗口。
频率缺测、重复时间、非有限值和任何并发失败仍停止采集。
训练中有无效窗口时不拟合候选、不启动依赖该候选的 holdout；holdout 无效也
不生成已通过的组件。恢复成功只允许后续无关采样，不放宽 ±30 MHz 或 1 秒间隔。

EcoServe 恢复版本可显式指定 `metering_execution=isolated_process`：相同
NVML 后端、原始采样器、100 ms 周期和 1 秒最大插值间隔在独立子进程中执行。
它同时启用原采样器已有的实际频率观测，另存方法源码、PID、生命周期和
窗口保护凭据。序列化 RPC 必须在服务及尾部测量之外。缺测仍使测量无效，
不放宽阈值，也不补造此前缺失的功率或频率。
CSV 的 `metering_execution` 明示采样执行方式。装载前的启动自检使用八卡
功率与利用率观测的共同时间区间；正式服务窗口仍按原始到达时钟固定为
150 秒，不使用此自检裁剪规则。

创建新 campaign（不会重写旧 campaign）：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/pdblend/.venv/bin/python \
  scripts/2026-09-23_prepare_resident_comparison.py \
  --out results/YYYY-MM-DD/resident-comparison-vN --allow-mixed-qualification
```

准备脚本默认只生成不可变输入。正式执行使用已有 GPU 队列的 `enqueue jobs.json`，
或显式 `--enqueue`。其他系统通过 `--inputs` 传入各自的配置、profile、
离线选择和资格凭据，不会因 runner 能启动而自动获得完整资格。
Mixed 不需要预测 profile，但仍必须通过现场原始证据验收。

CSV 可由 `python -m pdblend.bench.comparison_campaign export --campaign ...
--sessions ... --out results/compare.csv` 重建。`--watch --queue ...` 在新 receipt
出现或相关任务结束时刷新，空闲期间只检查文件元数据。它从实际租约发现新
session，并核验组合任务的 plan/source/parent/freeze 哈希链后接受新增轨迹。
新 watcher 首次读取实际文件计算 SHA；同一进程内，仅在设备、inode、大小、
mtime、ctime 全部不变且修改时间已稳定至少 2 秒时复用摘要，避免每个新窗口
重复读取历史大文件。新文件与变更文件仍读完整字节，任务终态清空缓存再核验。
缓存不会从 receipt 声明的 SHA 初始化，也不跨进程持久化；重启的首次全量核验
安排在非测量阶段。
32B LongBench 只有独立 calibration/tuning 确认后才生成 evaluation 轨迹；
未确认的容量不得用评测结果反推，阻碍会保留在 CSV 中。
