# 渐进优化实现与验收

本次修改将在线正确性、模型消费、积压预测和候选成本排序接入运行入口。异构 TP 的联合规划、增量能量路由和降低 M 容量下限采用显式开启及实测证据校验。在线 TP 重分片仍是后续独立阶段；`slow_reshard_tp` 的原生验收保护保持有效。

## 实现范围

| 环节 | 实现 | 验收边界 |
|---|---|---|
| 请求入口 | 全部候选先检查容量；执行中的重复 body/header request ID 返回 409 | CPU 反例与真实 HTTP 测试 |
| 终止与取消 | 区分 completed、rejected_before_engine、uncertain、cancelled_acknowledged；异常 EOF 保留 ownership；合法 EOS 可提前结束 | 原生所有 rank、代次、请求、KV 与传输确认后恢复 |
| Shield | 全部在途记录不受最近 60 秒历史截断；最近 token 时间参与停顿检测 | 长请求与近期停顿回归 |
| 模型入口 | `load_profile` 接收明确文件或 registry/version；planner/router 共用绑定对象 | 不选择 latest；组件、覆盖、版本与查询结果写入日志 |
| Forecast | 明确关联输入／输出；两侧条件统计；先验渐退；等待 prefill、剩余 decode、已占 KV | 截断或取消未确认的工作仍计入存量 |
| Planner | 全可行候选按 60 秒稳态能量加转换成本排序；每轮角色子问题缓存 | 第二候选反例、缓存开关差分；不是实测节能比例 |
| 重配置 | GPU 控制移到线程并按物理 GPU 互斥；原生排空；唤醒确认后准入；逐阶段记录 | 共同采样积分保留 gross energy；未配对时不产生实测额外 J |
| 联合 TP | 外层目标流量＋内层角色／频率；完整驻留容量计费；实际分流反馈 | Mixed 功率缺失时保留当前配置；全部角色／频率／准入就绪才发布 |
| 能量路由 | TTFT、TPOT、KV、覆盖筛选后的增量 J 排序 | 对照窗口与独立 holdout 校验；候选证据不完整则全体回退至时间评分 |
| M 下限 | 按模型、TP、SLO、负载域及 profile 绑定验收文件 | 未验证域保留原下限；不能用 CPU 反例降低 |

PDBlend 使用 `pdblend_runtime.serve` 提供原生队列、逐 rank 状态及取消接口。新的补测明确记录这个服务入口；历史普通 vLLM 入口的 timing/static 测量仍标明原来的资格范围，不自动继承正式能效资格。其他独立 baseline 的模型与实验数据保持隔离。

## 运行接口

`--profile` 可以指向显式 `pdblend_profile_selection_v1` 描述文件。组件和运行时审计的生成、绑定方式见 [profile 说明](../../src/pdblend/profile/optimization_profiles.md)。已验证的实际版本消费回执位于 `results/2026-09-23/progressive-optimization-profile-consumers-v1/completion.json`；这是 CPU 消费验证，不是新 GPU 性能结果。

`pdblend bench` 与 matrix 支持：

- `--tp-mode resident_hetero_tp`、`--topology-profiles`、`--resident-pools`；
- `--joint-resident`：显式启用常驻双 TP 池联合规划；
- `--incremental-energy-path`：经过独立验证的增量能量文件；
- `--transition-catalog-path`、`--transition-qualified-only`：转换成本和严格覆盖；
- `--capacity-floor-path`：限定负载域的 GPU 容量下限验收。

具体参数以 `pdblend --help` 和 `pdblend bench --help` 为准。所有组件与布局检查均在启动 GPU 引擎前执行。联合规划不能同时采用强制固定池计划或另一套动态 M 下限控制器。

转换目录的证据格式见 [转换成本](transition-cost-catalog.md)，路由增量能量见 [能量路由](online-energy-artifact.md)，阶段比较见 [能效验收](optimization-acceptance.md)。普通 benchmark 输出增加 `profile-selection.json`／TP 选择元数据、`native-cleanup.json`、`transition-measurements.json` 和请求终止／路由估算信息。

## GPU 队列与后处理

`scripts/2026-09-23_prepare_progressive_optimization.py` 冻结实际实现源码、固定镜像和输入，运行 CPU 容器预检，生成九个独立租约任务：

1. 三模型原生断流、取消、KV 清理、请求恢复以及调频／停车期间继续服务，分别占 2、2、4 卡。
2. 两组 `7B TP1 + 14B TP1 + 32B TP2`，合计八卡，分别采集 1500 与 2520 MHz。每组训练与独立 holdout 共用一次引擎生命周期；每点三次窗口。

补测包括 B2/B3 连续 decode 功率和同时包含 prefill probe、decode、功率积分的 Mixed 窗口。保存真实原生调度 batch 分布。功率模型只开放经验证的精确 batch、chunk、到达率及上下文域；没有分数 batch 或新频率证据时仍拒绝外推。独立的 native timing 和正式工作负载能耗验收仍必需。

新任务在已有任务终止并释放租约后开始。SamplingEpochs 在窗口边界处理成员退出与重新验证，拒绝测量过程中加入未批准负载。正式整机比较仍需独占八卡。

`scripts/2026-09-23_collect_progressive_results.py --watch` 仅汇总这批任务。它重新验证每个组件的训练、holdout 和哈希，合并互不重叠的频率域并生成明确 development 选择文件；不修改 registry、latest 或正式默认配置。失败和未通过的组件保留原始证据。

统计同时报告占用 GPU 秒、合格测量 GPU 秒、同步、加载、预热／清理和无效测量。不能把瞬时 GPU 利用率或租约占卡率当成有效采样率。

## 推广条件

CPU 通过不等于原生功能通过；原生功能通过不等于节能。阶段验收要求同三模型、三数据集、seed 701、轨迹、请求率、SLO、TP、模型权重与引擎身份，保留独立 baseline profile。检查 joint SLO、达标 goodput、J/达标 token、拒绝率和切换次数。goodput 不退化且节能超过计量与重复实验不确定性才通过相应阶段。

本次没有把旧静态功率差、合成反例、局部功能或未配对转换能量换算成正式节能比例。真实证据不足的路径继续保持验收保护。
