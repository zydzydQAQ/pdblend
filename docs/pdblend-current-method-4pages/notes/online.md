# 当前在线路由与 KV handoff：源码核对

核对对象为 2026-09-24 `/home/pdblend4` 当前工作区，未运行 GPU，未修改服务代码。代码优先于旧文档。下方行号对应读取时的当前源码。

## 核心事实与代码依据

| 事实 | 依据 |
|---|---|
| 请求记录固定 P、D、TP、PP、pool、generation；M 路径 P=D，PD 路径 P≠D | `src/pdblend/online/router.py:21-46, 532-560` |
| P/D 配对需 TP、PP、pool、generation、model_id 完全相等，PP=1；当前没有跨 TP KV shape remap | `src/pdblend/online/router.py:486-493` |
| 请求输入长度与控制器发布的阈值决定当前候选路径；PD 压力门启用时阈值取 max(pd_min_input_tokens, pd_threshold_tokens)；没有 M 时可选 PD | `src/pdblend/online/router.py:500-527` |
| 未加 SLO 评分的默认实例选择：P 优先未见首 token 的 prefill token 总量，D 优先在途序列数；M 优先在途序列数 | `src/pdblend/online/router.py:480-498` |
| 压力进入/退出存在 enter/exit、hold 和稳定窗口；实际路由使用已随方案提交的阈值，而非到达事件临时切换规划负载划分 | `src/pdblend/online/router.py:410-448, 505-509` |
| SLO 路由容量限制：输入>0、输入+最大输出不超 max_model_len；每个涉及实例上已拥有请求的输入+最大输出之和，再加新请求预算，不超模型提供的 KV capacity | `src/pdblend/online/router.py:200-220` |
| 同步代理记账是保守 admission 预算，不能称其为已提前执行 vLLM block allocation；max_num_seqs 是 running batch 上限，非所有 admitted 请求硬上限 | `src/pdblend/online/router.py:213-218` |
| PD 的 queue 为 P 上未见首 token 请求的逐项 prefill 时间求和，decode batch 为 D 上所有在途请求+1，context 用最大输入+输出预算；禁止超出 decode profile 覆盖域 | `src/pdblend/online/router.py:222-246` |
| H 是 P 首 token 到 D 首 token 的端点间隔，已经包含第一个 decode step；N=2 必须有相应 first-gap profile 才能证明 SLO，handoff_floor 不能替代测量覆盖 | `src/pdblend/online/router.py:248-286` |
| PD TTFT=queue+prefill；PD TPOT=(H+(N-2)*step)/(N-1)；M TTFT=queue+prefill+step，TPOT=step；N=1 TPOT=0 | `src/pdblend/online/router.py:287-293` |
| M spillover 还检查原有未见首 token 请求的 TTFT 与已输出请求的剩余 TPOT 预算，加入新 prefill 对已有 decode 的干扰 | `src/pdblend/online/router.py:294-309` |
| 可预测且满足 safety×SLO 的原路径优先；仅当原路径为 PD 且无安全 PD 时考虑安全 M；同类安全路径选最小 TTFT；无安全预测可保留容量可行的原路径并标记 unproven_slo | `src/pdblend/online/router.py:324-379` |
| 路由器不是“对所有 M 与 PD 全局 argmin”；候选路径先受阈值/当前池方案限定，再有定向 PD→M spillover | 同上 |
| M 虚拟队列模型只在 deadline_safety 启用时使用；维护最多 max_num_seqs 个虚拟槽位，FIFO 排列未见首 token 请求，prefill 暂停已有 decode 的服务时间 | `src/pdblend/online/deadline.py:8-63`，`router.py:236-240` |
| deadline 信号来自 admission/token/terminal，token 信号50ms合并；明确标注 observed_proxy_queue，非原生 scheduler 遥测 | `src/pdblend/online/router.py:147-173, 566-590, 627` |
| P 首 leg max_tokens=1 且非流式，提取权威 token ID；先向用户发出 P 首 token，再向 D 提交 prompt+[首token]，剩余预算 N-1；最终校验 usage 与 DONE | `src/pdblend/online/server.py:127-139, 182-201, 203-240`，`src/pdblend/engine/carry.py:68-104` |
| 当前 carry 协议限 token-ID prompt、greedy、ignore_eos=true、n=1，拒绝无法支持的 sampling/stop 语义；不应描述为任意通用 OpenAI 请求无损支持 | `src/pdblend/engine/carry.py:23-41` |
| N=1 将已选 PD 路径改为 P_ONLY，实际只在 P 执行普通请求，无远程 KV；resident 之前已获取的 P/D 预算仍按 request ID 到完成时释放 | `src/pdblend/online/server.py:100-111` |
| P2pNccl 通过相同 request ID 编码 P/D 地址，KV 由引擎直传；NIXL 用 kv_transfer_params 传递描述信息，均非代理中转 KV tensor | `src/pdblend/engine/client.py:63-96` |
| P2P 按层提交 KV；D 远程复用 prompt 长度-1 的 KV，使新增 carry token 在 D 本地执行；不能说所有物理复制都在 P 首 token 返回后才开始 | `engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:250-283, 310-349` |
| 旧传输模型为 fixed + tokens×kv_bytes_per_token/bandwidth；较长输出没有端点覆盖时用 transfer+step 估 H | `src/pdblend/profile/query/model.py:202-204`，`router.py:264-286` |
| 端点开发 profile 仅接受准确输出预算和请求频率、batch=1、context=input+output、输入插值域；不是加载下任意 batch、任意频率的合格模型 | `src/pdblend/profile/query/development_composite.py:258-275` |
| H 观测写入 request.route_estimate，源码没有看到利用此值在线 refit first-gap 的实现，勿写成持续在线训练 | `src/pdblend/online/router.py:577-584` |
| HTTP失败或客户端断流不等于KV已释放；状态 uncertain 会保留记账并 quarantine P/D，原生取消回执满足 request/topology/generation/rank/transfer 清理证据才释放 | `src/pdblend/online/router.py:592-643, 655-692`，`server.py:144-180` |

## resident TP 与能耗路由的实现边界

1. `ResidentRouter` 先收集各子池 `candidates`，保留子池当前提供的路径；`ResidentTPRouter` 原子保留每个所选实例上的 `input_tokens + max_tokens`，请求完成之前固定 generation。见 `router.py:800-832, 880-895`；`tp_modes.py:97-174, 176-188`。
2. target shares 由外层规划器给出。可行池内按最大欠额 `w_k (A+1)-A_k` 分发；目标池不可行时换另一可行池。见 `router.py:867-879`。这是请求流量权重，不是在线重写模型权重。
3. 可选增量能量评分只有显式传入 `incremental_energy_path` 时启用，见 `bench/run.py:318-323`。不是普通 prefill/decode 功率拟合直接构成逐请求增量能量。
4. 增量能量精确绑定 model、TP/PP、path、profile keys、P/D 当前频率、input/output、batch、context、queued prefill、reservation。相同时间窗长度的带请求/基线全路由 GPU 能量差为 ΔE；训练和独立 holdout 各至少3个样本，holdout 最大相对误差≤10%，候选排序不得逆转；不做插值。见 `energy_routing.py:18-21, 38-72, 90-139, 142-199`。
5. 若所有存留候选均有 qualified 能量，按 ΔE 比较；只要某候选无覆盖，则整个集合统一用 TTFT+N×TPOT 的时延分数。能量评分前先排除超出其时延约束的候选。见 `router.py:834-866`。这些仍只是 development/非 formal-qualified artifact，见 `energy_routing.py:195-198`。
6. **当前未统一的关键点：** `ResidentRouter._route_context` 仍用 `TTFT=queue+prefill+transfer+step`、`TPOT=step`，见 `router.py:761-781`。其最后传 `choice` 调用子 Router，绕过子 Router `_slo_route_choice`，见 `router.py:537-540, 888`。`PoolPlanner._prefill_pool` 也把 transfer 加入 TTFT，见 `planner/pool.py:375-395`。论文不能宣称普通 Router 的端点 H 保护已统一落地到整个 resident/池级规划层。
7. 默认配置为 slo_routing=True、shield_mode=budget_aware，但 deadline_safety=False；见 `bench/pdblend_runtime_options.py:10-16`。两 token 缺 profile 只不能证明 SLO，并非无条件拒绝 PD：无安全 M 时原路可继续并标记 `legacy_capacity_fallback_unproven_slo`。

## 简明数学表达

统一用输入长度 n、输出预算 m、预填充时间 P(n,f)、decode 步长 D(b,c,f)、排队时间 q、首间隔 h。避免引入大量集合和希腊字母。

\[
T_{\mathrm{first}}^{PD}=q_P+P(n,f_P),\quad
T_{\mathrm{token}}^{PD}=\frac{h+(m-2)D(b,c,f_D)}{m-1},\qquad m\ge2.
\]

\[
T_{\mathrm{copy}}(n)=t_0+\frac{n b_{KV}}{B},\qquad
h=\max\{T_{\mathrm{copy}}(n)+D(b,c,f_D),h_{\min}\}
\]

第二式是无完整端点覆盖时、m>2 的旧近似。若首间隔测量覆盖，h直接取测量预测与 floor 的最大值，不再加 D。若m=2且无覆盖，预测未知。

每实例 admission 预算：`sum_owned (n_r+m_r) + n+m <= C_i`；不是 native KV block 硬分配。

## 建议伪代码：忠实于普通 Router 的 SLO 模式

```text
输入请求(n,m)，当前角色、阈值、实例负载与profile
C0 ← 按已提交长度阈值枚举当前路径的兼容实例或(P,D)对
C1 ← 当C0为PD时的M备选，否则为空
分别删除 C0,C1 中不接受新请求或超过KV/长度预算的候选
对其余候选查询有覆盖域的TTFT/TPOT，并检查M中已有请求预算
G0,G1 ← 满足安全系数×SLO的候选
若 G0 非空：选 G0 中预测TTFT最小者
否则若 C0 原为PD且 G1 非空：选 G1 中预测TTFT最小者
否则若容量允许的 C0 非空：按旧负载规则选择并标记SLO未证实
否则拒绝
绑定实例对、拓扑代次，记账后执行：M直接运行；PD先P首token，再D续写
完成释放记账；未确认清理的失败保留所有权并暂停相关实例接单
```

## 论文草稿

见相邻 `sections/online-draft.tex`，正文约 1800 个中文字符（另含公式与伪代码）。主稿可将边界段和可选能量段缩写，避免抢占第三部分篇幅。
