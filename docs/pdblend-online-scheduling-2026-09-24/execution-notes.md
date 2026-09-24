# PDblend online 执行链路核对（2026-09-24 工作树）

本页依据当前源码，区分请求路由与实例内 vLLM 调度。未更改业务代码，未启动 GPU 实验。`engine_patches/vllm-0.10.1.1` 是仓库内当前可见的 P2P overlay；本页不将其等同于已核验的当前运行容器版本。

## 1. 分配结果与执行责任

- `Proxy.completions` 先调用 `router.dispatch(request_id, input_tokens, max_tokens)`；返回记录固定 `path, prefill_instance, decode_instance`，因此普通在线 PD 路径在请求到达时已同时选定 P 与 D，并非等 prefill 结束后才挑 D。[server.py](/home/pdblend4/src/pdblend/online/server.py:86)
- M 路由令 `p=d=m`；代理仅向 m 发送一次原始 completion，m 在自己的 vLLM 队列内完成 prefill、首 token 和后续 decode。绿色 prefill 与蓝色 decode 可与该实例上其他请求交错，不代表每个阶段重新经过全局 router。[router.py](/home/pdblend4/src/pdblend/online/router.py:416)、[server.py](/home/pdblend4/src/pdblend/online/server.py:140)、[server.py](/home/pdblend4/src/pdblend/online/server.py:242)
- “实例”是 GPU group，一个实例可用 TP×PP 张卡。代码 launcher 为每组启动独立 vLLM；本地 batch、chunked prefill、token step 分配由该实例 vLLM scheduler 完成。[launcher.py](/home/pdblend4/src/pdblend/engine/launcher.py:20)、[launcher.py](/home/pdblend4/src/pdblend/engine/launcher.py:84)
- native scheduler 在 admission 设置允许的请求阶段中调用 `super().schedule()`，记录本次 schedule queue，而不是 PDblend router 逐 token 选 GPU。[native_v1.py](/home/pdblend4/src/pdblend_runtime/native_v1.py:128)

## 2. PD 的首 token carry 协议

设原始 prompt 为 x（长度 L），请求输出预算 O≥2。

1. P 接收 x，`max_tokens=1, stream=False`，执行完整/chunked prefill，生成真实第一输出 y₁；代理通过 logprobs 的权威 token ID 读取 y₁，不对文本重新 tokenize。[server.py](/home/pdblend4/src/pdblend/online/server.py:182)、[carry.py](/home/pdblend4/src/pdblend/engine/carry.py:68)
2. P 的 HTTP 响应完成后，代理先向客户端发送 y₁ 的 SSE，调用 `router.first_token`，记录 TTFT；随后启动 D 的 HTTP 请求。[server.py](/home/pdblend4/src/pdblend/online/server.py:127)
3. D 输入为 `[x, y₁]`，输出预算为 O−1。P2P connector 给 D 标记的 external matched token 数为 `len(D_prompt)-1-num_computed_tokens`，即首次为 L：从 P 复用原 prompt 的 KV；D 对 carry token y₁ 做续算，产生 y₂，继续至 yO。[carry.py](/home/pdblend4/src/pdblend/engine/carry.py:88)、[p2p_nccl_connector.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:310)
4. D 流式结果接到 y₁ 后转发；最终 usage 合并为原始 prompt L、输出 O，要求 `[DONE]` 与准确 combined usage。[server.py](/home/pdblend4/src/pdblend/online/server.py:203)、[carry.py](/home/pdblend4/src/pdblend/engine/carry.py:96)
5. O=1 的特殊情况：已选定 P 后转换为 `P_ONLY`，把 sequence 计数从 D 转到 P，普通无标签单引擎生成，不发远程 KV。[server.py](/home/pdblend4/src/pdblend/online/server.py:100)

当前 PD carry 路径要求非空 token-ID prompt、greedy、`ignore_eos=true`、单 choice，并限制 stop/penalty/logprobs 等选项；不支持的请求在进引擎前返回 400 并释放预留。[carry.py](/home/pdblend4/src/pdblend/engine/carry.py:23)、[server.py](/home/pdblend4/src/pdblend/online/server.py:91)

## 3. KV transfer 的控制流与数据流

- 当前 launcher 默认 `P2pNcclConnector`、`kv_role=kv_both`、`PUT_ASYNC`。基准运行使用实例 specs 的 connector 显式构造 `PDTransfer` 并传给 Proxy，所以不能因 Proxy 构造函数缺省写着 Nixl 就把当前默认画成 Nixl。[launcher.py](/home/pdblend4/src/pdblend/engine/launcher.py:32)、[launcher.py](/home/pdblend4/src/pdblend/engine/launcher.py:71)、[run.py](/home/pdblend4/src/pdblend/bench/run.py:265)
- P2P 中两段 HTTP 使用相同 engine request ID，内含 P 和 D 的 ZMQ 地址；HTTP body 不携带 `kv_transfer_params`。两端引擎依 ID 判断本请求为 P、D 或普通 M。[client.py](/home/pdblend4/src/pdblend/engine/client.py:63)、[client.py](/home/pdblend4/src/pdblend/engine/client.py:67)、[p2p_nccl_connector.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:457)
- P 以 request+layer 为键，将 prompt KV 按层发送到 D 对应 rank；D 接收并注入 paged KV cache。数据直接走引擎间 P2P/NCCL，代理只处理请求、首 token 和流。[p2p_nccl_connector.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:228)、[p2p_nccl_connector.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:250)
- chunked prefill 情况会积累 block IDs，到原 prompt 全部 prefilled 的 step 才建立完整发送 metadata；因此不要把当前 connector 画成每一个 prefill chunk 都必然立即转发。[p2p_nccl_connector.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:366)
- 发送侧 compute stream gather KV 并记录 CUDA event，后台 PUT_ASYNC 排队发送；接收侧等待 tensor 后从 receive store pop，一次消费并注入，避免 decoder 留两份 prompt KV。[p2p_nccl_engine.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:219)、[p2p_nccl_engine.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py:266)
- Nixl 是可选适配：P 请求 `do_remote_decode=true`，P 响应 handoff 参数复制到 D 请求并改为 `do_remote_prefill=true`。[client.py](/home/pdblend4/src/pdblend/engine/client.py:50)
- 实例即使 runtime 改为 P/D/M，也可保留同一个支持双向 KV 的 engine；不应把 M/P/D 画成三种不同模型结构。[manifest.json](/home/pdblend4/engine_patches/vllm-0.10.1.1/manifest.json)

## 4. 时间指标与负载回馈

- `TTFT = first_token_s - submitted_s`，PD 的首 token 是 P 的 y₁。[router.py](/home/pdblend4/src/pdblend/online/router.py:45)
- `TPOT = (finished_s-first_token_s)/(completion_tokens-1)`，包含 y₁→y₂ 的首个间隙。[router.py](/home/pdblend4/src/pdblend/online/router.py:49)
- online `observed_handoff_s = first_decode_token_s - pd_handoff_started_s`，其中 handoff start 等于 P token 发送时的 `first_token_s`。这等价于代理观测的 y₁→y₂ first gap，含后续提交、D 排队、KV 等待/注入、D 首步等；不能标成纯物理 KV copy 时延，且已在 y₁ 前完成的 transfer 不在此区间内。[server.py](/home/pdblend4/src/pdblend/online/server.py:133)、[router.py](/home/pdblend4/src/pdblend/online/router.py:476)
- profile 的 `measure_handoff` 另有“同逻辑第二输出”差值指标 `PD(y₂到达)-M(y₂到达)`，标明 `includes_http_and_scheduling=true, physical_copy_time=false`，别与 online raw first gap 混淆。[handoff_timing.py](/home/pdblend4/src/pdblend/engine/handoff_timing.py:9)
- dispatch 时 `Qp[p] += L`、`Nd[d] += 1`；第一 token 出现时 `Qp[p] -= L`；成功完成时 `Nd[d] -= 1`。M 中 p=d 所以两类计数都归同一个 m。Qp 是已派发且未见首 token 的 prompt tokens，Nd 包含已预留但尚未开始 D 的 sequence，不等于实时 GPU running batch。[router.py](/home/pdblend4/src/pdblend/online/router.py:452)、[router.py](/home/pdblend4/src/pdblend/online/router.py:476)

## 5. 失败与安全回收

- Proxy 正常路径没有“PD 出错即重投 M”的自动重试。失败作为 SSE error 返回；未开始 engine 的拒绝直接释放，已提交而结果不确定则保留负载/ownership，并将涉及 P、D quarantine 停止接新请求。[server.py](/home/pdblend4/src/pdblend/online/server.py:149)、[router.py](/home/pdblend4/src/pdblend/online/router.py:499)
- 开启 native control 的实际 PDblend benchmark 会调用原生 cancel；必须拿到所有已提交引擎的 generation、TP/PP、所有 rank、清空 request/KV/transfer inventory 的 ACK，才能释放并解除 quarantine；失败则继续保留。[run.py](/home/pdblend4/src/pdblend/bench/run.py:323)、[server.py](/home/pdblend4/src/pdblend/online/server.py:170)、[router.py](/home/pdblend4/src/pdblend/online/router.py:526)、[router.py](/home/pdblend4/src/pdblend/online/router.py:551)
- `EngineClient.complete` 单独提供首字节前特定连接断开的一次重试；它不是 Proxy 的 M/PD 重新路由策略。[client.py](/home/pdblend4/src/pdblend/engine/client.py:190)
- connector 中 D 请求若被 vLLM preempt，已消费的远程 KV 不会再次等待；之后可本地重算 prefill。这是实例内部恢复，不是全局把请求迁到另一个 M 实例。[p2p_nccl_connector.py](/home/pdblend4/engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:100)

## 建图建议

图上同时画两条可视执行链：

- M：`Q_M → 绿色 prefill → 黄色 y₁ → 蓝色 y₂,y₃,…`，同一灰色 instance 内。
- PD：`Q_P → 绿色 prefill → 黄色 y₁ → client`；从 P 画灰色 KV 粗箭头直达 D；控制箭头标 `prompt+[y₁], O−1`；D 时间线为 `queue/KV load → 蓝色 y₂,y₃,…`。y₁ 与 y₂ 之间标灰色 first gap，并向调度器回馈 TTFT/TPOT/队列。
- 明确“选 P 和 D”发生在 route commit，一条逻辑 PD route 对应两个物理 instance，而不是一个特殊的“PD instance”。
- 首 token 释放的是 proxy prefill 负载计数；不要在这一点错误标注“所有 KV 都已释放”。native KV 生命周期与 request 结束/transfer 消费另行表示。
