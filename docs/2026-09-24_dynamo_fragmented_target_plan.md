# Dynamo 分段目标接线：开发接口与未完成事项

此计划只覆盖 Dynamo 私有开发路径。`dynamo-stationary-ipc-primitive-v2` 和 `dynamo-stationary-kv-functional-v1` 的冻结源码、输入和 job 保持原样；目标构造、真实服务和完整 TP 变换尚未通过 GPU 验收。

## 当前可复用与不能复用的接口

| 现有接口 | 可复用部分 | 必须补足的部分 |
|---|---|---|
| `stationary_service.StationaryCoordinator` | 完整 ASGI 生命周期 admission fence；真实 fresh drain；全 rank epoch/UUID；失败 quarantine；源 KV 重建后 generation +1 | 当前只允许 source/target UUID 完全相同、transfer=0。没有 HTTP export、consumer ACK、target 注册、目标加载或路由提交端点。不能直接放开 pin 的限制当作完整切换。 |
| `DynamoStationaryWorkerExtension` | 原 dense 参数 pin；同物理 GPU `StationaryOwner.export`；consumer ACK；原 owner 最后退出 | collective RPC 现有 export 根据 target UUID 选 owner。`describe` 仍依赖 dense `named_parameters()`；新接收的单片段 root 无法表示为完整原 Qwen shard，需新的 `export_roots`，不能让 IPC consumer 再转发。 |
| `SourceKvWorkspace` | 原 weight storage 保持；源 KV backing 清除/实际观察/重建；禁止源执行；完整 allocator/owner metadata | 仅源原 TP group，不能替目标初始化 KV。256 MiB 是开发恢复保护量，目标峰值、初始化临时量和板级准入仍待证明。 |
| `FragmentInventory.from_ipc` | 同卡原 owner consumer 引用；incoming allocation 必须恰好一个缺片段；binding 借用期间禁止 ACK | 缺片段 transport 收据的生产者尚不存在。不能手填 `completed=true` 或用 dense copy 获得资格。 |
| `build_qwen_meta_target` / `MetaTargetBinding` | Qwen 元数据；linear/embedding/LM-head 分段消费；无 dense dummy weights；保留 TP layer forward | 新私有 loader/worker 已接入，现只有 CPU ABI/故障合同测试，实际 CUDA、KV、native 请求及输出 golden 尚未验证。`require_activation()` 继续明确拒绝。 |

## 本轮新增的 owner 图

`stationary_owner_graph.OriginalOwnerGraph(plan, released_owner_refs)` 从整文件 SHA 绑定的 released 原始 receipt 创建完整原 owner 根；`json_path` 允许选择 service wrapper 内的 rank receipt，而哈希始终绑定完整 wrapper。原参数 storage pointer/bytes/segment、物理 UUID、PID/start ticks/boot ID、generation 都保留，不把逻辑切片字节当成可释放显存。

`plan_handoff(target_gpu_uuids, target_shapes)` 以 Qwen 的全局参数坐标覆盖每个目标，优先使用同卡根。结果只产生 `original_owner_ipc_export` 或 `direct_missing_fragment_transfer` 请求，包含原 producer/root/offset/length 和完整 graph SHA，没有 GPU 执行。IPC alias 永远指回 producer；已接收的片段只有在未来 transport 提供实际 `dynamo-direct-receive-owner-root/v1` 分配观察后才能注册成新根。这种 CPU 合同核对是独立步骤，不能证明收据中的 CUDA 声明真实。

`register_export(packet_ref, plan_ref)` 可消费现有原 dense `StationaryOwner` 的真实 packet，核对原 allocator 根、stride/offset、完整 planned views 和 generation。`acknowledge_release` 要求实际 packet 对应的 consumer、同步、视图释放；`retire_consumer` 另外检查该 PID/start/boot 身份确已退出。收到 ACK 不等于进程退出。无 clean ACK 的 crash 将相关原 owner quarantine，禁止借图声明源 KV 可恢复。

`snapshot()` 保留完整根、direct-receive provenance、export/ACK 和所有退休历史；`restore_snapshot` 重新读取绑定文件并复算，不能遗漏当前路由未用到的根。producer 退出、重启、root 缺失或文件 SHA 变化立即拒绝。新 receive owner 只有全部下游消费结束、实际退出和独立绑定 compute-PID 缺席观察都具备后才能退休；历史节点仍保留。`require_owner_restore` 只证明引用关系允许下一步，仍不执行 KV restore、不授予物理释放或 serving 资格。

文件 SHA 验真、OS 进程存活、历史报告的原 allocation、consumer clean ACK 是不同证据。每次真正切换前仍需向原 worker 获取新鲜 `lease.receipt()` / allocation attestation；不能用仍存活的 PID 推导 tensor 未被替换。

## 目标实际 worker 必须接入的顺序

1. **闭合 owner 图和资源准入。** 源 admission 关闭并全 rank drain，pin 原 weights，释放源 KV。记录同一阶段板级 CUDA/NVML free 与所有进程/root。由 `stationary_memory` 审计 bootstrap、transport、serving、rollback 四阶段增量；未知 context/NCCL/activation/workspace/allocator slack 保持未知，不填 0。
2. **独立目标进程和 TP group。** 新 `DynamoFragmentTargetWorker` / 私有 runner 使用独立 rendezvous 和明确目标 rank/UUID，全过程不可路由。源 TP group 和无关实例不参与目标 collective。保留现有 native base 源字节；源 admission fence 不能误暂停 peer。
3. **目标 load hook。** 在 GPUModelRunner 调用 checkpoint/dummy loader 之前切入私有 loader：建立 meta Qwen 结构，向原 producers 请求当前 root attestation 和同卡 exports，再消费实际 imports 和缺片段接收 allocation。禁止完整 dummy 权重、retained clone/cat/contiguous/host staging。缺片段暂存/通信 buffer 若存在，要单列实测峰值和时间。
4. **真实非权重与 KV 初始化。** 明确初始化 RoPE 等非权重 CUDA buffer，构建真实 attention backend、input batch 和目标 `KVCacheConfig`，调用目标原生 `initialize_kv_cache`，不是仅调用 meta model 或源 `initialize_kv_cache_tensors`。所有 register/cache 引用和实际 storage 必须有原始观察。
5. **真实 native forward 与黄金输出。** 先只开私有资格请求，沿 native scheduler → `GPUModelRunner.execute_model` → logits/sampling → SSE 的实际路径执行。比对预先冻结同模型同 TP ordinary engine 的逐 token golden；记录 prefill、decode、完成/失败和每 rank epoch。一次 `.forward` 或 CPU 数学一致不能替代 ordinary request，也不能把 peer 功能检查说成 SLO 资格。
6. **提交或回滚。** 所有 rank 的 binding/buffer/KV/golden 通过后才可原子切路由。失败先停止目标 admission并 drain；target binding 归还 inventory borrow，consumer 同步/清空/ACK、进程真实退出，原 owner 验新鲜参数，再恢复源 KV和generation、全 rank drain/ordinary输出。部分 rank 未知、consumer crash 或原 producer 丢失则隔离，不能构造成功 rollback。原 storage root 不能在后续 target 仍借用时关闭。

固定镜像 vLLM 0.10.1.1 的真实接缝：`gpu_worker.py:154 init_device` 在模型前按 `total_memory × utilization` 检查 free；`gpu_worker.py:209 load_model` 调 runner 原 loader；`gpu_worker.py:222 determine_available_memory` 依赖 `model_memory_usage` 和同卡其他进程用量稳定假设；`gpu_worker.py:285 initialize_from_config` 调 `initialize_kv_cache`。`gpu_model_runner.py:1948 load_model`、`:3180 initialize_kv_cache`、`:1505 execute_model` 需要私有桥接。不能删除原 free-memory guard，或把 IPC logical bytes填成真实 target model allocation来骗过默认 KV sizing。后续需显式有证据的目标 KV 配置与板级预算。

## 最小功能验收顺序（尚非可执行 job）

| 阶段 | 资源/权重装载 | 必需的真实结果 |
|---|---|---|
| 当前 source-only | 2 GPU；7B source TP1 与 peer TP1 各一次 | 原 KV backing 释放/重建、原 storage 不变、peer 实际服务、全库存清理。现已有独立冻结 job；不包含 target。 |
| 下一步 same-TP 目标 | 2 GPU；GPU0 上原 source 与新 target 进程，GPU1 为 peer；target 0 次完整模型装载 | 全参数同卡原 storage imports、真实 target buffer/KV/native 请求与固定 ordinary golden；目标结束后源 rollback。先消除 loader/KV/生命周期缺口，尚不叫 TP 变换。 |
| 缺片段传输 primitive | 最少 2 GPU、0 模型 | 每参数片段 canonical 坐标、actual CUDA/NCCL completion、严格 byte coverage、真实新 allocation owner和错误/崩溃清理。单列暂存；不能 full-copy。 |
| 首次 TP1→TP2→TP1 | 最少 3 GPU（2 张变换目标 + 1 张不相交 peer）；普通 TP2 golden 的装载/复用须事先绑定 | 目标 rank0 多片段执行、rank1 实际 missing transport、独立 communicator、完整跨轮 owner 图、全部 golden与失败恢复。参考 engine 的装载单列，不藏进服务能耗。 |

只有实际 source-only/IPC 成功及上述私有代码闭合后，才生成对应新功能 probe 的不可变准备包。此文没有可排 GPU 的假 target job。完整 Dynamo 还需同模型 profiles/history/predictor/loaded transition，以及原 1800/300/5 秒机制资格；150 秒正式窗不能替代。

## same-TP 私有桥接的 CPU 实现进展

`stationary_target_bootstrap.py` 只通过原子、不可覆盖的 JSON 文件交换实际目标 worker 的 PID/start ticks/boot、UUID、epoch 和原 owner 的完整 SHA 绑定 IPC packet。`stationary_target_loader.py` 注册独立 `dynamo_stationary_same_tp_v1` loader，直接消费 original-owner imports；checkpoint download、dense load_weights 均拒绝。构造 Qwen 时仅创建 meta 参数；linear/embedding/LM-head 沿已有分段绑定路径使用保留片段，RMSNorm 直接别名原 storage。

目标的非权重初始化明确区分两类。RoPE 的注册 `cos_sin_cache` 由固定实现创建；vLLM 0.10.1.1 `Attention` 的 `_q_scale/_k_scale/_v_scale/_prob_scale` 和 `q_range/k_range/v_range` 是未注册的标量 Tensor，不能靠 `model.buffers()` 发现。私有 loader 只在未量化、auto KV、关闭 scale 计算的固定构造契约下创建这七个标量，未知未注册 Tensor 拒绝。初始空 KV 占位仍由后续原生 KV 初始化替换。没有先创建完整 dummy weights，也没有从 meta weight 或保留片段 clone/contiguous/cat 重建 dense 参数。

`SameTpTargetWorker` 保留原 `init_device` 的真实 free-memory guard、独立 TP1 communicator、原 memory profiling、`initialize_from_config` 和 `GPUModelRunner.execute_model`。实际执行计数只在原 native execute_model 成功后递增；绑定已关闭或 KV 未就绪时拒绝请求。`stationary_target_service.py` 的 target 全程禁止 public mutation，只提供绑定的 status/golden/close。golden 使用原 `serve.generate_events(private=True)`，必须逐 token 相同且真实 native 执行计数增加；最终 drain 失败保留 raw、标记需要隔离，不算通过。close 的 receipt 必须对应实际 rank、UUID、epoch、PID 和 consumer 的同步/视图释放 ACK；ACK 后仍要求 target 真正退出，源 KV 才可恢复。

新 source 私有入口复用既有全 ASGI 生命周期 fence、fresh drain 和失败 quarantine，并新增 export_to_target/target_release_ack。导出的 packet 独立嵌套，避免旧 collective RPC 追加的 rank/transaction 字段改变已签名 packet 的内容。目标 ready 的实际进程必须仍存活；rank/UUID/epoch 或 RPC 错误不能沿用成功状态。

固定镜像 66 项 CPU 测试通过（2026-09-24），涵盖真实 vLLM 层/loader 注册/Attention 构造 ABI、owner 图的正整数与地址不重叠、container→host PID 映射、目标 close ACK、请求错误与最终 drain 失败。CPU oracle 不证明实际 IPC、CUDA KV、forward、数值 golden 或 physical peak。当前仍 **not GPU-ready**：尚无同驻留 source+peer 的完整目标启动/退出/回滚 harness；owner 图尚未通过真实 released wrapper、target packet 和 ACK 串入该 harness；目标四阶段 context/NCCL/activation/KV/allocator 峰值仍未知，不能填零。没有生成目标 GPU job，也不扩展已冻结 IPC-v3/source-KV-v2 包。
