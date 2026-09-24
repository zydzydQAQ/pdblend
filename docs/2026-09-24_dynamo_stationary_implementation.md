# Dynamo 原地权重实现边界

作者 [IV-C](https://arxiv.org/html/2408.00741v1#S4.SS3) 要求在目标 GPU 布置中尽量保留已在该 GPU 上的权重，并直接传输缺失部分。TP 变化后的通信组同步和实际停供也属于迁移成本。存储视图原语不等于可运行的目标引擎。

## 已实现、尚未接生产的部分

`src/pdblend_baselines/dynamollm/stationary_tensors.py` 把匹配后的物理 UUID 布置转成实际 Qwen 参数形状下的 BF16 切片。QKV、gate/up、行切分、词表和复制的 norm 分开处理；复制参数可选择同卡已有 rank，不强制从 rank 0 重传。每个目标参数必须无重复、无遗漏地覆盖。

`RetainedStorageLease` 持有原 worker 的 CUDA 参数视图，不克隆数据、不把权重放入主机内存。它核对 generation、UUID、参数形状、storage pointer 和 Tensor version，并在释放前保持存储引用。融合参数的多个视图不会被偷偷拼接成新张量。`held_view_bytes` 只描述这个原语持有的范围；规划量用 `planned_*`，不冒充实测传输量。

14 项 CPU 测试通过独立的 GQA 模型构造规则重建 TP1↔2、TP2↔4 的目标权重，并验证引用别名、替换/修改检测和释放行为。未运行 CUDA。`original_weight_retention_implemented`、`hardware_qualified`、`target_engine_activated` 均保持 false。

## 后续实际 serving 路径

1. 新增 Dynamo 自己的 worker/HTTP 扩展，不改已冻结的 `pdblend_runtime/serve.py`。源实例证明新鲜 drain 后，按事务 pin 参数存储；异常后不得把尚未释放的导出引用记成完整 drain。
2. 新生命周期需要证明同卡重叠的实际内存预算。现有 `SubprocessLifecycle.start` 拒绝物理 GPU 重叠；vLLM 0.10.1.1 也在初始 free-memory 检查拒绝给第二实例重复分配默认 0.85 容量。不能删除 guard 或借用另一个系统的 KV/profile。必须给出旧 KV 释放、保留权重、目标权重和目标 KV 的真实清单。
3. 目标在初始化中绑定同卡权重片段，仅把缺失片段纳入跨 GPU 传输。融合布局需要设备内重排时，单列本地拷贝量和时间。CUDA IPC 生产者必须活到消费者完成交接；不得在目标仍持有 IPC 引用时关闭旧进程。
4. 目标 TP 组完成初始化后，对同 TP 普通引擎黄金输出逐 token 比对；记录源最后服务、目标准备、参数传输、目标首次可服务、提交、旧实例回收和所有 rank drain。首次选择最小同模型 TP1→2→1，再扩展到目标模型与其合法方向。
5. GPU primitive 通过仍不能开放 150 秒比较。原 1800/300/5 秒周期下的实际三级动作、工作负载覆盖、带负载成本和完整八卡逐窗能耗必须分别通过。

当前 relay 仍创建新 dummy engine 并传完整目标权重。新的存储原语没有被接入它，因此现有冻结执行结果、当前队列 profile 和正式资格均不受影响。

## 2026-09-24 第二阶段：同卡 IPC 与所有者生命周期

新增模块均属于 `pdblend_baselines.dynamollm`；没有修改公共 `pdblend_runtime/engine`，也没有把新模块接入已有 relay、正式 comparison 或冻结 source。

- `stationary_ipc.py` 使用固定 PyTorch 2.7 的 CUDA reducer/export 与 `rebuild_cuda_tensor`。导出的是原参数 `detach`/`narrow` 的 allocation handle、storage offset、shape 和 stride，没有权重载荷、clone、cat 或主机中转。消费端必须是同一实际 UUID，匹配 plan、generation、进程 start ticks/boot ID，以及原参数 storage pointer/offset/stride；片段的正确性不能靠 shape 相同替代。
- owner 为每个 consumer/rank 只导出一次；部分导出失败仍保持原引用。consumer 必须先同步 CUDA 并清空视图，然后退出；仅收到 ACK 或超时不足以释放 owner。崩溃消费者可能留下 producer 侧引用计数，故 owner 明确进入 quarantine 并要求实际退出，不能声称 allocation 已回收。实际 NVML compute-empty 在外层执行。这遵循 [PyTorch 2.7 CUDA sharing 生命周期](https://docs.pytorch.org/docs/2.7/multiprocessing.html#sharing-cuda-tensors)。收到的 IPC tensor 不再转发给第三进程，后续消费者需直接由原 owner 导出。
- `stationary_worker.py` 提供 `DynamoStationaryWorkerExtension.dynamo_stationary_operation`。`pin` 消费真实参数 metadata 与新鲜 native drain ACK（含 scheduler、KV 与所有 rank），`export` 按物理 UUID 选 rank；`consumer_release_ack/release` 显式指定 source rank，适合 collective RPC 广播。持有或 quarantine 的事务计入 worker drain 的 active sessions，阻止旧权重修改/拓扑操作。它仍是未接 HTTP/生命周期控制的 worker 原语，调用者必须在 pin 前关闭 native admission，并在 owner 释放前保持关闭；尚未声称有可靠服务切换事务。
- `stationary_admission.py` 区分“一个原始 storage view 可表达”与“需要分段执行”。独立 NumPy oracle 验证 TP1↔2、TP2↔4 的 QKV、gate-up、行切分和词表线性算术；它不是 CUDA layer，也不保证不同浮点归约顺序逐 token 一致。显存准入使用实际当前 free bytes，包含缺失权重、目标 KV、workspace、CUDA context、TP communicator 和 guard，不提前抵扣仍存在的源 KV。

### 已核查的实际引擎阻碍

在固定镜像 `sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc` 以 CPU-only Docker 读取 vLLM 0.10.1.1 实现：

1. `model_executor/layers/linear.py` 的 `UnquantizedLinearMethod.create_weights/apply` 创建并消费单个 dense weight；`vocab_parallel_embedding.py` 同样消费一个 dense 参数。TP1→TP2 的 QKV 和 gate/up 各片段具有不同 source-offset 与 target-offset 差，不能用单个 affine view 表达。直接 `cat`、`.contiguous()` 或写入 dummy engine 会复制保留片段，不符合此任务的原存储要求。必须新增 Dynamo 专属分段 linear/embedding/LM-head 执行和参数加载路径，且不先分配一套完整 dummy weights。
2. `v1/worker/gpu_worker.py:init_device` 在构建模型前检查 `total_memory * gpu_memory_utilization`。现有 Dynamo `.85` 与源引擎权重/KV 并存时没有预算保证。现有 sleep level 1 将 weights 主机卸载，level 2 丢弃，并非“仅释放 KV 且继续保存原 weight allocation”。源实例需要自己的 KV teardown/rebuild hook 与真实 CUDA 分配清单；不能把 native `free_blocks == total_blocks` 当作 GPU KV backing 已释放。
3. PyTorch CUDA IPC 的单位是底层 cudaMalloc allocation，多个参数/空闲 allocator block 可能共享它。逻辑保留字节小不代表整块 allocation 可释放。新 codec 从实际 `torch.cuda.memory_snapshot()` 找到 allocation 并单列 backing bytes；查不到就拒绝预算声明。源 allocator 的非保留区域在 consumer 使用期间也不能随意释放/重用为另一参数。当前 lease 持有整个源参数库存，保护正确性但不实现显存节约。
4. 现有 `SubprocessLifecycle.start` 拒绝同物理 GPU 重叠，weight transport NCCL 也假设 unique GPU ranks。需要独立 owner/target 子生命周期：源停止 admission 并 drain；owner 只保留 storage，目标保持不可路由；新的目标 TP group 初始化且只传缺失片段；同模型同 TP ordinary engine 的逐 token golden 通过后才提交；失败时先隔离目标，再恢复源 KV/代际/时钟并 drain，恢复失败则停止该事务。其它实例不能进入这些 group、切换时不能暂停它们的 native 服务。当前没有实现这些原生 hooks，故 `release_kv`、`bind_target`、`activate` 全部明确拒绝。

### 最小 GPU 验收入口

`python -m pdblend_baselines.dynamollm.stationary_probe --config <frozen config> --out <attempt dir>`，配套冻结脚本为 `scripts/2026-09-24_prepare_dynamo_stationary_primitive.py`。只需 1 张空闲租约 GPU、两个 CUDA 进程与一个 CPU coordinator，不加载任何模型、不修改时钟，不使用 1890 秒窗口。固定执行两例：正常 consumer 同步/释放/退出，以及持有 IPC view 时异常退出。每例独立 owner，consumer 崩溃后 owner 退出，再由 NVML 验证该租约 GPU 无 compute PID，避免把泄漏交给下一例。

实际 tiny BF16 Qwen-shaped tensors 验证 TP1 原存储中属于 TP2 rank 0 的所有片段。v1 错误要求 NVML 能看到另一个真实 UUID；实际队列只暴露租赁的一张 GPU，因此在 CUDA 前失败。新 `dynamo-stationary-ipc-primitive-v2` 保留该失败记录，改为严格验证一张可见租赁卡，未执行的 rank 1 使用明确标记的合成未知 UUID，不声称真实 peer 存在。consumer import 前额外验证错误 UUID descriptor 被元数据门拒绝，不对未知 UUID 执行 CUDA。目标 TP2 engine 从未启动。校验用的 synthetic golden tensor 独立构造，明确不属于保留权重的传输。原始 packet、source/view identity、逐片段 equality、consumer exitcode、owner quarantine、进程组清理与 GPU 前后库存分别落盘。异常仅清理本任务创建的进程组，绝不杀死预先存在的 GPU PID。

该任务成功只允许 `same_gpu_hardware_primitive_qualified=true`；`full_tp_switch_qualified`、`original_dynamo_mechanism_qualified`、`target_engine_activated`、`energy_comparable` 和 `formal_eligible` 始终 false。下一步先实现上面的 KV/分段执行/独立 TP-group hooks，再排 7B TP1→2→1 功能验收，同时让另一个独立实例持续服务并验证输出/间隙。只有这些证据及 profiles/history/predictor/loaded transition 齐备，才值得运行保留原 1800/300/5 秒周期的长资格任务。

## 2026-09-24 第三阶段：分段计算与目标参数绑定

`stationary_layers.py` 实现真实 Torch 的分段 linear、embedding 和 LM-head 计算。QKV/gate-up/词表按目标输出行写入激活，o/down 按输入列求各片段贡献并累加；只有激活结果分配，不拼接权重。复制的 RMSNorm 参数直接引用原 view。缺失片段必须是单片段大小的独立 CUDA BF16 allocation，并附完整 direct-GPU 传输收据；本模块不实现或授予该传输的硬件资格。CPU oracle 通过单独命名入口，绝不能混作 CUDA inventory。

`BorrowableStationaryConsumer` 在新模块中继承既有 IPC consumer；inventory 与模型持有借用时，`close()` 拒绝声明 views 已释放。模型关闭先恢复 meta 占位参数、禁用分段方法并移除 norm hook，inventory 再清空片段，最后才能同步/关闭 consumer。旧 IPC 模块及已冻结 primitive job 完全不变。

`stationary_binding.py` 的 `build_meta_target` 只构造 meta 参数元数据，不分配 CPU/CUDA 完整 dummy weights。`MetaTargetBinding` 先检查全部参数、dtype、分片方向、bias 和别名，再逐层替换该模型自己的 `quant_method`；失败恢复所有原 meta 参数与 borrow。QKV bias 在分段输出中恰加一次，Row/Column/Embedding 的原 vLLM forward 与 TP 通信保持原位。通用 `.to/.cuda/_apply/load_state_dict` 被拒绝，以免后续无意复制保留 storage。显式 tied weights、量化、LoRA、deferred bias 及未知层暂不支持。

`build_qwen_meta_target(vllm_config, inventory)` 是固定 vLLM 0.10.1.1 的 Qwen2 专用构建入口：核对 BF16、无量化/LoRA、PP1/DP1、真实目标 TP rank/size 与模型几何后直接调用 Qwen 模型构造器，不经过 checkpoint/dummy loader。它返回仍不可激活的 binding，尚未接入线上 worker 生命周期。meta RoPE 等非权重 buffer 仍须独立初始化；不能把未运行 buffer/KV/TP/routing 代码称为 serving 已实现。

29 项新增 CPU 测试覆盖 TP1↔2、TP2↔4 的数值 oracle、词表边界、storage alias、非法/重复/缺失片段、参数版本变化、模型移动/加载拒绝和失败回滚。连同已有原语共 84 项通过。固定镜像另运行实际 QKV/MergedColumn/Row/Embedding/LM-head 构造器、forward 与 logits 入口的 CPU 测试通过；该测试把 TP rank/size 和 collective 显式设为单 rank CPU oracle，没有建立 TP group，也没有初始化 CUDA。

BF16 分段行 GEMM 的归约顺序与 dense GEMM 不同；CPU 数学一致不保证生产逐 token 一致。仍需 GPU kernel/分配轨迹和同模型 ordinary goldens、缺片段真实 transport、KV backing teardown/rebuild、目标 communicator 隔离、非权重 buffer 初始化及路由提交/回滚证据。所有新收据保留 `hardware_qualified=false`、`target_engine_activated=false`、`formal_eligible=false`，正式 Dynamo 比较门禁没有开放。

目标 dense `layer.weight` 在绑定后为 `None`，完整片段库存由 `FragmentInventory` 持有。因此现有 native `named_parameters()` metadata/weight export 不能直接描述这个目标；下一阶段必须提供 Dynamo 专属 inventory-aware metadata 与再次切换的 donor 图。后续消费者仍要直接从原 CUDA IPC owner 导入，不能把目标 import 的 IPC tensor 转发第三进程。该接口缺口也是 TP1→2→1 功能验收前的硬阻碍。

## 2026-09-24 第四阶段：源 KV backing 与独立服务事务

`stationary_kv.py`、`stationary_kv_worker.py` 与 `stationary_service.py` 提供源实例单步 KV 生命周期。公共 native server、引擎、profile 和旧 source 均不改。私有 service 复用真实 native drain，先等待所有已接纳 HTTP 请求的完整 ASGI 生命周期结束，再关闭 admission；每步核对新鲜 scheduler/all-rank generation 和实际 UUID。caller 不能提供伪造 drain。任一 rank 状态不明会保持 HTTP 写操作 fence 和 native admission 关闭，要求隔离任务拥有的进程，不能凭空声明 rollback 成功。

源只支持固定镜像的 eager Qwen2、PP1/DP1、无 sleep allocator、无 KV connector。保留全部原模型 weight storage，仅清除 `runner.kv_caches` 和 attention context 的 KV backing 引用；Python weakrefs 与源 allocator active/pending blocks 必须同时证明旧 backing 已释放。每个原参数的 pointer、storage bytes、stride、offset、version 保持不变。KV restore 调用原 vLLM `initialize_kv_cache_tensors`，校验原配置/shape/stride/dtype，不重新加载模型。所有 owner 关闭后才安装 generation +1，确认所有 rank ACK，恢复 ordinary serving。源 TP group 从未改变。

CUDA driver free 和 NVML board memory 在每个阶段实际记录，源 allocator block/segment 与完整 owner PID/start ticks/boot ID 一起落原始 receipt。释放的 allocator inactive block 不自动计入 driver free，逻辑 retained bytes 不冒充完整 owner 的物理占用。当前恢复要求原 KV 配置字节加 256 MiB 的开发保护量；它不是 allocator rounding/初始化 peak 的资格证明。新增 `stationary_memory.py` 的独立预算审计仍拒绝把未知 target context/NCCL/activation/workspace 当作 0；本 probe 不创建 target，不能据它宣称双 TP 实例峰值可承受。

冻结准备包 `results/2026-09-24/dynamo-stationary-kv-functional-v1` 为 2 GPU、2×7B TP1/native32，每个源/peer 各装载一次，1800 秒仅是任务超时上限。执行固定 seed 9701 的 128→16 token ordinary goldens，源 KV 真实释放时验证 public mutation HTTP 409，并让不相交 GPU 上的 peer 完成两个真实请求且 epoch 不变；源恢复后逐 token 比对，最终两个实例 drain、停止拥有的进程组并逐 UUID 验证 compute-empty。原始请求、各 phase、capability、identity binding、实际启动/ready 计数、失败和清理分别保留。`probe_on_resident` 也接受未来已有私有 Fleet，可避免这两次装载。peer 检查不是 SLO 资格或迁移成本拟合。

固定镜像 CPU 预检与 58 项测试通过，包含实际 vLLM KV 分配/reshape/bind ABI、实际 NativeWorker 动态 extension MRO、完整 service fence/失败 quarantine、双实例启动/清理合同。GPU 仍未执行；任务依赖新 IPC primitive 成功，且只由 root 调度。结果无论通过与否，仍不授予 consumer 绑定、target TP 初始化、missing-fragment transport、目标峰值显存、完整 TP 切换或原 1800/300/5 秒机制资格。150 秒正式窗不能替代这些机制验收。
