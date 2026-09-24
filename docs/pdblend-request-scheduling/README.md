# PDBlend 请求调度与实例执行图

根据 2026-09-24 当前工作区实现核对。三个并行审计分别覆盖 Router、原生 V1 scheduler、启动配置；本次不运行 GPU，不修改服务代码。

- [PNG](pdblend-scheduling.png)：展示版本，2880 × 2384。
- [SVG](pdblend-scheduling.svg)：可编辑矢量源，1800 × 1490。
- [PDF](pdblend-scheduling.pdf)：单页矢量导出。
- [生成脚本](build_figure.py)、[排版检查结果](validation.json)。

图参考用户提供的论文图风格：白底、衬线字体、细黑边、绿色 prefill、蓝色 decode、灰色等待/KV、黄色输出 token。没有复制参考图中的其他系统架构。图内使用英文标签，便于论文与幻灯片复用。

## 可以直接使用的说明文字

PDBlend 使用分层调度。请求到达代理后，Router 根据输入 token 数、当前角色表和已发布阈值选择 Mixed 或 PD 路径，再根据实例负载确定执行实例。Mixed 路径优先选择未完成序列较少的实例，平局时比较尚未产生首 token 的 prompt token 总量；PD 路径先比较 P 侧的 prefill 负载，再比较 D 侧的未完成序列数。实例内部由 vLLM V1 进行连续批处理：默认 waiting 队列采用 FCFS，每轮先安排 running 中的请求，再使用剩余 token、序列和 KV 预算接纳 waiting 请求。因此，系统不是全局 FCFS，也不是一个请求完整执行结束后才处理下一个。Mixed 实例可将不同请求的 prefill chunks 和 decode tokens 放入同一个 batch。PD 路径由 P 计算 prompt 并生成首 token y1，传递 prompt KV；D 接收原 prompt 加 y1，复用已传递的 KV，计算 y1 产生 y2，随后继续生成后续 token。

## 图 (a)：默认单池请求分发

定义 `L` 为输入 token 数，`K` 为要求的输出 token 数：

- `N_i = inflight_seqs`：派发后尚未完成的序列数，包括引擎 waiting 和 running。
- `U_i = inflight_prefill_tokens`：派发后尚未观察到首 token 的 prompt token 总量；不是逐 chunk 精确剩余量。
- `tau` 来自当前 plan；图中 1024 只是示例，不是固定常量。

普通 Router 的规则：有兼容 PD 且没有 M 或 `L >= tau` 时走 PD；否则优先 M；没有任何可接收路径则返回 503。没有全局 FIFO 等待队列。

| 路径 | 实例选择，按元组从左到右比较 |
|---|---|
| M | `argmin_m (N_m, U_m)` |
| PD | `argmin_(p,d) (U_p, N_d, U_d, (p_id,d_id))` |

PD 配对要求不同实例、同 model/TP/PP/pool/generation，且 PP=1。派发时 P、D 同时绑定；P 完成后不重新选择 D。

图中 **说明性快照**：M0…M3 的 `(N,U)` 分别为 `(3,2048),(1,1024),(1,0),(2,0)`；P0/P1 的 U 为 3072/1024；D0/D1 的 `(N,U)` 为 `(4,0)/(2,0)`。因此 512-token 请求选择 M2，2048-token 请求选择 P1→D1。若后者要输出 64 token，则 P 输出 y1、D 输出余下 63 token。这些数字不是运行日志。

实现依据：

- [入口与长度读取](../../src/pdblend/online/server.py)：`completions()`。token-ID prompt 取列表长度；字符串依赖调用者提供的 `prompt_tokens`，proxy 本身不进行 tokenizer 计算。
- [选择与记账](../../src/pdblend/online/router.py)：`_compatible_pd()`、`_least_pd()`、`choose()`、`dispatch()`、`token()`。
- [计划发布](../../src/pdblend/online/controller.py)：`execute()` 发布角色与 tau。

可选 ResidentRouter 不在主图中：它先在各子池生成候选，再通过 profile、KV reservation、并发等约束，可能结合目标流量份额选择池；配置合格增量能量数据时才可按能量打分。不能将此增强路径当成普通 Router 的默认算法。

## 图 (b)：实例内部 FCFS 的边界

当前标准 PDBlend 原生路径为 `bench.run` → `pdblend_runtime.serve` → `NativeScheduler` → vLLM V1 Scheduler。仓库固定 vLLM 0.10.1.1。

每轮迭代：

1. 遍历 running，处理当前应计算的 token。running 包括 decode，也包括尚未完成 prompt 的请求。
2. 有剩余 token budget、序列槽位和 KV 空间时，按 waiting 队列顺序接纳新请求。
3. 合成 batch 执行 model step，更新请求状态；完成者离开，其他请求继续推进。

默认 waiting policy 为 FCFS，但这不保证全局开始/完成顺序，也不等于所有 decode 都具有严格阶段优先级。资源不足或远端 KV 状态等会影响可调度性；KV 不足可触发抢占、回队和重算。

Launcher 默认 `max_num_batched_tokens=8192`、`max_num_seqs=256`，开启 chunked prefill，关闭 prefix caching。8192 是整个迭代共享的 token 预算。具体任务可以覆写这些值。

`native_mode='temporal'` 当前只是记录字段，`schedule()` 没有 temporal/spatial 调度分支。不能据此画成每次只能运行纯 P 或纯 D batch。图中叠放的 P/D 色块代表同一个 batch 中不同请求的 token，不代表同一 GPU 上两个独立计算流。

依据：

- [原生 scheduler](../../src/pdblend_runtime/native_v1.py)：`NativeScheduler.schedule()` 调用 `super().schedule()`。
- [原生服务注入](../../src/pdblend_runtime/serve.py)、[启动参数](../../src/pdblend/engine/launcher.py)。
- [固定上游 FCFS 默认配置](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/config/scheduler.py)。
- [固定上游 V1 scheduler](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/v1/core/sched/scheduler.py)、[请求队列](https://github.com/vllm-project/vllm/blob/v0.10.1.1/vllm/v1/core/sched/request_queue.py)。

## 图 (c)：P、D、M 的计算与时间线

P 请求实际设置 `max_tokens=1`。最后一个 prompt step 产生 y1，代理将该 token 输出给客户端。D 的 prompt 是 `X + y1`，输出预算为 `K-1`；connector 匹配原 prompt 的 KV，D 首步计算 y1 产生 y2，不重新计算整个原 prompt。KV 从引擎间传递，不经过 proxy 复制。

KV connector 在模型逐层执行时发送该层 KV，`wait_for_save()` 等待发送完成。因此图中的灰色 KV 条与 prefill 尾段重叠，不能解读为所有 KV 都在客户端收到首 token 后才开始发送；具体重叠时间没有实测。随后代理发起 D 请求，还可能有接纳、接收/装载和等待开销。

P 可以继续处理其他 prompt，同时 D 生成此前请求的后续 token。M 则在同一实例上完成一个请求的 prefill 和 decode；后来的请求可在预算允许时加入 batch。图中 P/D/M 均为稳定角色，换角色过渡期还可能保留以前绑定的请求。

时间轴仅表达先后关系与允许的批处理组合，不代表实测耗时；不同块的长度不能用于推导延迟或吞吐。图 (c) 与图 (a) 的负载快照分开：没有声称这是从该快照模拟得到的精确执行序列。客户端行的 TTFT 表示到达至首 token，TPOT 标记相邻后续 token 的间隔；首 token 到第二个 token 还包含 handoff 和 D 首步。

特殊情况：`K=1` 在选定 P 本地结束，不进行远端 KV handoff。当前多 token carry 协议对 token-ID prompt、greedy/ignore_eos 等有明确限制，详见 `carry.py`；图展示支持的普通路径。

依据：[代理两段执行](../../src/pdblend/online/server.py)、[首 token carry](../../src/pdblend/engine/carry.py)、[P2P connector](../../engine_patches/vllm-0.10.1.1/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py)。

## 重新生成

安装 `cairosvg`、`Pillow`、`PyMuPDF`，并确保系统有 Liberation Serif 字体：

```bash
python3 docs/pdblend-request-scheduling/build_figure.py
```

生成器检查文字越界和文字重叠；PDF 保留可搜索文字。出图后还需目视检查字体和箭头。本次已检查；这是文档制图验证，不是调度性能测试。
