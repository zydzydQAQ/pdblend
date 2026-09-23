# 八卡布局变换：排空、重建 TP、重新加载权重

图片：[eight-gpu-layout-transition-ideal.png](eight-gpu-layout-transition-ideal.png)。由内置 imagegen 按用户参考论文图风格生成。

这是用户授权制作的**理想执行流程图**。当前 PDblend 的 `slow_reshard_tp` 在 `src/pdblend/bench/tp_runtime.py:74` 被显式拒绝，尚无可执行且经 GPU 验证的在线重分片后端。`src/pdblend/control/reshard.py` 已有回执检查协调框架，但它不能代替真实引擎重建、逐 rank 排空和权重加载。图中没有声称发生过实际 GPU 变换实验。

## 同一组八张 GPU

| GPU | 原布局 | 新布局 | 新布局的 TP-shardable 权重 |
|---|---|---|---|
| G0 | TP1 Mixed | TP1 Mixed，继续服务 | W，16 GiB |
| G1 | TP1 Mixed | TP1 Mixed，继续服务 | W，16 GiB |
| G2 | TP1 Mixed | TP2 Prefill，rank 0 | W0，8 GiB |
| G3 | TP1 Mixed | TP2 Prefill，rank 1 | W1，8 GiB |
| G4 | TP1 Mixed | TP2 Decode，rank 0 | W0，8 GiB |
| G5 | TP1 Mixed | TP2 Decode，rank 1 | W1，8 GiB |
| G6 | TP1 Mixed | OFF，进程停止 | 0 |
| G7 | TP1 Mixed | OFF，进程停止 | 0 |

新布局按**实例**计为：2 个 TP1 Mixed、1 个 TP2 Prefill、1 个 TP2 Decode、2 张关闭卡。P、D 各有两个 rank，但不是各有两个独立实例。

W=16 GiB 是说明性假设，专指可均匀 TP 分片的张量部分，并非某个真实模型的测量结果。每层张量按 TP 规则形成 W0/W1；P、D 两个组各自持有完整模型，不是 P 只存一部分层、D 存另一部分层。

示意权重总量从 `8×16=128 GiB` 变为 `2×16+4×8=64 GiB`。这不代表总显存使用恰好减半：复制张量、KV 池、运行缓冲、通信缓冲等另外计入，每卡都必须满足实际 HBM 容量约束。Host/checkpoint 在 GPU 副本卸载后继续保留可恢复模型。

## 六个阶段

1. **旧布局**：八个 TP1 Mixed 实例。
2. **关闭准入并排空**：G2–G7 不接新请求，已绑定的排队及运行请求继续在原 GPU 完成；G0/G1 可在容量范围内承接新请求，超过容量的流量有界等待或施加背压，不承诺吞吐不变或 SLO 无损。
3. **释放旧资源**：每个旧 rank 确认 waiting=0、running=0、live KV=0、pending transfers=0，回执的事务/代次匹配后，才销毁旧工作进程并释放 HBM。已分配但无活跃块的 KV arena 要到 teardown 才释放。
4. **重建并加载**：在 G2/G3 和 G4/G5 上分别建立 TP2 workers/NCCL 通信组；从已校验 checkpoint/host cache 加载分片，也可在目标 rank 之间复制已校验的同一分片；建立空 KV 池。G6/G7 为 OFF，而不是权重驻留的 L1。
5. **验证**：核验模型及分片身份、输出探针、取消及 KV 释放；P/D 的 TP、PP、模型及代次兼容，目标负载有 profile/容量依据；测试 KV 也释放干净。仍不开放新请求准入。
6. **发布**：四个目标 rank 均通过后，协调器发布新路由代次并开放新 P/D 准入。G0/G1 上未完成请求保持原绑定。跨 TP 在线切换及原子路由发布在本图中属于理想要求。

## 请求排空算例

| GPU | waiting | running |
|---|---:|---:|
| G2 | 2 | 1 |
| G3 | 1 | 2 |
| G4 | 0 | 2 |
| G5 | 1 | 1 |
| G6 | 0 | 1 |
| G7 | 1 | 0 |
| 合计 | 5 | 7 |

本例没有取消，12 个旧请求全部正常完成：`(waiting,running,completed)=(5,7,0)→(2,4,6)→(0,0,12)`。各阶段总数均为 12。图中的取消清理回执是通用排空条件，不表示这 12 个请求实际被取消。

旧 TP1 KV 不迁入新 TP2。提交之后的新长请求才执行 TP2 P→D KV 接力：示意的对应关系为 G2/r0→G4/r0、G3/r1→G5/r1，实际连接器仍需验证逐 rank KV 布局兼容。P 返回 y1，D 使用 input⊕y1 生成剩余输出。

## 失败恢复与成本

提交前失败时，G2–G7 保持关闭准入；回滚需要重新加载并验证旧 TP1 服务，不能视为瞬时撤销。无法证明回滚完成的 GPU 应隔离并保留所有权。G0/G1 的可用服务能力仍受容量限制。

图只给出分阶段计时项，没有虚构测量数字：

`T_transition = T_drain + T_teardown + T_TP-init + T_weight-load + T_verify + T_publish`

每项代表该阶段的屏障耗时，内部可并行，不应把各 GPU 的并行耗时简单累加。实际还需测量过渡期间总能耗、相对基线的额外能耗、入口积压和请求延迟影响。当前普通 Planner 的唤醒/改频估价不能直接覆盖此类完整 TP 重建开销。

独立 subagent 的[最终机制与算术审查](review.md)确认请求数及权重计算一致。关闭准入还须拒绝迟到的旧路由派发；权重加载期间的临时副本、缓冲和通信峰值应计入容量检查。

提示词：[初稿](prompt.txt)、[局部修订](refinement-prompt.txt)。原有工程、GPU 服务与实验结果未改动。
