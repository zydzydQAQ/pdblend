# 理想八卡转换方案独立审计

结论：给定文字方案的算术、实例划分和资源生命周期自洽，未发现阻断性的逻辑错误。此结论针对方案与绘图提示词；未逐像素审阅最终图片，未运行 GPU，也不代表原生后端已实现或通过实测。

- **权重账正确。** 初态八个独立 TP1 Mixed：`8×16=128 GiB`。终态 G0/G1 各 16 GiB，G2–G5 各 8 GiB，G6/G7 为零：`2×16+4×8=64 GiB`。释放量为 G2–G5 的 `4×8` 加 G6/G7 的 `2×16`，合计 64 GiB。上述仅统计 TP-shardable GPU 权重；replicated tensors、KV、workspace、communicator 与加载临时缓冲另计，不能推出总显存恰好减半或启动峰值只有 64 GiB。
- **实例和分片正确。** G2/G3 是一个 TP2 P，G4/G5 是另一个独立 TP2 D；两组分别持有完整模型的 W₀/W₁，两组之间保存同分片的副本。G0/G1 仍是两个 TP1 Mixed。终态共四个服务实例、六个活动 GPU rank、两个停止的 GPU worker；并非四个 TP2 实例或一个 TP4 组。G6/G7 的 OFF 指进程停止和权重释放，不代表物理断电。
- **请求守恒正确。** G2–G7 的 waiting 合计 `2+1+0+1+0+1=5`，running 合计 `1+2+2+1+1+0=7`。三阶段 `5+7+0=2+4+6=0+0+12=12`；中间阶段可由三个 waiting 开始运行、六个请求完成得到。关闭新准入不应禁止已有 waiting 被调度。若旧请求实际发生取消，图中 completed 应说明是正常完成数，或改为 terminal 数并分列取消；当前数例可按全部正常完成理解。
- **释放屏障正确。** G2–G7 每个旧 rank 对同一旧 epoch 确认 waiting/running/live KV/pending transfers 均为零后，才能停止 worker、释放其 HBM。live KV 为零不等于已分配 KV arena 已释放；后者由 teardown 完成。准入关闭须同时拦截旧路由的迟到请求，ACK 须对应受保护的同一转换事务，避免屏障检查后再次进入请求。
- **重建与发布正确。** 旧资源释放后，创建独立 TP2 communicators，从已验证的同模型 checkpoint 加载分片；host fan-out 或跨 GPU 复制同分片均不改变上述最终驻留账。必须检查每卡启动峰值和最终显存预算。四个新 rank 全部通过模型/分片身份、warmup、golden、取消与 KV 清理探针，释放测试 KV 后，才以原子路由 epoch 提交开放准入。G0/G1 的既有请求保持原绑定。
- **KV 与失败语义正确。** G2/r0→G4/r0、G3/r1→G5/r1 是提交后新请求在兼容 TP2 KV 布局下的 P→D 传输，不是旧 TP1 KV 迁移。失败后保持受影响实例关闭；已 teardown 的 TP1 必须重新加载并验证才能恢复，不能瞬时回滚。G0/G1 只在剩余容量内服务，溢出受背压，不能宣称无停顿或 SLO 不变。
- **实现边界已核对。** `src/pdblend/bench/tp_runtime.py:74–75` 在 `TPMode.SLOW_RESHARD` 分支直接抛出 `UnsupportedTPMode`，提示需要 GPU-qualified native transaction backend。图必须保留 IDEAL/目标设计标识，不能作为当前 PDblend 在线 TP 重分片能力或时间、能耗实测证据。

验证：CPU 断言通过，请求总数 12、初态权重 128 GiB、终态权重 64 GiB、释放量 64 GiB。
