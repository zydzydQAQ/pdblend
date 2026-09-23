# PDblend Planner：数值算例与 GPU 交互细节图

最终图片：[pdblend-planner-detail.png](pdblend-planner-detail.png)。图像由内置 imagegen 生成，沿用用户参考图的论文插图样式。

## 同一个例子贯穿规划、路由和重规划

- 8 张 GPU，TP=PP=1。图中显式设置 `min_m_instances=2`，仅允许 L1 停驻，规划窗口 60 s，切换收益门槛 5%。这不是默认 PDblend 策略配置。
- 到达率 12 req/s，输入 256/512/2048/4096 各占 25%，初始输出 128。阈值 1024 将 Mixed 与 PD 各分到 6 req/s。
- SLO 为 1 s / 20 ms，安全因子 0.85，因此模型筛选线为 850 ms / 17 ms。表中的 latency proxy 不等同于客户端实测 TTFT。
- 全枚举选择 C：P2+D1+M2+L1×3。P=2100 MHz，D=M=2520 MHz。A→C 的 60 s 模型成本从 46.328 kJ 降到 42.711 kJ，后者包含 34 J 切换成本；收益 7.8083%，超过 5%。
- 阈值 4096 会让 Mixed 流量增加到 9 req/s，示例候选 E 因 TPOT 和 Mixed 尾部约束失败，说明阈值和资源数须共同规划。

## GPU 映射与请求计数

当前 `assign_roles` 在 A 的 G0–G4=M、G5–G7=L1 上应用 C 后，保留 G0/G1=M，将 G2/G3 设为 P、G4 设为 D。

- 短请求 Rs=256/128：G0 有 2 个预留序列、G1 为 0，故选择 G1；预留序列 0→1→0。
- 长请求 Rl=2048/128：G2 待首 token 的输入总量 4096，G3 为 1024，故选择 G3→G4；G3 pending 1024→3072→1024；G4 reserved 8→9→8。
- G3 的真实首 token y1 由代理先发给客户端。G4 接收原始 2048 个位置的 KV，以扩展到 2049 个 token 的 prompt 续接，剩余输出预算 127。
- 图中的 112 MiB 来自合成模型的 57344 bytes/token × 2048。它是模型算例，不是新做的 GPU 传输测量。
- `reserved sequences` 包含 P 阶段已预留的 D 请求，不是原生 decode batch 大小。`pending prompt tokens` 不是原生 waiting 队列长度。
- 图中的到达票据和引擎队列为机制示意；当前实现没有单独由 Planner 消费的中央 FIFO。Planner/Shield 使用代理记录与计数，图不声称它们直接读取所有引擎内部队列。

## 动态重规划

完成输出统计从 128 增至 256，而到达率及输入分布保持不变时，重新评估 C 得到 TPOT=22.434 ms，当前方案不可行。全枚举选择 D=P2+D2+M2+L1×2，TPOT=15.567 ms。

按上述 GPU 顺序，新增 D 是 G7。执行 L1 unpark、设频后开放准入；新的长请求可选择 G7，已派发请求保留原实例。阈值保持 1024。

这次模型成本增加：46.850→49.189 kJ，其中唤醒成本为 3.4 J。因为当前配置不可行，先恢复可行性，不能用“未节能”阻止这次扩容。

图中规划周期 10 s、保护检查周期 1 s。恢复阶段以 hold≥30 s、两个独立稳定规划窗口及保护状态允许为条件，再关闭 G7 准入、等待代理计数排空、返回 L1；不承诺固定恢复时刻，不将代理排空等同于证明原生 KV/传输资源全部释放。

## 核算、来源与提示词

三个 subagent 分别完成了数值复算、请求/控制逻辑核对、布局建议；主 agent 整合并生成图像。

- [数值审计](data/AUDIT.md)
- [完整结果与源文件哈希](data/example.json)
- [CPU 复现脚本](data/reproduce_example.py)
- [初次生成提示词](prompt.txt)
- [连接线修订提示词](connector-refinement-prompt.txt)

核心来源为 `src/pdblend/control/planner.py`、`control/forecast.py`、`control/controller.py`、`control/shield.py`、`proxy/router.py`、`proxy/server.py`、`engine/carry.py` 和 `tests/pdblend/synthetic.py`。

CPU 复现命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pdblend/.venv/bin/python /home/pdblend4/docs/pdblend-planner-detail/data/reproduce_example.py
```

所有性能/功率数字来自仓库合成模型，表中成本采用当前规划器的 `H × power + switch_energy`。独立的额外 KV 能耗未被该模型单独验证，不能把模型预测节能率理解为实测收益。相关增量能耗敏感性上界已列入数值审计。
