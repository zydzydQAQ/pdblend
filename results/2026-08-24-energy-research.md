# 能耗目标下的 Selective PD 分离：调研纪要

- 日期：2026-08-24
- 范围：`papers/` 下四篇论文，以及 2025-2026 年相关后续工作
- 问题：在不超过 TTFT、TBT（或 TPOT）SLO 的前提下，以能耗为优化目标，使用 selective / hybrid PD 分离是否还有可做空间

**结论：** 还有，而且切口比 2024 年那四篇更清楚。不要再做「纯 PD 分离 + DVFS」或「纯共置 + DVFS」——这两条已经被占满。真正空着的，是在 TTFT/TBT SLO 约束下，以能耗为目标，决定哪些请求、在什么负载下该分离、哪些该共置。

---

## 1. 结论摘要

可以把现有工作放在两个正交轴上：**架构（共置 / 全分离 / hybrid）x 目标（goodput / 能耗）**。

- DistServe、Splitwise 把 PD 拆开，优化的是 goodput、吞吐、成本或**峰值装机功率**，不是运行能耗（Wh）。
- DynamoLLM、mu-Serve 以功耗/能耗为目标，但架构仍是共置（或根本不是 LLM PD）。
- 2025 年后，TaiChi 在 **goodput** 上把共置与分离接成 hybrid；VoltanaLLM / DualScale 在 **已经全分离** 的集群上做相位感知 DVFS。
- 测量工作 *Revisiting Disaggregated LLM Serving* 表明：公平 GPU 数下，全分离即使能独立调频，能耗 Pareto 仍可能差于共置。

因此，交叉点仍空着：

> **能耗最优的 selective / hybrid PD**：只在「拆开省下的相位级 DVFS 能量 > 权重副本 + KV 传输 + 池闲置能量」时才分离。

energy-optimal 的混合配比一般不等于 TaiChi 的 goodput-optimal 配比。

---

## 2. 四篇论文对照

| 论文 | 会议 | 架构 | 优化目标 | SLO | 核心旋钮 |
|---|---|---|---|---|---|
| DistServe | OSDI'24 | 全 PD 分离 | per-GPU **goodput** | TTFT + TPOT | P/D 实例数、并行策略、带宽感知放置 |
| Splitwise | ISCA'24 | 全 PD 分离 + mixed 溢出池 | 吞吐 / 成本 / **峰值功耗预算** | TTFT + TBT + E2E | 异构机型、decode 功率封顶、P:D 配比 |
| DynamoLLM | HPCA'25 | **共置** vLLM，按请求类型分池 | **能耗 (Wh)** | TTFT + TBT | 实例数、TP、GPU 频率 |
| mu-Serve | ATC'24 | 多模型算子切分，**不是 LLM PD** | **功耗** | 端到端延迟 | 算子敏感度放置 + DVFS |

源文件：

- `papers/osdi24-zhong-yinmin.pdf` - DistServe
- `papers/isca24-Splitwise.pdf` - Splitwise
- `papers/hpca25-DynamoLLM.pdf` - DynamoLLM
- `papers/atc24-microserve.pdf` - mu-Serve

### 2.1 DistServe：拆开 PD 耦合，目标是 goodput

抓住两件事：

1. **Prefill-decode 干扰**：一个 prefill 进入 decode batch，两边延迟都会被拉高。
2. **资源 / 并行耦合**：共置时没法给 prefill 用高 intra-op（压 TTFT）、给 decode 用高 batch / replication（压 TPOT）。

解法是物理拆开 P/D，各自搜索并行方案，最大化「满足双 SLO 的每 GPU 请求率」。论文自己也承认：

- SLO 松、追求吞吐时，chunked-prefill（Sarathi）可能更好；
- GPU 很少、资源受限时，分离甚至做不动；
- 长上下文下 KV 传输绝对量变大，但相对 prefill 计算（近似平方）仍可接受。

几乎不谈能耗。多一份权重副本、KV 传输、P/D 池闲置，对 goodput 可能划算，对 Wh 不一定。

### 2.2 Splitwise：最接近「能耗」，优化的却是装机功率

把两阶段差异刻画得很清楚：prefill 吃算力和功耗，decode 吃带宽、对 power cap 几乎不敏感。于是：

- decode 可以放到 A100，或把 H100 封到约 70% TDP；
- 集群按吞吐 / 成本 / **峰值功率** 配 P:D；
- mixed pool 只是队列溢出时的回退，不是按请求选择「分离还是共置」。

这和「在不超过 TTFT/TBT 的前提下最小化积分能耗」不是同一个问题。峰值功率决定能装多少卡；能耗决定电费和碳。论文明确写了：优化的是 provisioned power，不是 dynamic energy。

### 2.3 DynamoLLM：能耗 + 双 SLO，架构仍是共置

四篇里和本问题最贴的一篇：**在 TTFT/TBT SLO 下最小化能量**。主要发现：

- 短请求、低负载可以低频 + 小 TP；
- 长请求、紧 SLO 必须高频 + 大 TP；
- 最低功耗点不等于最低能量点（跑太慢，积分能量反而高，U 形曲线）；
- 同一 GPU 上混跑不同长度，频率会被最紧的请求绑死。

旋钮是 **实例数 / TP / 频率**，池按 SS/MM/LL 划分，但是 **共置实例**。引用了 Splitwise，却 **没有把 PD 分离当成能耗旋钮**。共置带来一个解不掉的约束：一块 GPU 只能有一个频率，prefill 要高频时 decode 也被迫高频。

### 2.4 mu-Serve：功耗感知 serving 先驱，但不是现代 LLM PD

证明 GPU 频率-延迟是饱和曲线、功耗近似线性，所以 SLO 有余量时降频能省电。贡献包括：算子敏感度放置、生成式长度预测 + SJF、MIAD 调频。局限：

- SLO 是 **p99 端到端延迟**，不是 TTFT/TBT；
- 模型偏小（gpt2-large 一级），没有 continuous batching 下的 P/D 干扰模型；
- 频率粒度为整卡，多模型共卡时被最敏感算子卡住——和 DynamoLLM「被最紧请求卡住」是同一类耦合，只是发生在算子层而不是阶段层。

---

## 3. 四篇合在一起的缺口

它们共同指向一个没被联合优化的耦合：

> DistServe 拆的是 **并行 / 资源耦合**；DynamoLLM 调的是 **频率**，但没拆架构；Splitwise 拆了架构，但调的是 **装机功率**；mu-Serve 调频率，但不区分 P/D。

共置时，频率被更敏感的那一阶段（通常是 prefill / TTFT）决定，decode 的能量被浪费。全分离可以给 P 高频、D 低频，但要付三笔账：

1. **权重双份** -> 静态 / 空闲功耗翻倍倾向
2. **KV 传输** -> 延迟之外还有 CPU / DRAM / NIC 能量
3. **P/D 池不平衡** -> 空闲 GPU 仍可能吃 30-50% TDP

对 **能耗** 来说，全分离不是默认最优，共置也不是。需要的是 **selective / hybrid PD**：只在拆开的相位级 DVFS 收益盖过上述能量税时才分离。

---

## 4. 2025-2026 已占掉的点（必须避开）

后续工作把「全分离 + 能耗」和「hybrid + goodput」都做了，**但没有把两者乘在一起**。

| 工作 | 年份 / 出处 | 做了什么 | 没做的 |
|---|---|---|---|
| **TaiChi** | arXiv:2508.01989 | Hybrid PD：P-heavy / D-heavy 实例，请求级 latency shifting，**最大化 goodput** | 能耗；energy-optimal 配比不等于 goodput-optimal |
| **VoltanaLLM** | arXiv:2509.04827 | **已分离** 集群上 iteration 级调频 + decode 路由，相对静态最高频约省 36% | 不决定 colocate vs disagg |
| **DualScale** | arXiv:2602.18755 | 全分离上粗粒度放置 + 细粒度 DVFS；相对 DistServe，prefill / decode 能量约降 39% / 48% | 同样假设架构已是 PD |
| **Revisiting Disagg** | arXiv:2601.08833 | 公平 GPU 数下的 Pareto：**独立调频也打不赢共置能耗** | 只是测量，没有 selective 系统 |
| **GreenLLM** | arXiv:2412.20322 | 异构旧卡跑 decode，优化 **碳 / embodied carbon** | 不是同构卡上的能量最优混合 |
| **Kairos** | arXiv:2607.02043 | 把 prefill **偏转到 decode 节点** 降 TTFT 尾部 | 目标是延迟不是能量 |
| **EcoServe** | OSDI'26 | 弱互连上 **时间维部分分离**（不传 KV）提 goodput | 不是能耗目标 |

两篇尤其关键：

- **TaiChi** 已经证明：紧 TTFT + 松 TPOT -> 共置更好；紧 TPOT + 松 TTFT -> 分离更好；**双 SLO 均衡时两者都不是最优**，需要 hybrid。它用 SLO slack 去 **挪延迟换 goodput**。若做 hybrid，必须换目标：用同一套 slack 去降频 / 少开机，换能耗。
- **Revisiting** 给出直接动机：在 2xA100、公平 GPU 数下，全分离即使能独立 DVFS，**能耗 Pareto 仍差于共置**。这不是说分离永远更费电，而是说「永远全分离 + 调频」作为能耗方案不成立。这正是 selective 的实验由头。

TaiChi 的 hybrid 骨架可概括为：

- 实例类型：P-heavy（快 prefill、高干扰 decode）与 D-heavy（低干扰 decode、慢 prefill）
- 三个滑块：两类实例比例、各自 chunk size
- 调度：flowing decode（把还能扛干扰的 decode 迁到 P-heavy）+ length-aware prefill（短 prefill 可放到 D-heavy）
- 目标：最大化 90% SLO attainment 下的 goodput

能耗版需要重写目标函数和启发式，而不是复用 latency shifting。

---

## 5. 仍可做的点

### 5.1 点 A（最值得）：能耗最优的 selective PD，约束是 TTFT + TBT

问题形式：

```text
min  E(架构, 频率, 路由)
s.t. P(TTFT <= S_P) >= 90%
     P(TBT  <= S_D) >= 90%
```

决策可以有三层，不必一次做完：

1. **集群层（分钟级）**：P-only / D-only / Mixed 三类实例的数量 + 各自基线频率 + TP。低负载、松 SLO -> 全 Mixed + 低频（DynamoLLM 风格）；高负载、紧 TBT -> 拉出 D-only 并降频；紧 TTFT -> 拉出 P-only 高频。
2. **请求层**：短 prefill、KV 小、TBT slack 大 -> 共置（免传输能量）；长 prefill、会打爆 decode 迭代 -> 分离，好让 D 卡停在能量甜区。
3. **迭代层**：Mixed 卡上用 chunked prefill 控制干扰；P/D 卡上可沿用 VoltanaLLM 式调频。增量不在调频算法本身，而在 **调频作用在三类实例上**。

和 TaiChi 的本质差别：TaiChi 把「已经满足 SLO 的请求」变慢，把 GPU 时间让给快违规的请求（max goodput）。能耗版是把「已经满足 SLO 的余量」变成 **更低频率 / 更少在线卡 / 更少 KV 传输**（min energy，goodput 只要达标）。

**为什么 energy-optimal blend 不等于 goodput-optimal blend（故事核心）：**

- Goodput 几乎不计静态功耗，倾向于多拆实例、多传 KV。
- 能量要付权重副本、空闲卡、传输路径（Revisiting 的 CPU/DRAM 分解）。
- 延迟对频率单调，能量是 U 形：D-only 再降频可能更省，P-only 降频可能因时间变长而更费。
- 因此 TaiChi 搜出来的 R_PD 和 chunk size，直接拿来跑能耗会偏「拆得太多」。

### 5.2 点 B：把「频率耦合」写成 PD 该不该拆的判据

相对好讲的 mechanistic contribution。

共置 GPU：f = max(f_TTFT, f_TBT)。交互场景里往往 f_TTFT 远大于 f_TBT（人眼阅读约 50-100 ms/token 就够），decode 被绑在 prefill 的频率上。这就是 DynamoLLM 在共置里省不干净的那部分，也是 Splitwise「decode 可 power cap」在 **动态能量** 上的对应物。

干净实验（A100 即可）：

- 共置 + 统一频率（DynamoLLM）
- 全分离 + 相位独立频率（VoltanaLLM / DualScale）
- 按（输入长度, 负载, SLO 松紧）选择分离或共置

预期：只在 **TTFT 紧、TBT 松、prefill 足够长、负载中高** 的区域，分离才省能；短请求 / 低负载 / 双 SLO 都松时，共置反而更省。这张「该不该拆」的相图，目前没有人画过。

### 5.3 点 C：KV 传输能量作为一等公民（辅助点）

DistServe 把传输当延迟开销；Revisiting 已经测到更深存储层级会抬高非 GPU 能量。selective 的一个直接好处是 **短请求根本不传**。适合做消融，但主贡献不要落在「又一种 KV 传输路径」上。

### 5.4 点 D：负载驱动的架构切换

DynamoLLM 在负载变化时改 **池大小和频率**，架构始终共置。Splitwise mixed pool 是 **吞吐溢出** 才把机器改成 mixed。能耗版应该是：低谷关掉 P/D 专用池、并回 Mixed 并降频；高峰且 TBT 吃紧时再拆出 D-only。切换开销（权重加载、KV 迁移）必须进能量模型，否则和 DynamoLLM 的 scale-in/out 分不开。

---

## 6. 不建议再做的方向

- **全 PD + DVFS**：VoltanaLLM、DualScale 已做。
- **共置 + 请求类型池 + DVFS**：就是 DynamoLLM。
- **算子级频率 / 多模型 multiplexing**：mu-Serve。
- **异构卡 / 旧卡跑 decode 降碳**：Splitwise + GreenLLM。
- **Hybrid PD 冲 goodput**：TaiChi。
- **Prefill 偏转到 D 节点冲尾部 TTFT**：Kairos。
- **弱网时间维分离冲 goodput**：EcoServe。

实验里必须同时对比：

- DynamoLLM（共置能耗）
- VoltanaLLM / DualScale（分离能耗）
- TaiChi（hybrid goodput；能耗版要改成报 Wh，而不是只报 goodput）

否则审稿人会认为只是换了目标函数复现 TaiChi。

---

## 7. 建议的问题陈述与实验注意

建议 claim：

> 在 TTFT 与 TBT SLO 约束下最小化 LLM serving 能耗时，共置会把 GPU 频率绑在更敏感的阶段上，全分离则引入权重副本、KV 传输和池闲置的能量税。两者的能量最优区随负载、长度分布和 SLO 松紧而变。需要按请求 / 按负载选择共置或分离的 serving 系统，联合实例组合与相位感知调频，在达标 SLO 的前提下降低能耗。

最小可区分贡献三块：

1. **相图**：何时分离更省、何时共置更省（相对 Revisiting 的「分离从不省」给出条件和规模）。
2. **联合控制器**：三类实例配比 x 每类频率，目标是能量不是 goodput。
3. **请求级路由**：用预估的「干扰能量 vs 传输 + 空闲能量」，而不是 TaiChi 的 latency shift。

实验注意：

- **公平 GPU 数**：分离至少 2 卡时，共置 baseline 也要用同样卡数（Revisiting 的批评）。
- **能量用积分功率得到 Wh**，不要只报瞬时功率或 TDP；峰值功率是 Splitwise 的目标，不是这里的目标。
- A100-40GB 足够做 7B/13B 的相图和路由；70B 级可以靠 profiling + 仿真补（DistServe / Splitwise 都是这条路）。
- SLO 必须同时约束 TTFT 与 TBT/TPOT，达标率沿用 90% 较易与 DistServe / TaiChi / DynamoLLM 对齐。

---

## 8. 一句话

四篇论文把「分离换 goodput」和「共置换能耗」做成了两条平行线；TaiChi 把线在 goodput 上接上了，VoltanaLLM / DualScale 把线在全分离能耗上接上了。还空着的交叉点，就是 **能耗目标下的 selective PD**。这一点仍然成立，而且比 2024 年更好讲——因为正反两面的极端都已经有人做完了。
