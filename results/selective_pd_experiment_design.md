# Selective PD 本机实验设计

## 1. 目标

基于 `selective_pd_energy_research.md` 验证两个 motivation：

1. prefill 与 decode 对 GPU 频率的敏感度不同，共置会把整卡频率绑在更严格的阶段上。
2. 全分离还要支付 KV 传输、第二份权重和空闲池能量；在 TTFT、TBT 双 SLO 下，最低 GPU Joule 的架构随工作负载和互连能力改变。

优化问题：

```text
min  GPU_Energy(architecture, placement, frequency)
s.t. P(TTFT <= S_TTFT) >= 90%
     P(TBT  <= S_TBT)  >= 90%
```

## 2. 平台

- 镜像：`pdblend:vllm-cu128`
- vLLM：0.11.2
- PyTorch：2.9.0+cu128
- CUDA runtime：12.8
- 模型：Qwen2.5-7B-Instruct，FP16
- GPU0：NVIDIA A100-PCIE-40GB，150–250 W，最高 SM 1410 MHz
- GPU1：Tesla V100-PCIE-32GB，100–250 W，最高 SM 1380 MHz
- CPU：2× Xeon Gold 6138，80 逻辑 CPU
- 内存：440 GiB
- GPU 拓扑：同一 NUMA 节点内的 PCIe `NODE`，无 NVLink

关键限制：`nvidia-smi topo -p2p` 的双向读写均为 `NS`，CUDA peer access 双向为 `False`。P2P NCCL 在首个 KV 张量传输时报 `unhandled cuda error`。获批准后，真实 PD 改用 NIXL 1.4.0，并强制 UCX `self,sm,tcp,cuda_copy` 走 host-staged/TCP 路径。

## 3. 公平性

- 每种架构都占用相同的 A100+V100 两张物理卡。
- Colocated 为两台独立 P+D server，按 2:1 容量权重分流。
- PD 同时测试 `A100=P,V100=D` 和 `V100=P,A100=D`。
- 全部关闭 vLLM prefix caching，避免重复 trace 在第二次运行中命中 KV。
- 使用相同 prompt token、到达时间、输出长度和随机 seed 做 paired comparison。
- 服务启动和模型加载不计入 steady-state 能量；两卡空闲功耗计入请求窗口。

异构双卡不能同时常驻 P-only、D-only、Mixed 三池。因此本机实验验证 workload epoch 级架构选择和 oracle selective 上界，不声称实现请求级在线 hybrid。

## 4. 指标

- TTFT：客户端请求开始到首个生成 token。
- TBT/ITL：相邻生成 token 的到达间隔。
- TPOT：每个请求所有 TBT 的均值。
- SLO attainment：请求级 TTFT 和 token 级 TBT 分别统计，二者均至少 90% 才可行。
- Gross GPU energy：20 Hz NVML 功率在首个请求调度到最后 token 间做梯形积分。
- 派生指标：J/request、J/output-token、每卡均值/峰值功率、利用率和负载时 SM 频率。

RAPL 在本机不可见，报告不包含 CPU、DRAM 和 PCIe 交换芯片能量。对 NIXL host-staged PD 而言，这会低估实际总能量。

## 5. 相位频率实验

在每张卡分别执行：

- prefill-heavy：2048 input、1 output、并发 1。
- decode-heavy：32 input、256 output、并发 1。
- A100：615、930、1410 MHz。
- V100：600、900、1380 MHz。
- 每格 2 次有效重复；若变异大，额外诊断 suite 做 3 次确认。

目的不是证明最低频一定最省，而是找到满足相位 SLO 的能量甜点，并量化共置统一频率的机会成本。

## 6. 等资源 crossover

工作负载：

- short：128 input、128 output
- prefill-heavy：2048 input、128 output
- balanced：1024 input、256 output
- decode-heavy：128 input、512 output

每类包含低、中的固定 Poisson 到达率。最高频配置用于隔离架构与传输税；两个 medium 角点另跑 eco 频率：

- Colocated eco：A100 930、V100 900 MHz
- A100-P eco：A100 1410、V100-D 600 MHz
- V100-P eco：V100 1380、A100-D 615 MHz

每格先运行同形状 warmup，再执行 2 次 paired trace。SLO 阈值来自最高频 colocated 的 pooled p90，tight 为 1.1×，loose 为 1.5×。

## 7. 数据与验证

每个运行目录包含：

- `manifest.json`：软件、GPU、拓扑和配置
- `idle_power.csv` / `idle_summary.json`
- `logs/`：每个 vLLM server 与代理日志
- `<case>__r<n>/requests.jsonl`：逐请求、逐 token 时间戳
- `<case>__r<n>/power.csv`：逐卡 NVML trace
- `<case>__r<n>/complete.json`：延迟与能量汇总
- `suite_summary.json`

聚合产物：

- `results/experiments/summary/summary.csv`
- `results/experiments/summary/aggregated.csv`
- `results/experiments/summary/summary.json`
- `results/selective_pd_motivation_pilot.md`

单元测试覆盖功率积分、percentile、TTFT/TBT 汇总和配置 case 唯一性。真实 smoke 覆盖单卡、双 mixed、NIXL 1P1D；每次锁频退出时恢复默认。

## 8. 解释与扩展

本机结果对应“异构、无 P2P、host-staged KV”区域。若 PD 因 TTFT 传输税而不可行，正确结论是 selective 策略应在此区域选择 colocated，而不是把结果外推为 PD 普遍无效。

论文级扩展需要：

1. 至少两张同代、CUDA P2P 可用的 GPU，最好再增加 NVLink/RDMA 对照。
2. 25%/50%/75% 三档校准负载，至少 5 次重复。
3. 固定毫秒 SLO 与相对 SLO 同时报出。
4. 使用 RAPL 或机架功率计补上 CPU、DRAM 和网络能量。
5. 至少 3 张卡实现 P-only、D-only、Mixed 同时常驻，验证请求级 selective 路由。

按本次 pilot 墙钟时间估算，完整扩展矩阵约需 8–12 小时。
