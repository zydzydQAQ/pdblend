# Selective PD 本机实验

本目录是在 `pdblend:vllm-cu128` 容器中运行的 TTFT、TBT 与 GPU 能量实验框架。主模型为 `/models/Qwen2.5-7B-Instruct`，服务端固定 FP16、vLLM 0.11.2、关闭 prefix caching。

## 本机限制

- GPU0：A100-PCIE-40GB；GPU1：V100-PCIE-32GB。
- 两卡无 NVLink，且 CUDA P2P 读写均不支持。
- `P2pNcclConnector` 会在首次传输时报 NCCL CUDA error，不能用于本机结果。
- 经用户批准，PD 使用 NIXL 1.4.0，并设置 `UCX_TLS=self,sm,tcp,cuda_copy`，使 KV 经主机路径传输。
- NVML 只测 GPU 能量，不包含 CPU、DRAM 与 PCIe 交换芯片；因此 PD 能量是偏乐观估计。

## 文件

- `launch.py`：启动和回收单卡、双 mixed、1P1D 服务。
- `proxy.py`：容量加权 colocated 代理和 NIXL prefill→decode 代理。
- `loadgen.py`：固定 token 长度的 Poisson 流式负载，记录每个 token 时间。
- `power_sampler.py`：20 Hz NVML 采样与梯形能量积分。
- `run.py`：YAML suite 编排、锁频、warmup、重复和结果归档。
- `analyze.py`：聚合最新 suite，计算双 SLO attainment 与 oracle 选择。
- `plot_results.py`：从聚合 CSV/JSON 可复现地生成报告所用 PNG。
- `configs/pilot.yaml`：本次 motivation pilot 的完整配置。

## 运行

启动双卡容器：

```bash
bash docker/run_dual_gpu.sh
```

首次在镜像中启用 host-staged PD：

```bash
uv pip install --python /opt/venv/bin/python \
  -r experiments/selective_pd/requirements-extra.txt
```

测试：

```bash
PYTHONPATH=/workspace python -m unittest \
  experiments.selective_pd.test_harness
```

依次运行：

```bash
PYTHONPATH=/workspace python -m experiments.selective_pd.run --suite phase_a100
PYTHONPATH=/workspace python -m experiments.selective_pd.run --suite phase_v100
PYTHONPATH=/workspace python -m experiments.selective_pd.run --suite crossover_colocated
PYTHONPATH=/workspace python -m experiments.selective_pd.run --suite crossover_a100_prefill
PYTHONPATH=/workspace python -m experiments.selective_pd.run --suite crossover_v100_prefill
PYTHONPATH=/workspace python -m experiments.selective_pd.analyze
PYTHONPATH=/workspace python -m experiments.selective_pd.plot_results
```

每个 suite 写入 `results/experiments/<timestamp>-<suite>-<hash>/`。分析器按 suite 选择最新目录，旧目录保留用于审计；`*_confirm` 诊断 suite 不进入最终结论。

## 口径

- TTFT：客户端发出 HTTP 请求到首个生成 token。
- TBT：相邻生成 token 到达客户端的时间差。
- 可行配置：TTFT attainment 和 token 级 TBT attainment 均至少 90%。
- 主能耗：首个请求调度到最后一个 token 的两卡 gross GPU Joule。
- 启动、模型加载与架构切换能量不计入 steady-state 主结果，需另做切换实验。
