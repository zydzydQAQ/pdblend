# PDblend 迁移与环境复现指南

> **当前入口（2026-09-23）**：本文第 1–8 节保留早期 7B `results/v2` 迁移流程。当前三模型 seed701 工作流以 `results/2026-09-22/three-model/IMPLEMENTATION-STATUS.md`、实际 queue receipt 和 `results/2026-09-23` 冻结 manifest 为准。当前运行必须固定 vLLM 0.10.1.1、Torch 2.7.1、CUDA 12.8.1 和 8×L20；7B、14B、32B 三个模型都必须具备权重、tokenizer、verification receipt 和独立 profile/predictor，功能工作负载统一使用 seed701。

本文档固化在另一台机器上复现本项目（8×GPU LLM serving 能耗控制评测）所需的全部环境信息。
目标读者是人类或自动化 agent（如 Codex）：按顺序执行 §1–§5 可完成旧版 7B 迁移，§6 说明旧版 profiling；三模型任务还需要各自 7B/14B/32B 权重、tokenizer/model verification manifest 和 Dynamo predictor artifact。

## 0. 迁移清单

复现需要 **4 样东西**，缺一不可：

| # | 内容 | 源位置 | 大小 |
|---|------|--------|------|
| 1 | 本仓库（含 .git） | `/home/pdblend4` | ~60 MB |
| 2 | Docker 镜像 `pdblend:l20-cu128-vllm-v1` | 本机 docker daemon | ~28 GB（导出后） |
| 3 | 镜像构建上下文（重建用，含 vLLM 补丁） | 本仓库 `Dockerfile` + `engine_patches/vllm-0.10.1.1/` | <1 MB |
| 4 | 模型权重与 tokenizer | `/home/models/Qwen2.5-{7B,14B,32B}-Instruct` | 按模型 manifest |

注意：Dockerfile 和 vLLM 补丁已收进本仓库（`Dockerfile` + `engine_patches/vllm-0.10.1.1/`），
只迁移本仓库即可走 §3 路线 B 重建镜像；`/home/pdblend3/docker/Dockerfile.v1-p2p` 是同一次构建的原始副本。

## 1. 目标机要求

- GPU：8× NVIDIA L20（当前 profile 所基于的硬件；其他型号/数量 → 必须按 §6 重新 profiling）
- 驱动：≥ 535（本机 580.126.09），支持 CUDA 12.8 runtime
- Docker + nvidia-container-toolkit（`docker run --gpus` 可用）
- 宿主机 Python ≥ 3.10（仅用于跑 CPU 测试/脚本；GPU 流程全部在容器内）
- 空闲端口：8000（proxy）、8101–8108（每实例一个，`base_port 8100 + gpu_index`）
- 容器内需能锁 GPU 频率：`--cap-add SYS_ADMIN`（NVML locked clocks）

## 2. 仓库

```bash
git clone <repo> pdblend4        # 或整体 rsync，含 .git、datasets/、results/
cd pdblend4
```

CPU 冒烟测试（不需要 GPU，验证代码完整性）：

```bash
pip install -e '.[test]'
python -m pytest tests -q        # gpu/historical 标记的用例会自动 skip
```

## 3. Docker 镜像（二选一）

### 路线 A：直接导出/导入（推荐，快）

```bash
# 源机：导出并压缩（28 GB → ~9 GB）
docker save pdblend:l20-cu128-vllm-v1 | zstd -3 -T0 > pdblend-l20-v1.tar.zst

# 校验文件完整性
zstd -t pdblend-l20-v1.tar.zst

# 目标机：恢复成镜像（不用先解压，边解边载）
zstd -d pdblend-l20-v1.tar.zst | docker load

# 如需解压回 tar（会得到一个 28 GB 的 .tar）
zstd -d pdblend-l20-v1.tar.zst
```

### 路线 B：用仓库内 Dockerfile 重建

构建上下文已在本仓库根（`Dockerfile` + `engine_patches/`，与 `/home/pdblend3/docker/Dockerfile.v1-p2p` 同源）：

```bash
cd /home/pdblend4        # 本仓库根
docker build -t pdblend:l20-cu128-vllm-v1 .
# 非阿里云网络覆盖镜像源：
#   --build-arg PIP_INDEX=https://pypi.org/simple --build-arg APT_MIRROR=archive.ubuntu.com
```

镜像要点（已固化在 Dockerfile 中，重建时勿改）：

- 基础镜像 `nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04`（digest `a99a1860...`）
- venv `/opt/venv`：`vllm==0.10.1.1`、torch 2.7.1、nixl 1.4.1（可选，失败则用 P2pNcclConnector）、
  nvidia-ml-py、aiohttp、numpy、pyyaml、pytest、matplotlib
- **transformers 必须钉 `4.55.2`**（tokenizers 0.21.4、hf_hub 0.34.4）：不钉会装到 5.x，
  删掉 vLLM 0.10.1.1 依赖的 `all_special_tokens_extended`
- **vLLM 补丁（关键，不可省）**：`engine_patches/vllm-0.10.1.1/` 在打 vLLM 官方包之后按
  manifest.json 覆盖 2 个文件并做 sha256 校验：
  - `p2p_nccl_engine.py`：修 P2pNccl 流序错误（否则 remote-prefill 输出乱码）和接收端 OOM
  - `p2p_nccl_connector.py`：kv_both 支持，使实例可在运行时被改写为 P/D/M 角色（pdblend 核心机制）
- 镜像内会 COPY 本仓库 src 并 pip install（/workspace），但运行时靠 `PYTHONPATH` 指向挂载进来的本仓库，互不影响

## 4. 模型权重

```bash
# 目标机放置为 /models/Qwen2.5-7B-Instruct（容器内路径由 PDBLEND_MODELS_DIR 决定）
# 完整文件：config.json、generation_config.json、tokenizer{,_config}.json、vocab.json、
#           merges.txt、model-0000{1..4}-of-00004.safetensors、model.safetensors.index.json
```

引擎按 `MODELS_DIR/<--model 参数>` 解析路径（`src/pdblend/engine/launcher.py:15`），
默认 `PDBLEND_MODELS_DIR=/models`。当前三模型 functional/native jobs 会挂载 14B/32B 和 Dynamo predictor；只有运行旧版 7B v2 矩阵时，才可按旧流程省略未用模型。

## 5. 启动容器（评测入口）

从当前运行中容器反推的完整命令（路径按目标机实际 checkout 位置替换）：

```bash
docker run --rm --name pdb2-matrix-v2 \
  --gpus all --network host --ipc host --shm-size 16g \
  --cap-add SYS_ADMIN --security-opt label=disable \
  -v /home/pdblend4:/home/pdblend4 \
  -v /home/models:/models \
  -w /home/pdblend4 \
  -e PDBLEND_MODELS_DIR=/models \
  -e PYTHONPATH=/home/pdblend4/src \
  pdblend:l20-cu128-vllm-v1 \
  python -m pdblend.cli matrix results/v2/eval-7b-v2/spec.json
```

要点：host 网络（实例间 NCCL/ZMQ 直连）、host IPC + 16G shm（vLLM 需要）、
`--security-opt label=disable`（SELinux 环境下挂载可写）、`--rm` 自动清理。
仓库挂在与宿主机相同的路径，日志/spec 里的相对路径才能对上。

## 6. Profiling：何时重做、怎么做

`results/v2/profile-7b/profile.json` 是**本机 8×L20 + vLLM 0.10.1.1** 的实测仿射性能/功耗模型；
`results/v2/eval-7b-v2/spec.json` 的 `capacity_rps`、`gpus: "0..7"` 均由它推出。

满足以下**全部**条件可直接复用现有 profile，跳过节余步骤：

- GPU 型号为 L20，数量 8 张，且频率档可用 900/1200/1500/1800/2100/2520 MHz
- 镜像/引擎版本不变（vllm 0.10.1.1 + 同一补丁）

任一不满足（换 GPU 型号、卡数变化、升级 vLLM）→ 按顺序重做：

```bash
# 1. 进容器（同 §5 的 docker run，命令换成 bash）
# 2. 重新 profile：约 150 点网格，2 张卡（第 2 张做 KV 传输对端），可加 --resume 断点续跑
python -m pdblend.cli profile --model Qwen2.5-7B-Instruct --gpus 0,1 \
  --out results/v2/profile-7b
# 3. 重新生成评测 spec（capacity_rps、速率点全部由新 profile 推导；卡数不同改脚本里的 gpus）
python scripts/gen_spec_v2.py
# 4. （可选）先跑 quick-five 冒烟：sharegpt-x0.5 的 mixed/distserve_static/dynamollm/ecoserve/pdblend
# 5. 全量 156 点：python -m pdblend.cli matrix results/v2/eval-7b-v2/spec.json
```

频率档查询：`nvidia-smi -q -d SUPPORTED_CLOCKS`；目标档不同可用 `--freqs` 覆盖。

## 7. 迁移后验证清单（按序冒烟）

1. `python -m pytest tests -q`（CPU，宿主机或容器均可）
2. `python -m pdblend.cli gate-kv --gpus 0,1`（G0：跨实例 KV 传输，验证补丁生效）
3. `python -m pdblend.cli gate-park --gpu 0`（G1：sleep/wake，验证锁频与 dev-mode 接口）
4. `python -m pdblend.cli bench --gpus 0,1,2,3 ...` 单点试跑（sharegpt-x0.5-mixed）
5. 全量 matrix

## 8. 已知坑

- `scripts/matrix-watch-v2.sh` 根据脚本位置计算工程根目录，可在不同 checkout 路径直接使用
- 锁频需要特权：容器内 `nvidia-smi -lgc` / NVML locked clocks 依赖 `--cap-add SYS_ADMIN`；
  跑完评测镜像内脚本会 reset，但异常退出后建议在宿主机执行 `nvidia-smi -rgc`
- 引擎启动要求 `VLLM_SERVER_DEV_MODE=1`（/sleep、/wake_up）和 `--enable-sleep-mode`，
  launcher 已自动注入（`src/pdblend/engine/launcher.py:76-94`），不要手动去掉
- host 网络模式下 8000/8101–8108 被占用会直接失败，跑前检查
- profile/spec 绑定硬件与引擎版本：混用他机 profile 跑出的能耗/SLO 结论无效
