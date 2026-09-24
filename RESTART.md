# pdblend4-v3：新 8×L20 部署与实验复现

本手册适用于 **Linux x86_64、Python 3.10、8×NVIDIA L20（每卡约 46 GiB）**。
主线复现使用 Qwen2.5-7B/14B/32B-Instruct，三数据集 Alpaca/ShareGPT/LongBench，
seed 701，150 秒服务窗以及完整排空尾部；每模型原始 12 个倍率点，共 36 个 PDblend 点。
四个 baseline 的复现入口见下文。

`pdblend4-v3` 保存最新维护源码；当前机器正在运行的历史矩阵使用冻结的 `d4fe2c6a…` 源码。
新入口默认冻结当前 checkout 并重新计算启动契约；`--source-mode historical` 则选择历史源码。
新机器使用新的 GPU UUID、新租约、新输出目录，**不导入或启动源机器的 queue.json**。
现有 profile 在新机器上只用于 development 重跑，`formal_eligible=false`；它不证明新机已校准，
也不继承旧机器的能耗或性能验收。每台机器的正式结论仍需要独立校准与验收。

## 1. 交付文件与固定环境

源码之外的交付目录在源机 `/home/pdblend4-v3-release/`。传输该目录及 `/home/models/`：

| 内容 | 用途 |
|---|---|
| `pdblend4-v3.bundle` | 分支源码与 Git 历史；也可以使用已取得的同名 Git 分支 |
| `image/pdblend4-v3-engine.tar.zst`、相邻 `.json` | 当前实际运行镜像及归档 SHA256 |
| `inputs/pdblend4-v3-inputs.tar.gz`、清单与 `.sha256` | profile、轨迹、配置、冻结源码、predictor 等不可变运行输入 |
| `external/pdblend4-v3-inputs.tar.gz`、清单与 `.sha256` | baseline 所需的外部原始到达记录，保留其原绝对路径 |
| `validation/` | 源机实际 CPU、镜像、模型与迁移验证记录 |
| 独立 `/home/models/` | Qwen 三模型权重，以及 BERT/旧 predictor 输入；约 104 GiB |

镜像 tag 为 `pdblend:l20-cu128-vllm-v1`。源机 image ID：

```text
sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc
```

真实版本是 **vLLM 0.10.1.1 / Torch 2.7.1+cu126 / torch.version.cuda=12.6**；
CUDA 基础镜像是 **12.8.1**。tag 中的 cu128 不代表 Torch 编译版本。
`VLLM_USE_V1=1`，NCCL KV 传输，BF16，8192 上下文，关闭 prefix caching。
当前补丁由 `engine_patches/vllm-0.10.1.1/manifest.json` 定义。
宿主控制器单独使用 Torch 2.7.0+cu128 / Transformers 4.51.3。

**根目录旧 `pdblend-l20-v1.tar.zst` 是另一镜像，不能用于本次恢复。**
严格身份依据包括镜像 RootFS、Config、全部 Python 包和补丁 SHA；Docker 存储后端可能使用不同
image ID，导入工具按镜像层与配置核验，实验入口绑定目标 Docker 实际解析出的身份。

## 2. 目标机准备与源码恢复

目标机须已具备可工作的 NVIDIA 驱动、Docker Engine、NVIDIA Container Toolkit，以及
`git`、`rsync`、`zstd`、`build-essential`、Python 3.10 和对应 `venv` 支持。
运行用户需要 Docker 访问权限与实验所需的 NVML 锁频权限。源机驱动为 580.126.09。
宿主无需照搬另一台机器的 CUDA toolkit 或 `.venv`。安装宿主驱动/容器运行时后，按第 4 节
真实八卡 CUDA 检查判断是否可用。不要在 GPU 实验期间执行 Docker 重启或 `systemctl daemon-reload`。

先将源机交付目录复制到目标机 `/srv/pdblend4-v3-release/`（以下统一记为此路径）；
`SOURCE_HOST` 替换为实际源机 SSH 地址：

```bash
rsync -aH --partial --info=progress2 SOURCE_HOST:/home/pdblend4-v3-release/ /srv/pdblend4-v3-release/
```
为了保持不可变输入的绝对路径与 checksum，项目恢复到 **`/home/pdblend4`**，模型恢复到
**`/home/models`**；这里是本版的明确路径约束，不使用符号链接绕过。

```bash
# 在目标机执行；/home/pdblend4 必须不存在，避免覆盖已有工程。
git clone -b pdblend4-v3 /srv/pdblend4-v3-release/pdblend4-v3.bundle /home/pdblend4
cd /home/pdblend4
git status --short --branch
```

权重从源机完整复制，保留 `pdblend-model-manifest.json` 与来源清单；将 `SOURCE_HOST`
替换成实际源机 SSH 地址（这些 rsync 命令在目标机执行）：

```bash
rsync -aH --partial --info=progress2 SOURCE_HOST:/home/models/ /home/models/
```

## 3. 校验并恢复实验输入和模型

```bash
cd /srv/pdblend4-v3-release/inputs
sha256sum -c pdblend4-v3-inputs.tar.gz.sha256
cd /home/pdblend4
python3 scripts/2026-09-25_release_assets.py extract \
  /srv/pdblend4-v3-release/inputs/pdblend4-v3-inputs.tar.gz --project /home/pdblend4

cd /srv/pdblend4-v3-release/external
sha256sum -c pdblend4-v3-inputs.tar.gz.sha256
cd /home/pdblend4
python3 scripts/2026-09-25_release_assets.py extract \
  /srv/pdblend4-v3-release/external/pdblend4-v3-inputs.tar.gz --project /home/pdblend

python3 scripts/2026-09-25_release_assets.py verify-models \
  --models-root /home/models --out /srv/pdblend4-v3-release/target-model-verification.json
```

提取器使用标准库，逐项验证归档 SHA、文件清单及内容，拒绝路径穿越、符号链接和覆盖不同内容。
与已有文件相同则保留，故可重复提取。模型验证会重新读取全部约 104 GiB 数据，并与本分支
固定的模型 manifest 对照；只数分片或读取 manifest 不算通过。
外部输入包只恢复历史绑定的到达记录，不会创建旧工程源码或旧虚拟环境。
输入包提供实际 runtime 消费闭包，完整历史审计原件仍保留在源机 44 GiB 的 results 中。

## 4. 导入相同 Docker，安装宿主控制器

```bash
cd /home/pdblend4
python3 scripts/2026-09-25_environment.py import \
  --archive /srv/pdblend4-v3-release/image/pdblend4-v3-engine.tar.zst
python3 scripts/2026-09-25_environment.py verify --hardware
bash scripts/2026-09-25_bootstrap_host.sh
```

`import` 先检查镜像归档 SHA 和固定身份，再导入；同名 tag 如指向不同内容会拒绝覆盖。
`verify --hardware` 在空闲目标机检查八张 L20 的真实 BF16 CUDA 运算，并检查镜像包版本、补丁、
层和配置、`pip check`。这个检查会使用 GPU，应在开始实验之前执行。

bootstrap 在本项目新建 `.venv`，消费 `requirements/pdblend4-v3-host.lock`，安装当前源码并
执行 `pip check`。默认依赖联网下载固定版本；本交付没有包含完整宿主 wheelhouse，不能将
Docker 镜像包等同于离线宿主依赖。不要复用 `/home/pdblend/.venv`。
源码重建 Docker 方式见 [环境说明](requirements/README.md)，但重建后应作为另一个镜像验证；
要保持当前镜像全部系统层相同，采用上面的导入路径。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src CUDA_VISIBLE_DEVICES='' \
  .venv/bin/python -m pytest -q -m 'not gpu and not historical' \
  tests/pdblend tests/independent_baselines
```

`historical` 测试需要完整旧实验原始记录，与新机 runtime 最小包分开。不可把被跳过的 GPU 或
历史测试写成新机器的通过记录。

## 5. 立即开始最新 PDblend 的 36 点复现

在工程根目录执行；输出目录必须是全新的。使用日期目录和自己的 run-id，下面用固定示例方便核对。

```bash
cd /home/pdblend4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=src

# 仅检查输入闭包、计划规模和输出路径，不运行 Docker 或 GPU。
.venv/bin/python scripts/2026-09-25_reproduce_v3.py prepare \
  --out results/2026-09-25/v3-reproduce-01 --dry-run

# 在准确运行镜像内执行三模型 CPU 启动前检查，冻结本分支源码；不入队、不占 GPU。
.venv/bin/python scripts/2026-09-25_reproduce_v3.py prepare \
  --out results/2026-09-25/v3-reproduce-01

# 确认本机八卡空闲后，建立新队列并按八卡独占方式顺序执行。
.venv/bin/python scripts/2026-09-25_reproduce_v3.py start \
  --out results/2026-09-25/v3-reproduce-01
```

`prepare` 会读取目标 GPU UUID、复核镜像、绑定全套源码，按原 tuning 数据重算实际启动契约。
`preparation.json` 记录源码/输入 SHA、CPU preflight、模型和输出队列。服务窗、轨迹、seed、SLO
保持原值；当前缺完整证书的 low-M、切换成本、增量能量配置保持原来的关闭状态。
只先运行 7B 时在 `prepare` 加 `--models 7b`；每个模型必须使用自己的 profile/trace。

若要逐点复跑当前接受的历史 d4fe 实现，另建输出目录并在 `prepare` 加
`--source-mode historical`。不要把默认 workspace 结果标为旧源码结果，也不要覆盖任何历史回执。
两个模式都显式记录旧 profile 的来源，不宣称已完成新机正式校准。

结果写入新目录的 `campaign.json`、`jobs.json`、`queue.json` 和 `queue-attempts/`，逐窗原始请求、
功率、native-result 和 receipt 以具体 job payload 的输出路径为准。
原始 36 点与自适应扩点是不同范围；上述命令先复现原 36 点，不自动沿用源机的边界搜索状态。

运行结束后可从新目录导出独立 CSV（baseline 使用自己的 campaign 与 queue-attempts 路径）：

```bash
.venv/bin/python -m pdblend.bench.comparison_campaign export \
  --campaign results/2026-09-25/v3-reproduce-01/campaign.json \
  --sessions results/2026-09-25/v3-reproduce-01/queue-attempts \
  --out results/2026-09-25/v3-reproduce-01/compare.csv \
  --analysis-policy all_recorded_windows/v1
```

## 6. baseline 与后续实验

四个 baseline 保持独立实现及其历史冻结配置，不能用 PDblend 最新源码替换它们的算法。
五系统标准矩阵共有 180 个逻辑点：PDblend 36 点，四 baseline 各 36 点。
PDblend 队列完成并释放八卡后，执行下列命令；不要同时启动两个整机能耗队列。

```bash
cd /home/pdblend4
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src
.venv/bin/python scripts/2026-09-25_reproduce_baselines_v3.py prepare \
  --out results/2026-09-25/v3-baselines-01 --dry-run
.venv/bin/python scripts/2026-09-25_reproduce_baselines_v3.py prepare \
  --out results/2026-09-25/v3-baselines-01
.venv/bin/python scripts/2026-09-25_reproduce_baselines_v3.py start \
  --out results/2026-09-25/v3-baselines-01
```

默认执行三个模型、四个 baseline 的 144 个标准点。可用 `--models 7b` 与
`--systems mixed distserve ecoserve dynamollm` 选择子集。每组使用自己的已绑定冻结源码，
启动前在真实镜像内核对原生部署、profile 和 predictor/hash；DynamoLLM 还核验真实 Azure
到达记录。DynamoLLM 的 1800/300/5 秒控制周期保留；150 秒标准观测不代表通过其完整
长周期机制验收。所有新测量进入新的输出和整机独占队列；baseline 同样不继承新机正式资格。

边界扩展、native profile 补测及严格容量搜索的设计见
[Profile 全量复测与单次观测边界](docs/2026-09-24_profile_saturation_round.md)、
[恢复实现](docs/2026-09-24_pdblend_recovery_implementation.md)。旧机的历史 status/queue/lease
不属于新机运行配置；正式容量搜索也不能由单次 seed701 观测替代。

## 7. 本次新队列的平滑暂停和恢复

```bash
# 只阻止领取下一个 job；当前 job 会完成自己的所有窗口、排空和清理。
touch results/2026-09-25/v3-reproduce-01/worker.stop
.venv/bin/python scripts/2026-09-22_gpu_campaign_queue.py \
  --db results/2026-09-25/v3-reproduce-01/queue.json status
```

待 worker 退出、当前租约和 owned 容器/clock cleanup 均完成后，再移除自己创建的 stop 文件并恢复：

```bash
rm results/2026-09-25/v3-reproduce-01/worker.stop
.venv/bin/python scripts/2026-09-22_gpu_campaign_queue.py \
  --db results/2026-09-25/v3-reproduce-01/queue.json worker \
  --workers 1 --idle-exit --stop-file results/2026-09-25/v3-reproduce-01/worker.stop
```

不删除失败 attempt，不清空队列，不全局杀容器，不把源机 UUID 写入新机命令。
源机旧队列恢复手册仅归档于 [旧恢复文档](docs/RESTART-queue-2026-09-23.md)，不可用于新机初始化。

## 8. 本版已执行的验证

已完成当前镜像真实导出/导入往返、三模型全量 SHA、3472 项 CPU 回归，以及最终工具 20 项专项检查。
还在隔离目录恢复两份输入包，隐藏源机原始 results，并以模拟新 GPU UUID 完成全部 180 点的
真实镜像 CPU 预检。日志与限制见 [发布验收](docs/releases/pdblend4-v3/README.md)。
目标服务器尚未连接，八卡 CUDA、服务与性能实跑应在目标机执行，不能继承源机验证结果。
