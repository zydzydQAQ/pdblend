# 三模型迁移与复现

当前入口是 [README](README.md)、[实时状态](results/2026-09-23/status/current.md)和
[队列恢复手册](RESTART.md)。本轮工作负载统一 seed 701；profiler 每点三次窗口重复仍保留。

## 必须一并迁移的内容

- 工程源码、Dockerfile、引擎补丁，以及每个实验引用的冻结源码和 manifest。
- 当前 Docker 镜像，校验 image ID 为
  `sha256:1c2d0bf96dfa752394a6aa4b5398a6105dcf060936a484a89729dcab6f9d9acc`。
  它绑定 vLLM 0.10.1.1、Torch 2.7.1、CUDA 12.8.1；同名 tag 或重新构建成功不能代替身份校验。
- `/home/models/Qwen2.5-{7B,14B,32B}-Instruct` 的权重、配置、tokenizer 和验证凭据。
- 三模型分别 tokenize 的 prepared 数据及其 manifest、原始语料和各模型独立 predictor。
- 各系统独立 profile、原始 samples、训练/holdout 计划、校准版本和全部引用证据。
- queue、attempt、租约历史、原始请求/SSE、功率、动作、清理记录和归档索引。
- `results/runs.csv`、`profile_points.csv` 与历史删除身份 CSV；`raw_pruned` 历史行不再具有原始证据。

不要把整个 results 当缓存删除。失败 attempt 也可能含当前校准引用的唯一有效样本。
原始 artifact 带绝对路径和 checksum；迁移优先保留原路径。改变路径时建立新的映射和验证
凭据，不直接修改已冻结 manifest。旧唯一镜像迁移包仍须保留。

## 环境与恢复核验

目标机需要可用的 NVIDIA 驱动、Docker/NVIDIA container runtime 和锁频权限。先读取 GPU
UUID、拓扑、显存、功率/频率能力及当前进程，确认与本轮 8×L20 的实际身份是否一致。
镜像导入后用 `docker image inspect` 核验上述 image ID，再检查容器实际 Torch/vLLM/CUDA
版本、引擎补丁 checksum 和三个模型/tokenizer hash。

当前功能任务通过队列取得 GPU UUID、容器内索引、独立端口和 concurrency-environment。
按 [RESTART](RESTART.md) 先核对已有 worker/lease，再恢复 worker；不能另外直接启动旧
`results/v2` 矩阵占用同一批 GPU。容器、锁频和清理只能操作各自租约范围。

CPU 回归、真实引擎启动/输出、KV/取消/恢复检查、模型级 profile 校准和系统机制验收是不同
阶段。PDBlend 只使用 PP1，7B/14B 的 TP1/2/4 和 32B 的 TP2/4 仍需各自证据；正式整机
总能耗比较必须独占八卡。具体阻塞项以实时状态内的 receipt 和 scope 为准。

## Profile 复用

按 system、model、TP、PP、角色、频率和实测覆盖域复用有效样本，只补缺失点。除这些
维度外，还必须检查 image/source、GPU UUID/拓扑、模型/tokenizer、原始样本 checksum 和
独立 holdout；相同 GPU 型号、相同 vLLM 版本两个条件不足以判定可以正式复用。

显式选择 [校准版本](results/2026-09-23/calibration-versions-v1/report.md)，使用
[`load_version`](src/pdblend/profile/versions.py) 的覆盖域和资格门禁。
[最新补测审计](results/2026-09-23/incremental-wave-closeout-v1/report.md)的局部通过也不能
扩大旧版本的可查询范围。换硬件、引擎或 workload 域后应评估受影响点并补测，不能跨模型
或跨系统借用 profile，也不必无条件重跑所有已有测量。

## 历史索引

旧栈结果、混合 profile、半截断数据和 CPU replay 不参与新栈三模型正式排名。本轮已退役目录
按[结果保留规则](docs/RESULTS.md)仅留历史指标/身份 CSV；有当前有效引用的原始证据继续保留。
早期迁移说明完整保留于
[归档](results/archive/docs-2026-09-23/MIGRATION-before-current-rewrite.md)，其中旧评测命令、
profile 复用条件和全卡清理示例不适用于当前并行租约流程。
