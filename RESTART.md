# RESTART — 三模型 seed701 队列恢复手册

本手册用于当前三模型队列的平滑恢复。当前状态入口是
[`results/2026-09-23/status/current.md`](results/2026-09-23/status/current.md)。
运行 `python3 scripts/2026-09-23_live_status.py` 可按实际队列和验收凭据刷新。

当前调度状态是 `results/2026-09-22/three-model/queue.json`。它是会被 worker heartbeat、lease
claim 和完成 receipt 更新的**可变调度状态**，不是 immutable receipt；需要保留的 immutable
证据在各 attempt、raw、manifest 和冻结源码目录中。

当前运行约束固定为：Qwen2.5-7B-Instruct、Qwen2.5-14B-Instruct、Qwen2.5-32B-Instruct，
`pdblend:l20-cu128-vllm-v1`（vLLM 0.10.1.1）、Torch 2.7.1、CUDA 12.8.1、8×L20，
功能工作负载使用 seed 701。三模型的权重、tokenizer、model verification receipt 和独立
profile/predictor 必须逐一存在并通过 identity 校验。

旧版 7B `results/v2` 手册仅作历史证据，原文归档为
[`RESTART-legacy.md`](results/archive/docs-2026-09-23/RESTART-legacy.md)，SHA256 为
`a44591dda529a9066cde9105e75793458bc26eeb9d45ebbe7f5a79483188645b`。

## 1. 先读取当前状态

```bash
cd /home/pdblend4
cat results/2026-09-23/status/current.md 2>/dev/null || true
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python scripts/2026-09-22_gpu_campaign_queue.py \
  --db results/2026-09-22/three-model/queue.json status
ps -eo pid,stat,etime,args | rg '[2]026-09-22_gpu_campaign_queue.py'
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
```

worker PID 会变化；以进程参数和当前 worker receipt 为准。确认 queue 中的 active lease、attempt
目录和 worker 日志，不能只根据进程是否存在判断可以重启。

## 2. 平滑 drain

如果 worker 仍在运行，创建 stop 文件只会阻止领取新任务，不会中断当前租约：

```bash
touch results/2026-09-23/remaining-parallel-execution-v1/worker-v3.stop
```

随后反复读取 queue status、active lease、worker 日志和 GPU 状态，等待当前任务自然结束，
并确认 owned process/clock cleanup receipt 已清空。不要删除 attempt 目录、raw、manifest 或
queue 条目；失败和未完成 attempt 由恢复流程保留以供审计。

只有在确认没有 owned active lease、GPU 已释放且 worker 已退出后，才可移除这个单一 stop 文件：

```bash
python3 - <<'PY'
from pathlib import Path
p = Path("results/2026-09-23/remaining-parallel-execution-v1/worker-v3.stop")
if p.exists():
    p.unlink()
PY
```

禁止使用旧手册中的 `rm -rf` 批量清理；当前恢复必须依靠 queue lease 和 attempt 幂等性。

## 3. 使用当前 worker 恢复

先查看当前 CLI 帮助，确认部署中的 worker 参数，再从仓库根目录启动；不要复用旧 7B
matrix runner：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python scripts/2026-09-22_gpu_campaign_queue.py \
  --db results/2026-09-22/three-model/queue.json worker --help
```

当前配置为八个 worker。六成员 profile 波次需要至少六个可领取任务的 worker，少于六个会使
已启动成员等待尚未启动的成员。当前进程信息保存在
`results/2026-09-23/remaining-parallel-execution-v1/worker-v3.json`；恢复时保留相同 stop-file：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python scripts/2026-09-22_gpu_campaign_queue.py \
  --db results/2026-09-22/three-model/queue.json worker --workers 8 \
  --stop-file results/2026-09-23/remaining-parallel-execution-v1/worker-v3.stop \
  >> results/2026-09-23/remaining-parallel-execution-v1/worker-v3.log 2>&1
```

worker 必须使用 lease 提供的 GPU UUID、local index、端口和 concurrency-environment；不要在
命令行手写物理 GPU，也不要复用旧容器名或旧 attempt 输出目录。

## 4. 重启后核验

```bash
cat results/2026-09-23/status/current.md 2>/dev/null || true
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  /home/pdblend/.venv/bin/python scripts/2026-09-22_gpu_campaign_queue.py \
  --db results/2026-09-22/three-model/queue.json status
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
tail -50 results/2026-09-23/remaining-parallel-execution-v1/worker-v3.log
```

接受新任务前，确认三模型 identity、source/image hash、seed701、profile/predictor 路径和
lease port window 均来自当前 queue payload。development smoke、CPU replay 或普通请求
不能提升为 formal 或 energy qualification；相应的 `formal_eligible` 和
`energy_comparable` 必须由 acceptance receipt 明确证明。

## 5. 不可删除的恢复证据

保留所有 raw samples、model/tokenizer manifests、verification receipts、predictor artifacts、
冻结源码、attempt receipts、concurrency-environment snapshots 和 queue lease history。历史
整理只能使用 `results/archive/` 中带 hash 的 cleanup tombstone，并逐项递归核对引用；不要
移动或改写绝对路径、原始数据或 immutable manifest。
