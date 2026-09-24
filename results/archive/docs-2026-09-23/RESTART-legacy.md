# RESTART — PDblend eval-7b-v2 矩阵实验重启手册

> **历史入口**：本手册只适用于 2026-09-21 的旧 7B `results/v2/eval-7b-v2` 矩阵。当前三模型 seed701、native baseline 和 Dynamo functional 任务不应使用本手册；请从 `results/2026-09-22/three-model/IMPLEMENTATION-STATUS.md` 和 queue receipt 进入。

## 0. 一句话

矩阵 runner 幂等续跑：`results/v2/eval-7b-v2/<点名>/summary.json` 存在即跳过。任何时候停掉、用 §3 原命令重拉，都会从第一个未完成点继续。**改代码或改 spec 必须重启 runner 才生效**（进程只在启动时读一次）；`scripts/build_compare_csv.py` 是离线脚本，随改随用。

## 1. 停机时状态（2026-09-21 15:00）

- 工作区 `/home/pdblend4`：代码、语料（`datasets/prepared/2026-09-21-7b-v2-half`）、profile（`results/v2/profile-7b/profile.json`）、spec（`results/v2/eval-7b-v2/spec.json`）、结果全部在此。唯一外部依赖：`/home/models` 模型权重挂载 + 镜像 `pdblend:l20-cu128-vllm-v1`。
- 进度 **16/156**。点顺序：sharegpt-x0.5 五策略快速组 ✓ → alpaca x0.1–x0.9 mixed ✓ → sharegpt x0.1–x0.2 mixed ✓；`sharegpt-x0.3-mixed` 曾发生引擎退出，已归档并会从原 spec 重跑。其余点按 spec 顺序继续。
- 主判定指标：**j_per_token（全程能耗÷输出 token）**，门槛 joint_slo_rate≥0.9；对比表 = `results/v2/eval-7b-v2/compare.csv`。
- **冻结纪律**：baseline（mixed/distserve_static/dynamollm/ecoserve）测完即冻结。禁止改动共享规划模型参数（`control/planner.py` 的 safety/rho_decode/dwell_s 及 decode/prefill 模型、`profile.json`）——否则已测 baseline 与 static_best 全部失效。pdblend 独占路径可改：`control/policies/__init__.py` 的 pdblend 条目、`control/controller.py`、`control/shield.py`。
- 每个新点落盘：summary.json / outcomes.jsonl / power.jsonl / **util.jsonl（SM 利用率）/ freq.jsonl（频率）** / controller.jsonl / logs/。前 13 个点（重启前测的）没有 util/freq，compare.csv 里对应列为空。

## 2. 停止

```bash
docker stop pdb2-matrix-v2
# 清理残点（无 summary.json 的目录 = 被中断的点，重启后自动重跑）：
for d in /home/pdblend4/results/v2/eval-7b-v2/*/; do [ -f "$d/summary.json" ] || rm -rf "$d"; done
# 必须确认 8 卡显存全部归零再重启：
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
```

## 3. 重启（逐字命令）

```bash
cd /home/pdblend4
setsid nohup docker run --rm --name pdb2-matrix-v2 \
  --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
  --network host \
  -v /home/pdblend4:/home/pdblend4 -v /home/models:/models \
  -e PYTHONPATH=/home/pdblend4/src -e PDBLEND_MODELS_DIR=/models \
  -w /home/pdblend4 \
  pdblend:l20-cu128-vllm-v1 \
  python -m pdblend.cli matrix results/v2/eval-7b-v2/spec.json \
  >> /home/pdblend4/results/v2/logs/eval-7b-v2-matrix.log 2>&1 &
```

## 4. 验证重启成功

2 分钟内：
```bash
docker ps --format '{{.Names}} {{.Status}}' | grep pdb2-matrix-v2   # Up
tail -5 /home/pdblend4/results/v2/logs/eval-7b-v2-matrix.log
# 日志应打印 [HH:MM:SS] <点名>: {...}，且是 spec 中第一个无 summary.json 的点
```
首个新点完成（约 6–8 分钟）后：
```bash
ls /home/pdblend4/results/v2/eval-7b-v2/<点名>/
# 应有 summary.json + util.jsonl + freq.jsonl（缺后两者说明跑的是旧代码）
```

## 5. 看进度 / 出对比表

```bash
# 一次性快照（容器状态 + 完成数 + 最近点 + 错误行）：
bash /home/pdblend4/scripts/matrix-watch-v2.sh

# 重建 compare.csv（幂等；CPU-only，容器内跑，几秒）：
docker run --rm --ulimit nofile=65536:65536 --network none \
  -v /home/pdblend4:/home/pdblend4 -e PYTHONPATH=/home/pdblend4/src -w /home/pdblend4 \
  pdblend:l20-cu128-vllm-v1 \
  python scripts/build_compare_csv.py results/v2/eval-7b-v2
# stdout 末尾打印 pdblend PASS/LOSE 计数与失利点名单；全表在 results/v2/eval-7b-v2/compare.csv
```

## 6. 注意事项

- **残点/错点**：runner 遇异常会在点目录写 error.txt 并继续下一点。要重测某点：删该点整个目录，再按 §3 重启（或直接等 matrix 自然跑到它——只会跑缺失点）。
- **pdblend 无需二次重启**：暖启动代码（warm_start + hold_initial + margin=0.08）已在当前镜像代码里，且只触达 pdblend 路径；baseline 有 summary.json 不会被重测。只有当你又改了 pdblend 侧代码时，才按 §2/§3 再重启一次。
- **判定口径**：compare.csv 一律从 outcomes.jsonl 原始记录重算（ttft/tpot 百分位、joint、goodput、j/token），跨点口径一致，**判定以 compare.csv 为准**；summary.json 的 slo 块由 runner 进程内代码计算，新旧点可能有细微口径差（nearest_rank 改造），不影响 compare.csv。
- **单点 smoke**（优化循环用；需 GPU 空闲，先停矩阵）：
  ```bash
  docker run --rm --ulimit nofile=65536:65536 --gpus all --cap-add SYS_ADMIN --ipc=host --shm-size=16g \
    --network host -v /home/pdblend4:/home/pdblend4 -v /home/models:/models \
    -e PYTHONPATH=/home/pdblend4/src -e PDBLEND_MODELS_DIR=/models -w /home/pdblend4 \
    pdblend:l20-cu128-vllm-v1 \
    python -m pdblend.cli bench --policy pdblend --profile results/v2/profile-7b/profile.json \
      --corpus datasets/prepared/2026-09-21-7b-v2-half --dataset sharegpt --rate 8.11 --scale 0.5 \
      --duration 300 --seed 701 --out results/v2/smoke/<自定义名>
  ```
- **测试**：改码后跑 `docker run --rm --ulimit nofile=65536:65536 --network none -v /home/pdblend4:/home/pdblend4 -e PYTHONPATH=/home/pdblend4/src -w /home/pdblend4 pdblend:l20-cu128-vllm-v1 python -m pytest tests/pdblend/ -q`，要求 67 passed + 1 skipped。
- 别动无关容器（如其他终端的 focused_joliot）。git 提交用 `git -c user.name=pdblend -c user.email=noreply@local commit ...`。
- 战役文档：`results/v2/eval-7b-v2/OPTIMIZATION-LOG.md`（判定标准 + 优化迭代记录）；baseline 全量完成后补 `BASELINES-FROZEN.md` 快照。
