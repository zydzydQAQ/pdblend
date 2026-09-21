# PDblend 优化循环日志 — eval-7b-v2

判定标准（每点）：joint≥0.9 且 mean_power ≤ 四个 baseline（mixed / distserve_static / dynamollm / ecoserve）中 SLO 通过者的最低功率。

## 迭代 1 — 暖启动 + 初始计划保持 + 滞回加宽（2026-09-21）

### 失利证据（sharegpt-x0.5 快速首组，冷启动版 pdblend）

| policy | mean_power (W) | j/tok | joint | ttft90 (s) |
|---|---|---|---|---|
| mixed | 2121 | 0.843 | 1.000 | 0.17 |
| distserve_static | 1942 | 0.772 | 1.000 | 0.29 |
| dynamollm | 1296 | 0.517 | 1.000 | 0.24 |
| ecoserve | 1271 | 0.508 | 1.000 | 0.28 |
| **pdblend（冷启动）** | **1336** | **0.532** | 1.000 | 0.99 |

→ 胜 mixed（-37%）与 distserve（-31%），**负于 dynamollm（+3.1%）与 ecoserve（+5.1%）**。

### 诊断（controller.jsonl 计划轨迹）

1. **冷启动税**：t=0–41s fail-open M8@2520（~2500W）→ 300s 窗口均摊 ≈ +116W。
2. **计划抖动**：380s 内 13 次 plan（forecast rate 1.9→8.0 rps 爬升 100s、output_mean 27→330 爬升 150s；margin=3% 抵不住 Poisson EWMA 噪声）。每次切换有 wake/drain 成本且功率震荡。
3. **自身反证**：t=319 收敛到 M4+4L1@2100 预测 1129W——若从 t=0 就持有离线计划即可赢过两个赢家。

### 改动（全部 pdblend 独占路径，baseline 代码路径不变）

- `policies/__init__.py`：Policy 新增 `warm_start: bool`、`margin: Optional[float]`；pdblend 条目 `warm_start=True, margin=0.08`（其余 policy 默认 False/None）。
- `bench/run.py`：`policy.freeze or policy.warm_start` → t=0 用 `offline_forecast(trace)` 出初始计划；`hold_initial=policy.warm_start` 传入 Controller。
- `control/controller.py`：新增 `hold_initial` 字段；未 inform（<20s 且 <30 样本）时持有暖计划，不再 fail-open 全 M@2520。
- 滞回 margin 3%→8% 仅 pdblend（planner_config 里 replace），抑制爬升期抖动；freeze/baseline 的 PlannerConfig 不受影响。

### 验证

- pytest tests/pdblend2：67 passed + 1 skipped（新增 2 测试：hold_initial 持有暖计划 / 无 hold_initial 回退 fail-open）。
- 离线 sanity（scripts/warm_plan_sanity.py，27 点）：pdblend 暖计划 = static_best 计划（同源离线预测），例如：
  - sharegpt-x0.5: M5+3L1@2100（模型预测 1350W；实测稳态通常低于预测）
  - alpaca x0.1–x0.7: M1–M7@1800 + L1 补齐（434–1797W）
  - sharegpt x0.6–x0.9: 纯 PD（2P5D/2P6D，1558–2317W）；longbench x0.2–x0.7: 重 P 池（2P1D→6P2D）
  - longbench x0.8–x0.9 / alpaca-x0.9: M8@2100/2520
- 生效时机：当前 runner 在跑 baseline（旧代码）；baseline 冻结后停 runner 重启（skip 已完成点），pdblend 27 点用新代码。

### 待办（若迭代 1 仍有负点）

- Shield 降档提速：只升受压池、冷却 30s→15s（shield.py）。
- 停机深度：off 被 L1 严格支配（35.4W/35s vs 25.9W/0.028s）→ pdblend 禁选 off。
- PD 门控复核：减半语料 sharegpt mean 784 tok 下 min_pd_input_tokens=256 是否仍合理。
- 排程：JSQ / drain_timeout 30s。
