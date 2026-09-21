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

### 分段实测（sharegpt-x0.5-pdblend 的 power.jsonl × 计划段对齐）

| 布局 | f_M | 段长(s) | 预测W | 实测W(去头3s) |
|---|---|---|---|---|
| 8M（冷启动） | 2520 | 40.9 | — | 1848 |
| 1P3D+4L1 | 2520 | 20.6 | 1075 | 1095 |
| M4+4L1 | 2100 | 10.3 | 1141 | 1094 |
| 1P3D1M+3L1 | 2520 | 20.5 | 1271 | 1298 |
| **M5+3L1（=暖计划）** | 2100 | 20.4+20.4 | 1359/1371 | **1270/1280** |
| 2P4D+2L1 | 2520 | 20.4 | 1397 | 1436 |
| 2P5D+1L1 | 2520 | 32.2 | 1523 | 1514 |
| 2P4D | 2520 | 91.9 | 1393 | 1304 |

读数：
1. 冷启动税精算：(1848−1270)×40.9/300 ≈ **+79W**。
2. 模型对 M 主导布局**系统性高估 6–9%**（1359→1270、1393→1304），对 PD 布局高估 2–6%；static_best 同源模型，相对比较仍自洽，暂不动校准（冻结期禁改共享参数）。
3. trace 是恒定 8.11 rps 齐次泊松——forecast "爬升" 纯属 EWMA 暖机偏差。**全程持有 M5+3L1@2100 在能量上是正确的**；margin=0.08 会挡住预测差 <8% 的切换（如 →2P4D 仅 2.5%）。
4. **迭代 1 预期**：窗口均值 ≈1270–1290W → 胜 dynamollm（1296），与 ecoserve（1271）持平/险胜。若 compare 判 LOSE，下一杠杆是容量模型校准方向：M4+4L1@2100 实测仅 1094W（10s 段），规划器却以稳态模型判 M4 不足——若 TPOT 高估同幅存在，x0.5 的最优解其实是 M4。

### 待办（若迭代 1 仍有负点）

- Shield 降档提速：只升受压池、冷却 30s→15s（shield.py）。
- 停机深度：off 被 L1 严格支配（35.4W/35s vs 25.9W/0.028s）→ pdblend 禁选 off。
- PD 门控复核：减半语料 sharegpt mean 784 tok 下 min_pd_input_tokens=256 是否仍合理。
- 排程：JSQ / drain_timeout 30s。
