# PDblend 优化循环日志 — eval-7b-v2

**判定标准（2026-09-21 用户拍板修订）**：每点达标 = joint_slo_rate≥0.9 且 **j_per_token（全程能耗÷输出 token）≤ 四个 baseline（mixed / distserve_static / dynamollm / ecoserve）中 SLO 通过者的最低 j_per_token**。window_j/token 与 mean_power 仅作副列。全量对比表 = `compare.csv`（`scripts/build_compare_csv.py` 从原始落盘件统一重算，幂等）。

**口径切换对快速首组的复判**：pdblend 0.532 vs ecoserve 0.508 / dynamollm 0.517 / distserve 0.772 / mixed 0.843（j/tok）——与功率口径同序，**仍负于 ecoserve（+4.8%）与 dynamollm（+3.0%）**，迭代 1（暖启动）目标不变。

**runner 重启（14:37）**：run.py 补落 util.jsonl/freq.jsonl（采样器本就在采，此前未写盘）；重启后第 14 点（alpaca-x0.9-mixed）起全部点带原生 SM 利用率/频率；前 13 点 compare.csv 记 N/A。暖启动代码随之加载（仅触达 pdblend 路径，baseline 行为不变）。

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

- pytest tests/pdblend：67 passed + 1 skipped（新增 2 测试：hold_initial 持有暖计划 / 无 hold_initial 回退 fail-open）。
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

## 迭代 2 — pdblend 独占校准 profiler（创新点）+ x0.5 正面交锋（2026-09-21 晚）

### 结论（compare.csv 口径）

| policy | mean_power(W) | j/tok | joint |
|---|---:|---:|---:|
| ecoserve（原最佳 baseline） | 1271 | 0.5078 | 1.0 |
| pdblend 迭代 1（旧代码落盘） | 1279 | 0.5097 | 1.0 |
| **pdblend 迭代 2（最终栈）** | **868.7** | **0.3503** | **0.9971** |

**判定：WIN，-31.0%。** ttft_p90 0.864s（SLO 5.0，余量 5.8×）、tpot_p90 0.0753s（SLO 0.15，余量 2×）、终态 M3+off5、全程 plan 10 次有界。

### 根因（回答"为什么规划器选不中 M4"——profiler 三件套）

1. **冻结 profile 原始测量过期**（主因）：引擎构建已变。同 grid 点 decode step(32,1024,2100)：冻结拟合 61ms vs 当夜新测 **28.3ms**（差 2.2×）；kv_capacity 453392→411824；静态功耗也漂移（parked 25.9→37.0W、off 35.4→36.0W——**off 不再被 L1 支配**，迭代 1 待办中"禁选 off"的结论随新数据反转）。
2. **冻结 json 为旧版 3 系数拟合**：B 与 B·ctx 在 3 值 ctx 网格上共线，beta 被压成 0，batch 增长全塞进注意力项。
3. **排队不动点放大器**：γ 高估 ~30% → Little 分母贴边（0.246）→ 稳态 batch 估出 2.2×（55 vs 实测 ~25）→ rho 容量门 569<653 判 M4"队列爆炸"。功率高估 6-9% 为连带误伤（以为 B=55，decode_power 随 B 涨）。

### 路线（用户拍板）：pdblend profiler 独立，作为创新点

baseline 继续用冻结的 results/v2/profile-7b/profile.json（全程未动）；pdblend 家族（27 主点 + 12 消融点，spec.json per-point 覆盖）用当夜重测的 6 频校准 profile：**results/v2/profile-7b-pdblend/**（34.8min 全量校准，decode_time 残差 6.6-14.2%，prefill 1.2-5%）。接线零代码改动（matrix.py 的 defaults+point 合并机制），gen_spec_v2.py 已同步。原"容量松动/home override"方案作废——胜利来自模型本身，无任何单点特权选型。

### 验证门（重启矩阵前全部通过）

- **真值对照**：三次固定布局实测（fixed-m5-2100 1279.9W / fixed-m4-2100 1097.6W / fixed-m4-2520 1274.7W），新模型预测功率误差 **+2.3% / +2.3% / +3.7%**，可行性与排序全部正确。
- **39 点 argmin 表**（scripts/audit_pdblend_profile.py → profile-7b-pdblend/audit.json）：无 fallback、无荒谬布局；x0.5 argmin = M2+off6@1800 = 706.6W（6 频比 2 频解锁 1800MHz 档）。
- **异 seed 稳健**（回应过拟合疑虑）：M4+4L1@2100 在 seed 701 / 1701 分别 1097.6W/0.4376 与 1097.9W/0.4341，joint 均 1.0——选型依据是负载档位而非特定 trace。
- **全栈先行验证**（adaptive-calibrated，2 频 profile + 完整控制栈）：911.2W / 0.3673 / joint 0.998，终态 M3+off5。

### x0.5 重跑控制轨迹（controller.jsonl）

首条 forecast rate=7.18（bootstrap 先验生效，旧代码从 2.0 爬坡）；初始计划即 argmin **M2+off6@1800**（t=0 home 落位）；ramp 期 shield 两级提级（clock→2520、M2→M3）兜住瞬态 TTFT 压力；随后稳定，终态 M3+off5。驻锚 + plan_hold_30s + down_votes_2 将抖动压在 ramp 期（plan 10 次 vs 迭代 0 的 15 次，且全部发生在前 ~100s）。

### 协调记录（双会话）

另一会话的 stage1 队列（calibration-2100-2520 → fixed-m4 两 seed → adaptive-calibrated）与本轮优化并行，其 2 频校准与先行验证结果被直接复用；其 driver 脚本曾两次自动恢复 matrix，均在 pdblend profile 就绪前被按 §2 规程暂停清理（残点零损失）。6 频校准完成后 matrix 于 ~21:31 恢复，x0.5-pdblend 为首个重跑点。GPU 争抢事故一次（profile-6f 首跑因并发容器内存不足失败，重跑成功）。

### 后续

- 26 个 pdblend 矩阵点（索引 130-155）将由 runner 断点续跑，全部使用 6 频校准 profile；消融点同。
- 长期（freeze 窗口外）：baseline 共享 profile 已过期 2.2×，重新 profiling 后 dynamollm/ecoserve 的数字也会变——报告时需在论文口径中说明 profiler 版本。
- decode_power 弱仿射（残差 14-32%）未影响真值门（±3.7% 内），若后续点出现功率排序误判可考虑分段功率拟合。

### 停机交接（21:50）

矩阵于 44/156 点完成处按 §2 停止（残点已清、显存归零），GPU 让给 codex 修正 profiler。恢复：codex 完工后按 RESTART.md §3 重启即可（幂等续跑；pdblend 家族 39 点已接线 spec 中的 profile-7b-pdblend 路径，若 codex 更新该文件内容无需再改 spec；若改路径需同步 spec.json 与 gen_spec_v2.py）。
