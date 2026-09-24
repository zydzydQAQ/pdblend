# SLO 容量边界只读判定

`pdblend.bench.slo_capacity` 读取已完成、不可变的 calibration/tuning 回执；不启动 GPU、不入队、不修改结果，也不提升 profile 资格。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/pdblend/.venv/bin/python \
  -m pdblend.bench.slo_capacity --ledger /absolute/path/capacity-ledger.json
```

ledger 使用以下结构。`point` 应绑定准备阶段冻结的期望 point；`receipt` 绑定实际执行窗口的回执。下面的路径与 SHA 为格式示例，须替换成真实绑定。

```json
{
  "schema": "pdblend-slo-capacity-ledger/v1",
  "config": {
    "series_id": "Qwen2.5-7B-Instruct/pdblend/alpaca",
    "slo_ttft_s": 1.0,
    "slo_tpot_s": 0.1,
    "required_repeats": 3,
    "min_requests_per_trial": 100,
    "relative_tolerance": 0.05,
    "initial_rate": 1.0
  },
  "series": {
    "model_id": "Qwen2.5-7B-Instruct",
    "system": "pdblend",
    "dataset": "alpaca"
  },
  "trials": [
    {
      "repeat_id": "seed-8801",
      "point": {"path": "/absolute/path/point.json", "sha256": "REAL_POINT_SHA256"},
      "receipt": {"path": "/absolute/path/receipt.json", "sha256": "REAL_RECEIPT_SHA256"}
    }
  ],
  "multipliers": [0.25, 0.5, 0.75, 1.0, 1.1]
}
```

输出到 stdout，包含 `state.action`、`state.next_rate_scale`、通过的下界、失败的上界，以及收敛后才产生的 `frozen_evaluation_grid`。空 `trials` 可获取初始采样点。旧 audit 仅保存 raw-reference mapping 的 digest 时，可在 trial 中显式提供 `raw_refs`；它必须与原 audit 的 digest 完全一致。

每个 rate 至少三次独立重复，每次至少 100 个请求。完整观察必须满足 100% 请求成功、联合 SLO 达标率至少 90%、TTFT/TPOT P99 均不超过冻结 SLO。一致失败构成上界，一致通过构成下界；相对区间宽度不超过 5% 才能收敛。缺失测量不构成失败上界；已知 pass/fail 冲突和非单调边界要求调查。任何 incomplete 证据继续阻塞收敛，追加成功重复不会消除旧缺口；当前没有自动 supersession 或删除失败证据的接口。

读取器校验期望 point、窗口 artifacts、raw-reference mapping、测量 gates、reset/drain、八卡身份、trace split/SLO/请求分母。每条 trace 必须能从 `capacity_workloads` 冻结的 workload family、corpus、独立 anchor 和已知版本生成器完整重放；point 与 inputs 都必须绑定同一 `capacity_workload_family`。point 的 scale、seed、repeat_id、SLO、服务时长、protocol 和 output_workload 必须一致，系统必须属于该 family。全系列固定 family SHA、源码、配置、profile 和八个 GPU UUID，不能只用相同服务时长或 `rate_rps/scale` 隐藏不同输入。重复身份使用 family 的三个固定 seed；同一原始窗口换路径或重复标签仍不能算独立重复。测量有效性与 profile 资格分离，报告始终保留 `formal_eligible=false`。

跨系统只读比较入口：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /home/pdblend/.venv/bin/python \
  -m pdblend.bench.capacity_comparison \
  --ledger mixed=/absolute/mixed-ledger.json \
  --ledger distserve=/absolute/distserve-ledger.json \
  --ledger ecoserve=/absolute/ecoserve-ledger.json \
  --ledger dynamollm=/absolute/dynamollm-ledger.json \
  --ledger pdblend=/absolute/pdblend-ledger.json
```

只有五个系统来自同一 workload family、模型、数据集、八卡硬件和 SLO 协议，所有区间都收敛，且 PDblend 的通过下界严格大于四个 baseline 的失败上界，才输出 `observed_boundary_lead=true`。区间相接／重叠、缺系统、缺失败上界、未收敛或缺有效测量均不能宣布领先。这是重复 calibration/tuning 的观测边界比较，不是精确容量或正式资格，不产生 GPU 执行／队列写入。

实际入队仍需独立 preparation 接口：冻结 calibration/tuning 轨迹并保持共同 rate 锚点，按 state 建立缺失重复或倍增／减半／二分的新 point，执行后把原始回执追加到新版本 ledger。收敛后，应将固定网格应用到独立 evaluation 轨迹。现有 `comparison_campaign.prepare` 和 `independent_dispatch.validate` 使用 evaluation；现有 recovery A/B 也是 evaluation，不能作为容量选点输入。应新增显式容量实验 scope，不能重标旧回执或放宽现有 evaluation 校验。
