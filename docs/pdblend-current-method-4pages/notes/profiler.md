# Profiler：实现核对与论文用法

本笔记只读核对当前工作区代码，不代表本次重新执行 GPU 实验，也没有假定某个未给出的 profile 文件已经通过全部门槛。论文应把“提供了该实现”与“当前实验已有完整合格数据”分开。

## 1. 建议采用的主叙事

Profiler 是离线的 collection → calibration → query 三层：按模型、TP/PP、频率和实际请求形状采集时延/功率，同时记录 KV 容量、静态功率、唤醒/调频耗时和 P/D 交接；仅以 training 拟合，再冻结模型，以独立 holdout 审核，在线通过有覆盖域的统一接口读取。论文可用 native timing 的简洁公式讲模型，说明历史模型仍保留兼容接口，而不把历史 affine 公式冒充唯一当前模型。

## 2. 可核对事实及代码锚点

| 事实 | 当前代码位置（行号） |
| --- | --- |
| profile identity 含系统、模型、引擎、硬件、TP、PP、role、workload；原始记录绑定模型/tokenizer 和环境 | `src/pdblend/profile/collection/profiler.py:103-160` |
| 历史 decode collection 至少 3 次、settle ≥2 s、measure ≥5 s；完成所有 prompt prefill 后再等待 16 个背景 token | `collection/profiler.py:123-127,271-295` |
| 连续 decode 窗口提取实际 token 增量、有效 context、步时和各卡功率，取重复中位数；不会把名义 prompt 直接当稳态 context | `collection/profiler.py:303-369` |
| 历史 Mixed 是在已有 decode 上注入 prefill probe，记录 TTFT、重叠 decode 最大 stall；不等价于一个完整 Mixed 功率模型 | `collection/profiler.py:438-468,829-833` |
| native timing 读取各 rank CUDA runner events，要求 TP rank 请求/形状序列一致，logical step 取所有 rank 的最大 GPU elapsed；decode 必须每请求调度 1 token，异构 context 被排除在这个 scalar-context 模型外 | `collection/native_timing_audit.py:20-49` |
| native timing 检查频率实测、采样时窗、客户端完整性及 native drain；训练/holdout window 与 shape 必须不相交 | `collection/native_timing_audit.py:52-86,89-97` |
| native timing prefill 用 [1,n,n²]；decode 用 [1,B,Bc]，内部 n/c 除 8192，时延单位 ms；NNLS 非负拟合；decode 覆盖域是实际训练点 convex hull | `collection/native_timing_audit.py:98-128` |
| native timing scalar 查询 exact frequency；超出实际区间/convex hull 报 missing_profile；prefill marginal 去掉 intercept | `query/native_timing.py:33-60` |
| native power training 3 次重复取均值；context 节点为实际连续窗口的 time-weighted context；低 batch 1/2/3 要求精确节点，B≥4 可在共同覆盖 context 内线性插值；频率不插值 | `query/native_power_components.py:17-78`; `collection/native_power_audit.py:54-68,154-173` |
| native power 的独立 holdout 在 candidate freeze 后采，重放时训练重拟合结果必须逐字等于冻结 candidate | `query/native_power_components.py:81-133` |
| 当前 native prefill 功率是 request-cycle mean，而非 active CUDA kernel power；因此 `prefill_power_w` 明确拒绝将它与 CUDA prefill time 相乘 | `query/native_power_components.py:29-32`; `query/native_composition.py:110-114` |
| 完整周期/布局功率用独立组件，依赖拓扑、负载、数据集、频率、采样周期等精确域；不能冒充任意配置的通用 energy model | `query/native_cycle_model.py:71-79,96-106`; `query/native_layout_model.py:43-64,84-96` |
| runtime contract：`prefill_seconds`, `prefill_marginal_seconds`, `step_seconds`, `decode_power_w`, coverage, static/wake, transfer；启用 PD/DVFS 时需相应 runtime 组件 | `query/runtime.py:9-41` |
| version 显式选择，禁止 latest；formal usage 重新检查资格，而非凭调用者写一个标记 | `query/versions.py:299-364`; `query/native_composition.py:147-228` |

上表未加 `src/pdblend/profile/` 前缀的代码位置均相对此目录。

## 3. KV transfer 与 transition 口径（供在线算法章节）

1. 历史基础 `PerfModel.transfer_seconds(n)` 是 `fixed + n * kv_bytes_per_token / bandwidth`，见 `query/model.py:198-200`。它是可拟合的固定项加字节量项，但训练目标不应称为纯 NCCL copy latency。
2. `collection/profiler.py:607-631` 对相同 prompt 分别跑 M 和 PD，以 carry-first-token 的 second-output overhead 标定。新 native composition 直接在 512/2048/7168 三个节点的 holdout-qualified training 中位数上构建 CurveIndex，见 `query/native_composition.py:91-108,127`；不是同一个 affine formula。
3. `collection/native_handoff.py:38-71` 给出更清楚的端点间隔 `H = t(D首token) - t(P首token)`，拆为 `client dispatch + D submit-to-first`；该值包含 HTTP、调度、KV 处理和首个 D step，明确 `physical_copy_time=False`。
4. 该 handoff 的历史提取 candidate 仍为 diagnostic；`native_handoff.py:134-160,217-226` 明确 `component_qualified=False`、没有修改 router threshold、短输出/跨频率的独立实测尚需覆盖。不能把这些 helper 的存在写成已部署了一个完整新 handoff predictor。
5. static/wake/clock/transfer runtime 各项有三次训练、一份 holdout，中位数预测；见 `collection/native_runtime_audit.py:267-299`。`query/native_composition.py:107` 的频率切换耗时取两个方向中较大者。
6. runtime raw 的能量是八卡绝对功率积分，不是增量 transition energy；`collection/native_runtime_audit.py:261-265` 明确 `incremental_energy_j=None`。不能宣称当前 profiler 已测出精确的 TP 重配增量能耗。

## 4. 适合四页方法正文的三个公式

用 `n` 表示 prompt/chunk token 数，`B` 表示 decode batch，`c` 表示每请求上下文，`f` 表示 GPU 频率；将代码内部的归一化和 ms/s 换算吸收入系数：

```latex
\begin{align}
T_p(n,f)&=a_f+b_fn+d_fn^2,\\
T_d(B,c,f)&=u_f+v_fB+w_fBc.
\end{align}
```

这些是当前 **native timing** 模型；历史 `PerfModel` decode 还可含 `B²` 或 split-B1/有界表修正，正文无需罗列全部兼容实现。NNLS 对上述系数施加非负约束，独立 holdout 评估相对误差。

```latex
\bar P={1\over t_1-t_0}\int_{t_0}^{t_1}\sum_{g\in G}P_g(t)\,dt,
\qquad E=\bar P(t_1-t_0).
```

此式定义测量而非假定任何阶段功率可任意组合；功率与时间必须用同一个测量边界。

```latex
H=t_{D,1}-t_{P,1}=T_{\mathrm{dispatch}}+T_{\mathrm{D\ submit\to first}}.
```

这第三式可放在线算法节，用文字说明完整交接间隔需要同时保护首个 inter-token gap；不要把 H 与 `transfer_seconds` 的历史差值口径混写成已经完全一致。

## 5. 不宜写进正文的过度声明

- 不要声称“profile 都是 O(1) query”：compiled 历史索引有 O(1) 契约，但 native timing 检查凸包时做 halfspace scan，native power 也做节点扫描（`query/native_timing.py:53-55,69-73`; `query/native_power_components.py:53-78`）。稳妥写法是离线拟合、内存中查询、无在线 GPU profiling。
- 不要把任何采样完成等同 formal qualified：`collection/profiler.py:686-706` 直接保留 independent holdout/correctness/campaign 缺口。
- 不要把 model/TP 的离线 profile 覆盖解释为已经对任意在线 TP weight change 做独立校准。

## 6. 论文草稿

可直接参照同目录 `../sections/profiler-draft.tex`。其正文约 740 个中文字（另有英文术语）、三个展示公式。若全稿超页，可删“历史 profile”句和第二段最后一句；必须保留“测量边界一致”和“覆盖域外拒绝预测”两点。
