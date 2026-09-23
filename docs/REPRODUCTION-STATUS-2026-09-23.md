# 三模型五系统复现：代码发布说明

本次 GitHub 提交保存截至 2026-09-23 的实现、CPU 回归测试、运行脚本和方法文档。
它不是正式实验结果发布，也不表示校准、所有自动机制或整机能耗比较已经完成。

当前执行目标为 Qwen2.5-7B、14B、32B × Alpaca、ShareGPT、LongBench × x0.5 ×
Mixed、DistServe、DynamoLLM、EcoServe、PDBlend，共 45 个短窗口比较点；统一使用 seed 701、
每点 300 秒。DynamoLLM 原控制周期的长机制验收，以及 PDBlend 四种 TP 模式的相同长轨迹
消融，单独组织。300 秒的短窗口结果不能用于宣称动态 TP 节能收益。

本轮环境为 vLLM 0.10.1.1、Torch 2.7.1、CUDA 12.8.1、8×L20。各系统保持独立的策略、
profile、预测器和成本模型；公共部分负责 GPU 租约、请求轨迹、计量、身份校验与清理。
首批 DistServe 只允许已通过资格检查的 symmetric TP、PP1，不称为原论文的完整复现。
PDBlend 保持 PP1，按固定 TP、离线 TP、异构常驻池、慢时间尺度重分片依次验证。

功能请求成功、CPU 搜索通过、局部校准通过与正式可比较是不同状态。正式点仍须满足 profile
覆盖域和 provenance、真实自动动作、请求与 KV 清理、数据集 SLO、独占八卡总能耗归属等门槛。
任何缺口保持 `inconclusive`，不进入正式排名；代码中存在某个 runner 也不代表对应 GPU 验收通过。

## 当前实现与证据边界

截至 18:44，已完成 7B Mixed 独立 calibration/tuning 倍率标定；三个数据集 x0.5 的
seed701、300 秒评价轨迹已冻结，首批配置共 45 点，尚无正式通过点。独立 dispatcher
按系统选择各自 runner，并在缺少校准、覆盖域、动作或计量资格时拒绝提升为正式结果。

7B 的 TP1+TP2 三卡常驻池已完成 195/195 请求，两个池分别服务 57/138 个请求。
32B 常驻池首次启动的 KV rank 端口重叠已修复，重试已入队；修复不能替代 GPU 重试。
32B 固定/离线 TP 与强制 P/D 功能窗口完成，但后者仍须独立的 KV/output golden 检查。
EcoServe 7B/14B 自动 macro 任务结束，32B 任务执行中；各自动动作与 CSV 校准仍分别核验。

六成员增量 profile 使用 2+1+1+2+1+1 卡分组。每次成员退出都在窗口边界暂停、清理并重新
检验并行干扰；当前不允许任意新成员加入正在测量的波次，所以提前释放的卡可能暂时空闲。
同一模型尽量保持常驻。14B TP1 复用 36 个训练点，只补 24 个长域独立 holdout；短输入
分段采样目前是 experimental，不因采样成功就取得正式资格。Dynamo 保持独立采样与预测器。

## 保存与复现范围

- 提交源码、测试及小型独立 fixtures、配置和运行脚本，以及可编辑文档与最终阅读版本。
- 新增的 `results/2026-09-22/three-model/`、`results/2026-09-23/`、prepared 语料、模型、
  predictor 权重和冻结源码副本保留在实验机。已有 Git 历史中的结果不因此成为正式新栈证据。
- 不提交重复文档 ZIP、构建缓存、逐页预览和 worker 日志。HTML 方法文档所需的两个 Noto CJK
  字体是本地构建依赖，需另行放入 `docs/pdblend-method/assets/`；许可证随源码保留。
  LaTeX 文档使用系统 TeX 字体，其依赖见对应 README。

现场状态由 `scripts/2026-09-23_live_status.py` 根据队列与 receipt 更新。README 中的本地结果
链接、RESTART 与 MIGRATION 的路径是实验机操作入口；仅克隆 Git 仓库不会取得这些原始证据。
复现实验前必须恢复相应模型、数据和 profile 的带 hash 迁移包，不能以默认 7B 或其他系统
profile 补齐缺失项。

CPU 检查：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q \
  -m 'not gpu and not historical' tests/pdblend tests/independent_baselines
```

需要本地历史资产的测试和 GPU 验收另行执行；CPU 测试结果不能替代实验 receipt。
