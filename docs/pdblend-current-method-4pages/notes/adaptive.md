# Adaptive pool configuration：代码依据与写作边界

本节根据 2026-09-24 工作区 `src/pdblend` 当前代码。以下行号均为本次读取的实文件行号。已确认用户的 **TP weight change** 应解释为两个常驻异构 TP pool 的**请求流量权重改变**；不能写成已经实现在线模型权重重分片。

## 建议的章节逻辑

统一优化目标置于本节开头，随后严格分为 frequency control、role change、TP weight change。这样不需要第四小节，也能避免三处重复解释时延约束和切换成本。

以 `c` 表示一个配置（各角色数量、频率、PD/M 阈值），`H` 是保持窗口，`P(c)` 是当前预测负载下整池功率，`E(c_0,c)` 是转换的额外能量：

```latex
\begin{equation}
 c^*=\arg\min_{c\ \mathrm{feasible}}
 \{H P(c)+E(c_0,c)\}.
\end{equation}
```

可行条件包括 TTFT、TPOT 安全余量、容量、KV、模型覆盖以及已有请求归属约束。现有配置可行时，仅在新目标低于 `(1-margin) H P(c_0)` 时切换。默认 `H=60 s`、`margin=0.03`，避免将这些可配置默认值误当成所有实验参数。

## 精确证据

| 结论 | 实现位置 |
|---|---|
| 配置包含 P/D/M 数量、频率、阈值，以及 idle/L1/off 停车状态 | `planner/pool.py:1–5,19–23,34–55` |
| 默认摊销 60 秒、相对节省阈值 3% | `planner/pool.py:117–148` |
| 所有可行候选逐一加自己的转换成本，不能先选最低稳态功率再决定是否切换 | `planner/pool.py:722–767` |
| 转换成本优先精确目录；qualified-only 缺证据时返回无穷成本；普通模式允许标记过的旧估计 | `planner/pool.py:677–720` |
| 转换目录精确匹配源/目标配置、模型、TP/PP 和 profile；使用配对测量增量能量 | `planner/transitions.py:145–213` |
| 对各 role 已测离散频点枚举；P 默认偏好 2100/2520 MHz；不是局部梯度或邻域搜索 | `planner/pool.py:22,656–670` |
| 完整可行性包含角色合法性、SLO、M 干扰风险、已有 backlog 所属分支保留；停车功率加入总功率 | `planner/pool.py:523–602` |
| 控制器默认普通周期 10 秒、快速 tick 1 秒，属于可覆盖默认值 | `online/controller.py:30–45` |
| budget-aware Shield 将受压 PD prefill 映射到 P、decode 映射到 D，M 路径映射到 M；只提高这些角色频率 | `online/shield.py:223–233,337–374` |
| 短暂同步 token 间隙仅触发一次有界调频探测，不自动证明要增加容量 | `online/shield.py:213–221,267–289` |
| deadline safety 专门提高 M，按频率升序找最小能消除风险的频点；找不到则取上限 | `online/controller.py:526–545` |
| 事件驱动 deadline worker 至多约每 50 ms 评估；普通降频须连续安全周期解除 floor | `online/controller.py:574–605` |
| 每个实例的所有 TP GPU 按同一 MHz 写时钟；已知频率相同不重复写；预期 role/generation/transition 复核 | `online/controller.py:173–208` |
| GPU 调节在工作线程执行，按 physical GPU ID 获取线程锁和跨进程文件锁 | `online/gpu_actions.py:17–39` |
| 角色映射保留能够保留的已有角色；剩余实例按驻留状态/负载选择，减少不必要变化 | `planner/pool.py:857–884` |
| **ACTIVE→ACTIVE 角色变化不 drain**：直接调频、发布新 role，已有请求仍绑定原实例 | `online/controller.py:436–443` |
| **ACTIVE→parked 才 drain**：先关闭准入；proxy prefill tokens/sequences 清零；native drain；再停进程/复位时钟/停车 | `online/controller.py:290–321` |
| drain 的完整核查包括所有 ranks、代次、原生队列、已保留 KV、传输及 block 归还 | `online/native_control.py:15–49,74–80` |
| 唤醒 off 实例需要 start+ready，L1 要 unpark；调频及 native resume 之后再发布准入 | `online/controller.py:324–338` |
| 多个变更任务并行，等待全部完成；失败则相关实例停止准入，不假装整体成功 | `online/controller.py:353–396` |
| 最小保持、冷却、多次独立预测投票制约普通降配；安全恢复可以绕过 | `online/controller.py:769–806` |
| 双常驻 TP 池必须为不同 TP、相同模型/引擎/硬件版本、GPU 不重叠，并各自有独立 profile | `planner/topology.py:185–226` |
| 权重网格默认 0.0:0.1:1.0，加入当前份额及最近实际分流比例；每个份额调用每个池的内层规划 | `planner/topology.py:193,235–240,282–328` |
| 每个池的 forecast 只缩放未来到达量；已有 backlog 按原 pool 保留 | `planner/forecast.py:101–108` |
| 即使池的新流量份额为 0，全部驻留副本与未使用 GPU 仍计费 | `planner/topology.py:301–306,330–336` |
| joint planner 的当前配置独立重新评价并施加节省阈值 | `planner/topology.py:339–363` |
| coordinator 收集实际分流，受 Shield/保持策略约束；所有池 role/clock/admission 就绪后才发布新份额 | `online/resident_control.py:44–120` |
| weighted deficit：在当前可行池中最大化 `target_share*(dispatched_total+1)-pool_count` | `online/router.py:867–881` |
| 份额发布、实际分配计数以及 feedback 重置 | `online/router.py:741–755,893–895` |
| `slow_reshard_tp` 当前在线入口显式拒绝，缺 GPU-qualified native transaction backend | `online/tp_runtime.py:67–76` |

## 容易误写的边界

1. 不能把 **ACTIVE 角色改变** 与 **停车/停止/拓扑重建** 都叫排空切换。当前角色是新请求路由策略；在途请求仍在原实例上执行。
2. 不能将 `frequency_domain` 解释为硬件共享时钟域。`profile/collection/native_frequency_domain.py` 的该字段绑定的是 profile 测量频率集合；物理 GPU 的控制互斥是另一机制。当前并没有任意重叠 TP pool 的共享 DVFS 联合优化，resident 布局反而要求 GPU 不重叠。
3. 常规频率规划是离散全枚举；“局部调整”只能指按受压 role 做快速动作、只对变化实例写硬件，不要凭空引入邻域搜索。
4. budget-aware Shield、deadline safety、joint resident 等为显式能力/配置路径，不保证每个入口默认启用。文稿用“启用……时”，减少默认策略误断。
5. `native_layout.py:17–28,73–95` 是单独的 canonical 32B TP2/M4 整机功率修订，不能描述成全部 P/D/parking/resident 候选都已由整机实测覆盖；那里甚至拒绝 backlog。
6. `controller.py:428` 日志写 `native_drain_resume_and_proxy_publication` 不能替代逐分支代码；不要据日志推断 ACTIVE→ACTIVE 也 drain。

## 建议的统筹伪代码（忠于现有机制，避免承诺固定的三级串行搜索）

```latex
\begin{algorithm}[t]
\caption{自适应池配置}
\begin{algorithmic}[1]
\Require 当前配置，绑定的 profile，请求历史与在途工作
\State 更新到达预测、请求长度及原池 backlog
\If{出现时延风险}
  \State 执行受压角色调频；持续风险下恢复容量
\Else
  \State 枚举配置；启用联合 TP 时外层同时枚举流量份额
  \State 删除违反覆盖、容量、KV 或 SLO 的候选
  \State 按保持窗口能量加转换成本选择候选
  \State 检查节省阈值、保持时间与降配确认
  \State 保留既有请求归属，执行角色、频率和停车动作
  \State 等待全部池就绪后发布目标流量份额
\EndIf
\State 记录物理状态和实际分流，作为下一轮反馈
\end{algorithmic}
\end{algorithm}
```

该算法是论文层面的控制汇总，实际普通 Controller 会先 ordinary gating 再套 Shield；ResidentCoordinator 有 Shield guard 并将快环留给调用方，不能宣称整个仓库只有一条固定控制函数。

## 论文草稿

见 `../sections/adaptive-draft.tex`。正文采用三个子节，以分流权重解释 TP weight change，模型重分片边界只留一句。
