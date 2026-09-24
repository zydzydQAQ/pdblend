# 当前在线路由代码审计（2026-09-24）

以 `/home/pdblend4/src/pdblend/online/router.py` 当前工作树为准。`src/pdblend/proxy/router.py` 只是此模块的兼容别名。本笔记不把历史方法文档当作实现事实。

## 1. 实际默认值需要分三层看

- `Router` 类本身：`pressure_gate_enabled=False`（router.py:97）；`slo_routing_enabled=False`（:101）。
- 当前 homogeneous PDblend benchmark runner：`pdblend_runtime_options.py:10–15` 的 `DEFAULTS` 设 `slo_routing=True`、`slo_routing_safety=0.85`、`handoff_floor=0`；`run.py:146–155` 调用 `configure_slo_routing`，所以用户通常运行的 PDblend 默认经过 SLO 路由层。非 PDblend baseline 的 `control_options` 强制关闭此层（runtime_options.py:44–48）。
- `pdblend` 策略本身并没有压力门：`online/policies.py:62–64` 未覆盖 `dynamic_m_floor=False`。显式实验 `pdblend_dominance`（:80–85）才设 `dynamic_m_floor=True`，进而在 `Controller.__post_init__`（controller.py:90–98）打开 pressure gate，最短 PD prompt 设置为 1024。不应把实验门控画成所有 PDblend 都启用的默认逻辑。
- Resident 多 TP pools 另有 `ResidentRouter`：`scoped_control_options` 强制关闭上述 SLO 路由，显式要求开启则报错（runtime_options.py:61–67）。该分支不是把 homogeneous SLO 层简单堆在外层。

## 2. 阈值分支与候选池

定义 `L=input_tokens`、`O=max_tokens`、`τ=pd_threshold_tokens`。

`_pool(role)`（router.py:380）只返回当前角色相符且 `accepting=True` 的实例。parked 不在 M/P/D 池；quarantined 实例会被置 `accepting=False`。这代表已经提交且当前可接新请求的角色表，不是控制器正在评估但尚未应用的计划。

兼容 PD 对集合（router.py:389–396）：

```
C_PD = {(p,d): p∈P_pool, d∈D_pool, p != d,
                 pp_p = 1,
                 (tp, pp, pool_id, generation, model_id)_p
                   = (tp, pp, pool_id, generation, model_id)_d}
```

因为目前 P2P KV 不支持 shape remap，P/D 必须成对兼容；不能选完 P 再随意挑不同拓扑的 D。`profile_key` 不参与这里的身份相等比较。

`choose(L)`（router.py:403–421）：

1. 标准：`long = (L >= τ)`；显式实验 pressure gate 模式：`long=(L>=max(1024,τ))`，1024 也可由配置改写。
2. 有兼容 PD 对，且（没有接受新请求的 M 或 long）→选 PD。
3. 否则有 M→选 M。
4. 否则有兼容 PD→选 PD；无可用路径→None。

关键边界：没有 M 时短请求也走 PD；没有完整兼容 P/D 对时长请求可以走 M。阈值比较为 `>=`。

`candidates(L)`（router.py:423–430）先调用 choose 决定分支，再枚举这个分支的全部实例或全部兼容对。其返回不是一个提前锁定的具体实例。因此当前默认 SLO 层会有机会覆盖 choose 的负载式首选实例。

## 3. 没开 SLO 层时，prefill/decode 如何 assign

计数记号：

- `U_i=inflight_prefill_tokens`：已提交但尚未看到第一个输出 token 的请求，其输入 token 数之和。
- `N_i=inflight_seqs`：已提交但尚未完成的 sequence 数；PD 时该计数记在已经选好的 D 上，包括仍在 P 等待/执行 prefill 的请求，不等同于 D 的实际 running batch。

M：`m*=argmin_m (N_m,U_m)`，prefill_instance=decode_instance=m*（router.py:386–387,416–418）。JSQ 主序；同队列数再看待 prefill token。

PD：`(p*,d*)=argmin_(p,d)∈C_PD (U_p,N_d,U_d,(p,d))`（router.py:398–401）。先平衡 P 上未首 token 的输入工作量，再平衡 D 已分配 sequences；是在兼容对集合上做词典序选择，不是两个互相独立的 argmin。对通常纯 D 实例，U_d 一般为零。

没有预测模型时，这些规则不直接检查硬 KV capacity；进入默认 SLO 层后才有下面的容量过滤。

## 4. 当前 homogeneous 默认 SLO 层：真正最终 assignment

入口 `_slo_route_choice(L,O)`（router.py:227–282），仅在 dispatch 未提供显式 choice 且 enabled=True 时执行（:435–443）。

令 C0=阈值所选分支的所有候选。若 C0 是 PD，额外建立 CM=所有 M 备选；若 C0 是 M，不建立反方向 PD 备选。令 A0=C0 中容量通过的候选；G0=A0 中预测可用且 safe 的候选；GM=CM 中容量通过、预测可用且 safe 的候选。

最终决策严格为：

1. 若原分支 PD 且 G0 为空且 GM 非空：选 `argmin_(c∈GM)(TTFT_hat(c),c)`，新请求 PD→M spillover。
2. 否则若 G0 非空：选 `argmin_(c∈G0)(TTFT_hat(c),c)`。原始 M 和原始 PD 都如此；这可以覆盖 legacy 的 JSQ 或 P/D 负载排序。
3. 否则若 A0 非空：保留容量可接受的历史分支。legacy preferred 若仍属于 A0 就选它；否则选 `argmin_(c∈A0)(U_prefill,N_decode,c)`。记录 `legacy_capacity_fallback_unproven_slo`。
4. 否则拒绝：`no_capacity_or_proven_alternative`，dispatch 返回 None，HTTP 层返回 503。

必须正确区分：“预测超 SLO/缺模型”不等于一律拒绝。只要原分支还有通过容量的路径，就允许带 unproven 标记回退。但是仅有容量通过且未证明安全的 M 备选不足以授权 PD→M。

如果 C0 一开始为空，`original_pd=False`、CM 为空，直接拒绝。如果 C0 为 PD 但所有 PD 均容量不通过，只有可证明安全的 M 才能接手；没有这种 M 就拒绝。

当前不会因为 M 过载就逐请求反向从 M spillover 到 PD；是否改变阈值和池数由规划器/控制器负责。

## 5. 容量、安全与时间预测

硬容量过滤 `_route_capacity`（router.py:132–153）：

- `L>0` 且 `L+O<=max_model_len`。
- 模型给出有限正的 KV capacity。
- 路径每个端点都接受新请求且不在 quarantine。
- 路径每个端点的已拥有请求 `Σ(L_i+O_i)+L+O<=KV capacity`。采用完整输出上限做保留量，P/D 都检查。

`max_num_seqs` 不是硬 admission 排队长度；它只限制下面预测的 running-batch 覆盖。超过它会预测不可用，而容量足够的历史路径仍可 fallback（router.py:148–150,168–169；测试 test_slo_routing_recovery.py:142–151）。

`_slo_route_prediction`（router.py:155–225）：

```
Q_P = Σ_{r.prefill_instance=P 且 r.first_token_s=None} prefill_seconds(L_r,f_P)
T_pf = prefill_seconds(L,f_P)
B_D = #{r.decode_instance=D} + 1
C_D = max(L+O, max_{r.decode_instance=D}(L_r+O_r))
s = step_seconds(B_D,C_D,f_D)
```

当前频率必须在 profile 中；decode shape 必须 `decode_supported`；未知/越界不能授权新路径。这里 Q 是按请求逐项模型预测求和，不把排队总 token 当成一个超长 prompt。

M：`TTFT_hat=Q_P+T_pf+s`；`TPOT_hat=s`（O>1），否则 0。

PD：`TTFT_hat=Q_P+T_pf`，因为第一个输出来自 P，KV transfer/第一个 D token 延迟位于第一个 inter-token gap，不在 TTFT。

- O=2：h 必须是显式测量 `pd_first_gap_seconds(L,O,f_P,f_D,B_D,C_D,p,d)`；h=max(测量值,handoff_floor)。该测量已经包含首个 D decode，不再加 s，也不再加 transfer。没有此测量时预测 unavailable。
- O>2：h=max(transfer_seconds(L)+s,handoff_floor)，标注 `legacy_transfer_plus_step`。
- O>1：`TPOT_hat=[h+(O−2)s]/(O−1)`。
- O=1：TPOT_hat=0；最终服务层转 P_ONLY。

safe（router.py:249–253）：

```
TTFT_hat <= safety * TTFT_SLO
TPOT_hat <= safety * TPOT_SLO
incumbent_ttft_safe and incumbent_tpot_safe
```

默认 safety=0.85。对 M，还要检查新 prefill 是否让现有请求超预算（:201–213）：

- 已拥有且还未首 token：`age_since_submit+Q_P+T_pf+s<=0.85*TTFT_SLO`。
- 已首 token且输出上限>1：`[age_since_first+T_pf+max(0,O_i−tokens_so_far_i)*s]/(O_i−1)<=0.85*TPOT_SLO`。

这层是模型驱动的 development_policy（router.py:109–113,126），不是 SLO 硬保证。观测/预测/边界应在图和文字中分清。

## 6. 已分配请求的生命周期

dispatch（router.py:452–470）同步保留 P 的 U+=L、D 的 N+=1，创建 RequestRecord，锁定 `path,prefill_instance,decode_instance,tp,pp,pool_id,generation`；通知 Forecaster listener 请求到达。D 在 prefill 开始前已 assign，不是 prefill 结束以后再找 D。

首 token（router.py:476–488）：U_P-=L；PD 时该首 token 来自 P；第二个 token 是第一次由 D 生成的 token，记录 `first_decode_token_s` 与 observed handoff。

完成（:490–518）：释放 D 的 N；若首 token 尚未到则也释放 P 的 U。取消/失败且 native ownership 不确定时不释放账户，而是隔离涉及端点，等待完整 native cancellation ACK 后恢复（:526–539）。不能画成失败后随意把在途请求从 PD 迁回 M。

server.py:100–111 的例外：如果路由 PD 但 O=1，server 在 engine submit 前把 D sequence accounting 转到所选 P，设置 `path=P_ONLY`、decode_instance=P，无远程 KV。理由 `single_token_no_remote_kv`。

## 7. 压力门只属显式实验策略，且与计划提交一致

set_pressure_state（router.py:313–351）高压触发：shield active 或 m_pressure>=enter 或 decode/prefill risk。默认 enter=.75、exit=.55、hold=30s、stable windows=2（Policy/Controller）。满足低压稳定窗口和 hold 到期才退出。

它更新的是计划侧的 pressure mode。choose 不直接读取 `_pd_pressure_active`；控制器必须先评估并提交匹配的 τ/角色方案，路由才改变，防止路由分流偏离规划模型。`test_router.py:51–70` 明确验证 pressure alone 不会把 2048 从 M 变 PD，必须 set_roles 提交新阈值。

实验 planner：pressure active=False 时不评估有 PD 的计划；M+PD 共存仅 τ=1024；纯 PD 需要 workload input p95>=2048（pool.py:481–487,594）。这不能套在标准 pdblend 的所有运行上。

## 8. 主图应标的条件

1. 分支阶段与实例赋值阶段分开：`Committed role table + τ`→阈值候选分支→容量/SLO层→最终 `(P,D)` 或 `(M,M)`。
2. M 是同一引擎本地 P+D；PD 是单独 P 实例和 D 实例的组合，不是一个名叫 PD 的单体实例。
3. 兼容对约束：同 TP/PP、pool、generation、model；当前 PP=1。
4. 绿色 P 产生第一个黄色 output token；灰色 KV/handoff 后 D 蓝色 token steps；TTFT 截止于 P 第一个 token，第一个长 TPOT gap 包含 handoff。
5. 当前 homogeneous 默认 SLO 开启：安全候选优先按预测 TTFT 最小；负载选择放在关闭 SLO/安全候选不可用的回退说明中。
6. `PD→M spillover = only new request + proven-safe M + no safe original PD`，不要画在途 KV 反向迁移箭头。
7. 无 M→PD 接全部；無 compatible PD→M；无容量原分支且无可证明安全替代→503；有容量但预测缺失/不安全→原分支 unproven fallback。
8. O=1→P_ONLY；O=2 必须有 measured first gap 才能证明 PD 安全。
9. 独立虚线框标实验 pressure gate / resident multi-TP，别混入标准默认流程。

## 9. 审计证据与本地验证

重点阅读：

- `/home/pdblend4/tests/pdblend/test_router.py:10`（M JSQ）、`:21`（阈值）、`:33`（PD-only）、`:51`（压力与计划一致）、`:82`（兼容对）。
- `/home/pdblend4/tests/pdblend/test_slo_routing_recovery.py:37`（新请求 spillover 不改旧 ownership）、`:54`（handoff 与输出长度）、`:72`（incumbent约束）、`:95`（缺模型 fallback）、`:142`（running batch 非硬 admission）、`:154`（硬容量）、`:176`（two-token coverage）、`:193`（first gap 不双计）。
- `/home/pdblend4/tests/pdblend/test_pdblend_runtime_options.py:142`（baseline禁用）、`:207`（resident禁用）。

尝试运行上述 3 个 test 文件：默认 PATH 无 python；python3 存在但未安装 pytest，因此没有声称本机 pytest 通过。未安装依赖、未改源代码。

另外使用纯标准库 + 当前 Router 的 Python3 定向 smoke，三个断言均通过：1）两个 M 都预测安全时，legacy JSQ 选 a、SLO min-TTFT 改选 b；2）PD 两个 D 的 TTFT 相等且均安全时，legacy least-seqs 选 d2、SLO 路由按 tuple 打破平局选 d1；3）原 PD 路径 KV 容量耗尽且 M 可证明安全时，新请求落到 M。这验证当前默认 SLO 的实例选择确实可能覆盖负载均衡次序；尤其 PD TTFT 模型不含 D queue，所以通过安全门后同 TTFT 的 D tie-break 是标识符序，并非最少 D sequences。
