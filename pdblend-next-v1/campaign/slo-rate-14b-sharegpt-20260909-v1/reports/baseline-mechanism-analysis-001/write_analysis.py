"""Combine quantitative and mechanism evidence without changing the experiment."""
import csv
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
names={'mixed':'Mixed','distserve':'DistServe','ecoserve':'EcoServe','dynamollm':'DynamoLLM'}
stats=list(csv.DictReader((HERE/'quantitative.csv').open()))
observations=list(csv.DictReader((ROOT/'reports/final-001/observations.csv').open()))
power=json.loads((HERE/'power-mechanisms-evidence.json').read_text())
latency=list(csv.DictReader((HERE/'latency-mechanisms.csv').open()))
assert power['passed']
assert json.loads((HERE/'evidence.json').read_text())['passed']
assert json.loads((HERE/'implementation-evidence.json').read_text())['all_runtime_config_feature_sets_constant_within_node_system']

lines=['# PDBlend 相对基线的能耗、SLO attainment 与机制分析','',
'本次最稳固的结论是：PDBlend 在已测同机、同 rate 点上的八卡能耗和每个达标请求能耗均低于四个基线；SLO attainment 的相对优势集中在 DynamoLLM。相对 Mixed、DistServe、EcoServe，部分负载下达标率相同且能耗更低，接近 PDBlend 边界时则出现服务质量取舍。现有证据支持“较少常驻实例、带延迟约束的路由和频率控制”这一整套配置的能效优势，尚不能把全部收益归给单个算法。','',
'**统计口径。**共 18 个常规同机、同 rate 条件：A 0.5× 有 4 个、B 旧 1× 参考有 6 个、C 2× 有 8 个；每个条件分别与四基线比较，形成 72 对。三个 PDBlend 边界确认只作单列证据，不与首测混合、不取最佳值。主能耗是全部八卡在 100 秒到达加实际排空及控制尾段的积分；没有只计 PDBlend 两张服务卡。达标分母包括全部请求，两个有效超时仍算失败；每个达标请求能耗为 E/good，有效吞吐为 good/实测主窗口秒数。所有比较都在同机、同 trace、同 SLO 下进行。','',
'**已测达标点的定量结果。**以下主表限定 PDBlend 首测达标率 ≥90% 的 15 个条件（A3、B5、C7），避免用已经失守的点说明合格运行收益。范围是逐点同机比率的最小至最大值，不是跨机总能耗之比或平均提升。“高/同/低”是 PDBlend 相对基线的达标率点数；单种子不能据点数检验统计显著性。','',
'| 基线 | 八卡能耗降低范围 | 每个达标请求能耗降低范围 | 达标率差范围（百分点） | 达标率高 / 同 / 低的点数 |',
'|---|---:|---:|---:|---:|']
for system,name in names.items():
    selected=[r for r in stats if r['baseline']==system and r['subset']=='PDB_at_least_90_regular']
    assert sum(int(r['paired_points']) for r in selected)==15
    def low(k):return min(float(r[k+'_min']) for r in selected)
    def high(k):return max(float(r[k+'_max']) for r in selected)
    counts=[sum(int(r[k]) for r in selected) for k in ['slo_better_and_energy_lower_count','slo_equal_and_energy_lower_count','slo_worse_count']]
    lines.append(f"| {name} | {low('energy_reduction_pct'):.2f}%–{high('energy_reduction_pct'):.2f}% | {low('Jgood_reduction_pct'):.2f}%–{high('Jgood_reduction_pct'):.2f}% | {low('slo_delta_pp'):+.2f} 至 {high('slo_delta_pp'):+.2f} | {' / '.join(map(str,counts))} |")
lines += ['',
'在全部 18 个常规条件中，PDBlend 相对 DynamoLLM 的达标率为 11 点更高、7 点相同、0 点更低，同时 18 点能耗和 E/good 均更低。相对另外三者没有达标率更高的常规点，优势是同等或较低达标率下的能效，而不是普遍的延迟或容量领先。完整含边界的区间及逐 host 结果保留在 quantitative.csv；主表筛选不隐藏边界失守。','',
'**最能说明同等质量节能的例子：C 机 2×、1.50 rps。**五系统重放相同 122 个请求；PDBlend、DistServe、EcoServe 均有 121 个达标。','',
'| 系统 | 达标率 | 八卡能耗 (kJ) | 每个达标请求 (J) | goodput (req/s) | 平均八卡功率 (W) | 主窗口 (s) |',
'|---|---:|---:|---:|---:|---:|---:|']
case=[]
for system,name in [('pdblend','PDBlend'),*names.items()]:
    r=next(r for r in power['rows'] if r['node']=='C' and r['rate_rps']==1.5 and r['repeat']==1 and r['system']==system)
    case.append(r)
    lines.append(f"| {name} | {r['slo_attainment']:.2%} | {r['energy_j']/1000:.3f} | {r['energy_per_good_request_j']:.2f} | {r['goodput_measurement_rps']:.5f} | {r['average_all8_power_w']:.2f} | {r['duration_s']:.2f} |")
lines += ['',
'该点 PDBlend 相对 DistServe、EcoServe 的达标率相同，八卡能耗及 E/good 分别低 32.54%、33.44%，goodput 分别相差 −0.41%、+0.75%。这些微小吞吐差只是本次观测。相对 Mixed，节能 61.01%，达标率低 0.82 个百分点；相对 DynamoLLM，节能 46.23%，达标率高 13.11 个百分点、goodput 高 31.70%、E/good 低 53.34%。相同达标数并不表示延迟分布完全相同。','',
'同等达标率的例子覆盖其他机器：A 0.25 rps 五系统均 100%，PDBlend 对四基线节能 27.70%–45.76%；B 0.75 rps 的 PDBlend、Mixed、DistServe、EcoServe 均 100%，PDBlend 对后三者分别节能 57.06%、29.59%、34.81%；B 1.25 rps 与 EcoServe 均 99.02%，PDBlend 节能 38.85%。这些都是同机配对结果。','',
f'![C 机 1.50 rps 的质量、能量、平均功率和测量时长]({HERE}/C-r1.5-energy-slo-mechanisms.png)','',
'功率图灰色部分是整段测量中采样利用率为零的 GPU 所消耗的平均功率，彩色部分为其他 GPU；它是按实测卡组分账，不是对“空闲损耗”的单因素因果估计。','',
'**原因一：较少常驻实例降低了整机功率基础值。**本次 PDBlend 为 GPU6/7 两个 mixed TP1 服务实例，GPU0–5 无服务实例。四基线均保留八个 TP1 模型副本；EcoServe 的 macro 缩减只移出调度成员，idle parking 只释放频率锁，没有卸载模型、释放原生引擎或给 GPU 断电。','',
'C 1.50 rps，PDBlend 六张未服务卡合计平均 188.38 W，各卡约 24.8–35.0 W；EcoServe 五张采样利用率为零的驻留卡合计 385.42 W，DistServe 对应五卡为 373.27 W。PDBlend 与 EcoServe 总平均功率差 302.85 W，其中按这两组“零采样利用率卡”分账的差为 197.04 W；两组卡的数量和物理身份不同，这只是会计分解，不能说卸载模型单独贡献了 197.04 W。负载集中到两实例还能以接近的整机工作吞吐完成同一 trace；合批利用率改善是合理解释之一，但没有同原生引擎和同批次预算的消融，不能单独量化。','',
'**原因二：PDBlend 确实执行了受延迟约束的频率选择。**候选域是 900/1500/2100 MHz；规划器先检查新请求 TTFT、单步 TPOT、已有请求余量、KV 和性能模型覆盖，再比较预计增量能耗，decode 阶段也继续检查各请求余量。C 1.50 rps 的硬件时钟采样中，GPU6/7 约 42.21%/35.19% 的主窗口处于 1500 MHz，其余主要在 2100 MHz。该占比来自按主窗口裁剪的采样保持估计，不是精确事件驻留时间。','',
'低频选择与较低功率相符，但本轮没有关闭 DVFS 的同布局对照，不能独立分配节能贡献。频率并不随负载或 SLO 倍率单调变化：C 1.75 rps 的 1500 MHz 比例反而更高，C 0.25 rps 则主要为 2100 MHz。基线 idle reset 后的 2520 MHz 读数是释放频率锁后的实际硬件状态，不是把服务候选域偷偷扩到 2520 MHz。','',
'**原因三：相对 DynamoLLM，失分差异主要发生在准入等待。**DynamoLLM 按输入和预测输出长度划分形状池，请求只能使用本类或逐维更大类别的合格实例；本次长输入、长输出类 LL 只有一个实例。代表点所有 DynamoLLM 的 TTFT-only 失分都集中在该 LL 实例。下面的等待均值只针对已完成但 TTFT 超标的请求，不能解释为全部请求的平均等待，更不是 GPU prefill kernel 时间。','',
'| C 机 rate | PDBlend TTFT-only 失分数 | DynamoLLM TTFT-only 失分数 | Dynamo 送入后端前等待均值 (s) | Dynamo 后端首 token 均值 (s) |',
'|---:|---:|---:|---:|---:|']
for rate in [1.5,1.75,2.0]:
    p=next(r for r in latency if r['host']=='C' and float(r['rate_rps'])==rate and r['system']=='pdblend')
    d=next(r for r in latency if r['host']=='C' and float(r['rate_rps'])==rate and r['system']=='dynamollm')
    lines.append(f"| {rate:.2f} | {p['completed_ttft_only']} | {d['completed_ttft_only']} | {float(d['ttft_miss_pre_forward_mean_s']):.3f} | {float(d['ttft_miss_backend_first_token_mean_s']):.3f} |")
lines += ['',
'C 1.50 rps 的 16 个 Dynamo TTFT-only 失分全在 LL/GPU0；它承接 56/122 个请求和 82,640 个输入 token，平均送入后端前等 13.099 秒，其中首次规划至进入动作阶段占 13.096 秒。该实例实测频率几乎一直为 2100 MHz，记录更支持形状池集中和准入等待造成的限制。PDBlend 两个 mixed 实例都可以接收这些请求形状，且该档仅 1 个 TTFT-only 失分。这支持其路由和准入组合减少了失分请求数；仍需取消形状池限制的同布局消融才能单独归因。','',
'Dynamo 在 C 1.50、1.75 rps 还各有 1 个 TPOT-only 失分。代表点中的 TPOT-only 主要对应同一条只有 3 个输出 token 的请求，不能以此概括整个解码阶段吞吐不足。PDBlend 自身同样存在准入等待，优势是这些代表点中失分请求更少，并非所有请求都完全没有排队。','',
'更低的平均功率与更短的测量尾段同时影响能耗，满足 E_PDB/E_baseline = (平均功率_PDB/平均功率_baseline) × (时长_PDB/时长_baseline)。C 1.50 rps 的分解如下；它是恒等式，不是互相独立的因果贡献。','',
'| 基线 | PDB 平均功率降低 | PDB 主窗口时长变化 | PDB 能耗降低 |','|---|---:|---:|---:|']
for s,n in names.items():
    r=next(r for r in power['pairs'] if r['node']=='C' and r['rate_rps']==1.5 and r['baseline']==s)
    lines.append(f"| {n} | {r['average_power_reduction_pct']:.2f}% | {r['duration_change_pct']:+.2f}% | {r['energy_reduction_pct']:.2f}% |")
lines += ['',
'**为什么不能写成“各项都比所有基线好”。**PDBlend 只用两实例的配置带来了更低功率，也有更小的服务容量和准入选择空间。A1.00、B1.50、C2.00 rps 的首次达标率分别为 83.95%、88.52%、81.10%，同档 Mixed、DistServe、EcoServe 全部仍 ≥90%。它们更高负载的容量没有继续测；本轮没有容量优于三者的证据。','',
'C 1.75 rps，PDBlend 对 EcoServe 节能 26.67%，但达标率低 6.25 个百分点、goodput 低 17.33%，按达标请求计的节能幅度为 21.78%；到 2.00 rps，该幅度只有 4.71%，PDBlend 达标率已经降至 81.10%，EcoServe 仍为 100%。PDBlend C2 的超时请求在送入后端前等了约 104.75 秒，也表明其当前资源配置和准入策略已经失守。不能用该点的较低总能耗声称可同质量替代 EcoServe。','',
'**归因范围与可用于论文的表述。**本次基线是仓库内冻结策略的机制复现，非官方系统部署。PDBlend 实际关闭了 PD 拆分、动态池和慢拓扑功能；2048 是单轮 batched-token 预算，模型上下文上限仍为 8192，基线启动预算为 8192；1.5 秒是空闲后恢复服务频率域的确认超时上限，不是固定睡眠或周期。不同常驻实例数、原生适配器、批次预算和频率控制同时变化，所以本轮支持完整配置的系统级效果。','',
'实际配置还保留了继承 startup8192 性能 profile、changed chunk profile 未认证的兼容性标记。本机输出/频率/排空资格和原始计量通过，不代表预算改变后的性能预测模型已经重新校准。这可能影响在线可行性判断，尚不能据此断定某个 SLO 失分就是 profile 误差造成。','',
'要分别量化机制贡献，需要在相同原生实现、实例数和 batched-token 预算下对比 DVFS 开关，再独立对比闲置模型保留/卸载、Dynamo 形状池路由/共享路由及 2048/8192 预算。本轮没有执行这些消融，因此不宣称某个机制贡献了多少百分比。A/C 与旧 B 的 SLO 倍率绑定不同物理机器；只有一个到达种子，没有独立种子置信区间。','',
'适合结果章节的结论是：在 PDBlend 达到 90% SLO 目标的已测工作点中，相对 Mixed、DistServe、EcoServe、DynamoLLM 的八卡能耗分别降低 44.54%–61.01%、26.26%–37.15%、26.67%–40.30%、28.42%–47.09%。PDBlend 对 DynamoLLM 的达标率不降低，最高提高 13.11 个百分点；相对另外三者，在部分点保持相同达标率，接近容量边界时需牺牲服务质量。原始记录与实现审查将优势关联到更少常驻实例带来的低基础功率、受延迟约束的频率/路由选择，以及相对 DynamoLLM 更少的长请求准入失分。','',
'可复核文件：','']
for filename,label in [('quantitative.csv','逐机器、基线、统计子集的定量区间'),('regular-pairs.csv','72 组常规配对比较'),('confirmation-pairs.csv','单列确认比较'),('power-mechanisms-evidence.json','62 次新测的功率、频率和时间分解'),('implementation-mechanisms.md','实际实现与关闭功能审查'),('implementation-evidence.json','93 个运行配置及控制记录索引'),('evidence.md','25 个代表点、2,955 个请求的失分机制'),('request-mechanisms.csv','逐请求等待和失分数据'),('quantitative-evidence.json','定量复算证据')]:
    assert (HERE/filename).exists()
    lines.append(f'- [{label}]({HERE/filename})')
(HERE/'ANALYSIS.md').write_text('\n'.join(lines)+'\n')
print(HERE/'ANALYSIS.md')
