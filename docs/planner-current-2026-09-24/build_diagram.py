#!/usr/bin/env python3
"""Code-native vector diagram; numeric labels read from the current CPU replay."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
FONT = HERE.parent / 'pdblend-method/assets/NotoSansCJKsc-Regular.otf'
FP = FontProperties(fname=str(FONT))
data = json.loads((HERE / 'example.json').read_text())
fig, ax = plt.subplots(figsize=(16, 20))
fig.subplots_adjust(0, 0, 1, 1)
ax.set(xlim=(0, 160), ylim=(0, 200))
ax.axis('off')
bg, ink, muted, line = '#f8fafc', '#182c40', '#526477', '#a6b7c8'
fig.patch.set_facecolor(bg)
colors = {'M': '#e3edfc', 'P': '#e2f3ee', 'D': '#f4e9fa', 'off': '#e9edf1'}


def text(x, y, s, size=11, color=ink, ha='left', va='top', weight='normal'):
    return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va,
                   fontproperties=FP, linespacing=1.6, weight=weight)


def box(x, top, w, h, title, body='', color='#ffffff', size=10.5, title_size=12):
    ax.add_patch(FancyBboxPatch((x, top-h), w, h, boxstyle='round,pad=0.02,rounding_size=1',
                              linewidth=.8, edgecolor=line, facecolor=color))
    text(x+2, top-1.5, title, title_size)
    if body:
        text(x+2, top-6, body, size)


def arrow(x1,y1,x2,y2,label=None):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,
                                linewidth=1.15,color=muted))
    if label:
        text((x1+x2)/2, (y1+y2)/2+1.8, label, 9, ha='center')


def section(y, number, title, note=''):
    text(5,y,number,14,color='#226e92')
    text(12,y,title,14)
    if note:
        text(155,y-1,note,9,color=muted,ha='right')


text(5,196,'PDblend 当前 Planner：从负载到部署，再到每个请求',23)
text(5,189.7,'源码版本：2026-09-24 工作区  ·  默认策略 pdblend  ·  例子为 synthetic profile 的 CPU 复算',11,color=muted)
ax.plot([5,155],[185.5,185.5],color=line,lw=.8)

section(182.5,'01','三个层次，一条反馈闭环')
box(5,175,45,21,'Planner：决定资源配置',
    '通常每 10 s：P/D/M 数量、频率、τ\n选择 H=60 s 内的预测能量较小方案\n只输出计划，不逐 token 组 batch',color='#eaf2fa')
box(57.5,175,45,21,'Controller：把计划落到实例',
    '30 s 保持；缩容/降频确认 2 次\n角色重写 / 调频 / 排空 / 停驻 / 唤醒\nShield 每 1 s 观察并可覆盖普通计划',color='#edf5f1')
box(110,175,45,21,'Router → Engine：执行请求',
    '每请求选 M 或 P→D，并预留容量\nEngine 管实例内队列与 token batch\n完成、首 token、在途工作反馈给控制器',color='#f2edf8')
arrow(50,164.5,57.5,164.5)
arrow(102.5,164.5,110,164.5)
ax.plot([132,132,27,27],[154,150.6,150.6,154],color=muted,lw=1)
arrow(27,150.6,27,154)
text(80,149.7,'反馈：到达 / 输入输出配对 / 剩余输出 / 待 prefill / 已占 KV',9.5,ha='center',color=muted)

section(144.5,'02','一次规划如何选出方案','固定 TP 的 PoolPlanner 主路径')
box(5,137,46,23,'① 构造 Forecast',
    '到达率：短 EWMA + 近期真实计数\n输入均值 / p95；同请求配对输出\n保留所有未完成请求及原分支\n存量工作摊入未来 60 s 的负载')
box(57,137,46,23,'② 枚举布局 × 频率 × 阈值',
    '实例数总和 = slots；P/D 成对存在\n正式策略 M ≥ min(4, slots)\nP 高频；D/M 遍历 profile 档位\n共存时 τ ∈ {0, 1024, 4096}')
box(109,137,46,23,'③ 分支分别评估与过滤',
    '输入 ≥ τ → PD；其余 → M\nTTFT、TPOT ≤ 0.85 × SLO\n检查利用率、batch、KV、coverage\nM 被 prefill 干扰的超标比例 ≤10%')
arrow(51,125,57,125)
arrow(103,125,109,125)
arrow(132,114,132,109)
box(109,109,46,24,'④ 对全部可行方案重新排序',
    'J = 60 × 预测总功率 + 切换能量\n功率包括所有 active 和停驻实例\n切换成本：合格测量优先，或旧估计\n因此最低稳态功率不一定获选',color='#eaf2fa')
box(57,109,46,24,'⑤ 重评当前方案，决定是否切',
    '当前可行：总成本必须严格省 >8%\n当前不可行：先选可行方案恢复容量\n无可行候选：最高能力 fallback\nfallback 本身不承诺满足 SLO',color='#fff4e6')
box(5,109,46,24,'⑥ 交给 Controller 安全执行',
    '普通变化先过保持 / 确认 / 保护门控\n之后 Shield 可提频或补 P/D/M\n停车：禁新准入 → 排空 → 停驻\n唤醒：就绪 → 设频 → resume → 准入')
arrow(109,97,103,97)
arrow(57,97,51,97)

section(81,'03','具体例子：第二名为什么赢','8×TP1；18 req/s；输入四种等概率；输出 64')
text(5,74.8,'输入 256 / 512 / 2048 / 4096；SLO = 1 s / 20 ms；筛选线 = 850 ms / 17 ms；当前 A = M8 @2520 MHz',10.5)
text(5,70.2,'τ=1024 ⇒ M：9 req/s，平均输入 384；PD：9 req/s，平均输入 3072。全枚举得到 40 个可行候选。',10.5)
cols=[6,30,88,108,130,151]
headers=['候选','配置 / 频率','功率 W','TPOT ms','切换 J','60s kJ']
for x,h in zip(cols,headers): text(x,64.7,h,10,ha='right' if x>=88 else 'left',color=muted)
table=[('A 当前','M8；M=2520','A_current'),
       ('B','M6 + off2；M=2520','B_mixed6'),
       ('C 稳态最低','P2 D1 M4 off1；M=2100','C_lowest_steady'),
       ('D 最终选择','P2 D1 M4 off1；全2520','D_selected'),
       ('F 淘汰','M5 + off3；TPOT超过17','F_mixed5')]
for i,(label,desc,key) in enumerate(table):
    y=60.4-i*4.4
    if key=='D_selected':
        ax.add_patch(Rectangle((5,y-3.5),150,4.4,facecolor='#e0f1e9',edgecolor='none'))
    row=data['rows'][key]; p=row['plan']
    vals=[label,desc,f"{p['power_w']:.3f}",f"{p['tpot_s']*1000:.3f}",f"{row['switch_j']:.1f}",f"{row['total_energy_j']/1000:.3f}"]
    for x,val in zip(cols,vals): text(x,y,val,10,ha='right' if x>=88 else 'left')
text(6,37.7,'C 的 60 s 稳态节省仅 2.470 J，却多计 47.6 J 改频成本，D 胜出；D 相对 A 模型收益 13.81% > 8%。',10.5)

section(32.3,'04','计划落卡与负载变化','以下每次重规划均为无 backlog 的说明截面')
for i,role in enumerate(['M','M','M','M','P','P','D','off']):
    x=5+i*10.1
    ax.add_patch(FancyBboxPatch((x,19.8),9.1,7,boxstyle='round,pad=.02,rounding_size=.7',
                               facecolor=colors[role],edgecolor=line,lw=.6))
    text(x+4.55,25.8,f'G{i}\n{role}',9.5,ha='center')
text(5,17.8,'短请求 256/64 → G1：M 预留序列 0→1→0\n长请求 2048/64 → G5→G6：P pending 1024→3072→1024\nD 预留 8→9→8；P 的首 token 转发后，D 继续其余 63 token',9.6)
box(89,26.8,66,19.3,'输出增长：资源数不变，也可以重新分流',
    '64→128：旧 TPOT 18.281 ms >17，τ 1024→4096\nPD 9→4.5 req/s，M 9→13.5；新 TPOT 15.576 ms\n128→256：0 个可行候选，fallback M8；TPOT 17.408 ms',size=9.6,title_size=10.5,color='#fff4e6')
text(5,4.5,'注意：功率、延迟、收益均为模型预测；旧切换估计的 0 J 不等于真实切换无成本。详细公式、代码位置与复算数据见同目录 README / example.json。',8.5,color=muted)

for ext in ('png','svg','pdf'):
    fig.savefig(HERE / f'planner-current.{ext}',dpi=180,facecolor=fig.get_facecolor())
print('Wrote planner-current.png / .svg / .pdf')
