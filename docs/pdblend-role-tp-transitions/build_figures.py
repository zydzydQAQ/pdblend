"""Build exact, editable role / TP diagrams, then render PDF and PNG on CPU.

Dependencies for export: weasyprint==62.3 pydyf==0.10.0
                         PyMuPDF==1.24.14 Pillow==11.1.0
The SVG drawing itself uses only the Python standard library.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
FONT = ROOT.parent / 'pdblend-method/assets/NotoSansCJKsc-Regular.otf'
BOLD = ROOT.parent / 'pdblend-method/assets/NotoSansCJKsc-Bold.otf'
W = 1800
INK, MUTED, LINE = '#172c3a', '#4c6270', '#c9d3da'
GREEN, BLUE, GREY = '#d7ebdd', '#d8e7f7', '#edf0f2'
P_EDGE, D_EDGE, WARN = '#287343', '#286395', '#985329'


class SVG:
    def __init__(self, height, title):
        self.height, self.title = height, title
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{height}" viewBox="0 0 {W} {height}" role="img">',
            f'<title>{escape(title)}</title>',
            '<desc>GPU 与引擎边界明确的 PDblend 角色和张量并行变换示意；所有案例为布局示例，不是 GPU 实测。</desc>',
            '<defs><marker id="a" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#4c6270"/></marker></defs>',
            f'<rect width="{W}" height="{height}" fill="white"/>',
        ]
        self.text_boxes = []

    def text(self, x, y, text, size=26, color=INK, anchor='start', weight=400):
        lines = text if isinstance(text, (tuple, list)) else [text]
        for i, line in enumerate(lines):
            yy = y + i * size * 1.48
            self.parts.append(f'<text x="{x}" y="{yy}" font-family="DiagramSans, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(str(line))}</text>')
            self.text_boxes.append((x, yy, str(line), size, anchor, weight))

    def rect(self, x, y, w, h, fill='none', stroke=LINE, sw=1.5, dash=False):
        dashed = ' stroke-dasharray="7 5"' if dash else ''
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dashed}/>')

    def line(self, x1, y1, x2, y2, color=LINE, dash=False):
        dashed = ' stroke-dasharray="7 5"' if dash else ''
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="1.5"{dashed}/>')

    def arrow(self, x1, y1, x2, y2, both=False, dash=False):
        attrs = ' marker-start="url(#a)"' if both else ''
        attrs += ' stroke-dasharray="6 5"' if dash else ''
        self.parts.append(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{MUTED}" stroke-width="2.3" marker-end="url(#a)"{attrs}/>')

    def section(self, y, label, title, right=''):
        self.line(60, y-30, 1740, y-30)
        self.text(60, y+6, label, 27, P_EDGE, weight=700)
        self.text(112, y+6, title, 29, weight=700)
        if right:
            self.text(1740, y+5, right, 22, MUTED, anchor='end')

    def gpu(self, x, y, role, gpu, width=102, height=68, weight=''):
        if role == 'M':
            self.rect(x, y, width/2, height, GREEN, 'none')
            self.rect(x+width/2, y, width/2, height, BLUE, 'none')
        else:
            self.rect(x, y, width, height, {'P': GREEN, 'D': BLUE}.get(role, GREY), 'none')
        self.rect(x, y, width, height, stroke=INK, sw=1.5)
        self.text(x+width/2, y+28, f'{role} · {gpu}', 23, anchor='middle', weight=700)
        self.text(x+width/2, y+53, weight or ('TP1' if role != 'S' and role != 'F' else '备用'), 20, MUTED, anchor='middle')

    def token(self, x, y, label, role='M', width=135):
        if role in ('M', 'PD'):
            self.rect(x, y, width/2, 48, GREEN, 'none')
            self.rect(x+width/2, y, width/2, 48, BLUE, 'none')
            self.rect(x, y, width, 48, stroke=INK)
        else:
            self.rect(x, y, width, 48, {'P': GREEN, 'D': BLUE}.get(role, GREY), INK)
        self.text(x+width/2, y+32, label, 23, anchor='middle')

    def pool(self, x, y, width, height, title):
        self.rect(x, y, width, height, stroke=MUTED, dash=True)
        self.text(x+9, y-9, title, 21, MUTED)

    def engine(self, x, y, role, ids, tp, title=None):
        # Each outlined engine has exactly TP ranks. Small boxes are GPU ranks.
        bw, gap = 100, 14
        width = len(ids)*bw+(len(ids)-1)*gap+24
        height = 108
        self.rect(x, y, width, height, stroke=INK, sw=2)
        self.text(x+width/2, y+25, title or f'{role} · TP{tp}', 22, anchor='middle', weight=700)
        for rank, gpu in enumerate(ids):
            gx = x+12+rank*(bw+gap)
            self.gpu(gx, y+37, role, f'G{gpu}', bw, 60, 'W' if tp == 1 else f'r{rank} · W{rank}')
        if tp == 2:
            self.line(x+62, y+97, x+62, y+123, MUTED)
            self.line(x+176, y+97, x+176, y+123, MUTED)
            self.arrow(x+64, y+122, x+174, y+122, both=True)
            self.text(x+width/2, y+150, 'TP collective', 20, MUTED, anchor='middle')
        return width

    def save(self, filename):
        (ROOT/filename).write_text('\n'.join(self.parts+['</svg>']), encoding='utf-8')


def legend(s, y, tp=False):
    for x, role, desc in [(60,'P','Prefill'), (380,'D','Decode'), (700,'M','Mixed：本地 P+D')]:
        s.gpu(x, y, role, 'i', 92, 62, 'GPU')
        s.text(x+110, y+39, desc, 24)
    s.gpu(1160, y, 'F' if tp else 'S', 'i', 92, 62, '可分配' if tp else '停驻')
    s.text(1270, y+26, 'F：已释放、可重新分配' if tp else 'S：idle / L1 / off', 23)
    s.text(1270, y+57, '不等同于物理断电' if tp else '三种状态的显存占用不同', 21, MUTED)


def role_diagram():
    s = SVG(2500, 'Role change：TP1 的 GPU 增减、实例角色与 merge / split')
    s.text(60, 70, 'Role change｜TP1 的完整变换图', 43, weight=700)
    s.text(60, 112, '同一模型 · PP=1 · 每个引擎实例恰好占 1 张 GPU；双向箭头表示一对互逆变换。', 26, MUTED)
    legend(s, 140)

    s.section(254, 'A', '先固定单位：PD 是服务路径，P、D 分别是实例')
    s.engine(85, 309, 'M', [0], 1)
    s.text(230, 346, ['一个 M 实例', '同一引擎完成 P 和 D'], 24)
    s.text(230, 419, 'W：可分片模型权重', 22, MUTED)
    s.pool(590, 306, 400, 127, '一个 PD 服务单元 / PD pool 的最小配置')
    s.engine(605, 317, 'P', [0], 1)
    s.engine(851, 317, 'D', [1], 1)
    s.arrow(741, 375, 838, 375, dash=True)
    s.text(790, 356, 'KV', 22, MUTED, anchor='middle')
    s.text(1040, 330, ['M(n)：n 个独立 Mixed 实例', 'PD(p,d)：p 个 P + d 个 D 实例', 'PD 有效时 p≥1、d≥1；不要求 p=d。'], 26)
    s.text(1040, 451, '实框＝实例 / GPU；虚框＝逻辑服务池。', 23, MUTED)

    s.section(513, 'B', 'add / delete GPU：增加或减少单卡副本', '向右 add；向左 delete')
    rows = [(573, 'Mixed pool', 'M(n) + S ⇄ M(n+1)', [('M','G0'),('S','G1')], [('M','G0'),('M','G1')]),
            (696, 'PD pool：P 侧', 'PD(p,d) + S ⇄ PD(p+1,d)', [('P','G0'),('D','G1'),('S','G2')], [('P','G0'),('D','G1'),('P','G2')]),
            (819, 'PD pool：D 侧', 'PD(p,d) + S ⇄ PD(p,d+1)', [('P','G0'),('D','G1'),('S','G2')], [('P','G0'),('D','G1'),('D','G2')])]
    for y, label, formula, before, after in rows:
        s.text(76, y+23, label, 27, weight=700)
        s.text(76, y+63, formula, 23, MUTED)
        for i, (role, gpu) in enumerate(before):
            s.gpu(535+i*128, y-4, role, gpu)
        for i, (role, gpu) in enumerate(after):
            s.gpu(1160+i*128, y-4, role, gpu)
        s.arrow(962, y+31, 1080, y+31, both=True)
        s.text(1610, y+24, '+1 / −1', 24, anchor='middle')
        s.text(1610, y+58, '活动 GPU', 21, MUTED, anchor='middle')
    s.text(76, 937, '这里的 add/delete 是活动资源增减：当前 controller 唤醒或停驻既有 fleet 实例；外加新卡还需要配置和启动。', 23, MUTED)

    s.section(998, 'C', 'role change 与跨类型 merge / split', 'GPU 数守恒；每张卡都显示去向')
    for x, a, b in [(80,'P','D'),(640,'P','M'),(1200,'D','M')]:
        s.gpu(x, 1035, a, 'Gi', 102)
        s.arrow(x+127, 1068, x+255, 1068, both=True)
        s.gpu(x+280, 1035, b, 'Gi', 102)
        s.text(x+190, 1133, '同卡改角色，模型和 TP 不变', 23, MUTED, anchor='middle')

    s.text(78, 1167, '两张活动卡：两个 Mixed ⇄ PD', 26, weight=700)
    s.gpu(80, 1217, 'M', 'G0', 90)
    s.gpu(184, 1217, 'M', 'G1', 90)
    s.arrow(302, 1250, 422, 1250, both=True)
    s.pool(446, 1209, 260, 92, 'PD')
    s.gpu(459, 1221, 'P', 'G0', 90)
    s.gpu(603, 1221, 'D', 'G1', 90)
    s.arrow(555, 1253, 594, 1253, dash=True)
    s.text(80, 1338, '2M ⇄ P+D：一直是 2 个 TP1 引擎。', 23, MUTED)

    s.text(928, 1167, '一张 Mixed ⇄ PD：增加 / 归还一张卡', 26, weight=700)
    s.gpu(930, 1217, 'M', 'G0', 90)
    s.gpu(1034, 1217, 'S', 'G1', 90)
    s.arrow(1152, 1250, 1272, 1250, both=True)
    s.pool(1296, 1209, 260, 92, 'PD')
    s.gpu(1309, 1221, 'P', 'G0', 90)
    s.gpu(1453, 1221, 'D', 'G1', 90)
    s.arrow(1405, 1253, 1444, 1253, dash=True)
    s.text(930, 1338, 'M+S ⇄ P+D；也可以保留另一张 GPU。', 23, MUTED)

    s.text(80, 1403, 'PD 与 M 之间借卡 / 还卡', 25, weight=700)
    s.text(80, 1446, 'PD(p,d) + M ⇄ PD(p+1,d)   或   PD(p,d+1)', 25)
    s.text(980, 1403, 'PD 内部重平衡', 25, weight=700)
    s.text(980, 1446, '2P + D ⇄ P + 2D', 27)
    s.text(980, 1485, '通式：PD(p,d) ⇄ PD(p−1,d+1)，保留完整两侧。', 21, MUTED)

    s.section(1556, 'D', '逻辑 pool 的 merge / split', '改变集合边界；GPU、引擎副本数和 TP 都不变')
    s.text(78, 1622, 'M + M', 28, weight=700)
    s.pool(300, 1601, 170, 78, 'M pool A')
    s.pool(500, 1601, 170, 78, 'M pool B')
    s.token(317, 1616, 'a × M')
    s.token(517, 1616, 'b × M')
    s.arrow(737, 1640, 882, 1640, both=True)
    s.pool(960, 1601, 480, 78, 'M pool A∪B')
    s.token(981, 1616, 'a × M')
    s.token(1242, 1616, 'b × M')
    s.text(1485, 1644, 'M(a+b)', 27)

    s.text(78, 1762, 'PD + PD', 28, weight=700)
    s.pool(300, 1734, 170, 78, 'PD pool A')
    s.pool(500, 1734, 170, 78, 'PD pool B')
    s.token(310, 1749, 'P(p1) D(d1)', 'PD', 150)
    s.token(510, 1749, 'P(p2) D(d2)', 'PD', 150)
    s.arrow(737, 1773, 882, 1773, both=True)
    s.pool(960, 1734, 480, 78, '兼容的同模型、同 TP 的 PD pool')
    s.token(976, 1749, 'P(p1+p2)', 'P', 209)
    s.token(1210, 1749, 'D(d1+d2)', 'D', 209)
    s.text(1475, 1746, ['split 后每个子池', '都需要 P≥1、D≥1'], 22)

    s.text(78, 1880, 'PD + M', 28, weight=700)
    s.text(300, 1880, 'PD(p,d) + M(m)  ⇄  { P(p), D(d), M(m) }', 27)
    s.text(300, 1925, '这里只合并路由域，角色不变；若把 M 转入 P/D 角色，使用 C 中的借卡变换。', 23, MUTED)
    s.text(78, 1984, '逻辑分组为概念操作：当前无 merge()/split() 原语；隔离子池须配置池身份与路由，并处理存量请求。', 23, MUTED)

    s.section(2050, 'E', '执行顺序、停驻子状态与不可省略的边界')
    s.text(78, 2110, '活动角色互换：P / D / M', 25, weight=700)
    s.text(78, 2152, ['当前实现：改频 → 改角色表 → 新请求使用新角色。', '旧请求保持原绑定；无需权重重分片，不迁移活跃 KV。'], 23)
    s.text(78, 2246, '活动 → S：关闭准入 → 排空 → 停驻 / 停进程。', 23)
    s.text(78, 2285, 'S → 活动：唤醒或 start+ready → 设频 / resume → 准入。', 23)
    s.text(78, 2324, '停驻内部：idle ⇄ L1；idle ⇄ off；L1 ⇄ off。', 23)
    s.text(78, 2363, 'idle/L1 保留权重；off 停进程；不是同一种“空卡”。', 23, MUTED)
    s.text(955, 2110, '覆盖与约束', 25, weight=700)
    s.text(955, 2152, ['四状态 {P,D,M,S} 的 6 条双向边覆盖 12 个方向。', 'nP + nD + nM + nS = N；S 内另有 6 个停驻方向。', '删最后一个 P 或 D：必须替换、转 M 或关闭该 PD 路径。', '2 个 TP1 引擎 → 1 个双卡引擎，属于下一图的 TP 变换。', '停用资源需确认请求/KV/transfer 已清；native 检查需接入。', '多实例改角色当前不保证原子发布；仍须满足容量/SLO。'], 23)
    s.text(60, 2460, 'PDblend · 布局与机制示意 · 2026-09-23 · 源码依据与一般式见 README；本图未运行 GPU 实验。', 21, MUTED)
    s.save('role-change-tp1.svg')
    return s


def pd_snapshot(s, x, y, tp, free=False):
    # Same physical GPUs throughout: P uses 0/1, D uses 2/3.
    if tp == 1:
        s.engine(x, y, 'P', [0], 1)
        s.engine(x+320, y, 'D', [2], 1)
        s.arrow(x+139, y+66, x+301, y+66, dash=True)
        s.text(x+220, y+50, 'KV', 22, MUTED, anchor='middle')
        if free:
            s.gpu(x+145, y+112, 'F', 'G1', 95, 60, '可分配')
            s.gpu(x+254, y+112, 'F', 'G3', 95, 60, '可分配')
    else:
        s.engine(x, y, 'P', [0,1], 2)
        s.engine(x+320, y, 'D', [2,3], 2)
        s.arrow(x+251, y+66, x+307, y+66, dash=True)
        s.text(x+280, y+47, 'KV', 19, MUTED, anchor='middle')


def m_snapshot(s, x, y, tp, free=False):
    if tp == 1:
        s.engine(x, y, 'M', [0], 1)
        if free:
            s.text(x+175, y+70, '+', 28, anchor='middle')
            s.gpu(x+218, y+33, 'F', 'G1', 100, 62, '可分配')
    else:
        s.engine(x+40, y, 'M', [0,1], 2)


def main_tp_panel(s, x, y, kind, up):
    number = {('PD',True):'1',('M',True):'2',('PD',False):'3',('M',False):'4'}[(kind,up)]
    title = f'{number}  {"PD 两端" if kind == "PD" else "Mixed 实例"}的 TP {"增加" if up else "减少"}'
    s.text(x, y, title, 30, weight=700)
    before_tp, after_tp = (1,2) if up else (2,1)
    s.text(x, y+42, 'TP1 → TP2' if up else 'TP2 → TP1', 25, MUTED)
    if kind == 'PD':
        pd_snapshot(s, x+55, y+70, before_tp, free=up)
        ay = y+259 if up else y+224
        s.arrow(x+329, ay, x+329, ay+47)
        pd_snapshot(s, x+55, ay+65, after_tp, free=not up)
        fy = y+513
        s.text(x, fy, '活动 GPU：2 → 4；P、D 仍各 1 个引擎。' if up else '活动 GPU：4 → 2；G1、G3 归还资源池。', 23)
    else:
        m_snapshot(s, x+180, y+95, before_tp, free=up)
        s.arrow(x+339, y+(238 if up else 265), x+339, y+295)
        m_snapshot(s, x+180, y+323, after_tp, free=not up)
        fy = y+513
        s.text(x, fy, '活动 GPU：1 → 2；仍是 1 个 Mixed 引擎。' if up else '活动 GPU：2 → 1；G1 归还资源池。', 23)


def tp_diagram():
    s = SVG(2720, 'TP change：PD 和 Mixed 的 TP 增减与副本 merge / split')
    s.text(60, 70, 'TP change｜PD 与 Mixed 的 TP 增减', 43, weight=700)
    s.text(60, 112, '目标重建机制：当前 slow_reshard_tp 运行入口尚未支持；以下为结构与步骤示意。', 26, WARN)
    legend(s, 143, tp=True)
    s.text(60, 245, '主四图保持引擎数量不变：从空闲池取卡 / 向空闲池还卡。例子均为同模型、PP=1。', 25)
    s.line(900, 280, 900, 1420)
    main_tp_panel(s, 80, 306, 'PD', True)
    main_tp_panel(s, 965, 306, 'M', True)
    s.line(60, 861, 1740, 861)
    main_tp_panel(s, 80, 910, 'PD', False)
    main_tp_panel(s, 965, 910, 'M', False)

    s.section(1500, 'A', '一般式：所有增减方向与单侧 P / D 变换')
    s.text(80, 1560, 'R(t) + (u−t)F  ⇄  R(u)', 30, weight=700)
    s.text(80, 1604, 'R∈{P,D,M}；u>t；一个引擎占用 t → u 张 GPU。', 23)
    s.text(80, 1654, 'PD(t) + 2(u−t)F  ⇄  PD(u)', 30, weight=700)
    s.text(80, 1698, 'PD(t) = P(t) + D(t)；双侧对称扩缩。', 23)
    s.text(80, 1743, '1⇄2、2⇄4、1⇄4 同理；目标必须满足模型和显存约束。', 22, MUTED)
    s.text(970, 1560, '只改变 P 或只改变 D 的 TP', 27, weight=700)
    s.text(970, 1605, ['P(t) → P(u)：新请求必须配已就绪的 D(u)。', 'D(t) → D(u)：新请求必须配已就绪的 P(u)。', '增加和减少均适用；原对端可重建或换成兼容对端。', '当前不支持 P(1)→D(2) 或 P(2)→D(1) 接力。', '配对还要求 model / PP / pool / generation 一致。'], 23)

    s.section(1848, 'B', '固定 GPU 数的实例 merge / split', '副本数变化；不是主四图中的取卡 / 还卡')
    s.gpu(82, 1910, 'M', 'G0', 94)
    s.gpu(193, 1910, 'M', 'G1', 94)
    s.arrow(326, 1946, 438, 1946, both=True)
    s.engine(474, 1890, 'M', [0,1], 2)
    s.text(80, 2043, '2 × M(TP1) ⇄ 1 × M(TP2)', 26, weight=700)
    s.text(80, 2090, '4M(1) ⇄ 2M(2) ⇄ M(4)；反向 split 需加载完整副本。', 23)
    s.text(940, 1918, '2P(1) + 2D(1)  ⇄  P(2) + D(2)', 29, weight=700)
    s.text(940, 1964, ['P 侧单独合并、D 侧单独合并；共 4 张 GPU。', '合并后是 P、D 两个 TP2 组，不是一个 TP4。'], 24)
    s.text(940, 2043, '通式：n × R(t) ⇄ m × R(u)，n·t = m·u。', 24)
    s.text(940, 2090, '保持单实例时缩 TP 释放卡；另建完整副本才是 split。', 23, MUTED)

    s.section(2162, 'C', '所有 TP 变换共用的重建流程')
    stages = [('1  预检 / 锁卡','同模型、目标拓扑、HBM'), ('2  关闭并排空','旧请求在原实例结束'), ('3  重建 / 加载','释放旧资源；建立新 TP'), ('4  验证 / 发布','逐 rank 就绪后开放准入')]
    for i, (title, detail) in enumerate(stages):
        x = 80+i*426
        s.rect(x, 2205, 370, 90, '#f3f6f7', LINE)
        s.text(x+185, 2241, title, 26, anchor='middle', weight=700)
        s.text(x+185, 2275, detail, 21, MUTED, anchor='middle')
        if i < 3:
            s.arrow(x+381, 2250, x+413, 2250)
    s.text(80, 2345, '排空屏障：所有旧 rank 的请求、活跃 KV、pending transfer 均为 0；旧 TP 的活跃 KV 不迁入新 TP。', 23)
    s.text(80, 2388, '重建：workers / TP communicator → 加载已验证权重分片 → 空 KV 池 → 输出与取消 / KV 清理探针。', 23)
    s.text(80, 2431, '失败：保持关闭，重建并验证旧布局；恢复不确定则隔离 GPU。其他实例只能在自身容量内继续服务。', 23)
    s.text(80, 2497, '权重与通信', 25, weight=700)
    s.text(290, 2497, 'W 为可分片权重；TP2：W → {W0,W1}。P、D 各自模型完整；KV 与组内 collective 不同。', 22)
    s.text(80, 2540, '每卡显存', 25, weight=700)
    s.text(290, 2540, '可分片权重约 W/t + 复制张量 + KV + workspace + 通信 / 加载峰值；缩 TP 前必须装得下。', 22)
    s.text(80, 2583, '边界', 25, weight=700)
    s.text(290, 2583, 'TP∈{1,2,4}、PP=1 还需模型/profile 校验；切换已有不同 TP 常驻池，仅改变路由，不等于重建 TP。', 22)
    s.text(80, 2626, 'TP2 KV 对应：P/r0 → D/r0，P/r1 → D/r1；目标布局发布后，仅新请求使用这条兼容 KV 路径。', 22, MUTED)
    s.text(60, 2676, 'PDblend · 目标变换设计，未声称在线 TP 重分片已实现或经 GPU 验证 · 2026-09-23 · 详见 README。', 21, MUTED)
    s.save('tp-change.svg')
    return s


def render(figures):
    from weasyprint import HTML
    from PIL import ImageFont
    import fitz

    report = []
    combined = fitz.open()
    for name, svg in figures:
        markup = (ROOT/f'{name}.svg').read_text(encoding='utf-8')
        css = f'''@font-face{{font-family:DiagramSans;src:url("{FONT.as_uri()}");font-weight:400}}
        @font-face{{font-family:DiagramSans;src:url("{BOLD.as_uri()}");font-weight:700}}
        @page{{size:{W}px {svg.height}px;margin:0}}html,body{{margin:0;padding:0}}
        svg{{display:block}}'''
        html = '<!doctype html><meta charset="utf-8"><style>'+css+'</style>'+markup
        HTML(string=html, base_url=str(ROOT)).write_pdf(ROOT/f'{name}.pdf')
        pdf = fitz.open(ROOT/f'{name}.pdf')
        if len(pdf) != 1:
            raise RuntimeError(f'{name} unexpectedly spans {len(pdf)} pages')
        pdf[0].get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(ROOT/f'{name}.png')
        combined.insert_pdf(pdf)
        fonts, bounds, glyph_boxes = {}, [], []
        for x,y,t,size,anchor,weight in svg.text_boxes:
            key=(size,weight)
            if key not in fonts:
                fonts[key]=ImageFont.truetype(str(BOLD if weight==700 else FONT), size)
            tw=fonts[key].getlength(t)
            left=x-(tw/2 if anchor=='middle' else tw if anchor=='end' else 0)
            if left < 0 or left+tw > W or y > svg.height:
                bounds.append(dict(text=t, left=left, right=left+tw, baseline=y))
            b = fonts[key].getbbox(t, anchor='ls')
            glyph_boxes.append(((left+b[0], y+b[1], left+b[2], y+b[3]), t))
        overlaps = []
        for i, (a, text_a) in enumerate(glyph_boxes):
            for b, text_b in glyph_boxes[i+1:]:
                dx = min(a[2], b[2]) - max(a[0], b[0])
                dy = min(a[3], b[3]) - max(a[1], b[1])
                if dx > 1 and dy > 1:
                    overlaps.append(dict(first=text_a, second=text_b, width=dx, height=dy))
        chars=len(pdf[0].get_text())
        report.append(dict(figure=name, width=W, height=svg.height, pdf_pages=len(pdf), searchable_characters=chars, text_outside_canvas=bounds, overlapping_text=overlaps))
        if bounds or overlaps or chars < 500:
            raise RuntimeError(f'{name}: bad text bounds or missing searchable text: {report[-1]}')
        pdf.close()
    combined.set_toc([[1,'Role change at TP1',1],[1,'TP change',2]])
    combined.save(ROOT/'transitions.pdf', garbage=4, deflate=True)
    (ROOT/'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    figures = [('role-change-tp1', role_diagram()), ('tp-change', tp_diagram())]
    render(figures)
