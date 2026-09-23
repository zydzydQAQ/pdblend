"""Draw the five paper figures as editable, searchable SVGs. CPU only."""
from pathlib import Path
from html import escape

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
INK = '#163047'
MUTED = '#536878'
BLUE = '#23688d'
TEAL = '#167d7f'
ORANGE = '#bd7631'


class SVG:
    def __init__(self, height):
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}">',
            '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#536878"/></marker></defs>',
            '<rect width="960" height="100%" fill="#ffffff"/>']

    def text(self, x, y, lines, size=22, color=INK, anchor='middle', weight=400):
        if isinstance(lines, str):
            lines = [lines]
        for n, line in enumerate(lines):
            self.parts.append(f'<text x="{x}" y="{y+n*(size*1.48)}" font-family="DocSans, Noto Sans CJK SC, sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{escape(line)}</text>')

    def box(self, x, y, w, h, title, lines=(), color=BLUE, fill='#edf4f8', title_size=24):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{color}" stroke-width="1.8"/>')
        self.text(x+w/2, y+35, title, title_size, color, weight=700)
        self.text(x+w/2, y+69, lines, 19, INK)

    def arrow(self, points, color=MUTED, dashed=False):
        path = 'M ' + ' L '.join(f'{x},{y}' for x,y in points)
        dash = ' stroke-dasharray="8 6"' if dashed else ''
        self.parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.4"{dash} marker-end="url(#arrow)"/>')

    def line(self, x1, y1, x2, y2, color='#c8d7df', dashed=False):
        dash=' stroke-dasharray="7 6"' if dashed else ''
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{dash}/>')

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / name).write_text('\n'.join(self.parts + ['</svg>']), encoding='utf-8')


def overview():
    s=SVG(535)
    s.text(36,30,'OFFLINE / 离线',19,ORANGE,'start',700)
    s.box(36,55,278,162,'离线建模 / Modeling', ['采样 → 拟合 → 验证','time · power · overhead','+ valid range'],ORANGE,'#fff6e9')
    s.text(386,30,'PERIODIC / 周期决策',19,BLUE,'start',700)
    s.box(386,55,538,162,'选择性分离规划 / Planning', ['负载 + 模型 + SLO + 当前配置','比较全 M 与选择性 PD；决定资源与阈值','output: layout · clocks · parking · τ'])
    s.arrow([(314,135),(386,135)])
    s.text(350,107,'模型',17,MUTED)
    s.box(386,312,538,156,'在线协同 / Online coordination',['观测预测 · 安全执行 · 快速保护','按有效配置分流：M 或 P → D','输出 token；记录时延与完成情况'],TEAL,'#eaf6f3')
    s.arrow([(586,217),(586,312)])
    s.text(570,254,['Plan','配置与阈值'],18,BLUE,'end')
    s.arrow([(808,312),(808,217)],TEAL,True)
    s.text(824,260,['负载 / 延迟','feedback'],18,TEAL,'start')
    s.box(36,330,232,105,'请求 / Requests',['prompt + output budget'],TEAL,'#eaf6f3',22)
    s.arrow([(268,366),(386,366)],TEAL)
    s.text(326,349,'输入',18,TEAL)
    s.arrow([(386,416),(268,416)],TEAL)
    s.text(326,443,'token 流',18,TEAL)
    s.text(480,509,'模型供给 → 周期规划 → 在线执行 → 观测反馈；在线不自动重新训练模型',19,MUTED)
    s.save('01-overview.svg')


def modeling():
    s=SVG(525)
    s.text(480,29,'同一请求 / SAME REQUEST: 2048 input tokens → 128 output tokens',21,INK,weight=700)
    boxes=[('Prefill','输入长度 × 频率'),('Decode','batch × context × 频率'),('Mixed','阶段干扰与功率'),('Handoff / Switch','交接与配置切换')]
    for i,(title,line) in enumerate(boxes):
        s.box(20+i*237,56,218,101,title,[line],ORANGE,'#fff6e9',21)
        s.arrow([(129+i*237,157),(129+i*237,184),(480,184),(480,210)],ORANGE)
    s.box(215,210,530,99,'统一模型 / Cost model',['shape + layout → time, power, cost, coverage'],BLUE,'#edf4f8',24)
    s.arrow([(330,309),(330,348)])
    s.arrow([(672,309),(672,348)])
    s.box(40,348,412,109,'路径 A / Mixed',['同一实例读输入并生成输出','计入 mixed 干扰，无跨实例 KV 交接'],TEAL,'#eaf6f3',23)
    s.box(508,348,412,109,'路径 B / Selective PD',['P → KV handoff → D','分别配置频率，计入额外交接成本'],BLUE,'#edf4f8',23)
    s.arrow([(246,457),(246,480),(480,480)])
    s.arrow([(714,457),(714,480),(480,480)])
    s.text(480,513,'向 Planning 提供同口径预测 / Comparable predictions for planning',20,INK,weight=700)
    s.save('02-model.svg')


def planning():
    s=SVG(560)
    s.text(480,29,'同一负载、同一窗口、同一延迟要求 / SAME WORKLOAD, WINDOW AND SLO',20,INK,weight=700)
    s.box(28,61,354,131,'全 M / M-only',['全部请求 → M','归一化成本 / cost = 100'],TEAL,'#eaf6f3',25)
    s.box(464,61,468,131,'选择性 PD / Selective PD',['256 / 512 → M; 2048 / 4096 → P → D','阈值 / τ = 1024 tokens'],BLUE,'#edf4f8',25)
    s.arrow([(698,192),(698,223)])
    s.box(464,223,468,116,'只计算一次每项成本',['运行及停驻 78 + 交接 8 + 切换 4 = 90','交接仅计增量；不是实测结果'],ORANGE,'#fff6e9',23)
    s.arrow([(205,192),(205,362),(480,362),(480,397)],TEAL)
    s.arrow([(698,339),(698,362),(480,362)],BLUE)
    s.box(113,397,734,97,'先检查可行，再检查收益 / Feasibility before savings',['在预测达标且切换值得时选 90；若成本为 103 或违反 SLO，则拒绝'],BLUE,'#edf4f8',23)
    s.arrow([(480,494),(480,526)])
    s.text(480,551,'Plan: {P/D/M counts, clocks, parking, τ} → 在线执行 / Online execution',21,INK,weight=700)
    s.save('03-planning.svg')


def routing():
    s=SVG(645)
    s.box(20,16,920,91,'已生效配置 / Effective plan',['8 × TP1: P1 + D2 + M4 + L1; τ = 1024; requested output = 128'],BLUE,'#edf4f8',24)
    s.text(40,147,'短请求 / SHORT: 256 < 1024',20,TEAL,'start',700)
    s.box(370,119,244,74,'M',['一次完成全部输出'],TEAL,'#eaf6f3',21)
    s.arrow([(40,177),(350,177)],TEAL)
    s.arrow([(628,155),(926,155)],TEAL)
    s.text(780,142,'128 tokens → Client',19,TEAL)
    s.text(480,218,'长请求 / LONG: 2048 ≥ 1024',18,BLUE,weight=700)
    names=[('Client / Proxy',116),('P',440),('D',811)]
    for name,x in names:
        s.text(x,251,name,23,INK,weight=700)
        s.line(x,263,x,559,dashed=True)
    s.arrow([(116,286),(440,286)])
    s.text(278,271,'prompt[2048], budget = 1',18,MUTED)
    s.text(460,320,'P 保留原始 prompt 的 KV',18,ORANGE,'start')
    s.arrow([(440,357),(116,357)],TEAL)
    s.text(278,340,'首 token y1 → 客户端',19,TEAL)
    s.text(101,383,'TTFT 截止',17,TEAL,'end')
    s.arrow([(116,414),(811,414)])
    s.text(470,399,'prompt[2048] + [y1] → 2049 tokens; remaining budget = 127',18,MUTED)
    s.arrow([(440,463),(811,463)],ORANGE)
    s.text(626,449,'KV[2048] → D 复用 / reuse',18,ORANGE)
    s.arrow([(811,525),(116,525)],TEAL)
    s.text(468,509,'y2 … y128 → 合并为同一个逻辑输出 / one logical stream',18,TEAL)
    s.box(85,569,790,60,'最终 usage / Final usage: input = 2048; output = 128',[],BLUE,'#edf4f8',22)
    s.save('04-routing.svg')


def control():
    s=SVG(615)
    s.box(310,17,340,91,'观测与预测 / Observe',['到达、首 token、完成、队列状态'],TEAL,'#eaf6f3',24)
    s.arrow([(365,108),(365,137),(225,137),(225,164)],TEAL)
    s.arrow([(595,108),(595,137),(735,137),(735,164)],BLUE)
    s.box(30,164,390,108,'快速保护 / Fast protection',['秒级风险检查；升频或增加容量','也可提前请求重新规划'],TEAL,'#eaf6f3',24)
    s.box(540,164,390,108,'周期规划 / Periodic planning',['比较配置与阈值的完整成本','当前默认周期约 10 s'],BLUE,'#edf4f8',24)
    s.arrow([(225,272),(225,301),(480,301),(480,331)],TEAL)
    s.arrow([(735,272),(735,301),(480,301)],BLUE)
    s.box(227,331,506,91,'安全执行 → 有效配置',['路径容量就绪后发布；复用或停驻前排空'],BLUE,'#edf4f8',24)
    s.arrow([(480,422),(480,455)])
    s.text(480,481,'路由 M / PD + 频率与资源动作 → 引擎执行 → 新观测',21,INK,weight=700)
    s.arrow([(733,374),(950,374),(950,62),(650,62)],TEAL,True)
    labels=[('正常','stable'),('突发','burst'),('快速保护','protect'),('重新规划','replan'),('稳定后收缩','recover')]
    s.line(82,534,878,534,color=BLUE)
    for i,(zh,en) in enumerate(labels):
        x=82+i*199
        s.parts.append(f'<circle cx="{x}" cy="534" r="6" fill="{TEAL if i != 1 else ORANGE}"/>')
        s.text(x,568,zh,20,INK)
        s.text(x,597,en,18,MUTED)
    s.save('05-control.svg')


if __name__ == '__main__':
    for draw in (overview,modeling,planning,routing,control):
        draw()
    print('Built 5 SVG figures')
