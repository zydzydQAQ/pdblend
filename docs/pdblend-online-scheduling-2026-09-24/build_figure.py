"""Source-grounded, editable paper figure. CPU rendering only; no serving imports."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Polygon
from matplotlib.transforms import Bbox

OUT = Path(__file__).resolve().parent
REPO = OUT.parent.parent
W, H = 2400, 3370
INK, MUTED = '#161616', '#545454'
GREEN, BLUE, GRAY, GOLD = '#a4cb86', '#4778be', '#d5d5d5', '#ffdd79'
PEACH, LIGHT, RED = '#f9e5d8', '#f7f7f7', '#a73532'
plt.rcParams.update({'font.family': ['Liberation Serif', 'DejaVu Serif'], 'svg.fonttype': 'none',
                     'pdf.fonttype': 42, 'axes.unicode_minus': False})
fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor='white')
ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0,W), ylim=(H,0)); ax.axis('off')
texts = []

def text(x,y,s,size=27,bold=False,color=INK,ha='left',italic=False):
    # Consistent mathematical subscripts, including d (no Unicode subscript d).
    for old,new in [('Nᵈ',r'$N_d$'),('Uᵈ',r'$U_d$'),('Dᵈ',r'$D_d$'),
                    ('Uₚ',r'$U_p$'),('Pₚ',r'$P_p$'),('Nₘ',r'$N_m$'),
                    ('Uₘ',r'$U_m$'),('Mₘ',r'$M_m$'),('C₀',r'$C_0$'),
                    ('y₁',r'$y_1$'),('y₂',r'$y_2$')]:
        s=s.replace(old,new)
    t=ax.text(x,y,s,fontsize=size*.72,fontweight='bold' if bold else 'normal',
              color=color,ha=ha,va='baseline',fontstyle='italic' if italic else 'normal',zorder=5)
    texts.append(t); return t

def rect(x,y,w,h,fill='white',lw=1.15,dash=False,edge=INK):
    ax.add_patch(Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,linewidth=lw,
                           linestyle=(0,(5,4)) if dash else '-',zorder=1))

def line(x1,y1,x2,y2,color=INK,lw=1.1,dash=False):
    ax.plot([x1,x2],[y1,y2],color=color,lw=lw,ls=(0,(5,4)) if dash else '-',zorder=2)

def arrow(x1,y1,x2,y2,color=INK,lw=1.25,dash=False,both=False):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='<->' if both else '-|>',
        mutation_scale=13,color=color,linewidth=lw,linestyle=(0,(5,4)) if dash else '-',
        shrinkA=0,shrinkB=0,zorder=3))

def path(points,color=INK,dash=False,arr=True,lw=1.25):
    for a,b in zip(points[:-2],points[1:-1]):line(*a,*b,color,lw,dash)
    if arr:arrow(*points[-2],*points[-1],color,lw,dash)
    else:line(*points[-2],*points[-1],color,lw,dash)

def label(x,y,w,h,s,fill=LIGHT,size=27,bold=False,color=INK):
    rect(x,y,w,h,fill);text(x+w/2,y+h/2+size*.32,s,size,bold,color,ha='center')

def panel(y,s):
    line(40,y-31,2360,y-31,'#777',.8);text(40,y,s,32,True)

def blockrow(x,y,width,count,fill,labels=None,h=42):
    for k in range(count):
        label(x+k*width,y,width,h,labels[k] if labels else '',fill,24,color='white' if fill==BLUE else INK)

text(1200,49,'PDBlend: online routing, instance assignment, and execution',43,True,ha='center')
text(1200,85,'Current homogeneous PDblend runner  |  source snapshot: 24 September 2026',25,color=MUTED,ha='center')
for x,c,s in [(475,GREEN,'Prefill'),(750,BLUE,'Decode'),(1020,GRAY,'Wait / KV transfer'),(1435,GOLD,'Output token'),(1810,PEACH,'Control')]:
    rect(x,108,29,22,c);text(x+43,128,s,25)

# A: the controller supplies a committed layout; routing remains per-request.
panel(180,'(a) Two control timescales publish the layout used by every new request')
rect(40,206,500,220,LIGHT)
text(290,242,'Online observations',29,True,ha='center')
text(65,280,'Arrivals, input / output lengths, backlog',25)
text(65,318,'N: unfinished requests; U: prefill tokens',25)
text(65,356,'TTFT / TPOT, token progress, KV usage',25)
text(65,397,'Update on dispatch, token, and completion',24,color=MUTED)
arrow(540,316,600,316)
rect(600,206,1150,220)
label(619,223,618,44,'Planner + Controller  (~10 s)',PEACH,29,True)
label(1270,223,460,44,'Shield  (~1 s)',PEACH,29,True)
text(640,301,'Forecast → evaluate feasible layouts',27)
text(640,337,'Choose M / P / D counts, clocks, and threshold τ',25)
text(640,373,'Minimize modeled energy under SLO / capacity limits',24)
text(1290,301,'Observe latency / stall risk',26)
text(1290,337,'Raise clocks → wake capacity',25)
text(1290,373,'Can trigger an early replan',25)
text(930,412,'Apply role / clock / admission changes with transition guards',24,color=MUTED,ha='center')
rect(1820,206,540,220,LIGHT)
text(2090,242,'Profiles + SLO targets',29,True,ha='center')
text(1845,281,'Prefill time, decode step, KV capacity',25)
text(1845,319,'Transfer model, power, supported domain',24)
text(1845,357,'TTFT limit; TPOT limit',26)
text(1845,396,'Pressure gating is a policy variant',24,color=MUTED)
arrow(1820,316,1750,316)
arrow(1175,426,1175,456)
label(600,456,1150,44,'Committed state: roles + accepting flags + τ + current frequencies',PEACH,27,True)
text(45,481,'Feedback from all execution paths',24,color=MUTED,italic=True)
text(1820,480,'In-flight endpoints stay bound.',25,True)

# B: exact two-stage route selection, with SLO routing in the current runner.
panel(551,'(b) Per-request decision: select a path, evaluate candidates, then bind (prefill, decode)')
rect(40,581,335,189)
text(207,620,'1  Request R',30,True,ha='center')
text(62,662,'X: prompt; L = |X|',27)
text(62,702,'K: output-token budget',27)
text(62,744,'Read current layout / loads',24)
arrow(375,675,420,675)
rect(420,581,635,189)
text(737,620,'2  Baseline path and candidates C₀',29,True,ha='center')
text(444,661,'PD if a compatible pair exists AND',27)
text(469,699,'(no accepting M  OR  L ≥ τeff)',29,True)
text(444,740,'Otherwise M; neither available → HTTP 503',25)
arrow(1055,675,1100,675)
rect(1100,581,695,189)
text(1447,620,'3  SLO refinement (runner default: ON)',29,True,ha='center')
text(1124,661,'Check per-endpoint KV reservations + length',26)
text(1124,699,'Predict queued prefill, decode, and PD handoff',26)
text(1124,740,'Also test M alternatives only when C₀ is PD',25)
arrow(1795,675,1840,675)
rect(1840,581,520,189,PEACH)
text(2100,620,'4  Bind once at dispatch',29,True,ha='center')
text(1865,661,'M:  (p, d) = (m, m)',28,True)
text(1865,699,'PD: (p, d) = compatible pair',27,True)
text(1865,740,'Reserve Uₚ += L; Nᵈ += 1',26)

rect(40,798,775,370,LIGHT)
text(65,839,'Path scope and baseline assignment',29,True)
text(65,882,'τeff = τ; pressure variant: max(1024, τ)',28)
text(65,923,'M candidates: all accepting instances with role M',25)
text(65,963,'PD candidates: all accepting, compatible P × D pairs',24)
line(65,986,790,986,'#888',.7)
text(65,1024,'M load rule:   min (Nₘ, Uₘ)',30,True)
text(65,1066,'PD load rule: min (Uₚ, Nᵈ, Uᵈ, pair IDs)',29,True)
text(65,1106,'Used when SLO routing is OFF or for its fallback.',25)
text(65,1144,'U counts whole prompts until their first token arrives.',24,color=MUTED)

rect(845,798,950,370)
text(870,839,'SLO selection order  (current homogeneous runner)',29,True)
text(870,881,'A. Safe original candidates exist → min predicted TTFT',27,True)
text(870,923,'B. Else, original path = PD and a safe M exists:',27)
text(908,963,'spill the new request to M with minimum predicted TTFT',25)
text(870,1005,'C. Else, a capacity-admitted original candidate exists:',26)
text(908,1045,'keep baseline preference / fallback; SLO is unproven',25)
text(870,1087,'D. Else → reject (HTTP 503)',28,True)
text(870,1118,'TTFT ties use route IDs; the baseline load tie-break is not retained.',23,color=MUTED)
text(870,1150,'Safe: predicted TTFT / TPOT ≤ 0.85 × limits; protect M incumbents.',23,color=MUTED)

rect(1840,798,520,370,LIGHT)
text(1865,839,'Compatibility and boundaries',28,True)
text(1865,882,'P ≠ D; equal model, TP, PP,',25)
text(1865,920,'pool ID and generation; PP = 1',25)
text(1865,967,'PD binds both ends before P starts.',24,True)
text(1865,1007,'No re-selection of D after prefill.',25)
text(1865,1055,'Role changes affect new requests.',25)
text(1865,1095,'Parking closes admission, then drains.',24)
text(1865,1140,'A PD pair is two engine instances.',25,True)

# C: physical ownership and data plane.
panel(1223,'(c) Executing the assigned request: one Mixed engine, or two separate P / D engines')
for x,w,title in [(40,640,'Mixed pool: selected Mₘ'),(870,605,'Prefill pool: selected Pₚ'),(1730,630,'Decode pool: selected Dᵈ')]:
    rect(x+13,1252,w-13,318,GRAY,lw=.7)
    rect(x+7,1258,w-13,318,LIGHT,lw=.7)
    rect(x,1265,w-13,318)
    label(x+13,1278,w-39,43,title,PEACH,30,True)
    label(x+25,1430,w-63,38,'LLM weights + local KV cache', '#fff3cf',25)
    rect(x+25,1481,w-63,73,'#dce9f5')
    for j in range(4):label(x+43+j*79,1494,68,41,'…' if j==2 else 'GPU', '#dceccf',22)
    text(x+w-60,1524,'TP group',24,ha='right')

label(65,1338,228,64,'Prefill X → y₁',GREEN,28)
arrow(293,1370,323,1370)
label(323,1338,319,64,'Local decode y₂ … yK',BLUE,27,color='white')
text(65,1418,'Same request stays on M; no remote KV handoff',24)

label(895,1338,542,64,'Prefill X; max_tokens = 1 → y₁',GREEN,28)
text(895,1418,'First token is produced by P and sent to client',24)
label(1755,1338,567,64,'Reuse KV(X); consume y₁ → y₂ … yK',BLUE,27,color='white')
text(1755,1418,'Request to D: prompt = X + y₁; max_tokens = K − 1',23)
arrow(1462,1472,1730,1472,lw=2)
text(1595,1418,'Engine-to-engine',24,ha='center')
text(1595,1453,'P2P KV(X)',28,True,ha='center')
text(1595,1513,'May overlap the',23,ha='center')
text(1595,1544,'prefill tail',23,ha='center')
text(714,1365,'OR',32,True,ha='center')
text(40,1622,'Inside each engine: native vLLM V1 continuous batching; running work first, then eligible waiting work (default FCFS).',27)
text(40,1661,'Mixed batches may contain prefill chunks and decode tokens. On first token: Uₚ −= L. On confirmed completion: Nᵈ −= 1.',26)

# D: a single coherent six-request event sequence. Times are invented logical
# slots, not runtime measurements or values produced by a performance model.
scenario = [
    dict(id='R1', arrival=0,L=256,K=8,baseline='M',path='M',p='M0',d='M0',
         prefill=[0,2],outputs=[2,3,4,5,6,7,8,9]),
    dict(id='R2', arrival=1,L=2048,K=6,baseline='PD',path='PD',p='P0',d='D0',
         prefill=[1,5],outputs=[5,7,8,9,10,11],kv=[4,5],handoff_wait=[5,6],first_gap=[5,7]),
    dict(id='R3', arrival=2,L=512,K=5,baseline='M',path='M',p='M1',d='M1',
         prefill=[2,4],outputs=[4,5,6,7,8]),
    dict(id='R4', arrival=3,L=3072,K=5,baseline='PD',path='PD',p='P0',d='D0',
         prefill=[5,9],outputs=[9,11,12,13,14],kv=[8,9],handoff_wait=[9,10],first_gap=[9,11],queue=[3,5]),
    dict(id='R5', arrival=4,L=256,K=4,baseline='M',path='M',p='M0',d='M0',
         prefill=[4,6],outputs=[6,7,8,9]),
    dict(id='R6', arrival=5,L=2048,K=2,baseline='PD',path='M',p='M1',d='M1',
         prefill=[5,9],outputs=[9,10],reason='pd_first_gap_unavailable_spillover'),
]
for r in scenario:
    assert len(r['outputs']) == r['K']
    assert r['arrival'] <= r['prefill'][0] < r['prefill'][1] == r['outputs'][0]
    assert all(a < b for a,b in zip(r['outputs'],r['outputs'][1:]))
    assert r['baseline'] == ('PD' if r['L'] >= 1024 else 'M')
    if r['path']=='PD':
        assert r['p']=='P0' and r['d']=='D0'
        assert r['kv'][1] <= r['handoff_wait'][1] < r['outputs'][1]
        assert r['first_gap'] == r['outputs'][:2]
    else:
        assert r['p']==r['d'] and 'kv' not in r
assert sum(r['K'] for r in scenario)==30

panel(1722,'(d) Six arriving requests: route decisions → per-instance batches → per-request output tokens')
text(40,1762,'Fixed layout: M0–M3 + P0 + D0; τ = 1024; SLO routing ON. Only M0/M1 are expanded; M2/M3 serve background work.',26)
text(40,1798,'The shown safe candidates / TTFT rankings are assumed. Logical slots and bar widths are schematic, not measured or simulated.',25,color=MUTED)

card_notes=[('L < τ → M','Min predicted TTFT: M0'),
            ('L ≥ τ → PD','Compatible, predicted-safe P/D'),
            ('L < τ → M','Min predicted TTFT: M1'),
            ('L ≥ τ → PD','P queue included; D bound now'),
            ('L < τ → M','M0 safe for R1 + new R5'),
            ('PD prediction lacks first gap','Safe M1 → new-request spillover')]
for i,r in enumerate(scenario):
    x=40+i*390
    rect(x,1824,370,185,PEACH if r['id']=='R6' else LIGHT)
    text(x+185,1859,f"{r['id']} arrives @ {r['arrival']}",29,True,ha='center')
    text(x+185,1895,f"L = {r['L']}; K = {r['K']}",26,ha='center')
    text(x+185,1933,card_notes[i][0],24,ha='center',color=RED if i==5 else INK)
    text(x+185,1967,card_notes[i][1],22,ha='center')
    route=r['p'] if r['path']=='M' else 'P0 → D0'
    text(x+185,1996,f"Assign: {route}",27,True,ha='center')

X0, DX = 400, 136
def tx(t):return X0+DX*t
def segment(start,end,y,h,s,fill,size=25):
    label(tx(start),y,DX*(end-start),h,s,fill,size,color='white' if fill==BLUE else INK)
def decoding(rid,events,y,h=48):
    for n,t in enumerate(events,2):segment(t-1,t,y,h,f'{rid}: y{n}',BLUE,23)
def instance_frame(y,name,subtitle):
    rect(40,y,2320,96,LIGHT)
    line(385,y,385,y+96,'#777',.8)
    text(58,y+37,name,30,True)
    text(58,y+74,subtitle,23,color=MUTED)

text(40,2054,'Logical slot',27,True)
arrow(tx(0),2060,2360,2060)
for t in range(15):
    line(tx(t),2052,tx(t),2067)
    text(tx(t),2045,str(t),25,ha='center')
    line(tx(t),2198,tx(t),2806,'#dddddd',.65,True)
text(40,2110,'Request arrival',27,True)
text(40,2160,'Final assignment',27,True)
for i,r in enumerate(scenario):
    x=tx(r['arrival'])
    text(x,2103,r['id'],28,True,ha='center')
    arrow(x,2113,x,2133,color=RED if i==5 else INK)
    route=r['p'] if r['path']=='M' else 'P0/D0'
    text(x,2160,route,25,True,ha='center',color=RED if i==5 else INK)
text(1330,2103,'R6 is first classified as PD, then admitted to M1.',26,color=RED)
text(1330,2141,'Endpoints stay fixed after each final assignment.',25,color=MUTED)

# Each row inside one engine is a request; aligned colored blocks share a batch.
instance_frame(2200,'Mixed M0','R1 + R5')
segment(0,2,2200,48,'R1 prefill → y1',GREEN)
decoding('R1',scenario[0]['outputs'][1:],2200)
segment(4,6,2248,48,'R5 prefill → y1',GREEN)
line(tx(5),2248,tx(5),2296,'#526442',.8,True)
decoding('R5',scenario[4]['outputs'][1:],2248)
text(tx(9)+23,2235,'4–6: P(R5) + D(R1)',26,True)
text(tx(9)+23,2273,'6–9: D(R1) + D(R5)',26)
text(tx(4),2324,'New R5 joins M0; aligned rows belong to one batch.',24,color=MUTED)

instance_frame(2350,'Mixed M1','R3 + R6')
segment(2,4,2350,48,'R3 prefill → y1',GREEN)
decoding('R3',scenario[2]['outputs'][1:],2350)
segment(5,9,2398,48,'R6 prefill chunks → y1',GREEN)
for t in (6,7,8):line(tx(t),2398,tx(t),2446,'#526442',.8,True)
decoding('R6',scenario[5]['outputs'][1:],2398)
text(tx(10)+22,2386,'5–8: P(R6) + D(R3)',26,True)
text(tx(10)+22,2423,'R6 finishes locally @ 10',25)

text(40,2500,'P0 waiting',26,True)
segment(3,5,2473,36,'R4 waits for P budget',GRAY,23)
text(tx(5)+22,2499,'D0 was already selected for R4 at arrival @ 3.',25,color=MUTED)
rect(40,2525,2320,56,LIGHT)
text(58,2562,'Prefill P0',30,True)
segment(1,5,2525,56,'R2 prefill → y1 @ 5',GREEN,27)
segment(5,9,2525,56,'R4 prefill → y1 @ 9',GREEN,27)
text(tx(9)+24,2562,'P0 can accept more prompt work …',26,color=MUTED)

text(40,2648,'P0 → D0 KV',27,True)
for r in (scenario[1],scenario[3]):
    a,b=r['kv']; ready=r['handoff_wait'][1]
    segment(a,b,2624,40,r['id']+' KV',GRAY,23)
    arrow(tx(a)+20,2581,tx(a)+20,2624)
    path([(tx(b),2644),(tx(ready),2644),(tx(ready),2710)],lw=1.4)
text(tx(10)+26,2647,'KV transfer may overlap prefill.',25,color=MUTED)

instance_frame(2710,'Decode D0','R2 + R4')
segment(5,6,2710,48,'R2 wait',GRAY,23)
decoding('R2',scenario[1]['outputs'][1:],2710)
segment(9,10,2758,48,'R4 wait',GRAY,23)
decoding('R4',scenario[3]['outputs'][1:],2758)
text(420,2792,'D0 only produces y2 onward; y1 came from P0.',23,color=MUTED)
text(tx(11)+22,2742,'10–11: joint decode',25,True)
text(40,2850,'Same-column blocks within one instance = one continuous batch; M and PD instances can execute at the same time.',26)

text(40,2902,'Client outputs per request (30 tokens total): green outline = first token; blue outline = subsequent decode token',27,True)
for i,r in enumerate(scenario):
    y=2938+i*50
    route=r['p'] if r['path']=='M' else 'P0 → D0'
    text(40,y+7,f"{r['id']}   {route}   (K={r['K']})",25,True)
    arrow(tx(r['arrival']),y+13,2345,y+13,color='#999',lw=.75)
    for k,t in enumerate(r['outputs'],1):
        rect(tx(t)-19,y-17,38,30,GOLD,lw=1.65,edge='#638c43' if k==1 else BLUE)
        text(tx(t),y+5,f'y{k}',20,ha='center')
    if r['path']=='PD':
        a,b=r['outputs'][:2]
        arrow(tx(a)+21,y-3,tx(b)-21,y-3,color=MUTED,dash=True,both=True,lw=.9)
        text((tx(a)+tx(b))/2,y-12,'first gap',20,ha='center',color=MUTED)
        text(tx(a)-27,y-10,'P0',19,ha='right',color='#527d39')
text(40,3240,'R2: TTFT 1→5; first gap 5→7.   R4: TTFT 3→9 includes P waiting; first gap 9→11.   P produces 2 tokens; D produces 9.',25)
text(40,3281,'R6: missing PD first-gap coverage alone does not force M; here M1 is assumed capacity-safe, SLO-safe, incumbent-safe, and best by TTFT.',24,color=MUTED)
text(40,3318,'M2/M3 background load makes them less preferred in this example. R4 waits because earlier P work occupies the available scheduling budget.',24,color=MUTED)
text(40,3354,'K is fully generated in this example. All times are logical slots; no measured latency, throughput, or profile qualification is claimed.',24,color=MUTED)

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
boxes=[(t,t.get_window_extent(renderer)) for t in texts]
outside=[]; overlaps=[]
for t,b in boxes:
    if b.x0<0 or b.x1>W or b.y0<0 or b.y1>H:outside.append(t.get_text())
for i,(ta,a) in enumerate(boxes):
    for tb,b in boxes[i+1:]:
        if min(a.x1,b.x1)-max(a.x0,b.x0)>2 and min(a.y1,b.y1)-max(a.y0,b.y0)>2:
            overlaps.append([ta.get_text(),tb.get_text()])
report={'canvas':[W,H], 'text_items':len(texts),'outside_canvas':outside,'text_overlaps':overlaps,
        'timing_is_measured':False,'source_scope':'current working tree; homogeneous PDblend runner',
        'scenario_requests':len(scenario),'scenario_tokens':sum(r['K'] for r in scenario),
        'scenario_event_checks':'passed'}
(OUT/'validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
for ext in ('svg','pdf','png'):
    fig.savefig(OUT/f'pdblend-online-scheduling.{ext}',dpi=160,facecolor='white')
fig.savefig(OUT/'pdblend-online-scheduling-preview.png',dpi=75,facecolor='white')
# Export panel (d) separately from the vector canvas for easy reading and reuse.
panel_bbox=Bbox.from_extents(0,0,W/100,(H-1682)/100)
for ext in ('svg','pdf','png'):
    fig.savefig(OUT/f'pdblend-multi-request-detail.{ext}',dpi=160,facecolor='white',bbox_inches=panel_bbox)
fig.savefig(OUT/'pdblend-multi-request-detail-preview.png',dpi=80,facecolor='white',bbox_inches=panel_bbox)
(OUT/'multi-request-scenario.json').write_text(json.dumps(dict(
    kind='illustrative_event_sequence', timing_is_measured=False, performance_simulation=False,
    threshold_tokens=1024, requests=scenario),indent=2)+'\n')
files=['src/pdblend/online/router.py','src/pdblend/online/controller.py','src/pdblend/online/server.py',
       'src/pdblend/online/shield.py','src/pdblend/planner/pool.py','src/pdblend/bench/pdblend_runtime_options.py',
       'src/pdblend/bench/run.py','src/pdblend/engine/carry.py','src/pdblend_runtime/native_v1.py']
(OUT/'source-manifest.json').write_text(json.dumps({p:hashlib.sha256((REPO/p).read_bytes()).hexdigest()
    for p in files},indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
