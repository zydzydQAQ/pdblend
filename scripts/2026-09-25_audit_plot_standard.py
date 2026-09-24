#!/usr/bin/env python3
"""Read-only inventory of exported standard-rate observations; no performance selection."""
import csv, datetime as dt, hashlib, io, json, math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/home/pdblend4')
OUT = ROOT / 'results/analysis/plot-inventory-20260925-v1'
SOURCES = {
    'current_compare': OUT / 'compare-snapshot.csv',
    'historical_180': ROOT / 'results/analysis/full-comparison-20260924-v1/selected-points-180.csv',
    'cohort_192': ROOT / 'results/analysis/cohort-energy-20260924-v2/points.csv',
}
TZ = dt.timezone(dt.timedelta(hours=8))
MODELS = ['7B', '14B', '32B']
SYSTEMS = ['mixed', 'dynamollm', 'distserve', 'ecoserve', 'pdblend']
DATASETS = ['alpaca', 'sharegpt', 'longbench']
RATES = [0.25, 0.5, 0.75, 1.0]

def truth(x): return str(x).lower() == 'true'
def num(x):
    try: return math.isfinite(float(x))
    except (ValueError, TypeError): return False
def model(r): return r.get('model') or r['model_id'].removeprefix('Qwen2.5-').removesuffix('-Instruct')
def key(r): return (model(r), r['system'], r['dataset'], float(r['rate_scale']))
def sha(data): return hashlib.sha256(data).hexdigest()
def total(r):
    return num(r.get('energy_service_j')) and num(r.get('energy_tail_j'))
def writecsv(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

meta, loaded = {}, {}
for label, p in SOURCES.items():
    b = p.read_bytes()
    loaded[label] = list(csv.DictReader(io.StringIO(b.decode())))
    meta[label] = {'path': str(p), 'sha256': sha(b), 'rows': len(loaded[label]),
                   'mtime_beijing': dt.datetime.fromtimestamp(p.stat().st_mtime, TZ).isoformat()}
current = loaded['current_compare']
standard = [r for r in current if float(r['rate_scale']) in RATES]
historical = loaded['historical_180']
cohort = loaded['cohort_192']
cohort_by_receipt = {r['receipt_path']: r for r in cohort}
hist_receipts = {r['receipt_path'] for r in historical}

# Check immutable receipt bindings, plus small point/result/metering artifacts.
validation = {'historical_receipt_checks': 0, 'historical_artifact_checks': 0,
              'current_standard_recorded_receipt_checks': 0, 'errors': []}
for r in historical:
    p = Path(r['receipt_path'])
    if not p.exists(): validation['errors'].append('missing '+str(p)); continue
    b = p.read_bytes()
    validation['historical_receipt_checks'] += 1
    if sha(b) != r['receipt_sha256']: validation['errors'].append('hash '+str(p))
    receipt = json.loads(b)
    for name in ['point.json', 'result.json', 'run/comparison-metering.json']:
        if name not in receipt.get('artifacts', {}): continue
        ap = p.parent / name
        validation['historical_artifact_checks'] += 1
        if not ap.exists() or sha(ap.read_bytes()) != receipt['artifacts'][name]:
            validation['errors'].append('artifact '+str(ap))
for r in standard:
    if not truth(r['measurement_usable']): continue
    p = Path(r['receipt_path']); validation['current_standard_recorded_receipt_checks'] += 1
    if not p.exists() or sha(p.read_bytes()) != r['receipt_sha256']:
        validation['errors'].append('current receipt '+str(p))

historical_summary = []
for m in MODELS:
    for s in SYSTEMS:
        rr = [r for r in historical if model(r) == m and r['system'] == s]
        cr = [cohort_by_receipt[r['receipt_path']] for r in rr]
        historical_summary.append(dict(model=m, system=s, expected=12, request_slo_recorded=len(rr),
            service_energy_complete=sum(num(r['energy_service_j']) for r in rr),
            total_energy_complete=sum(truth(r['energy_measurement_complete']) for r in cr),
            all_requests_successful=sum(truth(r['all_requests_successful']) for r in cr),
            slo_pass=sum(truth(r['slo_pass']) for r in cr),
            original_formal_eligible=sum(truth(r['formal_eligible']) for r in rr),
            common_clock_pass=sum(r['common_clock_evidence']=='pass' for r in cr)))

groups = defaultdict(list)
for r in standard: groups[key(r)].append(r)
histgroups = {key(r): r for r in historical}
cells = []
for m in MODELS:
    for s in SYSTEMS:
        for d in DATASETS:
            for rate in RATES:
                rr = groups[(m,s,d,rate)]
                measured = [r for r in rr if truth(r['measurement_usable'])]
                h = histgroups[(m,s,d,rate)]
                hc = cohort_by_receipt[h['receipt_path']]
                cells.append(dict(model=m,system=s,dataset=d,rate_scale=rate,
                    configured_offered_rps=h['offered_rps'],seed=h['seed'],duration_s=h['duration_s'],
                    historical_recorded=True,historical_service_energy_complete=num(h['energy_service_j']),
                    historical_total_energy_complete=truth(hc['energy_measurement_complete']),
                    historical_slo_pass=truth(hc['slo_pass']),historical_original_formal_eligible=truth(h['formal_eligible']),
                    current_export_rows=len(rr),current_recorded_observations=len(measured),
                    current_prepared_rows=sum(r['status']=='prepared' for r in rr),
                    current_failed_without_usable_request_rows=sum(r['status']=='failed' and not truth(r['measurement_usable']) for r in rr),
                    current_service_energy_observations=sum(truth(r['analysis_energy_usable']) for r in measured),
                    current_total_energy_observations=sum(total(r) for r in measured),
                    current_slo_pass_observations=sum(truth(r['analysis_slo_pass']) for r in measured),
                    current_original_formal_observations=sum(truth(r['formal_eligible']) for r in measured),
                    measured_revisions=json.dumps(sorted({r['revision'] for r in measured})),
                    historical_receipt_path=h['receipt_path'],historical_receipt_sha256=h['receipt_sha256'],
                    current_measured_receipts=json.dumps([r['receipt_path'] for r in measured])))

revgroups = defaultdict(list)
for r in standard: revgroups[(model(r),r['system'],r['revision'])].append(r)
revisions = []
for (m,s,rev),rr in sorted(revgroups.items()):
    measured = [r for r in rr if truth(r['measurement_usable'])]
    unique = {key(r) for r in measured}
    repeated = {}
    for k, values in groups.items():
        if k[0] != m or k[1] != s: continue
        count = sum(r['revision']==rev and truth(r['measurement_usable']) for r in values)
        if count > 1: repeated[str(k)] = count
    revisions.append(dict(model=m,system=s,revision=rev,export_rows=len(rr),recorded_observations=len(measured),
        distinct_standard_conditions=len(unique),prepared_rows=sum(r['status']=='prepared' for r in rr),
        failed_rows=sum(r['status']=='failed' for r in rr),service_energy_observations=sum(truth(r['analysis_energy_usable']) for r in measured),
        total_energy_observations=sum(total(r) for r in measured),slo_pass_observations=sum(truth(r['analysis_slo_pass']) for r in measured),
        formal_observations=sum(truth(r['formal_eligible']) for r in measured),
        measured_points=json.dumps(sorted(r['point_id'] for r in measured)),duplicate_condition_observations=json.dumps(repeated)))

rows = []
for r in standard:
    measured=truth(r['measurement_usable'])
    rows.append(dict(model=model(r),system=r['system'],dataset=r['dataset'],rate_scale=r['rate_scale'],
        offered_rps=r['offered_rps'],seed=r['seed'],duration_s=r['duration_s'],point_id=r['point_id'],revision=r['revision'],
        status=r['status'],request_slo_recorded=measured,service_energy_complete=truth(r['analysis_energy_usable']),
        total_energy_complete=measured and total(r),slo_pass=truth(r['analysis_slo_pass']),formal_eligible=truth(r['formal_eligible']),
        is_energy_supplement=r.get('is_energy_supplement'),energy_supplement_classification=r.get('energy_supplement_classification'),
        historical_selected=r['receipt_path'] in hist_receipts,
        offered_requests=r['offered_requests'],successful_requests=r['successful_requests'],joint_slo_requests=r['joint_slo_requests'],
        energy_service_j=r['energy_service_j'],energy_tail_j=r['energy_tail_j'],
        receipt_path=r['receipt_path'],receipt_sha256=r['receipt_sha256']))

summary = dict(schema='pdblend-standard-plot-inventory/v1',generated_at=dt.datetime.now(TZ).isoformat(),
    audit_script='/home/pdblend4/scripts/2026-09-25_audit_plot_standard.py',
    agents_md='No AGENTS.md found in /, /home, /home/pdblend4 or workspace results tree.',sources=meta,
    scope={'models':MODELS,'datasets':DATASETS,'systems':SYSTEMS,'rates':RATES,'standard_conditions':180},
    definitions={'recorded':'measurement_usable / bound 150 s request metrics, independent of successful requests, energy gaps or formal qualification',
        'service_energy':'finite recorded 150 s service energy; current analysis_energy_usable',
        'total_energy':'service and tail both present; historical cohort energy_measurement_complete',
        'slo_pass':'all requests successful; unresolved=0; joint SLO >=90%; TTFT/TPOT P99 within thresholds',
        'no_selection':'All revisions and repeated observations retained; no best-performance selection and no cross-revision splicing.'},
    historical={'recorded_conditions':180,'service_energy_complete':sum(r['service_energy_complete'] for r in historical_summary),
        'total_energy_complete':sum(r['total_energy_complete'] for r in historical_summary),
        'slo_pass':sum(r['slo_pass'] for r in historical_summary),'formal_flags':sum(r['original_formal_eligible'] for r in historical_summary),
        'by_model_system':historical_summary},
    current_standard_export={'rows':len(standard),'status_counts':dict(Counter(r['status'] for r in standard)),
        'recorded_observations':sum(truth(r['measurement_usable']) for r in standard),
        'recorded_unique_conditions':sum(c['current_recorded_observations']>0 for c in cells),
        'service_energy_observations':sum(truth(r['analysis_energy_usable']) for r in standard),
        'total_energy_observations':sum(truth(r['measurement_usable']) and total(r) for r in standard),
        'any_revision_service_energy_conditions':sum(c['current_service_energy_observations']>0 for c in cells),
        'any_revision_total_energy_conditions':sum(c['current_total_energy_observations']>0 for c in cells),
        'missing_service_energy_conditions':[{k:c[k] for k in ['model','system','dataset','rate_scale']} for c in cells if not c['current_service_energy_observations']],
        'missing_total_energy_conditions':[{k:c[k] for k in ['model','system','dataset','rate_scale']} for c in cells if not c['current_total_energy_observations']],
        'by_revision':revisions},
    validation=validation,
    caveats=['Historical 180 remains reproducible, but is not the newest observation set.',
        'Full-comparison report captured 2026-09-24 14:33 and cohort report captured 18:32; compare.csv is newer and includes additional revisions/supplements.',
        'Current export does not establish that every raw result directory has been imported; other agents audit raw newer runs.',
        '47 historical formal flags = Mixed 36 and EcoServe 11, but Mixed common-clock evidence is unknown and method qualification is distinct from data availability.',
        'Historical old status failed 20 reflects missing energy/acceptance and does not mean absence of request/SLO measurement.',
        'Union energy coverage across revisions is an inventory fact, not permission to build a spliced curve.'])
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'agent-standard.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
writecsv('agent-standard-historical-groups.csv',historical_summary)
writecsv('agent-standard-cells.csv',cells)
writecsv('agent-standard-revisions.csv',revisions)
writecsv('agent-standard-observations.csv',rows)
md=['# 标准矩阵盘点（所有版本保留）','',f"审计时间：{summary['generated_at']}。输入为主控冻结 compare-snapshot.csv，{len(current)} 行，SHA256 {meta['current_compare']['sha256']}。",'',
    '标准坐标为 Qwen2.5 7B/14B/32B × Alpaca/ShareGPT/LongBench × Mixed/DynamoLLM/DistServe/EcoServe/PDblend × 0.25/0.5/0.75/1.0，共 180 点。历史完整矩阵 180/180 已有请求/SLO 记录，每模型每系统 12/12，每数据集 4 个倍率齐全。',
    '', '历史完整轮：150 秒服务能耗 160/180，服务+尾部能耗 157/180；不是全部点均 SLO 达标。', '',
    '| 模型 | 系统 | 已测 | 服务能耗 | 总能耗 | SLO通过 | 原formal标志 |', '|---|---|---:|---:|---:|---:|---:|']
for r in historical_summary:
    md.append(f"| {r['model']} | {r['system']} | {r['request_slo_recorded']}/12 | {r['service_energy_complete']}/12 | {r['total_energy_complete']}/12 | {r['slo_pass']}/12 | {r['original_formal_eligible']}/12 |")
cs=summary['current_standard_export']
md += ['',f"当前导出标准倍率共有 {cs['rows']} 行，其中已测 {cs['recorded_observations']} 次、prepared {cs['status_counts'].get('prepared',0)} 行、failed {cs['status_counts'].get('failed',0)} 行；去掉版本/重复后仍为 {cs['recorded_unique_conditions']}/180 个坐标。不能把 {cs['recorded_observations']} 次观测当成同一版本矩阵。",'',
    f"跨版本至少一次有能耗的坐标：服务 {cs['any_revision_service_energy_conditions']}/180，总能耗 {cs['any_revision_total_energy_conditions']}/180。这是库存覆盖，不能直接拼接曲线。",'',
    '| 模型 | 系统 | revision | 已测次数/坐标 | prepared | 服务/总能耗 |', '|---|---|---|---:|---:|---:|']
for r in revisions:
    md.append(f"| {r['model']} | {r['system']} | {r['revision'][:12]} | {r['recorded_observations']}/{r['distinct_standard_conditions']} | {r['prepared_rows']} | {r['service_energy_observations']}/{r['total_energy_observations']} |")
md += ['', '旧 full-comparison（9 月 24 日 14:33 快照）和 cohort-energy（18:32 快照）均已不是当前完整库存，但历史矩阵能完整追溯。', '',
    f"核验历史 receipt {validation['historical_receipt_checks']} 个、绑定 point/result/metering 文件 {validation['historical_artifact_checks']} 个；当前已测标准 receipt {validation['current_standard_recorded_receipt_checks']} 个。错误 {len(validation['errors'])} 个。", '',
    '47 个历史 formal 标志来自 Mixed 36 + EcoServe 11；Mixed 36 点仍缺统一实测频率证据。PDblend/DistServe/DynamoLLM 历史正式资格为 0。不要把观测数据可画图等同于论文机制完整资格。', '',
    '逐坐标与来源：agent-standard-cells.csv（180 行）；所有标准观测及计划：agent-standard-observations.csv；按版本汇总：agent-standard-revisions.csv；历史模型/系统汇总：agent-standard-historical-groups.csv。JSON 保存来源 SHA256、统计定义和验证结果。', '',
    '复核脚本：`/home/pdblend4/scripts/2026-09-25_audit_plot_standard.py`；只读冻结快照和原始证据，写自身 agent-standard 审计产物。']
(OUT/'agent-standard.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'current_standard_export':{k:v for k,v in cs.items() if k!='by_revision'},'validation':validation},ensure_ascii=False,indent=2))
