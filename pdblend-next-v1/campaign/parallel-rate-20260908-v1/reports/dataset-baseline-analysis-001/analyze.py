import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
H = OUT.parent / 'historical-existing-fixed-slo-scope-002'
NEW = Path('/root/workspace/pdblend-next-v1/campaign/slo-rate-14b-sharegpt-20260909-v1/reports/baseline-mechanism-analysis-001')
systems = ['pdblend', 'mixed', 'distserve', 'dynamollm', 'ecoserve']
datasets = ['alpaca', 'sharegpt', 'longbench']
sources = {}

def pin(p, expected=None):
    p = Path(p)
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, p
    sources[str(p)] = digest
    return digest

def read_csv(p):
    pin(p)
    return list(csv.DictReader(Path(p).open()))

def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

rows = read_csv(H / 'scientific-comparison-points.csv')
tokens = read_csv(H / 'request-counts.csv')
pin(H / 'scientific-comparison-overlay.json')
pin(NEW / 'ANALYSIS.md')
pin(NEW / 'regular-pairs.csv')
main = [r for r in rows if r['phase'] == 'main']
assert len(main) == 450
lookup = {(r['model'], r['dataset'], r['rate_rps'], r['system']): r for r in main}
pairs = []
for p in main:
    if p['system'] != 'pdblend' or p['scientific_comparison_eligible'] != 'True':
        continue
    for system in systems[1:]:
        b = lookup[p['model'], p['dataset'], p['rate_rps'], system]
        if b['scientific_comparison_eligible'] != 'True':
            continue
        for k in ['trace_sha256', 'content_pairing_sha256', 'n_requests', 'slo_ttft_s', 'slo_tpot_s']:
            assert p[k] and p[k] == b[k], (p['cell_id'], b['cell_id'], k)
        pairs.append(dict(model=p['model'], dataset=p['dataset'], rate_rps=p['rate_rps'], baseline=system,
            pdblend_cell=p['cell_id'], baseline_cell=b['cell_id'],
            pdblend_at_least_90=float(p['slo_attainment']) >= .9,
            pdblend_attainment_pct=100*float(p['slo_attainment']), baseline_attainment_pct=100*float(b['slo_attainment']),
            energy_reduction_pct=100*(1-float(p['energy_j'])/float(b['energy_j'])),
            energy_per_good_reduction_pct=100*(1-float(p['energy_per_good_request_j'])/float(b['energy_per_good_request_j'])),
            attainment_difference_pp=100*(float(p['slo_attainment'])-float(b['slo_attainment'])),
            goodput_change_pct=100*(float(p['goodput_measurement_rps'])/float(b['goodput_measurement_rps'])-1),
            trace_sha256=p['trace_sha256'], content_pairing_sha256=p['content_pairing_sha256']))
write_csv('historical-main-pairs.csv', pairs)
groups = defaultdict(list)
for p in pairs:
    if p['pdblend_at_least_90']:
        groups[p['dataset'], p['model'], p['baseline']].append(p)
summary = []
for (ds, model, base), ps in sorted(groups.items()):
    record = dict(dataset=ds, model=model, baseline=base, paired_points=len(ps))
    for metric in ['energy_reduction_pct', 'energy_per_good_reduction_pct', 'attainment_difference_pp']:
        record[metric+'_min'] = min(p[metric] for p in ps)
        record[metric+'_max'] = max(p[metric] for p in ps)
    for label, sign in [('higher', 1), ('equal', 0), ('lower', -1)]:
        record['attainment_'+label+'_points'] = sum(((p['attainment_difference_pp'] > 1e-9)-(p['attainment_difference_pp'] < -1e-9)) == sign for p in ps)
    summary.append(record)
write_csv('historical-qualified-ranges.csv', summary)

selected = [('14b','alpaca',6), ('14b','longbench',.5), ('14b','longbench',.75),
            ('32b','alpaca',2.5), ('32b','longbench',.3), ('32b','sharegpt',.6)]
representatives = []
mechanisms = []
for model, ds, rate in selected:
    selected_rows = [r for r in main if r['model']==model and r['dataset']==ds and float(r['rate_rps'])==rate]
    for r in selected_rows:
        record = {k:r[k] for k in ['model','dataset','system','rate_rps','cell_id','n_requests','good_requests',
            'slo_ttft_s','slo_tpot_s','slo_attainment','energy_j','energy_per_good_request_j',
            'ttft_avg_s','tpot_avg_s','goodput_measurement_rps','measurement_duration_s',
            'scientific_comparison_eligible','scientific_exclusion_reason','checkpoint_path']}
        record['mean_eight_gpu_power_w'] = float(r['energy_j'])/float(r['measurement_duration_s'])
        representatives.append(record)
        if r['scientific_comparison_eligible'] != 'True':
            continue
        ckpath = Path(r['checkpoint_path'])
        pin(ckpath)
        ck = json.loads(ckpath.read_text())
        root = ckpath.parent.parent/'cells'/r['cell_id']
        for filename in ['runtime_config.json','bench.csv','control.jsonl','power.csv']:
            p = root / filename
            assert str(p) in ck['artifacts'], p
            pin(p, ck['artifacts'][str(p)])
        cfg = json.loads((root/'runtime_config.json').read_text())
        bench = list(csv.DictReader((root/'bench.csv').open()))
        assert len(bench)==int(r['n_requests'])
        assert all(b['success']=='1' and int(b['generated_tokens'])==int(b['output_len']) for b in bench)
        counts = Counter()
        for b in bench:
            ttft = float(b['ttft_s']) >= float(r['slo_ttft_s'])
            tpot = float(b['tpot_s']) >= float(r['slo_tpot_s'])
            key = 'both' if ttft and tpot else 'ttft_only' if ttft else 'tpot_only' if tpot else 'good'
            counts[key] += 1
        assert counts['good'] == int(r['good_requests'])
        events = [json.loads(l) for l in (root/'control.jsonl').open()]
        admissions = {e['client_request_id']:e for e in events if e['kind']=='admission'}
        timings = {e['client_request_id']:e for e in events if e['kind']=='request_timing'}
        misses=[]
        for b in bench:
            if b['slo_ok']=='1': continue
            a=admissions[b['request_id']]; t=timings[b['request_id']]
            misses.append(dict(request_id=b['request_id'], input_tokens=int(b['input_tokens']),output_tokens=int(b['generated_tokens']),
                ttft_s=float(b['ttft_s']), tpot_s=float(b['tpot_s']), max_itl_s=float(b['max_itl_s']) if b['max_itl_s'] else None,
                route=a['plan']['routes'], pre_forward_wait_s=t['forward_started_s']-t['planned_arrival_s'],
                forward_to_first_token_s=t['first_token_s']-t['forward_started_s']))
        mechanisms.append(dict(cell_id=r['cell_id'], config_path=str(root/'runtime_config.json'),
            instances=[{k:i[k] for k in ['id','tp','role','gpus']} for i in cfg['instances']],
            config={k:cfg.get(k) for k in ['strategy','allow_pd','dvfs','protect_pending_decode','candidate_label','profile_compatibility']},
            miss_counts=dict(counts), misses=misses,
            commanded_frequency_event_counts=dict(Counter(str(y['requested']) for e in events for y in e.get('clock_outcomes',[])))))
write_csv('representative-points.csv',representatives)
(OUT/'request-mechanism-evidence.json').write_text(json.dumps(mechanisms,ensure_ascii=False,indent=2)+'\n')

# Scientific figure: each column is a distinct historical 14B workload/SLO.
# Show all rates, including PDBlend saturation. Invalid measurements create gaps.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors = dict(zip(systems,['#087f8c','#7d8597','#b86d18','#944bb0','#45883d']))
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
for j,ds in enumerate(datasets):
    for sys in systems:
        rr=sorted([r for r in main if r['model']=='14b' and r['dataset']==ds and r['system']==sys],key=lambda r:float(r['rate_rps']))
        rates=[float(r['rate_rps']) for r in rr]
        for i,(metric,scale) in enumerate([('slo_attainment',100),('energy_per_good_request_j',1)]):
            ys=[float(r[metric])*scale if r['scientific_comparison_eligible']=='True' else float('nan') for r in rr]
            axes[i,j].plot(rates,ys,'o-',ms=4,lw=2 if sys=='pdblend' else 1.3,label=sys,color=colors[sys])
    axes[0,j].set_title(ds+' | historical A / 14B / 1x')
    axes[0,j].axhline(90,color='grey',ls=':',lw=1)
    axes[0,j].set_ylim(0,105)
    axes[1,j].set_yscale('log')
    for ax in axes[:,j]:
        ax.set_xlabel('Offered rate (rps)'); ax.grid(alpha=.18)
axes[0,0].set_ylabel('SLO attainment (%)')
axes[1,0].set_ylabel('Eight-GPU joules / good request (log)')
axes[0,2].legend(fontsize=8)
fig.suptitle('Dataset-specific SLOs; paired traces; one arrival seed; engineering exclusions are gaps')
for ext in ['png','pdf','svg']:
    fig.savefig(OUT/f'historical-14b-dataset-curves.{ext}',dpi=180)

print(json.dumps({'main_rows':len(main),'eligible_pairs':len(pairs),'qualified_pairs':sum(p['pdblend_at_least_90'] for p in pairs),
                  'representative_rows':len(representatives),'raw_checked_measurements':len(mechanisms),'pinned_sources':len(sources)},indent=2))
(OUT/'source-manifest.json').write_text(json.dumps(dict(scope='Historical main 1x comparisons; newer ShareGPT results separately sourced',
    validation='Passed pair hash equality and selected raw artifact hashes, completion and SLO replay; not a new scientific qualification',
    source_sha256=sources),ensure_ascii=False,indent=2)+'\n')
