"""Separate source, profile and policy versions in rate reports."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from boundary_first_loss_v2 import next_boundary,paired_jobs

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def enrich(point,binding,config,checkpoint):
    host=json.loads((Path(binding['host_release'])/'manifest.json').read_text())
    point['controller_source_sha256']=host.get('common_controller_sha256',digest(host['files']))
    # Operator paths/ports identify artifacts, not the serving policy. Everything
    # else is retained; do not silently merge a changed capacity/cost/SLO policy.
    excluded={'journal','port','profiles','host_source_release','controller_source_release',
        'engine_source_release','candidate_label','profile_compatibility','transfer_evidence',
        'frequency_evidence','interconnect'}
    policy={k:v for k,v in config.items() if k not in excluded and k!='instances'}
    policy['instances']=sorted([dict(tp=i['tp'],gpus=sorted(i['gpus']),role=i['role'])
        for i in config['instances']],key=lambda i:i['gpus'])
    point['policy_sha256']=digest(policy)
    point['version_id']=digest(dict(source=point['controller_source_sha256'],
        profile=point['profile_sha256'],policy=point['policy_sha256']))
    point['version_label']=point['implementation_id']+' / profile '+point['profile_sha256'][:8]
    point['configured_gpu_count']=len({g for i in config['instances'] for g in i['gpus']})
    point['configured_instance_count']=len(config['instances'])
    point['configured_tp_sizes']=[i['tp'] for i in config['instances']]
    point['energy_measured_gpu_count']=8
    point['dynamic_pools_enabled']=bool(config.get('dynamic_pools'))
    point['slow_topology_enabled']=bool(config.get('slow_topology'))
    point['service_frequency_ceiling_mhz']=config.get('max_service_frequency_mhz',2520)
    duration=point['measurement_duration_s']
    point['completed_work_throughput_rps']=point['completed_work_requests']/duration
    point['generated_token_throughput_tps']=point['generated_tokens']/duration
    point['gpu_util_per_gpu']=point['verification']['gpu_util_per_gpu']
    point['utilized_gpu_count']=sum(x>0 for x in point['gpu_util_per_gpu'])
    return point

def groups(points):
    result=defaultdict(list)
    for x in points:
        if x.get('measurement_valid'):
            result[(x['model'],x['dataset'],x['version_id'])].append(x)
    return result

def boundary_records(points):
    records=[]
    for (model,dataset,version),values in sorted(groups(points).items()):
        decision=next_boundary([dict(x,implementation_id=version) for x in values])
        decision.update(model=model,dataset=dataset,version_id=version,
            version_label=values[0]['version_label'],system='pdblend',
            controller_source_sha256=values[0]['controller_source_sha256'],
            profile_sha256=values[0]['profile_sha256'],policy_sha256=values[0]['policy_sha256'],
            configured_gpu_count=values[0]['configured_gpu_count'],
            independent_baseline_boundary_required=False,
            measured_rates=sorted({x['rate_rps'] for x in values}),
            note='Configuration service boundary only; incomplete requests do not establish saturation.')
        records.append(decision)
    return records

def adaptive(out,points):
    decisions=boundary_records(points)
    jobs=[]
    for d in decisions:
        jobs.extend(dict(j,model=d['model'],dataset=d['dataset'],version_id=d['version_id'])
            for j in paired_jobs([d]))
    (out/'adaptive-next-plan.json').write_text(json.dumps(dict(planning_only=True,measured=False,
        decisions=decisions,five_system_trace_jobs=jobs),indent=2)+'\n')

def matrix_by_version(points,pairs):
    by_cell={p['cell_id']:p for p in points}
    grouped=defaultdict(list)
    for pair in pairs:
        point=by_cell[pair['cell_id']]
        grouped[(point['model'],point['dataset'],point['version_id'],pair['baseline_system'])].append(pair)
    result=[]
    for (model,dataset,version,baseline),values in sorted(grouped.items()):
        p=by_cell[values[0]['cell_id']]
        result.append(dict(model=model,dataset=dataset,version_id=version,version_label=p['version_label'],
            baseline=baseline,verified_executions=len(values),rates=sorted({v['rate_rps'] for v in values}),
            development_passes=sum(v['passed'] for v in values),
            strict_service_energy_passes=sum(v['strict_service_energy_pass'] for v in values),
            configured_gpu_count=p['configured_gpu_count'],profile_sha256=p['profile_sha256'],
            controller_source_sha256=p['controller_source_sha256'],policy_sha256=p['policy_sha256']))
    return result

def figures(out,originals,points,series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    styles={'mixed':'#B57A13','distserve':'#21886B','dynamollm':'#9061AF','ecoserve':'#58768A'}
    metrics=[('slo_attainment','SLO attainment (%)',100),
        ('energy_j','All-eight-GPU energy (kJ)',.001),
        ('energy_per_good_request_j','J / SLO-qualified request',1),
        ('completion_fraction','Prescribed work completion (%)',100),
        ('goodput_measurement_rps','Goodput over full window (req/s)',1),
        ('completed_work_throughput_rps','Completed work / full window (req/s)',1),
        ('generated_token_throughput_tps','Generated tokens / full window (tok/s)',1),
        ('ttft_avg_s','Mean TTFT (s)',1),('tpot_avg_s','Mean TPOT (ms)',1000),
        ('gpu_util','Mean eight-GPU utilization (%)',100)]
    def value(x,field,scale):
        v=x.get(field)
        if field=='completion_fraction':v=x['completed_work_requests']/x['n_expected']
        if v is None and field in ('completed_work_throughput_rps','generated_token_throughput_tps'):
            n=x.get('completed_work_requests' if field=='completed_work_throughput_rps' else 'generated_tokens')
            t=x.get('measurement_duration_s')
            if n is not None and t:v=n/t
        return v*scale if v is not None else float('nan')
    with PdfPages(out/'rate-curves.pdf') as pdf:
        for field,label,scale in metrics:
            fig,axes=plt.subplots(3,3,figsize=(16,11),layout='constrained')
            for i,m in enumerate(('7b','14b','32b')):
                for j,d in enumerate(('alpaca','sharegpt','longbench')):
                    ax=axes[i,j]
                    for system,color in styles.items():
                        rows=sorted([x for x in originals if x['model']==m and x['dataset']==d and x['system']==system],key=lambda x:x['rate_rps'])
                        for baseline_repeat in sorted({x.get('repeat',1) for x in rows}):
                            repeated=[x for x in rows if x.get('repeat',1)==baseline_repeat]
                            ax.plot([x['rate_rps'] for x in repeated],[value(x,field,scale) for x in repeated],
                                label=system+(' (frozen policy)' if baseline_repeat==1 else ' repeat '+str(baseline_repeat)),
                                color=color,marker='.' if baseline_repeat==1 else 's',lw=.9,
                                ls='-' if baseline_repeat==1 else '--',alpha=.65)
                        bad=[x for x in rows if not x['work_complete']]
                        ax.scatter([x['rate_rps'] for x in bad],[value(x,field,scale) for x in bad],marker='x',color=color,s=40,zorder=4)
                    current=[x for x in points if x['model']==m and x['dataset']==d and x['measurement_valid']]
                    for vi,version in enumerate(sorted({x['version_id'] for x in current})):
                        local=[x for x in current if x['version_id']==version]
                        color=('#CD423A','#772D36','#E37E50','#AB3DB5')[vi%4]
                        for rep in sorted({x['repeat'] for x in local}):
                            rows=sorted([x for x in local if x['repeat']==rep],key=lambda x:(x['rate_rps'],x['cell_id']))
                            ax.plot([x['rate_rps'] for x in rows],[value(x,field,scale) for x in rows],
                                label=f"PDB {local[0]['profile_sha256'][:6]} / r{rep} / {local[0]['configured_gpu_count']} GPU",
                                color=color,marker='o' if rep==1 else 's',ls='-' if rep==1 else '--',lw=1.5)
                            bad=[x for x in rows if not x['work_complete']]
                            ax.scatter([x['rate_rps'] for x in bad],[value(x,field,scale) for x in bad],marker='x',color='black',s=80,zorder=5)
                    if field=='slo_attainment':
                        ax.axhline(90,color='#555555',ls=':',lw=1)
                        ax.set_ylim(0,105)
                    ax.set_title(f'{m} / {d}');ax.set_xlabel('Offered rate (req/s)');ax.set_ylabel(label);ax.grid(alpha=.2)
                    ax.legend(fontsize=5.6,loc='best')
            fig.suptitle(f'{series}: source/profile/policy and repeats kept separate; x = incomplete work',fontsize=12)
            pdf.savefig(fig)
            fig.savefig(out/(field+'-rate.png'),dpi=160)
            fig.savefig(out/(field+'-rate.svg'))
            plt.close(fig)
