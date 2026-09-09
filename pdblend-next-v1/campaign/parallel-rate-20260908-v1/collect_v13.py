"""Read completed immutable checkpoints; recheck workload and eight-board energy."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import re
import inspect_exploration
import report_versions_v6 as report_versions
import inspect_baseline_v3 as inspect_baseline
import source_identity_v2 as source_identity
import audit_dynamic
import raw_metrics_v3
import report_scope

ROOT=Path(__file__).resolve().parent
AUDIT=ROOT.parent/'main-slo-improvement-v7'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def load_audit():
    # Load the existing independent arithmetic verifier, not a GPU producer.
    sys.path.insert(0,str(AUDIT))
    spec=importlib.util.spec_from_file_location('parallel_rate_raw_audit',AUDIT/'report.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.raw_metrics=raw_metrics_v3
    return module


def stage_paths():
    paths={p.parent/'status.json' for node in ('A','B','C')
        for p in (ROOT/node).rglob('declaration-order.json') if (p.parent/'status.json').exists()}
    if (ROOT/'B/lb025-repeat2-declaration.json').exists():
        paths.add(ROOT/'B/adjacent-control-p4-001/status.json')
    return sorted(paths)


def stage_inputs(status_path):
    directory=status_path.parent
    if directory==ROOT/'B/adjacent-control-p4-001':
        path=ROOT/'B/lb025-repeat2-declaration.json'
        declared=read(path)
        if declared['schema']!='B-adjacent-LB025-second-repeat-v1':raise ValueError('unknown B adjacent declaration')
        status=read(status_path) if status_path.exists() else dict(phase='waiting_for_original_queue',complete=False)
        status.update(model='32b',stage='screen_fixed2')
        return status,path,[declared['cell']]
    path=directory/'declaration-order.json'
    return read(status_path),path,read(path)


def collect(out,series):
    if out.exists():raise FileExistsError(out)
    audit=load_audit();p=audit.p
    code_sources={str(path):sha(path) for path in (Path(__file__),ROOT/'boundary_first_loss.py',ROOT/'inspect_exploration.py',ROOT/'report_versions_v6.py',ROOT/'inspect_baseline_v3.py',ROOT/'source_identity.py',ROOT/'source_identity_v2.py',ROOT/'audit_dynamic.py',ROOT/'raw_metrics_v3.py',ROOT/'raw_metrics_v2.py',ROOT/'report_scope.py',ROOT/'boundary_first_loss_v2.py',AUDIT/'report.py',AUDIT/'protocol.py',AUDIT/'raw_metrics.py')}
    originals=p.original_points()
    identity_cache={}
    for old in originals:
        if old['system']=='pdblend':continue
        key=(old['executed_source']['binding_path'],old['dataset'])
        if key not in identity_cache:identity_cache[key]=source_identity.original_identity(old)
        old.update(identity_cache[key])
    baseline={x['cell_id']:x for x in originals if x['system']!='pdblend'}
    original_pdb_by_pair={p.pair_identity(x):x['cell_id'] for x in originals if x['system']=='pdblend'}
    original_pdb_ids=set(original_pdb_by_pair.values())
    original_baselines_by_pair=defaultdict(dict)
    for x in baseline.values():original_baselines_by_pair[p.pair_identity(x)][x['system']]=x['cell_id']
    declarations={};checkpoints={};attempts=[];sources=dict(code_sources);origins=defaultdict(list)
    for status_path in stage_paths():
        status,order_path,order=stage_inputs(status_path);directory=status_path.parent
        if status.get('stage') not in ('screen_fixed2','screen_dynamic','confirm_dynamic','confirm_fixed2'):continue
        match=re.search(r'parallel-rate-(p[0-9]+)',order[0]['cell_id']) if order else None
        if match is None:match=re.search(r'(?:^|[-/])(p[0-9]+)(?:[a-z]*[-/]|$)',str(directory.relative_to(ROOT)))
        version=match.group(1) if match else 'p1'
        if version!=series:continue
        attempts.append(dict(model=status.get('model'),path=str(directory),pid=status.get('pid'),
            phase=status.get('phase'),declared=len(order),completed=len(status.get('completed',[])),
            attempted=len(status.get('attempted',[])),failed=status.get('failed',[]),
            error=status.get('error'),current_cell=status.get('current_cell'),
            node_lease_held=status.get('node_lease_held'),updated_s=status.get('updated_s',status.get('finished_s'))))
        sources[str(order_path)]=sha(order_path)
        for cell in order:
            cid=cell['cell_id']
            if cid in declarations and declarations[cid]!=cell:raise ValueError('conflicting cell declaration '+cid)
            declarations[cid]=cell
            origins[cid].append(dict(path=str(directory),status=status,cell=cell))
            cp=directory/'results/checkpoints'/(cid+'.json')
            if cp.exists():
                if cid in checkpoints and checkpoints[cid]!=cp:raise ValueError('duplicate measured cell '+cid)
                checkpoints[cid]=cp
    points=[];pairs=[]
    for cid,cell in declarations.items():
        cp_path=checkpoints.get(cid)
        if cp_path:
            cp=read(cp_path)
            required=[cp['receipt'],cp['binding'],cell['trace']['path'],*cp['artifacts']]
            missing=[path for path in required if not Path(path).is_file()]
        else:missing=[]
        if missing:
            point=(audit.inspect(cell,None) if 'original_point' in cell else inspect_exploration.inspect(audit,cell,None))
            point.update(status='awaiting_mirror',error='completed checkpoint seen; raw evidence not yet mirrored',missing_artifact_count=len(missing))
        else:
            point=(audit.inspect(cell,cp_path) if 'original_point' in cell else inspect_exploration.inspect(audit,cell,cp_path))
        point['candidate_series']=series
        original_id=original_pdb_by_pair.get(p.pair_identity(point))
        if original_id is not None:
            point.update(declared_original_cell_id=point['original_cell_id'],original_cell_id=original_id,new_rate=False)
        else:point['new_rate']=True
        point['service_qualified']=bool(point['measurement_valid'] and point['work_complete'] and point['slo_attainment']>=.90)
        if cp_path and point['measurement_valid']:
            binding=read(read(cp_path)['binding'])
            point['host_manifest_sha256']=sha(Path(binding['host_release'])/'manifest.json')
            config=read(binding['configs'][cell['dataset']])
            point['profile_sha256']=sha(Path(config['profiles']))
            report_versions.enrich(point,binding,config,cp_path)
            point.update(audit_dynamic.inspect(point,cp_path))
            point['failed_requests']=read(read(cp_path)['receipt'])['summary'].get('failed_requests')
        points.append(point)
        if not point['measurement_valid']:continue
        sources[str(cp_path)]=sha(cp_path)
        declared_baselines=cell.get('baseline_cell_ids',{})
        exact_baselines=original_baselines_by_pair.get(p.pair_identity(point),{})
        for system,bid in declared_baselines.items():
            if bid in baseline:p.need(exact_baselines.get(system)==bid,'declared original baseline is not the exact workload pair')
        for system,bid in exact_baselines.items():
            if bid not in baseline:continue
            old=baseline[bid]
            verdict=p.verdict(point,old)
            strict=bool(point['work_complete'] and old['work_complete'] and
                point['slo_attainment']>=.90 and old['slo_attainment']>=.90 and
                point['energy_j']<old['energy_j'] and point['energy_per_good_request_j'] is not None and
                point['energy_per_good_request_j']<old['energy_per_good_request_j'])
            pairs.append(dict(model=point['model'],dataset=point['dataset'],rate_rps=point['rate_rps'],
                seed=point['seed'],repeat=point['repeat'],cell_id=cid,
                implementation_id=point['implementation_id'],version_id=point['version_id'],profile_sha256=point['profile_sha256'],
                **{f'baseline_{k}':old[k] for k in ('version_id','controller_source_sha256','profile_sha256','policy_sha256')},
                baseline_repeat=1,baseline_new_rate=False,**verdict,strict_service_energy_pass=strict))
    report_scope.annotate(points,origins,p)
    baseline_points,baseline_sources=inspect_baseline.collect(audit,ROOT,points) if series in ('p4','p5','p6') else ([],{})
    sources.update(baseline_sources)
    new_valid=[x for x in baseline_points if x['measurement_valid']]
    duplicate_keys=defaultdict(list)
    for old in new_valid:duplicate_keys[(p.pair_identity(old),old['system'],old['repeat'])].append(old['cell_id'])
    for key,cids in duplicate_keys.items():
        p.need(len(cids)==1,'ambiguous fresh baseline attempts; explicit selection is required: '+str(cids))
    for point in points:
        if not point['measurement_valid']:continue
        old_systems={v['baseline_system'] for v in pairs if v['cell_id']==point['cell_id']}
        for old in new_valid:
            if old['system'] in old_systems or old['repeat']!=point['repeat'] or p.pair_identity(old)!=p.pair_identity(point):continue
            verdict=p.verdict(point,old)
            strict=bool(point['work_complete'] and old['work_complete'] and
                point['slo_attainment']>=.90 and old['slo_attainment']>=.90 and
                point['energy_j']<old['energy_j'] and point['energy_per_good_request_j'] is not None and
                point['energy_per_good_request_j']<old['energy_per_good_request_j'])
            pairs.append(dict(model=point['model'],dataset=point['dataset'],rate_rps=point['rate_rps'],
                seed=point['seed'],repeat=point['repeat'],cell_id=point['cell_id'],
                implementation_id=point['implementation_id'],version_id=point['version_id'],profile_sha256=point['profile_sha256'],
                **{f'baseline_{k}':old[k] for k in ('version_id','controller_source_sha256','profile_sha256','policy_sha256')},
                baseline_repeat=old['repeat'],baseline_new_rate=True,**verdict,strict_service_energy_pass=strict))
            old_systems.add(old['system'])
    matrix=[]
    for model in ('7b','14b','32b'):
        for dataset in ('alpaca','sharegpt','longbench'):
            subset=[x for x in points if x['model']==model and x['dataset']==dataset]
            for system in p.BASELINES:
                values=[x for x in pairs if x['model']==model and x['dataset']==dataset and x['baseline_system']==system]
                matrix.append(dict(model=model,dataset=dataset,baseline=system,declared_executions=len(subset),
                    verified_executions=len(values),development_passes=sum(x['passed'] for x in values),
                    strict_service_energy_passes=sum(x['strict_service_energy_pass'] for x in values),
                    measured_rates=sorted({x['rate_rps'] for x in values}),
                    pending_or_invalid=sum(not x['measurement_valid'] and x['required_execution'] for x in subset),
                    pending_baseline_pairs=sum(x['measurement_valid'] and not any(v['cell_id']==x['cell_id'] for v in values) for x in subset),
                    pdb_version_ids=sorted({x['version_id'] for x in subset if x['measurement_valid']}),
                    baseline_version_ids=sorted({x['baseline_version_id'] for x in values}),
                    final_configuration_selected=False,
                    original_logical_grid_size=10,
                    measured_logical_points=len({x['original_cell_id'] for x in subset if x['measurement_valid'] and x['original_cell_id'] in original_pdb_ids}),
                    service_qualified_executions=sum(x['service_qualified'] for x in subset)))
    out.mkdir(parents=True)
    valid=[x for x in points if x['measurement_valid']]
    count=Counter(x['status'] for x in points)
    result=dict(schema=2,created_s=time.time(),candidate_series=series,baseline_snapshot='snapshot-006',
        declarations=len(declarations),verified=len(valid),work_complete=sum(x['work_complete'] for x in valid),
        development_pairs_passed=sum(x['passed'] for x in pairs),strict_pairs_passed=sum(x['strict_service_energy_pass'] for x in pairs),
        measured_pairs=len(pairs),statuses=dict(count),attempts=attempts,points=points,pairs=pairs,matrix=matrix,
        new_baseline_points=baseline_points,new_baseline_verified=len(new_valid),new_baseline_declared=len(baseline_points),
        independent_seeds_confirmed=False,independent_seed_confirmation_required=False,original_450_unchanged=True,
        boundaries_certified=False,raw_work_and_all_eight_gpu_energy_recomputed=True)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(out/'frozen-baseline-points.csv',list(baseline.values()))
    write_csv(out/'scope.csv',[{k:x.get(k) for k in ('model','dataset','rate_rps','repeat','cell_id','status','scope_status','required_execution')} for x in points])
    write_csv(out/'new-baseline-points.csv',baseline_points);write_csv(out/'points.csv',points);write_csv(out/'paired-baselines.csv',pairs);write_csv(out/'matrix36.csv',matrix)
    grid=logical_grid(originals,points,pairs)
    write_csv(out/'logical-grid90.csv',grid)
    boundaries=report_versions.boundary_records(points)
    write_csv(out/'matrix36-by-version.csv',report_versions.matrix_by_version(points,pairs))
    report_versions.adaptive(out,points)
    (out/'boundaries.json').write_text(json.dumps(boundaries,ensure_ascii=False,indent=2)+'\n')
    report_versions.figures(out,originals+new_valid,points,series)
    lines=[f'# 三主机并行修复与rate补测：{series} 独立快照','',
        '本快照对已镜像的终态观测重新核对请求工作量、SLO、原始八卡功率积分和指标。未镜像不等于未运行。', '',
        f"已核验{len(valid)}次，完整工作{sum(x['work_complete'] for x in valid)}次；开发验收{sum(x['passed'] for x in pairs)}/{len(pairs)}对，严格服务与节能验收{sum(x['strict_service_energy_pass'] for x in pairs)}/{len(pairs)}对。",'',
        '| 模型 | 数据集 | rate | 重复 | 能耗kJ | SLO% | 完成/规定请求 | 状态 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for x in sorted(points,key=lambda x:(x['model'],x['dataset'],x['rate_rps'],x['repeat'])):
        v=x['measurement_valid'];energy=f"{x['energy_j']/1000:.3f}" if v else '—';slo=f"{x['slo_attainment']*100:.2f}" if v else '—'
        work=f"{x['completed_work_requests']}/{x['n_expected']}" if v else '—'
        lines.append(f"| {x['model']} | {x['dataset']} | {x['rate_rps']:g} | {x['repeat']} | {energy} | {slo} | {work} | {x['status']} |")
    lines+=['',f'本版本原90个逻辑点已核验 {sum(x["verified_executions"]>0 for x in grid)}/90。已有重复全部单列；未测点不使用历史版本补齐。', '', '| 模型 | 数据集 | 对比baseline | 有效观测 | 服务达标 | 开发通过 | 严格节能通过 | 未测/无效 |', '|---|---|---|---:|---:|---:|---:|---:|']
    for x in matrix:
        lines.append(f"| {x['model']} | {x['dataset']} | {x['baseline']} | {x['verified_executions']} | {x['service_qualified_executions']} | {x['development_passes']} | {x['strict_service_energy_passes']} | {x['pending_or_invalid']} |")
    lines+=['','原450点、旧版本与失败尝试保留。零GPU锁冲突失败单列于results.json的attempts，不计为性能观测。',
        '此阶段矩阵为进度汇总，最终配置尚未全部选定；按版本矩阵与每条曲线保留来源。原baseline的历史源码变更也分开标记。',
        '最新用户要求：首个完整SLO<90%的PDB点即停止上探，取消独立种子边界确认及区间加密；90%只作rate停止条件，主比较允许双方低于90%。新率未配齐五系统时不作完整节能结论。全八卡主能耗与重叠外层窗口不相加。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for path,digest in code_sources.items():
        if sha(path)!=digest:raise ValueError('audit source changed during reading '+path)
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('frozen evidence changed during reading '+path)
    manifest=dict(schema=1,created_s=time.time(),sources=sources,
        files={path.name:sha(path) for path in out.iterdir() if path.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('declarations','verified','work_complete','development_pairs_passed','strict_pairs_passed','measured_pairs','statuses')}))



def logical_grid(originals,points,pairs):
    rows=[]
    for old in originals:
        if old['system']!='pdblend':continue
        actual=[x for x in points if x['original_cell_id']==old['cell_id']]
        measured=[x for x in actual if x['measurement_valid']]
        statuses={system:[v['strict_service_energy_pass'] for v in pairs if v['baseline_system']==system and any(v['cell_id']==x['cell_id'] for x in actual)] for system in ('mixed','distserve','dynamollm','ecoserve')}
        rows.append(dict(model=old['model'],dataset=old['dataset'],rate_rps=old['rate_rps'],seed=old['seed'],
            original_cell_id=old['cell_id'],declared_executions=len(actual),verified_executions=len(measured),
            work_complete_executions=sum(x['work_complete'] for x in measured),service_qualified_executions=sum(x['service_qualified'] for x in measured),
            status='not_yet_measured' if not measured else ('all_available_repeats_service_qualified' if all(x['service_qualified'] for x in measured) else 'contains_incomplete_work_or_service_failure'),
            repeat_requirement_complete=bool(actual and all(x['measurement_valid'] for x in actual if x.get('required_execution',True)) and any(x.get('required_execution',True) for x in actual)),
            required_executions=sum(x.get('required_execution',True) for x in actual),
            scope_statuses=sorted({x.get('scope_status','undeclared') for x in actual}),
            strict_all_available_repeats={k:bool(v) and all(v) for k,v in statuses.items()},
            cell_ids=[x['cell_id'] for x in measured]))
    assert len(rows)==90
    return rows


def boundary_evidence(originals,points):
    result=[]
    for model in ('7b','14b','32b'):
        for dataset in ('alpaca','sharegpt','longbench'):
            for system in ('pdblend','mixed','distserve','dynamollm','ecoserve'):
                group=[x for x in (points if system=='pdblend' else originals) if x['model']==model and x['dataset']==dataset and (system=='pdblend' or x['system']==system) and x.get('measurement_valid')]
                byrate=defaultdict(list)
                for x in group:byrate[x['rate_rps']].append(x)
                cleanpass=[r for r,v in byrate.items() if all(x['work_complete'] and x['slo_attainment']>=.90 for x in v)]
                cleanfail=[r for r,v in byrate.items() if all(x['work_complete'] and x['slo_attainment']<.90 for x in v)]
                uncertain=[r for r,v in byrate.items() if not all(x['work_complete'] for x in v) or (any(x['slo_attainment']>=.90 for x in v) and any(x['slo_attainment']<.90 for x in v))]
                lower=max(cleanpass,default=None)
                upper=min((r for r in cleanfail if lower is not None and r>lower),default=None)
                result.append(dict(model=model,dataset=dataset,system=system,complete_service_pass_rates=sorted(cleanpass),complete_service_fail_rates=sorted(cleanfail),incomplete_or_disagreeing_rates=sorted(uncertain),
                    exploratory_pass_lower=lower,exploratory_fail_upper=upper,relative_width=(upper-lower)/lower if upper is not None and lower else None,
                    certified=False,independent_seeds_required=[1701,2701,3701],
                    note='Exploratory observations only; incomplete requests and HTTP503 do not establish saturation. No throughput plateau claim.'))
    return result


def figures(out,originals,points,series):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    styles={'mixed':'#B57A13','distserve':'#21886B','dynamollm':'#9061AF','ecoserve':'#58768A'}
    metrics=[('slo_attainment','SLO attainment (%)',100),('energy_j','All-eight-GPU energy (kJ)',.001),('energy_per_good_request_j','J / SLO-qualified request',1),('completion_fraction','Work completion (%)',100),('goodput_measurement_rps','Goodput over full measurement window (req/s)',1)]
    with PdfPages(out/'rate-curves.pdf') as pdf:
        for field,label,scale in metrics:
            fig,axes=plt.subplots(3,3,figsize=(15,10),layout='constrained')
            for i,m in enumerate(('7b','14b','32b')):
                for j,d in enumerate(('alpaca','sharegpt','longbench')):
                    ax=axes[i,j]
                    for system,color in styles.items():
                        rows=sorted([x for x in originals if x['model']==m and x['dataset']==d and x['system']==system],key=lambda x:x['rate_rps'])
                        def value(x):
                            v=x.get(field)
                            if field=='completion_fraction':v=x['completed_work_requests']/x['n_expected']
                            return v*scale if v is not None else float('nan')
                        ax.plot([x['rate_rps'] for x in rows],[value(x) for x in rows],label=system,color=color,marker='.',lw=.9,alpha=.65)
                        incomplete=[x for x in rows if not x['work_complete']]
                        ax.scatter([x['rate_rps'] for x in incomplete],[value(x) for x in incomplete],marker='x',color=color,s=40,zorder=4)
                    current=[x for x in points if x['model']==m and x['dataset']==d and x['measurement_valid']]
                    for rep in sorted({x['repeat'] for x in current}):
                        rows=sorted([x for x in current if x['repeat']==rep],key=lambda x:x['rate_rps'])
                        ax.plot([x['rate_rps'] for x in rows],[x[field]*scale if x.get(field) is not None else float('nan') for x in rows],label=f'PDBlend {series} repeat {rep}',color='#CD423A',marker='o' if rep==1 else 's',ls='-' if rep==1 else '--',lw=1.5)
                        bad=[x for x in rows if not x['work_complete'] and x.get(field) is not None]
                        ax.scatter([x['rate_rps'] for x in bad],[x[field]*scale for x in bad],marker='x',color='black',s=80,zorder=5)
                    if field=='slo_attainment':ax.axhline(90,color='#555555',ls=':',lw=1)
                    ax.set_title(f'{m} / {d}');ax.set_xlabel('Offered rate (req/s)');ax.set_ylabel(label);ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=7)
            fig.suptitle(f'{series}: each repetition shown; frozen baselines; x = incomplete work',fontsize=13)
            pdf.savefig(fig)
            if field=='slo_attainment':fig.savefig(out/'slo-rate-curves.png',dpi=120)
            plt.close(fig)


def write_adaptive_plan(out,originals,points):
    from boundary_first_loss import next_boundary,paired_jobs
    records=[];jobs=[]
    for model in ('7b','14b','32b'):
        for dataset in ('alpaca','sharegpt','longbench'):
            local=[]
            for system in ('pdblend',):
                rows=[dict(x,implementation_id=x.get('implementation_id',f'frozen-snapshot006-{model}-{system}')) for x in (points if system=='pdblend' else originals) if x['model']==model and x['dataset']==dataset and (system=='pdblend' or x['system']==system) and x.get('measurement_valid')]
                d=next_boundary(rows);d.update(model=model,dataset=dataset,system=system)
                records.append(d);local.append(d)
            jobs.extend(dict(j,model=model,dataset=dataset) for j in paired_jobs(local))
    (out/'adaptive-next-plan.json').write_text(json.dumps(dict(planning_only=True,measured=False,decisions=records,five_system_trace_jobs=jobs),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--series',choices=('p1','p2','p3','p4','p5','p6'),required=True)
    args=parser.parse_args();collect(args.out.resolve(),args.series)
