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
    return module


def stage_paths():
    result=[]
    for node in ('A','B','C'):
        base=ROOT/node
        for pattern in ('fixed-screen*/status.json','screen-p*/status.json',
                        'execution-p*/fixed-screen*/status.json',
                        'confirm-*/status.json','added-rates-*/runs/status.json'):
            result.extend(base.glob(pattern))
    return sorted(set(result))


def collect(out):
    if out.exists():raise FileExistsError(out)
    audit=load_audit();p=audit.p
    code_sources={str(path):sha(path) for path in (Path(__file__),AUDIT/'report.py',AUDIT/'protocol.py',AUDIT/'raw_metrics.py')}
    originals=p.original_points();baseline={x['cell_id']:x for x in originals if x['system']!='pdblend'}
    declarations={};checkpoints={};attempts=[];sources=dict(code_sources)
    for status_path in stage_paths():
        status=read(status_path);directory=status_path.parent
        if status.get('stage') not in ('screen_fixed2','screen_dynamic','confirm_dynamic','confirm_fixed2'):continue
        order_path=directory/'declaration-order.json'
        if not order_path.exists():continue
        order=read(order_path)
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
            point=audit.inspect(cell,None)
            point.update(status='awaiting_mirror',error='completed checkpoint seen; raw evidence not yet mirrored',missing_artifact_count=len(missing))
        else:
            point=audit.inspect(cell,cp_path)
        points.append(point)
        if not point['measurement_valid']:continue
        sources[str(cp_path)]=sha(cp_path)
        for system,bid in cell['baseline_cell_ids'].items():
            old=baseline[bid]
            verdict=p.verdict(point,old)
            strict=bool(point['work_complete'] and old['work_complete'] and
                point['slo_attainment']>=.90 and old['slo_attainment']>=.90 and
                point['energy_j']<old['energy_j'] and point['energy_per_good_request_j'] is not None and
                point['energy_per_good_request_j']<old['energy_per_good_request_j'])
            pairs.append(dict(model=point['model'],dataset=point['dataset'],rate_rps=point['rate_rps'],
                seed=point['seed'],repeat=point['repeat'],cell_id=cid,
                implementation_id=point['implementation_id'],**verdict,strict_service_energy_pass=strict))
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
                    pending_or_invalid=sum(not x['measurement_valid'] for x in subset)))
    out.mkdir(parents=True)
    valid=[x for x in points if x['measurement_valid']]
    count=Counter(x['status'] for x in points)
    result=dict(schema=1,created_s=time.time(),baseline_snapshot='snapshot-006',
        declarations=len(declarations),verified=len(valid),work_complete=sum(x['work_complete'] for x in valid),
        development_pairs_passed=sum(x['passed'] for x in pairs),strict_pairs_passed=sum(x['strict_service_energy_pass'] for x in pairs),
        measured_pairs=len(pairs),statuses=dict(count),attempts=attempts,points=points,pairs=pairs,matrix=matrix,
        independent_seeds_confirmed=False,original_450_unchanged=True,
        boundaries_certified=False,raw_work_and_all_eight_gpu_energy_recomputed=True)
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    write_csv(out/'points.csv',points);write_csv(out/'paired-baselines.csv',pairs);write_csv(out/'matrix36.csv',matrix)
    lines=['# 三主机并行修复与rate补测实测进度','',
        '本快照对已镜像的终态观测重新核对请求工作量、SLO、原始八卡功率积分和指标。未镜像不等于未运行。', '',
        f"已核验{len(valid)}次，完整工作{sum(x['work_complete'] for x in valid)}次；开发验收{sum(x['passed'] for x in pairs)}/{len(pairs)}对，严格服务与节能验收{sum(x['strict_service_energy_pass'] for x in pairs)}/{len(pairs)}对。",'',
        '| 模型 | 数据集 | rate | 重复 | 能耗kJ | SLO% | 完成/规定请求 | 状态 |',
        '|---|---|---:|---:|---:|---:|---:|---|']
    for x in sorted(points,key=lambda x:(x['model'],x['dataset'],x['rate_rps'],x['repeat'])):
        v=x['measurement_valid'];energy=f"{x['energy_j']/1000:.3f}" if v else '—';slo=f"{x['slo_attainment']*100:.2f}" if v else '—'
        work=f"{x['completed_work_requests']}/{x['n_expected']}" if v else '—'
        lines.append(f"| {x['model']} | {x['dataset']} | {x['rate_rps']:g} | {x['repeat']} | {energy} | {slo} | {work} | {x['status']} |")
    lines+=['','原450点、旧版本与失败尝试保留。零GPU锁冲突失败单列于results.json的attempts，不计为性能观测。',
        '新增rate五系统配对、独立种子及饱和边界未完成时不标为通过。总能量为全八卡主窗口，重叠外层窗口不相加。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for path,digest in code_sources.items():
        if sha(path)!=digest:raise ValueError('audit source changed during reading '+path)
    for path,digest in sources.items():
        if sha(path)!=digest:raise ValueError('frozen evidence changed during reading '+path)
    manifest=dict(schema=1,created_s=time.time(),sources=sources,
        files={path.name:sha(path) for path in out.iterdir() if path.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('declarations','verified','work_complete','development_pairs_passed','strict_pairs_passed','measured_pairs','statuses')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    collect(parser.parse_args().out.resolve())
