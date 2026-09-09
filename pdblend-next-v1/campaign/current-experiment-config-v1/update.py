"""Publish a host-local index of the actual frozen PDB experiment; no GPU controls."""
import argparse
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import socket
import time

CAMPAIGN = Path('/root/workspace/pdblend-next-v1/campaign')
MODELS = {
    '14b': ('A14B-deadline-matrix-v1', '20a6ee2113cf70a9db3072a9d59037fb1b9855d5dd2b9829a66a4c802bcac302',
            'deadline-phase-bridge-v1/A14B-attempt-001/status.json'),
    '32b': ('B32B-main-scale-fixed-window-v1', 'e603d153cc86a9427c9acee94177d940b5d911cba975c4eeb52bc5d71f0b58b7',
            'B32B-main-scale-bridge-v1/status.json'),
    '7b': ('C7B-deadline-matrix-v1', 'a0614080b2694da25eb0044911701d261f915b93c0779d5165ed5d623e7004ae',
           'C7B-deadline-matrix-v1/bridge-attempt-1/status.json'),
}
SLOS = {'alpaca': (1., .1), 'sharegpt': (5., .15), 'longbench': (15., .2)}
DEADLINE_SHA = 'fabdbaa4267f63ec208c59ceac5780592ed238918c008db3eadb01e6dcf64e50'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def cst(epoch):
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def expected_config(config, row, output):
    return dict(config, journal=str(output/'control.jsonl'), slo_ttft_s=row['slo_ttft_s'],
                slo_tpot_s=row['slo_tpot_s'], slo_scale=row['slo_scale'],
                slo_protocol='per-dataset-slo-v1', slo_attainment_target=.9)


def dimensions(spec):
    require(spec['protocol_id']=='per-dataset-slo-fixed-window-v2' and spec['measurement_schema']==3
            and spec['execute_baselines'] is False and spec['formal_eligible'] is False, 'wrong experiment scope')
    main = [r for r in spec['cells'] if r['phase']=='main']
    scale = [r for r in spec['cells'] if r['phase']=='scale']
    require(len(main)==60 and len(scale)==36, 'complete frozen declaration required')
    indexed = {r['cell_id']: r for r in main}
    require(len(indexed)==60 and len({r['cell_id'] for r in spec['cells']})==96, 'duplicate logical cell')
    datasets = {}
    for name, base in SLOS.items():
        m = [r for r in main if r['dataset']==name]
        s = [r for r in scale if r['dataset']==name]
        rates = sorted({r['rate_rps'] for r in m})
        scale_rates = sorted({r['rate_rps'] for r in s})
        require(len(rates)==10 and len(m)==20 and
                {(r['rate_rps'],r['arrival_seed']) for r in m}==set(itertools.product(rates,(701,1701))),
                'main rates or paired seeds differ')
        require(len(scale_rates)==3 and len(s)==12 and
                {(r['rate_rps'],r['arrival_seed'],r['slo_scale']) for r in s}
                ==set(itertools.product(scale_rates,(701,1701),(.5,2.))), 'scale rates/seeds differ')
        for row in m+s:
            require(row['slo_ttft_s']==base[0]*row['slo_scale'] and row['slo_tpot_s']==base[1]*row['slo_scale']
                    and row['trace_duration_s']==300, 'actual per-row SLO/window differs')
            if row['phase']=='scale':
                ref=indexed.get(row['reuse_main_cell_id'])
                require(ref and all(ref[k]==row[k] for k in ('dataset','rate_rps','arrival_seed','trace_sha256','n_requests')),
                        'scale workload is not the exact 1x reference')
            else:
                require(row['slo_scale']==1., 'main scale changed')
        datasets[name] = dict(main_rates_rps=rates, scale_rates_rps=scale_rates,
            base_slo_seconds=dict(ttft=base[0],tpot=base[1]),
            slo_scales=[dict(scale=k,ttft_seconds=base[0]*k,tpot_seconds=base[1]*k) for k in (.5,1.,2.)])
    return datasets


def process_evidence(pid, required_text, proc_root=Path('/proc')):
    require(type(pid) is int and pid>1, 'missing actual process PID')
    directory=proc_root/str(pid)
    command=(directory/'cmdline').read_bytes().replace(b'\0',b' ').decode()
    require(required_text in command, 'PID is not the declared experiment on this host')
    stat=(directory/'stat').read_text().rsplit(') ',1)[1].split()
    require(stat[0]!='Z', 'declared process is a zombie')
    return dict(pid=pid,start_ticks=int(stat[19]),command=command)


def build(model):
    name,manifest_sha,bridge_relative=MODELS[model]
    package=CAMPAIGN/name
    manifest_path=package/'package-manifest.json'
    require(sha(manifest_path)==manifest_sha, 'wrong frozen model package')
    manifest=read(manifest_path)
    for relative,digest in manifest['files'].items():
        require(sha(package/relative)==digest, 'frozen package changed: '+relative)
    spec=read(package/'runspec.json');require(spec['model']==model,'wrong host model')
    datasets=dimensions(spec)
    scope=spec['deadline_scope'];require(scope['sha256']==DEADLINE_SHA and sha(scope['path'])==DEADLINE_SHA,
                                        'user 24-hour deadline changed')
    protocol=read(scope['path'])
    require(protocol['arrival_seeds']==[701,1701] and protocol['minimum_requests'] is None
            and protocol['request_hard_timeout_s']==120 and protocol['drain_after_arrival_window_s']==120,
            'old request count/timeout/seed protocol')
    config_path=package/'inputs/controller.fixed.json';config=read(config_path)
    require(config['strategy']=='pdblend-joint' and config['node_gpus']==list(range(8)), 'wrong serving policy/node boundary')
    bridge_path=CAMPAIGN/bridge_relative;bridge=read(bridge_path)
    require(bridge['package']==str(package) and bridge['package_manifest_sha256']==manifest_sha
            and bridge['phase'] in ('main','scale') and not bridge.get('error'), 'bridge is not actively running this package')
    live=process_evidence(bridge['pid'],str(package))
    runner=process_evidence(bridge['child_pid'],str(package/'run.py'))
    observed=[]
    for row in spec['cells']:
        output=package/'cells'/row['cell_id'];path=output/'runtime_config.json'
        if path.exists():
            require(read(path)==expected_config(config,row,output), 'actual serving config mismatch: '+row['cell_id'])
            observed.append(dict(cell_id=row['cell_id'],path=str(path),sha256=sha(path),dataset=row['dataset'],
                slo_scale=row['slo_scale'],ttft_seconds=row['slo_ttft_s'],tpot_seconds=row['slo_tpot_s']))
    require(observed, 'no actual dispatched Controller configuration available')
    ledgers={}
    for phase in ('main','scale'):
        p=Path(scope['path']).parent/'phase-ledgers'/model/(phase+'.json')
        if p.exists():
            value=read(p)
            maximum=protocol['deadline_s']-(13 if phase=='main' else 7)*3600
            require(value['model']==model and value['phase']==phase and value['deadline_scope_sha256']==DEADLINE_SHA
                    and value['deadline_s']==min(value['started_s']+(8 if phase=='main' else 6)*3600,maximum),
                    'actual phase ledger differs')
            ledgers[phase]=dict(path=str(p),sha256=sha(p),record=value,deadline_cst=cst(value['deadline_s']))
    require(bridge['phase'] in ledgers, 'running phase ledger missing')
    return dict(schema=1,kind='host_local_current_experiment_index',written_s=time.time(),hostname=socket.gethostname(),model=model,
        purpose='Local index of the actual frozen queue and effective settings; this index itself is not a serving override.',
        package=dict(path=str(package),manifest_sha256=manifest_sha,runspec_sha256=sha(package/'runspec.json')),
        controller=dict(path=str(config_path),sha256=sha(config_path),host_release=config['controller_source_release'],
            engine_release=config['engine_source_release'],strategy=config['strategy'],instances=config['instances'],
            dvfs=config['dvfs'],park_idle=config['park_idle'],output_prior=config['output_prior'],
            scheduler_budget_ablation=config.get('scheduler_budget_ablation'),
            base_numeric_slo_is_overridden_by_frozen_rows=True),
        datasets=datasets,arrival_seeds=[701,1701],arrival_window_seconds=300,minimum_requests=None,
        request_hard_timeout_seconds=120,drain_after_arrival_window_seconds=120,joint_slo_target=.9,
        main_cells=60,additional_scale_cells=36,scale_one_references_reused=18,
        energy_scope='all eight GPU boards; idle, failed and incomplete offered work included; primary and outer windows reported separately',
        baseline_execution=False,baseline_policy='preserve existing results; no baseline reruns',formal_eligible=False,
        deadline=dict(path=scope['path'],sha256=DEADLINE_SHA,global_end_seconds=protocol['deadline_s'],global_end_cst=cst(protocol['deadline_s']),
            latest_main_end_cst=cst(protocol['deadline_s']-13*3600),latest_scale_end_cst=cst(protocol['deadline_s']-7*3600),
            phase_hours=protocol['phase_hours']),
        phase_ledgers=ledgers,bridge=dict(path=str(bridge_path),phase_at_verification=bridge['phase'],process=live,runner=runner),
        actual_configuration_verification=dict(verified_s=time.time(),count=len(observed),records=observed),
        updater=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)))


def publish(record, campaign=CAMPAIGN):
    campaign=Path(campaign);target=campaign/'current-experiment.json'
    stamp=f"{time.time_ns()}-{record['model']}";history=campaign/'current-experiment-history'/stamp
    history.mkdir(parents=True,exist_ok=False)
    if target.exists():
        require(not target.is_symlink() and target.is_file(),'unsafe current index target')
        (history/'previous.json').write_bytes(target.read_bytes())
    data=(json.dumps(record,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
    (history/'current.json').write_bytes(data)
    temp=campaign/('.current-experiment-'+stamp+'.tmp');temp.write_bytes(data);temp.replace(target)
    require(target.read_bytes()==data,'local configuration readback differs')
    report=dict(hostname=record['hostname'],model=record['model'],path=str(target),sha256=sha(target),
                history=str(history),actual_configurations_verified=record['actual_configuration_verification']['count'],
                serving_controls_sent=False,baseline_executed=False,frozen_package_changed=False)
    (history/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=[f"# 本机当前实验：Qwen2.5-{record['model'].upper()}\n",
        f"核验时间：{cst(record['written_s'])}；主机：{record['hostname']}。\n",
        '本文件及 current-experiment.json 是本机当前配置入口，指向已冻结且正在执行的实验。它们不作为运行中的服务覆盖参数。\n',
        f"实际包：{record['package']['path']}\n",
        '三个数据集各10个rate、两个种子701/1701；每格发压300秒，无1000请求下限。请求硬超时120秒，到达窗口结束后最多排空120秒。\n',
        '主矩阵60格；SLO scale使用0.5/1/2倍，低中高各三个rate，1倍复用18个主格，额外测36格。联合SLO目标90%。\n',
        '|数据集|TTFT/TPOT|主rate（请求/秒）|scale rate（请求/秒）|',
        '|---|---|---|---|']
    for name,row in record['datasets'].items():
        slo=row['base_slo_seconds']
        lines.append(f"|{name}|{slo['ttft']:g}s / {slo['tpot']*1000:g}ms|{', '.join(map(str,row['main_rates_rps']))}|{', '.join(map(str,row['scale_rates_rps']))}|")
    lines.extend(['',f"总截止：{record['deadline']['global_end_cst']}。主阶段最晚{record['deadline']['latest_main_end_cst']}；scale最晚{record['deadline']['latest_scale_end_cst']}。每模型实际阶段记录可能更早，见JSON。\n",
        '全部8张GPU的闲置、失败及未完成工作能耗保留；主测量与外围完整操作窗口分别报告。baseline保留，不重跑。\n',
        f"本次核对了{record['actual_configuration_verification']['count']}份已实际创建的完整Controller配置，均与对应格子的冻结参数一致。基础配置中的5秒/100毫秒由每格数据集与scale参数覆盖，实际阈值以上表及JSON为准。\n",
        f"JSON SHA256：{report['sha256']}\n"])
    markdown=('\n'.join(lines)+'\n').encode();(history/'CURRENT_EXPERIMENT.md').write_bytes(markdown)
    md_target=campaign/'CURRENT_EXPERIMENT.md'
    if md_target.exists():
        require(not md_target.is_symlink() and md_target.is_file(),'unsafe markdown target')
        (history/'previous.md').write_bytes(md_target.read_bytes())
    md_temp=campaign/('.current-experiment-'+stamp+'.md.tmp');md_temp.write_bytes(markdown);md_temp.replace(md_target)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=MODELS,required=True)
    parser.add_argument('--write',action='store_true');args=parser.parse_args();record=build(args.model)
    print(json.dumps(publish(record) if args.write else record,indent=2,ensure_ascii=False))
