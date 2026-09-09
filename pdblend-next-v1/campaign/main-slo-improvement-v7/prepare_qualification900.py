"""Freeze matched 300/300/300 fixed-two and measured-dynamic development runs."""
import argparse
import copy
import json
from pathlib import Path
import protocol as p


def prepare(args):
    p.need(not args.out.exists(), 'new matched qualification output required')
    fixed=p.read(args.fixed_release);dynamic=p.read(args.dynamic_release)
    p.need(dynamic['fixed_comparison_release']==p.ref(args.fixed_release)
           and fixed['binding']==dynamic['binding'] and fixed['model']==dynamic['model'],
           'dynamic arm must extend the exact fixed strategy and real initial owners')
    trace=p.read(args.trace);domain=p.read(args.domain)
    p.need(trace['schema']=='capacity-development-trace-v1' and trace['duration_s']==900
           and trace['demand_domain_sha256']==domain['sha256'] and trace['formal_eligible'] is False
           and trace.get('split')=='development',
           'exact predeclared development shape trace required')
    p.need([v['name'] for v in trace['phases']]==['low','high','low']
           and all(v['duration_s']==300 for v in trace['phases']), 'three complete 300-second stages required')
    source=p.read(args.load_spec);capacity=p.checked(dynamic['capacity_binding'])
    p.need(source['original_binding']==fixed['binding'] and capacity['identity']==
           p.checked(source['capacity_binding'])['identity'], 'same actual source/layout calibration required')
    p.need(capacity['deadline_s']==fixed['deadline_s']==dynamic['deadline_s']==p.DEADLINE,
           'original absolute deadline changed')
    p.need(domain in capacity['demand_domains'], '900s trace domain lacks the released empirical coverage')
    code=args.out/'code';code.mkdir(parents=True)
    for path in args.capacity_code.glob('*.py'):
        (code/path.name).write_bytes(path.read_bytes())
    for name in ('capacity_runtime.py','capacity_executor.py','capacity_backend.py',
                 'capacity_planner.py','capacity_certificate.py'):
        p.need(p.sha(code/name)==p.sha(Path(dynamic['host_release'])/name),
               'both qualification arms must use the exact released dynamic dependency set')
    driver=Path(args.load_spec).parent/'code/capacity_load_calibrate.py'
    (code/'capacity_load_calibrate.py').write_bytes(driver.read_bytes())
    packages={}
    for arm,release in [('fixed2',fixed),('dynamic',dynamic)]:
        config=copy.deepcopy(p.checked(release['configs'][arm][args.dataset]))
        config.pop('measurement_window_protocol',None);config.pop('transfers',None)
        config.update(slo_ttft_s=domain['slo_ttft_s'],slo_tpot_s=domain['slo_tpot_s'],
            port=args.port,capacity_integration_v1=arm=='dynamic')
        for key in ('capacity_inventory_path','capacity_lease_authority','capacity_job_path'):
            config.pop(key,None)
        path=args.out/arm/'config.json';p.write(path,config,exclusive=True)
        files=dict(release['files']);files.update(capacity['files'])
        files.update({str(v.resolve()):p.sha(v) for v in code.glob('*.py')})
        for v in (args.fixed_release,args.dynamic_release,args.trace,args.domain,args.load_spec,
                  args.capacity_code/'capacity_runtime.py',path,Path(__file__)):
            files[str(v.resolve())]=p.sha(v)
        host=Path(release['host_release'])
        if arm=='dynamic':
            for name in ('capacity_runtime.py','capacity_executor.py','capacity_backend.py',
                         'capacity_planner.py','capacity_certificate.py'):
                p.need(p.sha(code/name)==p.sha(host/name), 'qualification dynamic dependency differs from released host')
        p.need(all(p.sha(v)==h for v,h in files.items()), 'qualification source changed')
        spec=dict(schema='capacity-load-calibration-spec-v1',authorized=True,automatic_retries=False,
            mode='qualification900',arm=arm,original_binding=fixed['binding'],capacity_binding=dynamic['capacity_binding'],
            config=p.ref(path),host_release=str(host),files=files,common_executor=source['common_executor'],
            deadline_s=p.DEADLINE,demand_domain_sha256=domain['sha256'],gpus=source['gpus'],trace=p.ref(args.trace),
            stop_path=str(args.out/'STOP'),api_base=f'http://127.0.0.1:{args.port}',
            served_model=source['served_model'],same_trace_per_pair=True,full_output_and_original_deadlines_required=True,
            role='development function and full-cycle energy validation; not original main point')
        target=args.out/arm/'spec.json';p.write(target,spec,exclusive=True);packages[arm]=p.ref(target)
    p.write(args.out/'pair.json',dict(schema='capacity-900-pair-declaration-v1',model=fixed['model'],
        trace=p.ref(args.trace),domain=p.ref(args.domain),fixed_release=p.ref(args.fixed_release),
        dynamic_release=p.ref(args.dynamic_release),arms=packages,repeats_per_arm=1,
        initial_and_final_instances=2,automatic_retries=False,deadline_s=p.DEADLINE,
        phases_s=[300,300,300],source=p.ref(__file__)),exclusive=True)
    return p.ref(args.out/'pair.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('fixed-release','dynamic-release','trace','domain','load-spec','capacity-code','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--dataset',choices=p.DATASETS,required=True)
    parser.add_argument('--port',type=int,required=True)
    print(json.dumps(prepare(parser.parse_args())))
