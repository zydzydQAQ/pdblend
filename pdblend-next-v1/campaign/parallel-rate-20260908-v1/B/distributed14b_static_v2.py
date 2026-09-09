"""Assigned same-host fixed-SLO cells using the unchanged common executor."""
import argparse, asyncio, copy, csv, hashlib, importlib.util, json, math, os, signal, socket, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R / 'common/distributed14b-deployment-v1'))
import deploy
read, sha, ref, need = deploy.read, deploy.sha, deploy.reference, deploy.require
QUALIFIED_CACHE = {}

def write(p, value, exclusive=False):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with p.open('x') as f: json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    else:
        tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n'); tmp.replace(p)

def checked(r):
    need(sha(r['path']) == r['sha256'], 'frozen reference changed: ' + r['path'])
    return read(r['path'])

def flat(cell):
    if 'source_row' not in cell: return copy.deepcopy(cell)
    row = copy.deepcopy(cell['source_row']); row.update(cell_id=cell['cell_id'], repeat=cell['repeat'])
    return row

def load_release(path):
    release = read(path)
    need(release['schema'] == 'distributed14b-static-release-v2', 'unknown release')
    for p, h in release['files'].items(): need(sha(p) == h, 'release input changed: ' + p)
    jobs = checked(release['jobs']); parent = checked(jobs['parent'])
    need(jobs['parent']['sha256'] == '913a2d5834dbcc3466cff9d6a45e30cba574069ed50d36c89704ffff89b80438', 'assignment differs')
    need(release['node'] == jobs['node'] and jobs['model'] == '14b' and jobs['node'] in ('B', 'C'), 'assignment/model differs')
    base = checked(release['binding'])
    need(release['system'] == 'pdblend', 'v1 is PDB only; baselines require the later observed-boundary contract')
    need(base['hostname'] == socket.gethostname() and base['system'] == release['system'], 'actual host/system differs')
    declared = [flat(c) for c in jobs['pdb_cells']] if release['system'] == 'pdblend' else [copy.deepcopy(c) for c in jobs['baseline_cells'] if c['system'] == release['system']]
    by_id = {c['cell_id']: c for c in declared}
    need(len(by_id) == len(declared), 'duplicate declared IDs')
    need(len(release['rows']) == len(declared) and {c['cell_id'] for c in release['rows']} == set(by_id), 'whole assigned system group required')
    for row in release['rows']:
        need(row == by_id[row['cell_id']], 'scientific row differs')
        need(row['seed'] == 701 and row['trace_duration_s'] == 100 and row['slo_scale'] == 1, 'original protocol differs')
        need(sha(row['trace']) == row['trace_sha256'], 'trace changed')
    successor=checked(release['source_successor'])
    need(successor['schema']=='distributed14b-hardware-profile-successor-v1' and successor['jobs']==release['jobs'], 'explicit assigned successor required')
    need(successor['parent_controller_manifest']==jobs['common_controller_manifest'] and type(successor['max_service_frequency_mhz']) is int and successor['max_service_frequency_mhz'] in (2100,2400), 'exact parent and qualified maximum required')
    cpu=checked(successor['cpu_validation'])
    need(cpu['passed'] and any(s['manifest']==successor['host_manifest'] for s in cpu['sources']), 'actual source absent from common CPU proof')
    qualification=checked(release['qualification'])
    need(all(release['files'].get(p)==h for p,h in qualification['files'].items()), 'qualification raw outside release closure')
    validator=release['qualification_validator'];need(release['files'].get(validator['path'])==validator['sha256'], 'qualification implementation not frozen')
    cache_key=(release['qualification']['sha256'],validator['sha256'],release['binding']['sha256'])
    if cache_key not in QUALIFIED_CACHE:
        spec=importlib.util.spec_from_file_location('static_saved_qualification_'+validator['sha256'][:12],validator['path']);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        QUALIFIED_CACHE[cache_key]=module.verify(release['qualification'],base)
    need(QUALIFIED_CACHE[cache_key]['independently_recomputed'] is True, 'raw qualification has not been reconstructed')
    if release['system'] == 'pdblend':
        need(release['actual_arm'] == 'fixed2' and release['capacity_integration_v1'] is False, 'static capacity declaration differs')
        need(len(base['instances']) == 2 and all(i['tp'] == 1 for i in base['instances']), 'actual fixed2 geometry differs')
        need(ref(Path(base['host_release']) / 'manifest.json') == checked(release['source_successor'])['host_manifest'], 'actual qualified P10 source differs')
        for p in base['configs'].values():
            cfg = read(p); need(cfg.get('capacity_integration_v1') is False and 'capacity_binding_path' not in cfg and 'capacity_binding_sha256' not in cfg, 'dynamic capacity source leaked')
    return release, base

def arrivals(receipt, output):
    p = output / 'cells' / receipt['cell_id'] / 'bench.csv'
    result = dict(passed=False, errors=[], max_limit_s=1., p99_limit_s=.1)
    try:
        with p.open() as stream: rows = list(csv.DictReader(stream))
        summary = receipt['summary']
        need(len(rows) == summary['n_expected'] > 0, 'missing request timing')
        delays = []
        for row in rows:
            planned, actual, reported = [float(row[k]) for k in ('planned_arrival_s', 'actual_dispatch_s', 'dispatch_delay_s')]
            need(all(math.isfinite(x) for x in (planned, actual, reported)) and actual >= planned - 1e-6, 'invalid dispatch timing')
            delta = max(0., actual - planned); need(abs(delta - reported) <= 1e-6, 'dispatch delay mismatch'); delays.append(delta)
        delays.sort(); at = .99 * (len(delays) - 1); lo = int(at); hi = min(lo + 1, len(delays) - 1)
        p99 = delays[lo] + (delays[hi] - delays[lo]) * (at - lo)
        need(abs(summary['dispatch_delay_max_s'] - delays[-1]) <= 1e-6 and abs(summary['dispatch_delay_p99_s'] - p99) <= 1e-6, 'summary timing differs')
        result.update(actual_dispatch_lateness_max_s=delays[-1], actual_dispatch_lateness_p99_s=p99, passed=delays[-1] <= 1. and p99 <= .1)
        if not result['passed']: result['errors'].append('arrival timing engineering limit violated')
    except BaseException as exc: result['errors'].append(repr(exc))
    if p.exists(): result['raw_requests'] = ref(p)
    return result

def engineering(receipt, timing):
    s = receipt.get('summary', {}); errors = []
    for key in ('measurement_valid', 'work_complete', 'fixed_window_valid'):
        if s.get(key) is not True: errors.append('summary ' + key)
    if receipt.get('measurement_valid') is not True: errors.append('operation measurement invalid')
    for key in ('failed_requests', 'request_timeouts'):
        if s.get(key) != 0: errors.append(key + ' is nonzero or missing')
    if s.get('completed') != s.get('n_expected') or s.get('completed_work_requests') != s.get('n_expected'): errors.append('incomplete request count')
    if not timing['passed']: errors.extend(timing['errors'])
    return dict(passed=not errors, errors=errors)

def boundary_rate(receipt):
    s = receipt['summary']
    # A low SLO result is a capacity observation only after complete work.
    q = float(s['slo_attainment'])
    need(math.isfinite(q) and 0 <= q <= 1, 'invalid joint SLO fraction')
    return q


async def hardware_identity(node):
    import subprocess
    result=await asyncio.to_thread(subprocess.run,['nvidia-smi','--query-gpu=index,name,uuid','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    need(result.returncode==0,'actual GPU identity lookup failed')
    rows=[]
    for row in csv.reader(result.stdout.splitlines()):
        need(len(row)==3,'unexpected GPU identity row');rows.append(dict(index=int(row[0]),name=row[1].strip(),uuid=row[2].strip()))
    rows.sort(key=lambda x:x['index'])
    need(len(rows)==8 and {r['index'] for r in rows}==set(range(8)) and len({r['uuid'] for r in rows})==8,'actual eight-card UUID inventory missing')
    return dict(schema='distributed14b-hardware-identity-v1',node=node,hostname=socket.gethostname(),gpus=rows,observed_s=time.time(),read_only=True)

async def execute(a, state):
    release, base = load_release(a.release)
    host = Path(base['host_release']); common = deploy.adapter.load_runtime(host, Path(base['executor']).parent)
    os.environ['PYTHONPATH'] = ':'.join([str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps'])
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from ecopadg.serving.campaign import node_lease
    from ecopadg.measure.backends import PynvmlBackend
    import aiohttp
    need(not a.out.exists() and 'PDBLEND_NODE_LOCK_FD' not in os.environ, 'new output/fresh owner required')
    a.out.mkdir(parents=True); output = a.out/'results'; output.mkdir()
    write(a.out/'declaration-order.json', release['rows'], True)
    state.update(declaration=release['jobs'], release=ref(a.release), declared=len(release['rows']), remaining=[r['cell_id'] for r in release['rows']], actual_arm=release['actual_arm'], measurement_host=release['node'], first_complete_slo_miss_rate=None, skipped_saturated=[])
    def save(): state.update(updated_s=time.time()); write(a.out/'status.json', state)
    save()
    with node_lease() as lease:
        deploy.require_lease(lease); state['node_lease_held']=True; save()
        hardware=await asyncio.to_thread(PynvmlBackend, power_mode='instant')
        async with aiohttp.ClientSession(trust_env=False) as session:
            await common.identity(session,base)
            for row in release['rows']:
                if a.stop_requested or (a.out.parent/'STOP-static').exists(): state['phase']='stopped_at_boundary'; break
                boundary=state['first_complete_slo_miss_rate']
                if boundary is not None and float(row['rate_rps']) > boundary:
                    state['skipped_saturated'].append(dict(cell_id=row['cell_id'], rate_rps=row['rate_rps'], first_complete_slo_miss_rate=boundary)); save(); continue
                load_release(a.release)
                binding=copy.deepcopy(base); binding['output']=str(output)
                bp=a.out/'bindings'/(row['cell_id']+'.json'); write(bp,binding,True); common.validate_binding(binding)
                hardware_before_path=a.out/'hardware'/(row['cell_id']+'.before.json'); hardware_after_path=a.out/'hardware'/(row['cell_id']+'.after.json')
                write(hardware_before_path,await hardware_identity(release['node']),True)
                state.update(phase='running',current_cell=row['cell_id']); state['attempted'].append(row['cell_id']); save()
                rp=output/'operations'/row['cell_id']/'receipt.json'; cp=output/'checkpoints'/(row['cell_id']+'.json')
                def checkpoint(error=None):
                    receipt=read(rp); timing=arrivals(receipt,output)
                    artifacts={str(f):sha(f) for d in (rp.parent,output/'cells'/row['cell_id']) for f in d.rglob('*') if f.is_file()}
                    artifacts[str(bp)]=sha(bp)
                    for hp in (hardware_before_path,hardware_after_path):
                        if hp.exists():artifacts[str(hp)]=sha(hp)
                    record=dict(row=row,declaration=release['jobs'],binding=ref(bp),receipt=ref(rp),artifacts=artifacts,measurement_valid=receipt.get('measurement_valid') is True,work_complete=receipt.get('summary',{}).get('work_complete'),qualification_refs=release.get('qualification_refs',{}),qualification=release['qualification'],qualification_validator=release['qualification_validator'],hardware_identity=ref(hardware_after_path) if hardware_after_path.exists() else None,hardware_identity_before=ref(hardware_before_path),source_successor=release['source_successor'],release=ref(a.release),execution_rules=release['execution_rules'],arrival_timing_qualification=timing,actual_arm=release['actual_arm'],capacity_integration_v1=release.get('capacity_integration_v1'),measurement_host=release['node'],completed_s=time.time())
                    if error: record['execution_error']=error
                    write(cp,record,True); return receipt,timing
                try:
                    await common.run_one(session,binding,row,output,hardware)
                    write(hardware_after_path,await hardware_identity(release['node']),True)
                    receipt,timing=checkpoint(); state['completed'].append(row['cell_id']); gate=engineering(receipt,timing)
                    before_hw,after_hw=read(hardware_before_path),read(hardware_after_path)
                    expected_hw=checked(checked(release['qualification'])['hardware_identity'])
                    need(before_hw['gpus']==after_hw['gpus']==expected_hw['gpus'] and before_hw['hostname']==after_hw['hostname']==expected_hw['hostname'],'actual hardware UUID changed during cell')
                    write(a.out/'engineering-gates'/(row['cell_id']+'.json'),gate,True)
                    need(gate['passed'],'request/measurement/native/arrival failure stops expansion: '+str(gate['errors']))
                    if release['system']=='pdblend' and boundary_rate(receipt)<.9:
                        state['first_complete_slo_miss_rate']=min(float(row['rate_rps']),state['first_complete_slo_miss_rate'] or math.inf)
                except BaseException as exc:
                    if rp.exists() and not cp.exists(): checkpoint(repr(exc))
                    state['failed'].append(dict(cell_id=row['cell_id'],error=repr(exc))); raise
                finally:
                    excluded={x['cell_id'] for x in state['skipped_saturated']}; state['remaining']=[r['cell_id'] for r in release['rows'] if r['cell_id'] not in state['attempted'] and r['cell_id'] not in excluded]; save()
            state['remaining']=[r['cell_id'] for r in release['rows'] if r['cell_id'] not in state['attempted'] and r['cell_id'] not in {x['cell_id'] for x in state['skipped_saturated']}]
            state['declared_group_complete']=not state['remaining'] and not state['failed']
            state['rate_search_complete']=state['first_complete_slo_miss_rate'] is not None
            state['complete']=state['declared_group_complete'] and state['rate_search_complete']
            if state['complete']: state['phase']='complete'
            elif state['declared_group_complete']: state['phase']='needs_new_rate_declaration'
            write(a.out/'identity.final.json',await common.identity(session,base),True)
        state['node_lease_held']=False; save()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--release',type=Path,required=True); p.add_argument('--out',type=Path,required=True); p.add_argument('--run',action='store_true'); a=p.parse_args(); a.stop_requested=False
    if not a.run:
        release,_=load_release(a.release); print(json.dumps(dict(cpu_only=True,declared=len(release['rows'])))); return
    def stop(*_): a.stop_requested=True
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,stop)
    state=dict(pid=os.getpid(),started_s=time.time(),phase='starting',complete=False,attempted=[],completed=[],failed=[])
    try: asyncio.run(execute(a,state))
    except BaseException as exc: state.update(phase='failed',error=repr(exc)); raise
    finally:
        if a.out.exists(): state.update(finished_s=time.time(),node_lease_held=False); state.pop('current_cell',None); write(a.out/'status.json',state)

if __name__=='__main__': main()
