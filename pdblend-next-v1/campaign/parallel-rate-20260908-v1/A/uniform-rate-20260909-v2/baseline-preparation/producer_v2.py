"""Fresh-node baseline deployment using the frozen legacy creation and cleanup primitives.

This stage produces actual native identities and a correctness-only bootstrap.
The separate qualification producer must pass before any performance handoff.
"""
import argparse
import asyncio
import copy
import os
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
DEPLOY = p.REPO / 'campaign/AC-baseline-deployment-v1'
BINDER = p.REPO / 'campaign/AC-baseline-binding-v2'
POLICY = p.REPO / 'campaign/AC-baseline-100s-preparation-v1'


def files(path):
    return {str(x):p.sha(x) for x in Path(path).rglob('*') if x.is_file() and '__pycache__' not in x.parts}


def template(layout, destination):
    p.need(not destination.exists(), 'fresh template path required')
    historical = p.load(POLICY / 'bind.py', 'legacy_baseline_policy')
    hetero = layout == 'lb_distserve'
    cfg, _ = historical.configuration('14b', 'longbench' if hetero else 'alpaca', 'distserve' if hetero else 'mixed')
    inventory = p.read(DEPLOY / 'inputs/A-containers-before.json')
    old = [c for c in inventory if c['Name'].lstrip('/').startswith('pdb-v2-cal0_')]
    by_gpu = {int(next(v.split('=',1)[1] for v in c['Config']['Env'] if v.startswith('CUDA_VISIBLE_DEVICES='))):c for c in old}
    index = p.read(DEPLOY / 'inputs/engine-config-index.json')['A']
    engine_templates = {int(k.rsplit('_',1)[1]):p.read(DEPLOY / v['copied_path']) for k,v in index.items()}
    instances = []
    for n, row in enumerate(cfg['instances']):
        source = p.read(DEPLOY / 'inputs/A-distserve-longbench-tp2.json') if row['tp'] == 2 else engine_templates[row['gpus'][0]]
        p.need(source['model'] == '/models/Qwen2.5-14B-Instruct' and source['tp'] == row['tp']
               and source['max_model_len'] == source['max_num_batched_tokens'] == 8192 and source['max_num_seqs'] == 32,
               'historical native work limits changed')
        source = copy.deepcopy(source)
        source.pop('retained_weights', None)
        instance = dict(tp=row['tp'], gpus=row['gpus'], role=row['role'], engine_template=source,
            environment=[e for e in by_gpu[row['gpus'][0]]['Config']['Env'] if not e.startswith('CUDA_VISIBLE_DEVICES=')],
            mounts=by_gpu[row['gpus'][0]]['Mounts'])
        instances.append(instance)
    host = HERE / 'platform-domain/runtime-baseline-002'
    source_files = files(DEPLOY)
    for directory in (BINDER, POLICY):
        source_files.update(files(directory))
    source_files.update({str(host / f):h for f,h in p.read(host / 'manifest.json')['files'].items()})
    for path in (host / 'manifest.json', Path(__file__), ROOT / 'common/execution-until-complete-v1/run.py',
                 ROOT / 'A/uniform-rate-20260909-v1/bootstrap-spec.json',
                 HERE / 'platform-domain/profiles-baseline-domain2100.json'):
        source_files[str(path)] = p.sha(path)
    value = dict(schema='uniform-new-A-legacy-template-v2', node='Anew20260909', model='14b', layout=layout,
        expected_hostname='iZwz9274emxme9019d2sjgZ', image='sha256:d11407cd827a43a0dec8ad7d4d7037c97c39bbe93c6f4b4fd951c94e67509a8b',
        host_release=str(host), source_entry=str(DEPLOY / 'engines/A/engine.py'), instances=instances,
        profile=p.ref(HERE / 'platform-domain/profiles-baseline-domain2100.json'),
        clock_domain_mhz=[900,1500,2100], files=source_files, old_node_qualification_inherited=False,
        deployment_budget_s=720, cleanup_budget_s=120)
    p.save(destination, value)
    return p.ref(destination)


def terminal(reference, layout):
    saved = p.checked(reference)
    p.need(saved['node'] == 'Anew20260909' and saved['model'] == '14b' and saved['complete']
           and saved['finished_s'] and not saved['node_lease_held'], 'predecessor scope is not terminal')
    last = p.checked(saved['last_cell_status'])
    p.need(last['complete'] and last['finished_s'] and not last['node_lease_held'] and not p.active_owner(last)
           and not last.get('error') and not last.get('failed'), 'previous measurement still active or failed')
    contract = p.load(ROOT / 'common/uniform-rate-20260909-v2/contract.py', 'newA_baseline_boundary')
    decisions = {}
    for dataset in ('alpaca', 'longbench'):
        group = contract.resolve_group(saved['declaration'], '14b', dataset, actual_host='Anew20260909')
        observations = [p.checked(o) for o in saved['observations'] if p.checked(o)['dataset'] == dataset]
        decision = contract.select_group(group, observations)
        p.need(decision.get('pdb_boundary_complete') and decision['phase'] in ('baselines','complete'), 'PDB boundary incomplete')
        if layout == 'lb_distserve':
            missing = [t['row'] for t in decision.get('baseline_tasks', []) if t['action'] == 'execute']
            p.need(all(r['dataset'] == 'longbench' and r['system'] == 'distserve' for r in missing), 'resident baseline scope incomplete')
        decisions[dataset] = decision
    binding = p.checked(saved['binding'])
    p.need(binding['hostname'] == 'iZwz9274emxme9019d2sjgZ' and binding['model'] == '14b', 'foreign predecessor binding')
    return saved, decisions


def make_spec(t, terminal_ref, previous, out):
    deployment = out / 'deployment'
    p.need(not deployment.exists(), 'new deployment directory required')
    deployment.mkdir(parents=True)
    hetero = t['layout'] == 'lb_distserve'
    portbase, kvbase = (38200, 58000) if hetero else (38000, 57000)
    records = []
    configs = {}
    for n, row in enumerate(t['instances']):
        rid = ('uniforma2l' if hetero else 'uniforma2r') + str(n)
        config = copy.deepcopy(row['engine_template'])
        config.pop('retained_weights', None)
        config['weight_cache_root'] = str(deployment / 'weights')
        config.update(id=rid, tp=row['tp'], role=row['role'], port=portbase+n, kv_port=kvbase+32*n,
                      runtime_dir=str(deployment / 'native'), initial_generation=0)
        configs[rid] = config
        records.append(dict(id=rid, tp=row['tp'], gpus=row['gpus'], role=row['role'], port=config['port'],
            kv_port=config['kv_port'], url='http://127.0.0.1:'+str(config['port']),
            container_name='pdb-v2-'+rid, config=str(deployment / 'engines' / (rid+'.json')),
            engine_entry=t['source_entry'], image=t['image'],
            environment=row['environment']+['CUDA_VISIBLE_DEVICES='+','.join(map(str,row['gpus'])),'PYTHONDONTWRITEBYTECODE=1'],
            mounts=row['mounts'], native_kind='legacy_sync_put', scheduler_cache_observed=False))
    peers={i['id']:dict(host='127.0.0.1',tp=i['tp'],kv_port=i['kv_port']) for i in records}
    frozen=dict(t['files'])
    for i in records:
        configs[i['id']]['peers']=peers
        p.save(i['config'], configs[i['id']]);frozen[i['config']]=p.sha(i['config'])
    for ref in (terminal_ref, previous, p.ref(__file__)):
        frozen[ref['path']]=ref['sha256']
    spec=dict(schema='uniform-new-A-legacy-deployment-v2', protocol_id='per-dataset-slo-five-system-fixed-window-v1',
        model='14b', node='Anew20260909', layout='distserve-longbench' if hetero else 'resident',
        hostname=t['expected_hostname'], deadline_s=None, campaign_lifecycle='until_declared_complete_v1',
        out=str(deployment), host_release=t['host_release'], executor_release=str(ROOT/'common/execution-until-complete-v1'),
        instances=records, required_predecessors=[], predecessor_terminal=terminal_ref,
        previous_binding=previous['path'], pdb_binding=previous['path'], image=t['image'], files=frozen,
        supported_datasets=['longbench'] if hetero else ['alpaca','longbench'], source_entry=t['source_entry'],
        deployment_budget_s=t['deployment_budget_s'], cleanup_budget_s=t['cleanup_budget_s'])
    p.save(deployment/'deployment.json',spec)
    return p.ref(deployment/'deployment.json')


async def perform(t, spec_ref, out, state):
    helper=p.load(ROOT/'B/baseline-return-after-external-source-v1/execution.py','newA_legacy_executor')
    executor=helper.load_common(t['host_release'])
    from ecopadg.serving.campaign import node_lease
    deploy=p.load(DEPLOY/'deploy.py','newA_frozen_legacy_deployer')
    # The original bounded operation/cleanup budgets remain unchanged; only its
    # obsolete campaign-wide wall clock becomes a fresh per-stage allowance.
    deploy.DEADLINE=time.time()+2*(t['deployment_budget_s']+t['cleanup_budget_s'])
    binder=p.load(BINDER/'bind.py','newA_frozen_native_binder')
    spec=p.checked(spec_ref)
    p.need('PDBLEND_NODE_LOCK_FD' not in os.environ,'inherited hardware lease forbidden')
    try:
        with node_lease():
            state['node_lease_held']=True;p.save(out/'status.json',state)
            await deploy.launch(SimpleNamespace(spec=Path(spec_ref['path'])))
            receipt_ref=p.ref(out/'deployment/deployment-receipt.json')
            receipt=p.checked(receipt_ref)
            frozen=dict(spec['files'])
            frozen.update(files(out/'deployment/deployment-power'))
            for reference in (spec_ref, receipt_ref):frozen[reference['path']]=reference['sha256']
            instances,inventory=await binder.live(spec,receipt,executor,frozen)
            original=p.read(spec['previous_binding'])
            frozen.update(original['files'])
            output=out/'bootstrap';output.mkdir()
            p.save(output/'identity.json',inventory);frozen[str(output/'identity.json')]=p.sha(output/'identity.json')
            binding=dict(schema=1, protocol_id=spec['protocol_id'],model='14b',system='mixed',implementation_variant='correctness-only',
                hostname=spec['hostname'],deadline_s=None,campaign_lifecycle='until_declared_complete_v1',host_release=spec['host_release'],
                output=str(output/'results'),configs={},instances=instances,files=frozen,large_inputs=original.get('large_inputs',{}),
                window_s=100,seeds=[701],deployment_receipt=receipt_ref['path'],identity_file=str(output/'identity.json'),
                correctness_gate_required_before_performance=True,output_correctness_verified=False,formal_eligible=False,
                fresh_node_native_identity=True,old_node_qualification_inherited=False,qualified_profile_required=t['profile'])
            executor.validate_binding(binding)
            p.save(output/'binding.json',binding)
            state.update(binding=p.ref(output/'binding.json'), identity=p.ref(output/'identity.json'), deployment_receipt=receipt_ref)
    finally:
        state['node_lease_held']=False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--make-template',choices=('8tp1','lb_distserve'))
    parser.add_argument('--spec',type=Path,required=True)
    parser.add_argument('--predecessor-terminal',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--run',action='store_true')
    a=parser.parse_args()
    if a.make_template:
        print(template(a.make_template,a.spec));return
    t=p.checked(p.ref(a.spec))
    for path,digest in t['files'].items():p.need(p.sha(path)==digest,'frozen producer input changed: '+path)
    p.need(a.predecessor_terminal and a.out and a.run,'explicit predecessor/out/run required')
    p.need(socket.gethostname()==t['expected_hostname'],'wrong physical node')
    p.need(not a.out.exists() and not (HERE.parent/'STOP').exists(),'fresh operation required; stop must be absent')
    prior,decisions=terminal(p.ref(a.predecessor_terminal),t['layout'])
    a.out.mkdir(parents=True)
    spec_ref=make_spec(t,p.ref(a.predecessor_terminal),prior['binding'],a.out)
    state=dict(schema='uniform-v2-new-A-baseline-bootstrap-status',node='Anew20260909',model='14b',layout=t['layout'],
        pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],started_s=time.time(),complete=False,
        node_lease_held=False,spec=spec_ref,template=p.ref(a.spec),predecessor_terminal=p.ref(a.predecessor_terminal),
        group_decisions=decisions)
    p.save(a.out/'status.json',state)
    async def controlled():
        task=asyncio.current_task()
        for signum in (signal.SIGTERM,signal.SIGINT):asyncio.get_running_loop().add_signal_handler(signum,task.cancel)
        await perform(t,spec_ref,a.out,state)
    try:
        asyncio.run(controlled());state['complete']=True
    except BaseException as exc:
        state['error']=repr(exc);raise
    finally:
        state['finished_s']=time.time();p.save(a.out/'status.json',state)


if __name__=='__main__':main()
