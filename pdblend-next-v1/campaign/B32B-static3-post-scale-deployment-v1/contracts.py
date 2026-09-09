"""B-only exact static-three declaration and actual main150/scale90 terminal gate."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
C=ROOT.parent
HOST='iZwz9i5bte3xkpmcoes3t2Z'
DEADLINE=1788872770.0400891
PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
PDB=C/'B32B-five-system100-v1/binding.pdblend.r2.json'
PDB_SHA='02b87002d5278e3797eead73f281cf992e80b878144786d41c8184cee61c3c0f'
INVENTORY=C/'B32B-capacity-current-targets-v1/containers.inspect.json'
INVENTORY_SHA='c26b63a8624a4daa503783a991a18caace6263b7c19ee00487811c1f99e46094'
CANDIDATE=C/'B32B-capacity-after-eco-v1'
CANDIDATE_SHA='537aceb863d5f9df674c56bfa072e807b4c60d9ae9d68bba014ff7592b3da683'
HANDOFF=C/'B32B-main-to-scale-handoff-v2/attempt-001'
SCALE_SPEC=HANDOFF/'scale-bindings/spec.json'
SCALE_SPEC_SHA='69507c8e3631af5dea3129a9dde223c4c2968af1676b3b6040097cb628da9975'
PREVIOUS=HANDOFF/'scale-bindings/ecoserve/binding.json'
PREVIOUS_SHA='1b1b4280ce8dd4b8cd919cf5a1706aa9229eb94776fd1ef0af5b4cc1c9ee423f'
RELEASE=HANDOFF/'B32B-model-release.json'
RELEASE_SHA='3e99003eaf4c4074f00bc4e88055f6b4a4c6316e5f52f03582d0341b0363599e'
SCALE=C/'scale-only-continuation-B32B-v1'
CONTRACT_SHA='482dba053bb9fa95ec895c7bcac6c51311b0b04e6887fcfb346be42fb62fff2b'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok,why):
    if not ok:raise RuntimeError(why)


def merge(target,items):
    for p,h in items.items():
        require(p not in target or target[p]==h,'conflicting source SHA: '+p)
        target[p]=h


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m


def contract():
    require(sha(SCALE/'contract.py')==CONTRACT_SHA,'frozen B scale consumer changed')
    return load(SCALE/'contract.py','b_static3_scale_contract')


def process(pid):
    require(type(pid) is int and pid>0,'actual positive process ID required')
    try:
        p=Path('/proc')/str(pid);a=(p/'stat').read_text().rsplit(') ',1)[1].split()
        argv=(p/'cmdline').read_bytes();b=(p/'stat').read_text().rsplit(') ',1)[1].split()
        return bool(argv and a[0]!='Z' and b[0]!='Z' and a[19]==b[19])
    except (FileNotFoundError,ProcessLookupError):return False


def terminal(scale,parent,live=process):
    require(scale.get('model')=='32b' and scale.get('spec_sha256')==SCALE_SPEC_SHA
        and scale.get('release_sha256')==RELEASE_SHA and scale.get('protocol_id')==PROTOCOL
        and scale.get('deadline_s')==DEADLINE,'actual B scale identity/source differs')
    require(scale.get('complete') is True and scale.get('phase')=='selected_scale_groups_finished'
        and not scale.get('error'),'B scale has not completely finished')
    require(type(scale.get('finished_s')) in (float,int)
        and scale['started_s']<=scale['finished_s']<DEADLINE,'actual scale terminal time missing')
    require(not live(scale['pid']) and scale.get('steps'),'scale supervisor live or no producer steps')
    for step in scale['steps']:
        require(step.get('complete') is True and step.get('exitcode')==0
            and not step.get('unconfirmed_child') and not live(step['pid']),'scale producer/cleanup not terminal')
    require(parent.get('model')=='32b' and parent.get('complete') is True
        and parent.get('phase')=='complete' and not parent.get('error')
        and parent.get('deadline_s')==DEADLINE,'actual handoff not terminal')
    require(parent.get('scale_spec')==dict(path=str(SCALE_SPEC),sha256=SCALE_SPEC_SHA)
        and parent.get('release')==dict(path=str(RELEASE),sha256=RELEASE_SHA)
        and parent.get('scale_pid')==scale['pid'],'handoff did not execute this scale source')
    require(type(parent.get('finished_s')) in (int,float)
        and parent['started_s']<=scale['finished_s']<=parent['finished_s']<DEADLINE,'handoff terminal times differ')
    require(not live(parent['pid']),'handoff parent remains live')
    if parent.get('observer_pid'):require(not live(parent['observer_pid']),'old metadata observer remains live')
    for step in parent.get('steps',[]):
        require(step.get('complete') is True and step.get('exitcode')==0
            and not step.get('exit_unconfirmed') and not live(step['pid']),'handoff child not terminal')
    return True


def completed_scale():
    require(sha(SCALE_SPEC)==SCALE_SPEC_SHA and sha(RELEASE)==RELEASE_SHA,'actual scale/release reference changed')
    paths=[HANDOFF/'scale/status.json',HANDOFF/'status.json']
    scale,parent=[read(p) for p in paths];terminal(scale,parent)
    require(not any((p/'STOP').exists() for p in (ROOT,HANDOFF,HANDOFF/'scale')),'STOP blocks deployment')
    c=contract();released=c.released.verify_release(RELEASE,RELEASE_SHA,expected_model='32b',deep=True)
    checked=c.check_spec(read(SCALE_SPEC),RELEASE,RELEASE_SHA)
    expected={r['cell_id'] for r in read(read(SCALE_SPEC)['source']['path'])['cells'] if r['phase']=='scale'}
    points=[p for g in checked['groups'] for p in g['reused']]
    require(checked['selected_scale_cells']==90 and checked['reused']==90 and checked['pending']==0
        and len(points)==90 and {p['cell_id'] for p in points}==expected,'actual exact90 scale CP/source/producer incomplete')
    scan=c.released.v1.process_scan();require(scan['no_live_serving_child'],'live serving child prevents deployment')
    require(read(paths[0])==scale and read(paths[1])==parent,'terminal journals changed during verification')
    return dict(main_release=released,main_points=150,scale_points=90,checkpoints=points,
        terminal_files={str(p):sha(p) for p in paths},process_scan=scan)


def build(out):
    for path,digest in [(PDB,PDB_SHA),(PREVIOUS,PREVIOUS_SHA),(INVENTORY,INVENTORY_SHA),
                        (CANDIDATE/'manifest.json',CANDIDATE_SHA),(SCALE_SPEC,SCALE_SPEC_SHA),(RELEASE,RELEASE_SHA)]:
        require(sha(path)==digest,'fixed source changed: '+str(path))
    b=read(PDB);previous=read(PREVIOUS);inventory=read(INVENTORY)
    expected={r['Name'].lstrip('/'):r for r in inventory}
    require(len(expected)==2 and [(i['tp'],i['gpus']) for i in b['instances']]==[(2,[0,1]),(2,[2,3])],'original B two TP2 geometry required')
    require(previous['model']=='32b' and previous['system']=='ecoserve'
        and [(i['tp'],i['gpus']) for i in previous['instances']]==[(2,[0,1]),(2,[2,3]),(2,[4,5]),(2,[6,7])],'actual B four baseline residents required')
    instances=copy.deepcopy(b['instances']);files=dict(b['files']);merge(files,previous['files'])
    for i in instances:
        x=expected[i['container']['name']]
        require(x['Id']==i['container']['id'] and x['Image']==i['container']['image']
            and x['State']['Running'] is False and x['State']['Pid']==0,'actual retained stopped identity missing')
        config=x['Args'][x['Args'].index('--config')+1];cfg=read(config)
        require(x['Config']['Cmd']==['python3','-m','ecopadg.serving.engine','--config',config],'retained engine command differs')
        require(cfg['id']==i['id'] and cfg['tp']==2 and cfg['max_num_seqs']==32
            and cfg['max_num_batched_tokens']==8192 and cfg['max_model_len']==8192,'retained model/capacity differs')
        require(files.get(config)==sha(config),'original engine config not frozen')
        i.update(config=config,container_name=i['container']['name'])
    future=read(CANDIDATE/'prepared/deployment.future.json')
    third=copy.deepcopy(instances[0]);third.update(future['third'])
    third.pop('container',None);third.pop('provenance',None)
    third.update(config=future['third_config'],engine_config=future['third_config'],
        native_kind='v3',scheduler_cache_observed=True,scheduler_cache_count=1,
        service_budget_tokens=8192,restore_budget_tokens=8192)
    provenance=copy.deepcopy(instances[0]['provenance'])
    provenance.update(instance_id=third['id'],cuda_visible_devices='4,5')
    for n,h in read(CANDIDATE/'manifest.json')['files'].items():merge(files,{str(CANDIDATE/n):h})
    for path in [PDB,PREVIOUS,INVENTORY,INVENTORY.parent/'receipt.json',CANDIDATE/'manifest.json',RELEASE,SCALE_SPEC,SCALE/'contract.py']:
        merge(files,{str(path):sha(path)})
    require(third['gpus']==[4,5] and third['tp']==2,'exact third pair required')
    return dict(schema=1,model='32b',hostname=HOST,protocol_id=PROTOCOL,deadline_s=DEADLINE,
        operation='restore-two-create-third-after-model-scale',out=str(Path(out).resolve()),
        host_release=b['host_release'],previous_binding=str(PREVIOUS),original_pdb_binding=str(PDB),
        expected_containers=expected,expected_provenance={i['id']:i['provenance'] for i in instances},
        instances=instances,third_instance=third,third_expected_provenance=provenance,
        third_creation=future,files=files,large_inputs=copy.deepcopy(b['large_inputs']),
        model_main_release=str(RELEASE),model_main_release_sha256=RELEASE_SHA,
        original_scale_spec=dict(path=str(SCALE_SPEC),sha256=SCALE_SPEC_SHA),
        deployment_budget_s=720,cleanup_budget_s=120,new_container_creation_allowed=True,
        old_container_removal_allowed=False,output_correctness_verified=False,
        fresh_three_replica_correctness_required=True,performance_authorized=False,
        scale_completion_asserted=False)
