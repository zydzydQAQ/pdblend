"""Fail-closed registration of an actual same-config native reference, CPU only."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import shape_gate as g

ROOT=g.ROOT
PACKAGE=ROOT/'campaign/B32B-native-default-reference-execution-v2'
ATTEMPT=ROOT/'campaign/B32B-native-default-reference-attempt-002'
ORIGINAL=ROOT/'campaign/B32B-temporal-observation-attempt-003'
PEER=ROOT/'campaign/B32B-native-default-reference-result-peer-v1/analyze.py'
PEER_SHA='c0be09d8d2e6fb3bd231368cba1152960b147ca728338c29bc1c2af4c83b6950'
SCHEDULER_SHA='572cdcb93e1af27439cf31534a7d1777eaf70f4b73432abd14f08ddfcfe9a691'


def resolved(p):
    p=Path(p)
    if p.is_file():return p
    prefix=Path('/root/workspace/pdblend/new-results/campaigns/node-b-v9/quick32-v1')
    if prefix in p.parents:
        mirror=ROOT/'campaign/baseline-raw-mirror-v1/B'/str(p).lstrip('/')
        if mirror.is_file():return mirror
        if p==prefix/'interconnect.txt':return Path('/root/workspace/pdblend/new-results/campaigns/three-pool-v2/interconnect.txt')
    if p.name=='worker-config.json' and ATTEMPT in p.parents:
        return ATTEMPT/'results/worker-config.json'
    return p
def sha(p):return hashlib.sha256(resolved(p).read_bytes()).hexdigest()
def read(p):return json.loads(resolved(p).read_text())
def require(ok,msg):g.require(ok,msg)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify_emitted(events,native):
    outputs={rid:[] for rid in native};current=None;seen=set();count=0
    for event in events:
        if event['kind']=='executed_step':
            if current is not None:require(seen==set(current['request_ids']),'owner output missing from previous step')
            current=event;seen=set()
        elif event['kind']=='output':
            rid=event['request_id'];values=event['token_ids']
            require(current is not None and rid in current['request_ids'] and rid not in seen,'output SID/owner mapping differs')
            require(values[:-1]==outputs[rid] and len(values)==len(outputs[rid])+1
                and event['finished'] is (len(values)==64),'output prefix/finish differs')
            outputs[rid]=values;seen.add(rid);count+=1
    require(current is not None and seen==set(current['request_ids']) and count==256 and outputs==native,
        'actual owner does not emit complete native256')


def contract(package_sha,spec_path,spec_sha):
    require(Path(spec_path).resolve()==ATTEMPT/'spec.json','only explicit same-config native attempt002')
    require(package_sha=='b4e299bc703e2146582d3180d01d41eddb60dac2b91394033f5d0a1b66e7d4ad' and len(spec_sha or '')==64 and all(c in '0123456789abcdef' for c in spec_sha),'actual fixed reviewed package/spec SHA values required')
    return dict(schema=2,kind='native-default-temporal-oracle-inputs',protocol_id=g.PROTOCOL,
        package=dict(path=str(PACKAGE/'manifest.json'),sha256=package_sha),
        native_spec=dict(path=str(ATTEMPT/'spec.json'),sha256=spec_sha),
        original_observation_spec=dict(path=str(ORIGINAL/'observation-spec.json'),
            sha256='5cf79e2072a9870fe5b41727e271dd2fd13ddcb7b1565c3f258993acbdf60fe2'))


def verify(inputs):
    require(inputs==contract(inputs['package']['sha256'],inputs['native_spec']['path'],inputs['native_spec']['sha256']),
        'oracle contract identity/protocol/source changed')
    files={}
    def fixed(path,expected=None):
        path=Path(path).resolve();digest=sha(path)
        require(expected is None or digest==expected,'actual oracle input SHA differs: '+str(path))
        files[str(path)]=digest;return read(path)
    package=fixed(inputs['package']['path'],inputs['package']['sha256'])
    for name,digest in package['files'].items():
        path=PACKAGE/name;require(sha(path)==digest,'same-config reference implementation changed');files[str(path)]=digest
    spec=fixed(inputs['native_spec']['path'],inputs['native_spec']['sha256'])
    require(spec['reference_package_manifest_sha256']==inputs['package']['sha256'] and spec['results']==str(ATTEMPT/'results'),
        'actual native operation used another implementation/output')
    require(spec['deadline_s']==1788872770.0400891 and spec['model']=='32b','original model/deadline')
    for path,digest in spec['files'].items():
        require(sha(path)==digest,'actual frozen reference dependency changed');files[path]=digest
    ns=fixed(spec['observation_spec'],spec['observation_spec_sha256'])
    old=fixed(inputs['original_observation_spec']['path'],inputs['original_observation_spec']['sha256'])
    require(ns['requests']==old['requests'][:4] and ns['request_ids']==old['request_ids'],'original four UUID/body declarations changed')
    status=fixed(ATTEMPT/'results/status.json');child=fixed(ATTEMPT/'results/child/status.json')
    require(status['complete'] and status['observation_completed'] and status['measurement_valid']
        and status['all_original_restored'] and status['capture_complete'] and not status['errors'],'actual native terminal evidence incomplete')
    require(child['complete'] and child['cleanup_complete'] and child['completed_requests']==4 and not child['errors'],
        'actual independent native256/cleanup incomplete')
    native=fixed(ATTEMPT/'results/child/full-outputs.json')['token_ids_by_request_uuid']
    original=fixed(ORIGINAL/'results/child/full-outputs.json','4f8ecbfa9592109f8144d6bf6477f27f25e34cdb7d08589eb709b921a8b03b52')['token_ids_by_request_uuid']
    tokens={label:native[row['request_uuid']] for label,row in zip(g.LABELS,ns['requests'])}
    historical={label:original[row['request_uuid']] for label,row in zip(g.LABELS,ns['requests'])}
    g.exact_reference(tokens,historical) # No registration if any of 256 tokens differs.
    require(sha(PEER)==PEER_SHA,'frozen independent capture reader changed');files[str(PEER)]=PEER_SHA
    peer=load('temporal_oracle_raw_peer',PEER);peer.FILES=files
    events=peer.lines(ATTEMPT/'results/diagnostic-owner.events.jsonl')
    owner=[e for e in events if e['kind']=='executed_step']; ids={r['label']:r['request_uuid'] for r in ns['requests']}
    remap={r['request_uuid']:label for label,r in zip(g.LABELS,ns['requests'])}
    require([[e['prefill'],e['decode'],e['tokens'],[remap[r] for r in e['request_ids']]] for e in owner]==g.expected_shapes(),
        'native owner is not actual197/69 reference')
    require(all(not e['preempted'] and not e['blocks_to_swap_in'] and not e['blocks_to_swap_out'] and not e['blocks_to_copy'] for e in owner),
        'native shape moved/preempted KV')
    selection=[e for e in events if e['kind']=='default_selection']
    restored=[e for e in events if e['kind']=='default_selection_restored']
    require(len(selection)==len(restored)==1 and selection[0]['owner']==0
        and selection[0]['engine_chunked'] is True and selection[0]['scheduler_chunked'] is True
        and selection[0]['diagnostic_entry_override'] is True and selection[0]['scheduler_sha256']==SCHEDULER_SHA
        and restored[0]['config_objects_unchanged'] is True and restored[0]['engine_chunked'] is True,
        'official default selection/all-True original config not observed')
    actual=fixed(ATTEMPT/'results/diagnostic-identity.before.json')
    require(actual['installed_sources']==spec['installed_diagnostic_sources'] and
        actual['installed_sources']['/usr/local/lib/python3.10/dist-packages/vllm/core/scheduler.py']==SCHEDULER_SHA,
        'actual installed native source identity differs')
    require(actual['provenance']['tp']==2 and actual['provenance']['dtype']=='bfloat16'
        and actual['provenance']['cuda_visible_devices']=='2,3','actual numerical device configuration changed')
    peer.capture(ATTEMPT,native,ids,owner)
    verify_emitted(events,native)
    checks=peer.lines(ATTEMPT/'results/child/checks/http.jsonl')
    http=[r for r in checks if r['route']=='/native-reference']
    require(len(http)==1 and http[0]['status']==200 and http[0]['body']['requests']==ns['requests']
        and http[0]['response']['token_ids_by_request_uuid']==native,'independent step output does not match actual HTTP')
    # Reuse the new frozen prefill predicate validator, writing its derived copies
    # only to a fresh temporary audit directory; no original artifact is changed.
    def tracked_sha(path):
        path=Path(path).resolve();digest=sha(path)
        if ROOT in path.parents: files.setdefault(str(path),digest)
        return digest
    stub=types.ModuleType('adapter');stub.read=lambda p:fixed(p);stub.sha=tracked_sha
    previous=sys.modules.get('adapter')
    try:
        sys.modules['adapter']=stub
        prefill=load('qualified_prefill_raw',PACKAGE/'prefill_evidence.py')
        with tempfile.TemporaryDirectory(prefix='temporal-oracle-prefill-') as tmp:
            prefills=prefill.verify(spec,Path(tmp))
    finally:
        if previous is None:sys.modules.pop('adapter',None)
        else:sys.modules['adapter']=previous
    require(prefills['complete'] and prefills['records']==8 and prefills['all_worker_configs_true'], 'actual two-rank prefill/config predicate incomplete')
    for path in (ATTEMPT/'results/prefill-capture-live').glob('*'):
        files[str(path)]=sha(path)
    physical_path=ROOT/'campaign/B32B-temporal-solo-pair-capture-peer-review-v1/analyze.py'
    require(sha(physical_path)=='b6a0dceac44fd4f41b0dff9c8d49d2d8cbccbf06e482cdf0ff5d201212542a19','frozen physical reader changed');files[str(physical_path)]=sha(physical_path)
    physical=load('qualified_native_physical',physical_path);physical.BASE=ATTEMPT;physical.FILES=files
    energy=physical.physical_evidence(status)
    power_module=load('qualified_power_source',Path(__file__).with_name('power_source.py'))
    energy['instant_provenance']=power_module.audit_raw(ATTEMPT/'results/power',files)
    require(status['finished_s']<=spec['deadline_s'],'actual reference exceeded original deadline')
    for path,digest in files.items():require(sha(path)==digest,'oracle evidence changed during audit')
    return dict(schema=2,kind='registered-native-default-temporal-oracle',protocol_id=g.PROTOCOL,
        inputs=copy.deepcopy(inputs),full256_exact_original003=True,reference_tokens_by_label=tokens,
        reference_shape=g.expected_shapes(),actual_native_config_true=True,actual_prefill_predicate='paged_kv',
        branch_is_source_inference=True,physical_evidence=energy,files=files,
        legacy_single_vs_pair_exact=False,legacy_failure_preserved=True,performance_qualification=False,
        input_path_policy='original path labels; missing historical B baseline bytes may be read from the exact B raw mirror, interconnect from three-pool-v2; every expected SHA remains mandatory')


def load_registered(path,digest):
    require(sha(path)==digest,'registered oracle SHA differs')
    saved=read(path);actual=verify(saved['inputs'])
    require(saved==actual,'registered oracle differs from original raw revalidation')
    return actual
