"""Pure CPU mechanism check. Native oracle registration is a separate gate.

This module deliberately does not issue performance qualification by itself.
An actual, frozen same-configuration native reference must be registered first.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/root/workspace/pdblend-next-v1')
READER = ROOT / 'campaign/AC-baseline-binding-v2/gate_evidence.py'
PROTOCOL = 'legacy-temporal-default-trajectory-exact-v2'
LABELS = ('solo96', 'solo192', 'pair96', 'pair192')


def load_reader():
    require(hashlib.sha256(READER.read_bytes()).hexdigest()=='b84a113563f1b064be5ef7a8cbf2006b0790f3b60bb1be3851ce66114dd9e9e6','frozen original gate reader changed')
    spec=importlib.util.spec_from_file_location('shape_original_gate_reader', READER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def require(ok, why):
    if not ok: raise RuntimeError(why)


def difference(a, b):
    return next((dict(position_one_based=k, reference=x, observed=y)
        for k,(x,y) in enumerate(zip(a,b),1) if x!=y), None)


def expected_shapes():
    result=[]
    for label,n in [('solo96',96),('solo192',192)]:
        result.append([1,0,n,[label]])
        result.extend([[0,1,1,[label]] for _ in range(63)])
    result += [[1,0,96,['pair96']]] + [[0,1,1,['pair96']] for _ in range(4)]
    result += [[1,0,192,['pair192']]] + [[0,2,2,['pair96','pair192']] for _ in range(59)]
    result += [[0,1,1,['pair192']] for _ in range(4)]
    return result


def exact_reference(candidate, original):
    """All four independent native outputs must match fixed original003 by label.

    This is a necessary comparison only. It does not certify actual source,
    config, prefill branch, measurement or independent native execution.
    """
    require(set(candidate)==set(original)==set(LABELS), 'four full-label references required')
    for values in [*candidate.values(), *original.values()]:
        require(len(values)==64 and all(type(x) is int and x>=0 for x in values), 'complete native256 required')
    mismatches={label:difference(original[label],candidate[label]) for label in LABELS}
    require(not any(mismatches.values()), 'native256 does not exactly match original003')
    return dict(full256_exact=True, first_differences=mismatches, is_performance_qualification=False)


def temporal(checks, http, events, instances, reference_tokens):
    """Reconstruct original four temporal requests and controls without mutation.

    Caller must validate/reference-pin the native oracle separately. This pure
    result can never substitute for that registration or fresh process gates.
    """
    reader=load_reader(); require(len(instances)==4 and all(i['tp']==2 for i in instances), 'original B four TP2 layout')
    b=instances[1]; phase=checks['temporal']; require(phase.get('complete') is True, 'original temporal work incomplete')
    request_rows=checks.get('requests',[])
    require(len(request_rows)==27 and len({r['request_id'] for r in request_rows})==27, 'complete original27 declaration required')
    rows={}; values={}
    for label,original_label,n in [('solo96','temporal-single-reference',96),('solo192','temporal-single-reference',192),
            ('pair96','temporal-first',96),('pair192','temporal-second',192)]:
        found=[r for r in request_rows if r.get('label')==original_label and r.get('instance_id')==b['id']
            and len(r.get('body',{}).get('prompt',[]))==n and r['body'].get('max_tokens')==64]
        require(len(found)==1, 'missing or duplicate temporal request '+label)
        rows[label]=found[0]; values[label]=reader.valid_request(found[0],http)
    require(set(reference_tokens)==set(LABELS), 'all four registered reference labels required')
    for label in LABELS:
        require(len(reference_tokens[label])==64 and all(type(t) is int and t>=0 for t in reference_tokens[label]), 'full reference output required')
        require(values[label]==reference_tokens[label], 'actual temporal/native exact differs: '+label)
    require(phase['reference_token_ids']==[values['solo96'],values['solo192']]
        and phase['token_ids']==[values['pair96'],values['pair192']], 'original temporal summary differs from actual HTTP')
    old_diff=[difference(values['solo96'],values['pair96']),difference(values['solo192'],values['pair192'])]
    require(phase['first_differences']==old_diff, 'old single-versus-pair failure was not preserved')
    require(bool(checks.get('checks',{}).get('temporal_exact')) == (not any(old_diff)), 'original exact header changed')
    first,second=rows['pair96']['request_id'],rows['pair192']['request_id']
    allocated,held=phase['first_allocated'],phase['held']
    require(first in allocated['kv_allocations'] and allocated['running']==1, 'first real allocation absent')
    require(second not in held['kv_allocations'] and held['waiting']>=1 and held.get('admit_prefill') is False
        and held.get('admit_decode') is True and held.get('mode')=='temporal', 'closed prefill hold absent')
    for label,admit in [('close_prefill',False),('open_prefill',True)]:
        control=phase[label];command=control['command'];before,after=control['before'],control['after']
        require(command['mode']=='temporal' and command['role']=='mixed' and command['admit_prefill'] is admit
            and command['admit_decode'] is True and command['generation']==before['generation']+1
            ==after['generation']==after['acknowledged_generation'], 'actual temporal control/ACK changed')
        actual=[r for r in http if r.get('instance')==b['id'] and r.get('route')=='/control'
            and r.get('body')==command and r.get('response')==command and r.get('status')==200 and not r.get('error')]
        require(len(actual)==1, 'control has no unique actual HTTP receipt')
    require(held['generation']==phase['close_prefill']['after']['generation'], 'hold observed under another generation')
    for observation in (allocated,held):
        matching=[r for r in http if r.get('instance')==b['id'] and r.get('route')=='/runtime'
            and r.get('status')==200 and r.get('response')==observation and not r.get('error')]
        require(bool(matching), 'allocation/hold does not match actual runtime response')
    request_labels={row['request_id']:label for label,row in rows.items()}
    owned=[e for e in events[b['id']] if set(e.get('request_ids',[]))&set(request_labels)]
    require(all(set(e['request_ids'])<=set(request_labels) for e in owned), 'foreign request joined fixed reference shape')
    shapes=[[e['prefill'],e['decode'],e['tokens'],[request_labels[r] for r in e['request_ids']]] for e in owned]
    require(shapes==expected_shapes(), 'unsupported owner trajectory: exact frozen197/69 required')
    for label,row in rows.items():
        own=[e for e in owned if row['request_id'] in e['request_ids']]
        require(len(own)==64 and all(e['finished_s']>=e['started_s'] and row['dispatch_s']<=e['started_s']
            <=e['finished_s']<=row['finished_s'] for e in own), 'owner execution is outside actual request')
        if label.startswith('pair'):
            require(all(e.get('mode')=='temporal' and not (e['prefill'] and e['decode']) for e in own), 'temporal owner phase missing/overlapped')
    return dict(protocol_id=PROTOCOL, temporal_native_trajectory_exact=True, full_temporal_outputs=256,
        owner_nonempty_steps=197,pair_steps=69,legacy_single_vs_pair_exact=not any(old_diff),
        legacy_first_differences=old_diff, actual_request_ids={k:v['request_id'] for k,v in rows.items()},
        numerical_scope='same source/config/native scheduling shape exact; not bitwise KV or mathematical correctness',
        is_performance_qualification=False)


def inspect_fresh_gate(gate, binding, reference_tokens, power_evidence):
    """Full original non-temporal/native/measurement checks plus new shape check.

    Intentionally returns eligible=False until the independent native v2 raw
    registrar and explicit versioned binder consume this result. Never writes
    old flags or treats a supplied token dictionary as a registered oracle.
    """
    reader=load_reader();gate=Path(gate)
    original,files=reader.audit(gate,binding['instances'],'distserve',power_evidence)
    require(original['verified']['ordinary'] and original['verified']['pd'], 'all ordinary/PD/cancel checks retained')
    before,after=reader.identities(gate,binding['instances'])
    require(len(before)==len(after)==4, 'all four actual process identities required')
    require(binding.get('identity_file') and binding['files'].get(binding['identity_file'])==reader.sha(binding['identity_file']),
        'fresh complete identity file must be SHA-bound')
    inventory=reader.read(binding['identity_file']); by={r['Id']:r for r in inventory}
    for row in after:
        container=row['container'];actual=by[container['Id']]
        require(container['State']['Pid']==actual['State']['Pid'] and container['State']['StartedAt']==actual['State']['StartedAt'], 'gate is from another host process')
    checks=reader.read(gate/'checks/checks.json');status=reader.read(gate/'status.json');events={}
    for instance in binding['instances']:
        candidates=[(path,rec) for path,rec in status['events'].items() if Path(path).name==instance['id']+'.control.events.jsonl']
        require(len(candidates)==1,'actual owner file missing');path,rec=candidates[0];local=gate/Path(path).name
        require(reader.sha(local)==rec['sha256'],'actual owner SHA changed');events[instance['id']]=reader.lines(local)
    matched=temporal(checks,reader.lines(gate/'checks/http.jsonl'),events,binding['instances'],reference_tokens)
    require(reader.files(gate)==files, 'fresh gate changed during audit')
    return dict(schema=2,kind='temporal-default-trajectory-prequalification',protocol_id=PROTOCOL,
        original_gate=original,temporal=matched,files=files,
        eligible_systems={'ecoserve':False},pending='actual same-configuration native oracle registration and fixed versioned binding required')
