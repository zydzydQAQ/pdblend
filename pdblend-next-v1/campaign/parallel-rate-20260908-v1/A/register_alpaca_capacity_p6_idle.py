"""Register the declared idle replacement using each owner's original raw inventory.

The union is an evidence index for the unchanged numerical verifier. It is never
a physical runtime inventory, never a formal CP, and never replaces either run.
"""
from pathlib import Path
import copy
import json
import sys
import time

A=Path(__file__).resolve().parent
sys.path.insert(0,str(A/'load-p6-code-001'))
from capacity_executor import fixed,require,durable,sha
from capacity_certificate import ref,raw_measurement,idle_result,derive_group,build,validate

SELECTION=A/'load-p6-idle-inputs-001/selection.json'
FULL=A/'load-p6-full-002'
IDLE=A/'load-p6-idle-001'
OUT=A/'alpaca-capacity-certificate-p6-002'


def terminal_stage(output,expected_count):
    output=Path(output)
    status_ref=ref(output/'status.json');state=fixed(status_ref)
    require(state.get('complete') is True and state.get('cleanup_complete') is True
        and not state.get('error') and not state.get('cleanup_errors') and state.get('finished_s'),
        'source physical work or cleanup incomplete')
    require(not Path('/proc',str(state['pid'])).exists(),'physical source owner must exit')
    require(len(state['completed'])==expected_count
        and len({r['path'] for r in state['completed']})==expected_count,'actual source observation count differs')
    spec_ref=fixed(ref(output/'spec-reference.json'));sp=fixed(spec_ref)
    for path,digest in sp['files'].items():require(sha(path)==digest,'source spec dependency changed')
    inventory_ref=ref(output/'inventory.json');inventory=fixed(inventory_ref)
    require(inventory.get('schema')!='capacity-evidence-inventory-union-v1'
        and inventory['complete'] is True and inventory['transition_inflight'] is False
        and inventory['pid']==state['pid'],'actual source inventory/owner differs')
    require(set(inventory['initial_ids'])=={'nextv3a6','nextv3a7'}
        and {r['id'] for r in inventory['active_instances']}==set(inventory['initial_ids']),
        'source must return to exact original two instances')
    require(not any(e['kind'] in ('transition_failed','rollback_failed') for e in inventory['events']),
        'source physical transaction failed')
    cap=fixed(sp['capacity_binding'])
    require(inventory['identity']==cap['identity'],'source inventory physical identity differs')
    full_operation=raw_measurement(state['full_operation_measurement'])
    require(state['started_s']<=full_operation['measurement_start_s']
        <full_operation['measurement_end_s']<=state['finished_s'],'source full-operation window outside owner lifecycle')
    for reference in state['completed']:
        result=fixed(reference)
        require(result['complete'] is True and result['work_complete'] is True and result['native_idle'] is True,
                'source observation has incomplete work')
        require(not result.get('failed_requests') and not result.get('request_timeouts'),'source request failure')
    return dict(output=str(output),status=state,status_ref=status_ref,spec=sp,spec_ref=spec_ref,
        inventory=inventory,inventory_ref=inventory_ref,capacity=cap,full_operation=full_operation,
        full_operation_ref=state['full_operation_measurement'])


def verify_idle_mapping(reference,identity,stage,phase):
    """Reject cross-owner, cross-phase and wrong-window evidence before any union."""
    expected_path=Path(stage['output'])/phase/'result.json'
    require(Path(reference['path'])==expected_path,'idle result belongs to another source output/phase')
    require(reference in stage['status']['completed'],'idle result absent from actual owner completed set')
    result=fixed(reference);sp=stage['spec'];inventory=stage['inventory']
    expected_source=dict(original_binding=sp['original_binding'],capacity_binding=sp['capacity_binding'],
        config=sp['config'],host_manifest=ref(Path(sp['host_release'])/'manifest.json'))
    require(result['source']==expected_source and result['phase']==phase and result['phase_kind']=='idle',
        'idle source or phase mapping differs')
    require(inventory['pid']==stage['status']['pid'] and inventory['identity']==identity,
        'idle mapped to another physical owner')
    start,end=result['actual_idle_start_s'],result['actual_idle_end_s']
    require(stage['status']['started_s']<=start<end<=stage['status']['finished_s']
        and stage['full_operation']['measurement_start_s']<=start<end
        <=stage['full_operation']['measurement_end_s'],'idle exact window outside owner measurement')
    require(result['declared_idle_duration_s']==sp['matched_idle_duration_s']==60,
        'original matched sixty-second idle duration changed')
    for resident in result['resident_before']:
        iid=resident['id'];require(iid in inventory['known_instances'],'idle contains foreign owner instance')
        instance=inventory['known_instances'][iid]
        require(instance['id']==iid and sorted(instance['gpus'])==sorted(resident['gpus']),
            'idle physical instance/GPU mapping differs')
        if iid not in inventory['initial_ids']:
            require(instance.get('owner_id')==stage['capacity']['owner_id'] and instance.get('physical_proof'),
                'idle instance belongs to another physical owner or lacks native proof')
    return idle_result(reference,identity,inventory)


def inventory_union(stages,identity,verified_idle):
    known={};initial=set(stages[0]['inventory']['initial_ids'])
    for stage in stages:
        inv=stage['inventory']
        require(set(inv['initial_ids'])==initial and inv['identity']==identity,'source initial identity differs')
        for iid,instance in inv['known_instances'].items():
            require(iid==instance['id'] and instance.get('verified') is True,'unverified physical inventory entry')
            if iid in known:
                # Each actual owner records its own registration timestamp.
                # Every physical identity/provenance field must still match.
                ignored={'changed_s'} if iid in initial else set()
                require({k:v for k,v in known[iid].items() if k not in ignored}
                    =={k:v for k,v in instance.items() if k not in ignored},
                    'same instance ID has conflicting physical proof')
            else:known[iid]=copy.deepcopy(instance)
    return dict(schema='capacity-evidence-inventory-union-v1',identity=identity,
        evidence_index_only=True,physical_runtime_inventory=False,formal_checkpoint_eligible=False,
        complete=True,transition_inflight=False,initial_ids=sorted(initial),known_instances=known,
        active_instances=[known[iid] for iid in sorted(initial)],
        source_inventories=[dict(inventory=s['inventory_ref'],spec=s['spec_ref'],status=s['status_ref'],
            full_operation=s['full_operation_ref'],owner_id=s['capacity']['owner_id'],pid=s['status']['pid']) for s in stages],
        separately_verified_idle_mapping=verified_idle,
        semantics='Read-only index after every idle_result passed against its own actual terminal inventory; never an actual runtime snapshot.')


def main():
    require(not OUT.exists(),'new certificate output required')
    selected=fixed(ref(SELECTION));sources=selected['sources']
    require(Path(sources['full']['output'])==FULL and Path(sources['idle']['output'])==IDLE,
        'only the predeclared full run and idle replacement are allowed')
    stages={'full':terminal_stage(FULL,27),'idle':terminal_stage(IDLE,2)}
    full,new=stages['full'],stages['idle']
    require(full['spec_ref']==sources['full']['spec_ref'],'selected full source differs')
    require(new['spec_ref']['path']==sources['idle']['spec_path']
        and new['spec']['replacement_selection']==ref(SELECTION),'new idle spec/selection chain differs')
    identity=full['capacity']['identity'];capref=full['spec']['capacity_binding']
    require(new['capacity']['identity']==identity==selected['unchanged_identity'],'new idle physical identity differs')
    for key in ('profiles','demand_domain_sha256','host_release','matched_idle_duration_s'):
        require(new['spec'][key]==full['spec'][key],'new idle source/domain differs: '+key)
    require(fixed(new['spec']['config'])==fixed(full['spec']['config']),
        'new idle controller policy/profile differs')
    business=[reference for reference in full['status']['completed']
              if fixed(reference).get('phase_kind')!='idle']
    require(len(business)==21 and selected['business_results']==business,'original21 business observations changed or omitted')
    retained_transitions=[reference for cycle in (1,2,3) for reference in
        (ref(FULL/f'cycle-{cycle}-under_load-layout2to3/result.json'),ref(FULL/f'cycle-{cycle}-remove.json'))]
    require(selected['retained_cold_and_remove_results']==retained_transitions,
        'original three cold and three remove observations changed')
    excluded=[ref(FULL/f'cycle-2-idle-layout{layout}/result.json') for layout in (2,3)]
    require(selected['excluded_idle_pairs']==[excluded] and selected['unselected_original_pair']==excluded
        and selected['rejected_idle_results']==[excluded[1]],'original idle rejection/paired replacement scope changed')
    expected_pairs=[];verified=[]
    for owner,cycle in [('full',1),('full',3),('idle',1)]:
        pair={}
        for side,layout in [('source',3),('target',2)]:
            phase=f'cycle-{cycle}-idle-layout{layout}';reference=ref(Path(stages[owner]['output'])/phase/'result.json')
            verify_idle_mapping(reference,identity,stages[owner],phase)
            pair[side]=reference
            verified.append(dict(result=reference,owner=owner,phase=phase,inventory=stages[owner]['inventory_ref']))
        expected_pairs.append(dict(**pair,source_owner=owner,target_owner=owner))
    require(len(selected['idle_pairs'])==3,'exact three selected idle pairs required')
    for planned,actual in zip(selected['idle_pairs'],expected_pairs):
        require(planned['source_owner']==actual['source_owner'] and planned['target_owner']==actual['target_owner'],
            'selected pair owner differs')
        for side in ('source','target'):
            require(planned[side]['path']==actual[side]['path']
                and (planned[side].get('sha256') in (None,actual[side]['sha256'])),'selected idle evidence differs')
    OUT.mkdir()
    union=inventory_union(list(stages.values()),identity,verified)
    durable(OUT/'inventory-union.json',union);union_ref=ref(OUT/'inventory-union.json')
    durable(OUT/'terminal-source-selection.json',dict(schema='terminal-capacity-evidence-selection-v1',selection=ref(SELECTION),
        original_business_results=business,selected_idle_pairs=expected_pairs,
        inventory_union=union_ref,source_stages=union['source_inventories'],
        original_cycle2_idle_pair_preserved=True,failed_original_layout3_never_relabelled=True,
        selected_before_replacement_measurement=True))
    groups=[]
    def save(name,kind,members,**fields):
        path=OUT/(name+'.json');durable(path,dict(schema='capacity-evidence-group-v1',identity=identity,
            kind=kind,capacity_binding=capref,members=members,
            terminal_source_selection=ref(OUT/'terminal-source-selection.json'),
            registration_source=ref(Path(__file__).resolve()),**fields));return ref(path)
    for layout,key in [(2,'high2'),(3,'high3')]:
        group=save(f'layout{layout}','layout',[ref(FULL/f'cycle-{n}-{key}-layout{layout}'/'result.json') for n in (1,2,3)])
        derive_group(group,identity);groups.append(group)
    savings=[]
    for key in ['idle','low','low40']:
        members=([{side:pair[side] for side in ('source','target')} for pair in expected_pairs] if key=='idle' else
            [dict(source=ref(FULL/f'cycle-{n}-{key}-layout3'/'result.json'),target=ref(FULL/f'cycle-{n}-{key}-layout2'/'result.json')) for n in (1,2,3)])
        group=save('saving-'+key,'idle_savings' if key=='idle' else 'savings',members,
            **({'inventory':union_ref} if key=='idle' else {}));derive_group(group,identity);savings.append(group)
    group=save('savings-grid','savings_grid',savings,demand_domain_sha256=full['spec']['demand_domain_sha256'])
    derive_group(group,identity);groups.append(group)
    for operation in ['restore_cold','remove']:
        paths=[FULL/f'cycle-{n}-under_load-layout2to3'/'result.json' if operation=='restore_cold' else FULL/f'cycle-{n}-remove.json' for n in (1,2,3)]
        group=save(operation,'transition',[dict(result=ref(path),inventory=full['inventory_ref']) for path in paths],operation=operation,gpus=[5])
        derive_group(group,identity);groups.append(group)
    reference=build(identity,groups,OUT/'certificate.json');certificate=fixed(reference);validate(certificate,identity)
    durable(OUT/'qualification.json',dict(passed=True,created_s=time.time(),certificate=reference,
        full_operation=full['full_operation_ref'],replacement_full_operation=new['full_operation_ref'],
        source_status=full['status_ref'],source_inventory=full['inventory_ref'],
        terminal_source_selection=ref(OUT/'terminal-source-selection.json'),
        evidence_inventory_union=union_ref,requires_actual_900s_p6_validation=True,production_ready=False,
        source_scope='Original21 business and3 cold/remove observations retained; only predeclared bad idle pair replaced with actual same-source matched idle pair'))
    print(json.dumps(dict(certificate=reference,layouts=certificate['layouts'],transitions=certificate['transitions'],savings=certificate['savings'])))


if __name__=='__main__':main()
