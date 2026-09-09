"""Physical clock-write eligibility for the current live PDB topology."""
import hashlib
import json
from pathlib import Path
import time


def bootstrap_allowed(controller,gpus,frequency,proof,now):
    """Empty owned GPUs permit only scoped cold setup or post-stop release."""
    if not isinstance(proof,dict):return False
    release=proof.get('schema')=='capacity-clock-release-v1'
    if release:
        if frequency is not None:return False
        states=('stop_intent','start_intent','started_unpublished','ready_unpublished')
        event_kind='clock_release_proof'
    else:
        if frequency!=2520 or proof.get('schema')!='capacity-clock-bootstrap-v1':return False
        states=('start_intent',);event_kind='clock_bootstrap_proof'
    path=Path(controller.config.get('capacity_inventory_path','/missing')).resolve()
    if Path(proof.get('inventory_path','/missing')).resolve()!=path:return False
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=proof.get('inventory_sha256'):return False
    inventory=json.loads(raw);iid=proof.get('instance_id');known=inventory.get('known_instances',{}).get(iid,{})
    if (inventory.get('transition_inflight') is not True or known.get('state') not in states
        or not isinstance(inventory.get('initial_ids'),list) or iid in inventory['initial_ids']
        or known.get('owner_kind')!='created_for_cell' or known.get('transaction')!=proof.get('transaction')
        or set(known.get('gpus',()))!=set(gpus) or set(proof.get('gpus',()))!=set(gpus)
        or iid in controller.backend.instances or len(gpus)!=known.get('tp')):return False
    if set(gpus)&{g for i in controller.backend.instances.values() for g in i['gpus']}:return False
    rows=proof.get('rows',[])
    now=time.time()
    if (len(rows)!=len(gpus) or {r.get('gpu') for r in rows}!=set(gpus)
        or any(r.get('process_pids')!=[] or not 0<=now-r.get('at_s',0)<=1 for r in rows)):return False
    events=inventory.get('events',[])
    if not any(e.get('kind')==event_kind and e.get('instance_id')==iid
        and e.get('transaction')==known['transaction'] and set(e.get('gpus',()))==set(gpus)
        and e.get('rows')==rows for e in events):return False
    from capacity_executor import check_lease
    check_lease(authority=controller.config.get('capacity_lease_authority'),
        expected_inventory=controller.config['capacity_inventory_path'],expected_job_path=controller.config.get('capacity_job_path'))
    return all(0<=time.time()-r['at_s']<=1 for r in rows)


def evaluate(controller,gpus,frequency,reason,bootstrap=None):
    now=time.time();gpus=tuple(gpus);snapshot=controller.state.snapshot
    result=dict(allowed=False,snapshot_version=snapshot.version,requested_gpus=list(gpus),
        frequency_mhz=frequency,reason=reason,instances=[])
    if bootstrap is not None:
        try:allowed=bootstrap_allowed(controller,gpus,frequency,bootstrap,now)
        except (OSError,ValueError,TypeError,KeyError,RuntimeError,ImportError) as exc:
            allowed=False;result['error']='invalid physical ownership proof: '+str(exc)
        result.update(allowed=allowed,bootstrap_instance=bootstrap.get('instance_id') if isinstance(bootstrap,dict) else None,
            bootstrap_no_serving_work=True)
        return result
    backend=controller.backend;mapping={i.instance_id:i for i in snapshot.instances};covered=set()
    for iid,config in backend.instances.items():
        members=set(config['gpus'])
        if not members.intersection(gpus):continue
        if not members<=set(gpus):return dict(result,error='partial TP group write')
        instance=mapping.get(iid)
        if (instance is None or set(instance.gpus)!=members or instance.tp!=config['tp']
            or instance.role not in ('mixed','decode') or not instance.accepting
            or not 0<=now-instance.timestamp_s<=controller.planner.telemetry_ttl_s):
            return dict(result,error='missing/stale/untrusted current topology')
        covered.update(members)
        batch=max(1,len(instance.requests),instance.running+instance.waiting)
        item=dict(instance_id=iid,gpus=sorted(members),batch=batch,current_mhz=instance.frequency_mhz,
            ledger_requests=len(instance.requests),native_requests=instance.running+instance.waiting)
        result['instances'].append(item)
        if not instance.requests:
            raw=backend.last.get(iid,{})
            if (instance.running or instance.waiting or instance.reserved_kv_tokens or instance.kv_allocations
                or instance.reserved_transfer_bytes or instance.transfer_allocations
                or raw.get('active')!=0):return dict(result,error='idle work lacks complete ledger evidence')
            item['trusted_idle']=True;continue
        if instance.running+instance.waiting>len(instance.requests):return dict(result,error='native work missing request-shape ledger')
        if frequency is None:return dict(result,error='cannot park admitted or native work')
        points=[controller.planner.point(instance,r,frequency,batch) for r in instance.requests]
        if not all(points):return dict(result,error='target does not cover full current plus reserved batch')
        promises={r.pending_frequency_mhz for r in instance.requests if not r.emitted}
        if promises and promises!={frequency}:return dict(result,error='outstanding first-token phase frequency promise differs')
        if 'fallback' in reason:
            if controller.frequency_pending_budgets():return dict(result,error='pending admission retains route and clock ownership')
            from .pending_frequency import assessed
            if assessed(controller.planner,instance,frequency,now) is None:
                return dict(result,error='fallback lacks measured existing-prefix and switch-cost feasibility')
        item['target_profile_covered']=True;item['phase_promises']=sorted(promises)
    result['allowed']=set(gpus)==covered and bool(gpus)
    if result['allowed'] and any(not 0<=time.time()-mapping[item['instance_id']].timestamp_s
            <=controller.planner.telemetry_ttl_s for item in result['instances']):
        return dict(result,allowed=False,error='topology telemetry expired during physical eligibility check')
    if not result['allowed']:result['error']='GPU is outside current published topology'
    return result
