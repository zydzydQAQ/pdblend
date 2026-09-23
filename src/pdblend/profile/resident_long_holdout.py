"""Epoch-aware resident execution of the immutable 14B holdout-only v1 plan.

The original module and package are unchanged. The existing raw-window audit
still validates every accepted window and its actual qualification receipt.
"""
import asyncio
import copy
import json
from contextlib import asynccontextmanager
from pathlib import Path

from . import long_holdout_only as original,long_context_collect as lc,long_context_followup as lf,sampling_guard as guard
from .wave import atomic_json


async def run_existing(*,package,profiler,client,gpus,out,window_boundary=None,qualification_guard=None):
    package,out=Path(package),Path(out);manifest,candidate,plan=original.load_package(package)
    if lf.identity(profiler.raw)!=lf.identity(manifest) or len(gpus)!=1 or len(profiler.specs)!=1:
        raise ValueError('resident long holdout requires own 14B TP1 engine')
    for k in ('image_digest','vllm','torch','cuda','hardware_id'):
        if not manifest['environment'].get(k) or profiler.raw['environment'].get(k)!=manifest['environment'][k]:
            raise ValueError('resident long environment mismatch: '+k)
    if any(lc.point_capacity_error(p,profiler.raw['kv_capacity_tokens']) for p in plan['points']):
        raise ValueError('resident long holdout exceeds live capacity')
    qualifier=qualification_guard or guard.static_guard(profiler);before=copy.deepcopy(profiler.raw)
    child=copy.copy(profiler);child.out_dir=out;out.mkdir(parents=True,exist_ok=True);child.raw=copy.deepcopy(profiler.raw)
    for k in ('external_interference','parallel_interference','concurrency_environment','identity_sha256','evidence_bindings'):
        child.raw.pop(k,None)
    binding=dict(candidate_sha256=manifest['candidate_sha256'],plan_sha256=manifest['plan_sha256'],
        package_manifest_sha256=lc.digest(package/'manifest.json'),resident_wrapper_sha256=lc.digest(__file__))
    child.raw.update(prefill=[],decode=[],mixed=[],transfer=[],static={},decode_pending={},independent_holdout=True,
        measurement_plan_sha256=manifest['plan_sha256'],long_holdout_binding=binding,evidence_class='independent_long_context_holdout',
        config=dict(child.raw['config'],independent_long_context_holdout=True))
    if (out/'raw.json').exists():
        child.resume()
        if child.raw.get('long_holdout_binding')!=binding:raise ValueError('resident long resume binding differs')
    history=child.raw.setdefault('qualification_history',{})
    for prior in history.values():lf.qualification_receipt(out,prior)
    done=lc.resume_points(child.raw,out,plan['points'])
    result=dict(status='failed',complete=False,calibration_passed=False,expected_points=24,new_training_points=0,reused_training_points=36,
        independent_holdout=True,fit_performed=False,formal_eligible=False,energy_comparable=False,binding=binding,
        qualification_scope='per-window epochs; unchanged before/after each accepted window')
    try:
        for point in plan['points']:
            key=lc.point_key(point)
            if key in done:continue
            index=len(child.raw['decode_pending'].get(key,()));active={}
            @asynccontextmanager
            async def background(p,client_arg,plan_arg,tag):
                nonlocal index,active
                if window_boundary is not None:await guard.call(window_boundary,point,index,'long_holdout')
                stamp=guard.snapshot(qualifier);saved=guard.save_binding(out,stamp)
                history[stamp['qualification_sha256']]=saved
                child._lock(point['freq_mhz'],gpus)
                async with lc._background(p,client_arg,plan_arg,tag) as live:
                    yield live
                guard.unchanged(qualifier,stamp)
                active=stamp;index+=1
            def checkpoint(repeats):
                repeats[-1].update(qualification_sha256=active['qualification_sha256'],epoch_id=active['epoch_id'],layout_sha256=active['layout_sha256'])
                child.raw['decode_pending'][key]=repeats;child._checkpoint()
            row=await lc.collect_bounded_decode_point(child,client,gpus,point,purpose='independent_holdout_repair',
                previous=child.raw['decode_pending'].get(key,()),on_window=checkpoint,_background_factory=background)
            child.raw['decode'].append(row);child.raw['decode_pending'].pop(key,None);done.add(key);child._checkpoint()
            print(f'14B resident long holdout {len(done)}/24 {key}',flush=True)
        checked=lf.audit(candidate,child.raw,out,plan);atomic_json(out/'holdout-audit.json',checked);original.load_package(package)
        result.update(status='passed' if checked['complete'] else 'inconclusive',complete=checked['complete'],
            calibration_passed=checked['passed'],audit_sha256=lc.digest(out/'holdout-audit.json'),measured_points=len(done),
            queue_receipt_semantics='measurement_complete_only; calibration_reported_separately')
    except BaseException as exc:result.update(error=f'{type(exc).__name__}: {exc}');raise
    finally:
        child._checkpoint();result['raw_sha256']=lc.digest(out/'raw.json');atomic_json(out/'completion.json',result)
        if profiler.raw!=before:raise ValueError('resident long callback modified parent profile archive')
    return result
