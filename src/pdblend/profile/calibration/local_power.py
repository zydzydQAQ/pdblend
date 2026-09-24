"""Model-owned 14B TP4 local power validation; never promotes a full profile.

The shared power runner owns loading, collection, checkpointing and cleanup.
This adapter supplies the immutable twelve-point contract and an explicit
isolated-group gate, or a coordinator-provided multi-member ProfileWave.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
from contextlib import asynccontextmanager

from pdblend.profile.calibration import power_calibration as pc
from pdblend.profile.calibration.core import _checkpoint_points, evaluate_holdout
from pdblend.profile.identity import sha256_value
from pdblend.profile.query.model import PerfModel
from pdblend.profile.query.power_table import validate as validate_table
from pdblend.profile.collection.wave import ProfileWave, atomic_json

SCOPE = '14b_tp4_B1_B128_six_frequency_local_power_only'
IDENTITY = ('pdblend', 'Qwen2.5-14B-Instruct', 4, 1)


def source_hashes():
    from pdblend.source_inventory import implementation_hashes as inventory_hashes
    return inventory_hashes()


def read_bound(root, name, expected):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()) or pc.digest(path) != expected:
        raise ValueError('local power bound sample checksum/path mismatch')
    return json.loads(path.read_text())


def verify_training(training, root, model, base):
    if tuple(training.get(k) for k in ('system','model_id','tp','pp')) != IDENTITY or training.get('holdout_independent') is not False:
        raise ValueError('local power requires independent 14B training, never holdout')
    if (model.system,Path(model.model).name,model.tp,model.pp) != IDENTITY or tuple(model.freqs) != pc.FREQUENCIES:
        raise ValueError('wrong local power candidate identity/frequencies')
    if pc.timing_fields(model) != pc.timing_fields(base):
        raise ValueError('local power candidate changes existing timing')
    if training.get('identity_sha256') != sha256_value({k:v for k,v in training.items() if k != 'identity_sha256'}):
        raise ValueError('training archive identity checksum differs')
    count = 0
    for f in pc.FREQUENCIES:
        rows = [r for r in training['decode'] if r['freq_mhz'] == f]
        spec = model.decode_power_overrides.get(f)
        if not rows or not spec or len(rows) != len(spec['nodes']):
            raise ValueError('missing own training power nodes')
        validate_table(spec)
        for row in rows:
            if len(row['repeats']) != 3:
                raise ValueError('training requires three complete windows')
            for rep in row['repeats']:
                sample = read_bound(root,rep['samples_file'],rep['samples_sha256'])
                context = row['context_tokens'] + statistics.fmean((a+b)/2 for a,b in zip(sample['start_token_counts'],sample['end_token_counts']))
                power = statistics.fmean(sum(values) for _,values in sample['power'])
                if (not math.isclose(context,rep['effective_context_tokens'],abs_tol=1e-8) or
                    not math.isclose(power,rep['power_w'],rel_tol=1e-9) or rep['steady_window_s'] < 5 or
                    rep['min_steps'] < 8 or len(sample['power']) < 2 or not sample['frequency']):
                    raise ValueError('training power/context/window evidence differs')
                count += 1
            node = dict(batch=row['batch'],nominal_context_tokens=row['context_tokens'],
                context_min=min(r['effective_context_tokens'] for r in row['repeats']),
                context_max=max(r['effective_context_tokens'] for r in row['repeats']),
                power_w=statistics.fmean(r['power_w'] for r in row['repeats']))
            actual = [n for n in spec['nodes'] if (n['batch'],n['nominal_context_tokens']) == (row['batch'],row['context_tokens'])]
            if len(actual) != 1 or actual[0].keys() != node.keys() or any(not math.isclose(node[k],actual[0][k],rel_tol=1e-12,abs_tol=1e-9) for k in node):
                raise ValueError('candidate power node is not the frozen own training mean')
    return count


def validate_plan(plan, model, candidate_sha):
    expected = {(f,b) for f in pc.FREQUENCIES for b in (1,128)}
    if plan.get('candidate_sha256') != candidate_sha or len(plan['points']) != 12 or {(p['freq_mhz'],p['batch']) for p in plan['points']} != expected:
        raise ValueError('local power panel must be exactly B1/B128 at six frequencies')
    for p in plan['points']:
        if p['purpose'] != 'independent_power_holdout' or p['repeats'] != 3 or p['settle_s'] < 2 or p['measure_s'] < 5:
            raise ValueError('local power sampling gates changed')
        if p['batch']*(p['context_tokens']+p['max_tokens']) > .9*model.kv_capacity_tokens or p['context_tokens']+p['max_tokens'] > 8192:
            raise ValueError('local power reservation exceeds memory')
        for c in (p['context_tokens'],p['context_tokens']+p['max_tokens']-1):
            if not model.decode_supported(p['batch'],c,p['freq_mhz']) or not model.decode_power_supported(p['batch'],c,p['freq_mhz']):
                raise ValueError('local power reservation outside frozen support')


def bind_original_timing(original, prior, base, candidate_sha):
    completion = json.loads((original/'completion.json').read_text())
    if (completion.get('complete') is not True or completion.get('independent_holdout') is not True or
        completion.get('candidate_sha256') != candidate_sha or completion.get('raw_sha256') != pc.digest(original/'raw.json')):
        raise ValueError('original timing is not complete and checksum-bound')
    manifest = json.loads((original/'frozen-fit.json').read_text())
    if manifest.get('candidate_sha256') != candidate_sha:
        raise ValueError('original timing manifest belongs to another candidate')
    sources = None
    raw = json.loads((original/'raw.json').read_text())
    files = [original/'completion.json',original/'raw.json',original/'frozen-fit.json']
    if completion.get('combined_holdout_sha256'):
        if pc.digest(original/'combined-holdout.json') != completion['combined_holdout_sha256']:
            raise ValueError('combined original timing checksum mismatch')
        raw = json.loads((original/'combined-holdout.json').read_text()); files.append(original/'combined-holdout.json')
        sources = {}
        for source,binding in raw['evidence_sources'].items():
            if binding['raw_sha256'] != pc.digest(prior/'raw.json'):
                raise ValueError('original timing inherited source mismatch')
            sources[source] = prior
    audit = pc.timing_component(evaluate_holdout(raw,base,original,expected_plan=manifest['plan'],evidence_roots=sources))
    return audit, files


def prepare(*, candidate, plan, base_candidate, training_raw, prior_holdout, out, original_holdout=None):
    candidate,plan,base_candidate,training_raw,prior_holdout,out = map(Path,(candidate,plan,base_candidate,training_raw,prior_holdout,out))
    if out.exists():
        raise FileExistsError('prepare into a new immutable directory')
    model,base = PerfModel.load(candidate),PerfModel.load(base_candidate)
    training = json.loads(training_raw.read_text())
    checked = verify_training(training,training_raw.parent,model,base)
    planned = json.loads(plan.read_text()); validate_plan(planned,model,pc.digest(candidate))
    if planned['training_raw_sha256'] != pc.digest(training_raw):
        raise ValueError('local plan bound to different training')
    prior = json.loads((prior_holdout/'raw.json').read_text())
    if any(prior.get(k) != training.get(k) for k in ('system','model_id','model_hash','tokenizer_hash','tp','pp')):
        raise ValueError('prior evidence belongs to another model')
    if prior.get('holdout_candidate_sha256') != pc.digest(base_candidate):
        raise ValueError('prior evidence bound to another timing candidate')
    if prior.get('identity_sha256') != sha256_value({k:v for k,v in prior.items() if k != 'identity_sha256'}):
        raise ValueError('prior evidence identity checksum differs')
    if json.loads((prior_holdout/'frozen-fit.json').read_text()).get('candidate_sha256') != pc.digest(base_candidate):
        raise ValueError('prior manifest bound to another timing candidate')
    _checkpoint_points(prior,prior_holdout)
    inputs = dict(base_candidate=base_candidate,training_raw=training_raw,prior_raw=prior_holdout/'raw.json',
                  prior_manifest=prior_holdout/'frozen-fit.json',candidate_proposal=candidate,plan_proposal=plan)
    timing = dict(passed=None,status='not_bound',scope='full_original_timing_pending',formal_eligible=False)
    if original_holdout is not None:
        timing,files = bind_original_timing(Path(original_holdout),prior_holdout,base,pc.digest(base_candidate))
        for index,file in enumerate(files):inputs['original_timing_'+str(index)] = file
    out.mkdir(parents=True)
    (out/'candidate.json').write_bytes(candidate.read_bytes());(out/'power-plan.json').write_bytes(plan.read_bytes())
    pc.write_immutable(out/'retained-timing-audit.json',timing)
    manifest = dict(schema=1,scope=SCOPE,system='pdblend',model_id=training['model_id'],tp=4,pp=1,
        model_hash=training['model_hash'],tokenizer_hash=training['tokenizer_hash'],candidate_sha256=pc.digest(out/'candidate.json'),
        plan_sha256=pc.digest(out/'power-plan.json'),timing_audit_sha256=pc.digest(out/'retained-timing-audit.json'),
        inputs={k:dict(path=str(v.resolve()),sha256=pc.digest(v)) for k,v in inputs.items()},
        training_windows_checked=checked,implementation_sha256=source_hashes(),original_timing_environment=prior['environment'],
        original_failed_rows_preserved=True,timing_fields_unchanged=True,holdout_used_for_fit_or_selection=False,
        full_profile_qualified=False,formal_eligible=False,energy_comparable=False,
        concurrency_modes=['isolated-host','coordinated-wave'],required_remaining=['fresh_local_power_panel','full_original_timing_binding','unvalidated_other_power_shapes'])
    if original_holdout is not None:
        from pdblend.profile.calibration.local_mixed_repair import plan_repair
        manifest['original_holdout']=str(Path(original_holdout).resolve())
        manifest['required_remaining'].remove('full_original_timing_binding')
        manifest['mixed_repair']=plan_repair(manifest)
        if manifest['mixed_repair'] is not None:
            pc.write_immutable(out/'mixed-repair-plan.json',manifest['mixed_repair'])
            manifest['mixed_repair_plan_sha256']=pc.digest(out/'mixed-repair-plan.json')
            manifest['required_remaining'].append('four_fresh_mixed_points')
    pc.write_immutable(out/'manifest.json',manifest)
    load_package(out)
    return manifest


def load_package(package):
    package = Path(package); m = json.loads((package/'manifest.json').read_text())
    if m.get('scope') != SCOPE or tuple(m.get(k) for k in ('system','model_id','tp','pp')) != IDENTITY or m['implementation_sha256'] != source_hashes():
        raise ValueError('local power package identity/source changed')
    for name,key in [('candidate.json','candidate_sha256'),('power-plan.json','plan_sha256'),('retained-timing-audit.json','timing_audit_sha256')]:
        if pc.digest(package/name) != m[key]:raise ValueError('local power package checksum changed')
    for binding in m['inputs'].values():
        if pc.digest(binding['path']) != binding['sha256']:raise ValueError('local power immutable input changed')
    if m.get('mixed_repair') is not None:
        from pdblend.profile.calibration.local_mixed_repair import plan_repair
        if (pc.digest(package/'mixed-repair-plan.json')!=m['mixed_repair_plan_sha256'] or
            json.loads((package/'mixed-repair-plan.json').read_text())!=m['mixed_repair'] or
            plan_repair(m)!=m['mixed_repair']):
            raise ValueError('local mixed repair plan/reference changed')
    model = PerfModel.load(package/'candidate.json');base = PerfModel.load(m['inputs']['base_candidate']['path'])
    if pc.timing_fields(model) != pc.timing_fields(base):raise ValueError('local power changed original timing')
    plan = json.loads((package/'power-plan.json').read_text());validate_plan(plan,model,m['candidate_sha256'])
    if plan['training_raw_sha256'] != m['inputs']['training_raw']['sha256']:raise ValueError('local plan training identity changed')
    return m,plan,model


def isolated_inventory(document, *, uuids, started_s, now_s, manifest_sha256):
    if (document.get('allocated_gpu_uuids') != uuids or len(set(uuids)) != 4 or
        document.get('lease_manifest_sha256') != manifest_sha256 or not document.get('lease_id') or
        not isinstance(document.get('last_updated_s'),(int,float)) or not 0 <= now_s-document['last_updated_s'] <= 15):
        raise ValueError('isolated group requires a fresh lease/UUID/manifest inventory')
    inventory = document.get('inventory',[])
    if len(inventory) != 8 or len({g['uuid'] for g in inventory}) != 8 or not set(uuids) <= {g['uuid'] for g in inventory}:
        raise ValueError('isolated group needs all eight physical GPU inventory entries')
    if document.get('peer_jobs') or any(g.get('pids') for g in inventory if g['uuid'] not in uuids):
        raise ValueError('isolated group observed external GPU activity; use an explicit coordinated wave')
    if any(s.get('peers') for s in document.get('peer_snapshots',[]) if s.get('at_s',0) >= started_s):
        raise ValueError('peer activity changed during isolated measurement')


class IsolatedGroup:
    def __init__(self, profiler):
        self.profiler=profiler;self.started_s=time.time();self.snapshots=[];self.failure=None
        self.root=Path(profiler.out_dir);self.receipt=None
    def check(self):
        path=self.root/'concurrency-environment.json';payload=path.read_bytes();doc=json.loads(payload)
        isolated_inventory(doc,uuids=self.profiler.raw['environment']['gpu_uuids'],started_s=self.started_s,
                           now_s=time.time(),manifest_sha256=pc.digest(self.root/'manifest.json'))
        sha=hashlib.sha256(payload).hexdigest();p=self.root/'samples'/('isolated-environment-'+sha+'.json')
        p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes(payload)
        entry=dict(samples_file=str(p.relative_to(self.root)),samples_sha256=sha)
        if not self.snapshots or self.snapshots[-1]!=entry:self.snapshots.append(entry)
    async def qualify_external(self, profiler, fleet):
        from pdblend.engine.client import EngineClient
        self.check();spec=profiler.specs[0];profiler._lock(2100,spec.gpus)
        async with EngineClient(spec.instance_id,spec.base_url) as client:
            row=await profiler._decode_batch(client,spec.gpus,8,1024,64,'local-power-isolated-check')
        row['freq_mhz']=2100;_checkpoint_points(dict(prefill=[],decode=[row]),self.root);self.check()
        if any(round(r.get('mean_freq_mhz',0)) != 2100 for r in row['repeats']):
            raise ValueError('isolated probe frequency differs from requested clock')
        self.receipt=dict(complete=True,passed=True,measured_mode='isolated_group',parallel=False,cross_job=False,
            formal_eligible=False,energy_comparable=False,probe=row,qualification_frequency_mhz=2100,
            scope='single_group_probe_with_other_GPUs_idle',inventory_resolution_s=5)
        self.publish(False)
    def publish(self, finished):
        if self.receipt is None:return
        self.receipt.update(measurement_complete=finished,inventory_snapshots=list(self.snapshots))
        payload=json.dumps(self.receipt,sort_keys=True,indent=2)+'\n'
        sha=hashlib.sha256(payload.encode()).hexdigest()
        path=self.root/'samples'/('local-power-isolation-'+sha+'.json')
        if not path.exists():path.write_text(payload)
        self.profiler.raw['local_power_isolation']=dict(complete=True,passed=True,measurement_complete=finished,
            measured_mode='isolated_group',parallel=False,samples_file=str(path.relative_to(self.root)),samples_sha256=pc.digest(path))
        self.profiler._checkpoint()
    @asynccontextmanager
    async def measurement(self):
        parent=asyncio.current_task();self.check()
        async def monitor():
            try:
                while True:await asyncio.sleep(1);self.check()
            except asyncio.CancelledError:raise
            except Exception as exc:self.failure=exc;parent.cancel()
        watcher=asyncio.create_task(monitor())
        try:
            yield
            self.check();self.publish(True)
        finally:
            watcher.cancel();await asyncio.gather(watcher,return_exceptions=True)
            if self.failure is not None:raise RuntimeError('isolated group interrupted by changed inventory') from self.failure
    def write(self, phase, value):atomic_json(self.root/('local-power-'+phase+'.json'),value)


class Adapter:
    scope=SCOPE
    load_package=staticmethod(load_package)
    def __init__(self, concurrency_mode):self.concurrency_mode=concurrency_mode
    def needs_followup(self,package,out):
        from pdblend.profile.calibration.local_mixed_repair import needs_collection
        manifest,_,_=load_package(package)
        return needs_collection(package=package,out=Path(out)/'mixed-repair',manifest=manifest)
    async def after_samples(self,*,profiler,client,gpus,package,out):
        from pdblend.profile.calibration.local_mixed_repair import collect_existing
        manifest,_,_=load_package(package)
        return await collect_existing(profiler=profiler,client=client,gpus=gpus,package=package,out=out,manifest=manifest)
    def make_wave(self,profiler):
        if self.concurrency_mode=='isolated-host':return IsolatedGroup(profiler)
        wave=ProfileWave.from_environment()
        if (wave is None or len(wave.members)<2 or not wave.coordinator or not wave.cohort_id or
            wave.spec.get('keep_peers_resident_until_all_done') is not True or
            wave.spec.get('synchronize_parallel_windows') is not True):
            raise ValueError('coordinated local power requires a new synchronized multi-member resident wave')
        return wave
    def audit(self,package,raw,out,manifest,plan,model,binding):
        load_package(package)
        key='local_power_isolation' if self.concurrency_mode=='isolated-host' else 'external_interference'
        receipt=raw.get(key,{})
        evidence=read_bound(out,receipt.get('samples_file',''),receipt.get('samples_sha256'))
        if self.concurrency_mode=='isolated-host':
            qualified=evidence.get('passed') is True and evidence.get('measurement_complete') is True and evidence.get('parallel') is False
        else:
            qualified=evidence.get('complete') is True and evidence.get('cross_job') is True and (evidence.get('passed') is True or evidence.get('fallback')=='serial_cohort')
        power=pc.audit_power(raw,out,plan['points'],model,binding,windows_per_frequency=6)
        if not qualified:power['passed']=False;power['failures'].append(dict(metric='concurrency_qualification'))
        timing=json.loads((Path(package)/'retained-timing-audit.json').read_text())
        atomic_json(out/'power-only-audit.json',power);atomic_json(out/'reused-timing-audit.json',timing)
        result=dict(power=power,timing=timing,calibration_components_passed=power['passed'] and timing['passed'] is True,
            concurrency_qualified=qualified,validation_scope=SCOPE,full_profile_qualified=False,formal_eligible=False,
            energy_comparable=False,original_failed_rows_preserved=True,timing_fields_unchanged=True,
            unvalidated_other_power_shapes=True,binding=binding,
            component_receipts=dict(power=pc.digest(out/'power-only-audit.json'),timing=pc.digest(out/'reused-timing-audit.json')))
        if manifest.get('mixed_repair') is not None:
            from pdblend.profile.calibration.local_mixed_repair import audit_repair
            repaired=audit_repair(manifest=manifest,package=package,out=out)
            result.update(repaired_timing=repaired,
                calibration_components_passed=power['passed'] and repaired['passed'] is True,
                effective_timing_receipt='repaired-timing-audit',original_timing_passed=timing['passed'])
            result['component_receipts']['repaired_timing']=pc.digest(out/'mixed-repair/repaired-timing-audit.json')
        atomic_json(out/'composite-audit.json',result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare')
    for name in ('candidate','plan','base-candidate','training-raw','prior-holdout','out'):prep.add_argument('--'+name,type=Path,required=True)
    prep.add_argument('--original-holdout',type=Path)
    run=sub.add_parser('run');run.add_argument('--package',type=Path,required=True);run.add_argument('--model',required=True)
    run.add_argument('--gpus',type=int,nargs='+',required=True);run.add_argument('--base-port',type=int,required=True)
    run.add_argument('--out',type=Path,required=True);run.add_argument('--concurrency-mode',choices=['isolated-host','coordinated-wave'],default='isolated-host')
    a=p.parse_args()
    if a.command=='prepare':result=prepare(**{k:v for k,v in vars(a).items() if k!='command'})
    else:
        try:
            result=pc.run(package=a.package,model_path=a.model,gpus=a.gpus,base_port=a.base_port,out=a.out,panel_adapter=Adapter(a.concurrency_mode))
        except BaseException as exc:
            result=dict(status='failed',complete=False,error=f'{type(exc).__name__}: {exc}',formal_eligible=False,energy_comparable=False)
            atomic_json(a.out/'completion.json',result)
    print(json.dumps(result,indent=2),flush=True)
    if a.command=='run':raise SystemExit(0 if result.get('complete') else 1)


if __name__=='__main__':main()
