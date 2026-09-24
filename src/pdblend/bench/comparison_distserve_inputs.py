"""Own-profile, calibration-only qualification for a native DistServe group.

This admits a measured symmetric TP/PP1 search domain. It does not declare
unmeasured TP/PP layouts optimal, nor promote old profile qualification flags.
"""
from __future__ import annotations
from pathlib import Path

from .comparison_acceptance import _bound, _need, _equal
from .comparison_native_acceptance import native_topology
from .resident_session import digest, file_sha
from pdblend_baselines.distserve.deployment import search_deployment
from pdblend_baselines.distserve.stage_surface import StageSurface

PROTECTED=('pdblend_baselines/distserve/policy.py','pdblend_baselines/distserve/request_runtime.py',
           'pdblend_baselines/distserve/runtime.py','pdblend_baselines/distserve/planning.py',
           'pdblend_baselines/distserve/deployment.py','pdblend_baselines/distserve/stage_surface.py',
           'pdblend_runtime/native_v1.py','pdblend_runtime/serve.py','pdblend_runtime/kv.py')


def validate_distserve_inputs(point,identity,*,source_manifest,replay_search=True):
    failures={};checked=[];values={}
    def gate(name,fn):
        try:value=fn()
        except (ValueError,TypeError,KeyError,OSError,IndexError,AttributeError) as exc:
            failures[name]=str(exc);return None
        checked.append(name);return value
    inputs=point.get('inputs',{})
    instances=gate('native.topology',lambda:native_topology(point,identity))
    for name in ('system_config','offline_choice','calibration','profile_source_manifest'):
        values[name]=gate('binding.'+name,lambda name=name:_bound(inputs.get(name)))
    trace=gate('binding.trace',lambda:_bound(point['trace']))
    source=gate('binding.execution_source',lambda:_bound(source_manifest))
    config,choice=values['system_config'],values['offline_choice']
    def basic():
        _need(point['system']=='distserve' and point['seed']==701 and point['duration_s']==150,
              'native DistServe 150s/seed701 point required')
        _need(inputs.get('trace')==point['trace'] and trace.get('selection_split')=='evaluation',
              'same immutable evaluation trace required')
        for k in ('model_id','dataset','rate_rps','slo','seed','duration_s'):
            _need(trace.get(k)==point.get(k), 'trace identity differs: '+k)
        _need(config.get('system')=='distserve' and config.get('model_id')==point['model_id']
              and config.get('max_batch_size')==32 and config.get('request_timeout_s')==240.,
              'independent native baseline configuration differs')
        _need(choice.get('system')=='distserve' and choice.get('model_id')==point['model_id']
              and choice.get('status')=='ready_for_native_execution'
              and choice.get('selection_split')=='calibration' and choice.get('evaluation_used_for_selection') is False
              and choice.get('gpu_budget')==8 and choice.get('rate_rps')==point['rate_rps']
              and choice.get('slo')==point['slo'] and choice.get('frequency_mhz')==2520,
              'calibration-only eight-GPU deployment selection differs')
        selected=choice['selected'];tp=selected['tp'];n=selected['replicas']
        _need(selected['config']==[1,tp,1,tp,1] and selected['pp']==1
              and selected['total_gpu_count']==2*tp*n<=8 and len(instances)==2*n,
              'unsupported selected deployment topology')
        expected={f'dist-{i}-{role}' for i in range(n) for role in ('P','D')}
        _need(set(instances)==expected and all(s['tp']==tp and s['pp']==1 for s in instances.values())
              and choice['deployment']['pairs']==[dict(prefill=f'dist-{i}-P',decode=f'dist-{i}-D') for i in range(n)],
              'native fleet differs from selected pairs')
        _need(choice['calibration']['sha256']==inputs['calibration']['sha256']
              and Path(choice['calibration']['path']).resolve()==Path(inputs['calibration']['path']).resolve()
              and choice['calibration']['split']=='calibration' and values['calibration'].get('calibration'),
              'offline search did not bind its independent calibration corpus')
        return selected
    selection=gate('deployment.calibration_only',basic)
    def profiles():
        refs=inputs.get('profiles',[])
        _need(refs and {(r['path'],r['sha256']) for r in refs}==
              {(r['path'],r['sha256']) for r in choice['profiles']}, 'selected own profiles differ')
        tps=[]
        for ref in refs:
            artifact=_bound(ref);surface=StageSurface.load(ref['path'],frequency=2520)
            meta=surface.identity
            _need(meta.get('system')=='distserve' and meta.get('model_id')==point['model_id']
                  and meta.get('pp')==1 and meta.get('engine_revision')=='vllm-0.10.1.1'
                  and all(meta.get(k)==identity[k] for k in ('model_hash','tokenizer_hash','image_digest')),
                  'independent profile execution identity differs')
            _need(meta.get('source_revision')==values['profile_source_manifest']['source_sha256'],
                  'stage profile source binding differs')
            tps.append(meta['tp'])
        _need(len(tps)==len(set(tps)) and selection['tp'] in tps,'selected TP lacks an own stage profile')
        return tps
    tps=gate('profile.raw_holdout_audit',profiles)
    def continuity():
        old=values['profile_source_manifest']
        for manifest in (source,old):
            _need(digest(manifest['files'])==manifest['source_sha256'], 'source inventory hash differs')
        for name in PROTECTED:
            _need(source['files'].get(name) and source['files'][name]==old['files'].get(name),
                  'independent baseline/native mechanism changed since profiling: '+name)
            _need(file_sha(Path(source_manifest['path']).parent/name)==source['files'][name],
                  'executed baseline source bytes differ')
        return dict(profile_source_sha256=old['source_sha256'],execution_source_sha256=source['source_sha256'])
    inherited=gate('source.continuity',continuity)
    def search():
        params=config.get('search_parameters',{})
        _need(set(params)=={'max_per_gpu_rate','epsilon','sample_size'}, 'explicit offline search bounds required')
        rebuilt=search_deployment([r['path'] for r in inputs['profiles']],inputs['calibration']['path'],
            model=point['model_id'],rate_rps=point['rate_rps'],ttft_s=point['slo']['ttft_s'],
            tpot_s=point['slo']['tpot_s'],gpu_budget=8,frequency=2520,**params)
        _need(_equal(rebuilt,choice), 'offline choice differs from independent simulator replay')
        return dict(replayed=True,scope='measured_symmetric_tp_pp1_only',qualified_tps=tps)
    search_receipt=gate('deployment.simulator_replay',search) if replay_search else None
    return dict(preflight_ready=not failures,formal_eligible=False,checked_gates=checked,gate_failures=failures,
        config=config,choice=choice,source_continuity=inherited,search=search_receipt,
        observation_scope='native_symmetric_tp_pp1_single_observation',complete_reproduction=False)
