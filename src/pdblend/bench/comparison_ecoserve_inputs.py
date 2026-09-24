"""Scoped EcoServe author-lookup inheritance; never edits legacy qualification.

The pinned evidence catalog is the audited 7B/14B native result inventory.
Inputs bind those exact artifacts at any readable path. Whole-profile flags
remain false: this validator establishes the original lookup-table protocol
and inherited unchanged mechanisms for a fresh single-observation audit.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from .comparison_acceptance import _bound, _need, _equal, _finite
from .resident_session import digest, engine_signature
from pdblend_baselines.native_profile import audit as audit_profile

PROFILE_SOURCE = '1cb717dbd5a16be2389a3fc810eb4bfa9edabdfcfef0693c91e3d096e48d2825'
MECHANISM_SOURCE = '05cbf5dbffd0cde18436d1b876e2e94a6cf5cc448c4fb788f9d2d756bdee2c90'
PROTECTED = ('pdblend_baselines/ecoserve/policy.py', 'pdblend_baselines/ecoserve/controller.py',
    'pdblend_baselines/ecoserve/runtime.py', 'pdblend_runtime/native_v1.py',
    'pdblend_runtime/serve.py', 'pdblend_runtime/kv.py')
CERTIFIED = {'Qwen2.5-7B-Instruct': {'eco_profile_csv': '11c3e5d8682c8244af1c2bc03c467b765a10c58e2925518f48eb50e1b4fb719d', 'eco_profile_manifest': 'd162bf8fd733a72b9cc26ca6da5defd293c481f19053ed49236fac8251bd64cf', 'eco_mechanism_completion': '56e8f6763257f59b28e76f3209609a6e8f2ba197320cbca29cbd222d3839610b', 'eco_automatic_completion': '25bd288f385f6c4c3340d79dca220acd535706860e1b3621666b4d8282436003', 'eco_mechanism_review': '1b0c6e18eb445db56ed712c7dd8f5849600ef2588b0666b8ff5a66f1077a4bb2'}, 'Qwen2.5-14B-Instruct': {'eco_profile_csv': '8a8f44eff4adaf18ccb3d1f5d564f680d6f13e6630a37c08c1e43c6855f28682', 'eco_profile_manifest': '6afdd1b77040c8e956d8873efd5e8236d826dfcf7cde436b4be7402756b8c555', 'eco_mechanism_completion': '1357303c628b984fca9682902f0fd7712a057215d369626518234fbf17940682', 'eco_automatic_completion': '652e1f1ff4eb365b4ca2f2991e3f7f7226a09d35da9f333d4fb23294265178e4', 'eco_mechanism_review': '1b0c6e18eb445db56ed712c7dd8f5849600ef2588b0666b8ff5a66f1077a4bb2'}}
REVIEWED_WRAPPERS = {'pdblend/engine/launcher.py': '72150dbeb45f9f4693108d7ef1dfab9df323fc8241ba03e9398a23f9fbfa36c0', 'pdblend_baselines/ecoserve/run_native.py': 'a788cff7ecb019af211c88d9821bc3712126bc5850f81aa4c06f52358e0e830f', 'pdblend_baselines/ecoserve/mechanism_four.py': '6079ef14d97b2e68ee7ccdef09bdb5da9894839a9036fda05577ee4ea9c9d1a8'}
LIFECYCLE_MODE = 'cohort_cancel_and_serial_close/v1'
LIFECYCLE_REVIEW_SHA256 = 'f108a19075dc1e7b074e7c33a8a3d71685e97d1004eefc95a20559ea98ec341a'
# Explicitly reviewed wrapper extension. Populate only from the independently
# tested source bytes; inherited controller/policy/runtime remain PROTECTED.
REVIEWED_LIFECYCLE_WRAPPERS = {
    'pdblend_baselines/ecoserve/run_native.py': '4da45daa1b00f4cfc2de0966a6e2c8ce9c28abbfa0d46b04d86323d180ebf3d4',
    'pdblend_baselines/ecoserve/comparison_lifecycle.py': '13954b0897b07fdf403501e6fbcca0f397c664b125a584bcfa9c98cf0e20a489',
    'pdblend/bench/comparison_ecoserve_lifecycle.py': '88d0cc1d012674e950ade331ce1b3140d513e1dddecae9e129ef2f77429b239e',
}
LIFECYCLE_CHECKS = ('no_hidden_cohort_cancellation', 'exact_http_cancellation_causes',
    'read_only_cancellation_exception', 'serial_close_finishes_active_membership',
    'waiting_membership_cancelled_before_mutation', 'close_timeout_is_failure',
    'old_mode_unchanged', 'core_source_unchanged')


def _bytes(ref):
    _need(isinstance(ref,dict) and isinstance(ref.get('path'),str), 'file binding absent')
    data = Path(ref['path']).read_bytes()
    _need(hashlib.sha256(data).hexdigest() == ref.get('sha256'), 'file checksum differs')
    return data


def _source(ref, expected=None):
    value = _bound(ref); files = value.get('files', {})
    _need(files and value.get('source_sha256') == digest(files), 'source inventory checksum differs')
    if expected is not None:
        _need(value['source_sha256'] == expected, 'source is outside the reviewed inheritance range')
    return value


def eco_topology(point, identity):
    _need(point.get('system') == 'ecoserve' and point.get('model_id') in CERTIFIED,
          'only reviewed 7B/14B EcoServe evidence is available')
    engine_signature(identity)
    fleet = identity.get('fleet_gpu_uuids', [])
    _need(len(fleet) == len(set(fleet)) == 8 and all(str(u).startswith('GPU-') for u in fleet),
          'eight distinct physical GPU UUIDs required')
    _need(identity.get('dtype') == 'bfloat16' and identity.get('entrypoint') == 'pdblend_runtime.serve'
          and identity.get('worker_extension') == 'native_v1', 'native runtime/dtype differs')
    rows = identity['instances']
    _need(len(rows) == 8 and all(r['tp'] == r['pp'] == 1 for r in rows)
          and [u for r in rows for u in r['gpu_uuids']] == fleet, 'EcoServe requires complete fixed TP1 eight-card fleet')
    for row in rows:
        o = row['launch_options']
        _need(o.get('kv_connector') == 'P2pNcclConnector' and o.get('max_num_seqs') == 32
              and o.get('max_model_len') == o.get('max_num_batched_tokens') == 8192,
              'EcoServe native KV/capacity launch options differ')
    if point.get('engine_identity') is not None:
        _need(_equal(point['engine_identity'],identity),'point engine identity differs')
    return {r['instance_id']:r for r in rows}


def _config(point, identity, config):
    instances = eco_topology(point,identity)
    _need(config.get('system') == 'ecoserve' and config.get('model_id') == point['model_id'],
          'EcoServe config owner/model differs')
    expected = [dict(id=r['instance_id'],gpus=[identity['fleet_gpu_uuids'].index(u) for u in r['gpu_uuids']],
                     tp=r['tp'],pp=r['pp']) for r in identity['instances']]
    _need(config.get('instances') == expected, 'config instance/GPU inventory differs from launch')
    for key,value in dict(eco_macro_lower=2,eco_macro_upper=3,eco_scale_period_s=5.,
                         eco_history_window_s=60.,eco_state_poll_s=.05,eco_active_frequency_mhz=2520).items():
        _need(config.get(key) == value, 'frozen EcoServe parameter differs: '+key)
    count = config.get('eco_initial_instances')
    _need(type(count) is int and 2 <= count <= len(instances), 'explicit initial EcoServe member count required')
    _need('eco_park_frequency_mhz' not in config and config.get('park_idle',True) is True,
          'EcoServe must use original common-clock parking')
    _need(config.get('slo_ttft_s') == point['slo']['ttft_s']
          and config.get('slo_tpot_s') == point['slo']['tpot_s'], 'config SLO differs from point')
    _need(_finite(config.get('request_timeout_s')) and config['request_timeout_s'] > 0
          and _finite(config.get('eco_drain_timeout_s')) and config['eco_drain_timeout_s'] > 0,
          'explicit positive request/drain limits required')
    return instances


def validate_ecoserve_inputs(point, engine_identity, inputs=None, *, source_manifest=None):
    """Read only, CPU-only. Ready means startup may proceed, never formal energy.

    Required bindings: trace, system_config; the five CERTIFIED evidence names;
    eco_profile_source_manifest, eco_mechanism_source_manifest. source_manifest
    is the current frozen source binding (or inputs['source_manifest']).
    """
    inputs = point.get('inputs', {}) if inputs is None else inputs
    failures, checks, values = {}, [], {}
    def gate(name,fn):
        try: value=fn()
        except (ValueError,TypeError,KeyError,OSError,IndexError,AttributeError) as exc:
            failures[name]=str(exc);return None
        checks.append(name);return value
    config=gate('config',lambda:_bound(inputs.get('system_config')))
    if config is not None:
        gate('config.fleet_policy',lambda:_config(point,engine_identity,config))
    trace=gate('trace',lambda:_bound(inputs.get('trace',point.get('trace'))))
    def trace_check():
        _need(inputs.get('trace',point.get('trace')) == point.get('trace'), 'input trace differs from frozen point')
        _need(point.get('seed') == 701 and point.get('duration_s') == 150 and trace.get('seed') == 701
              and trace.get('duration_s') == 150 and trace.get('selection_split') == 'evaluation', '150s evaluation seed701 required')
        for key in ('model_id','dataset','rate_rps','slo'):
            _need(trace.get(key) == point.get(key) and point.get(key) is not None, 'trace identity differs: '+key)
        _need(trace.get('requests'), 'nonempty frozen requests required')
        previous=-1.
        for index,row in enumerate(trace['requests']):
            prompt,n,at=row.get('prompt'),row.get('max_tokens'),row.get('arrival_s')
            _need(isinstance(prompt,list) and 1 <= len(prompt) <= 7168
                  and all(type(t) is int and t >= 0 for t in prompt) and type(n) is int and 2 <= n <= 512
                  and len(prompt)+n <= 8192 and _finite(at) and 0 <= at < 150 and at >= previous
                  and row.get('idx',index) == index, 'trace request outside author lookup/native context domain')
            previous=at
    gate('trace.domain',trace_check)
    def certified():
        expected=CERTIFIED[point['model_id']]
        for key,checksum in expected.items():
            _need(inputs.get(key,{}).get('sha256') == checksum, 'unreviewed inherited artifact: '+key)
            if key == 'eco_profile_csv': _bytes(inputs[key])
            else: values[key]=_bound(inputs[key])
    gate('inheritance.bound_catalog',certified)
    def profile():
        csv=inputs['eco_profile_csv'];manifest=values['eco_profile_manifest'];meta=manifest['metadata']
        _need(Path(inputs['eco_profile_manifest']['path']).resolve() == Path(csv['path']+'.manifest.json').resolve(),
              'author CSV sidecar path differs')
        proof=audit_profile(csv['path'])
        _need(proof['valid'] and proof['system']=='ecoserve' and manifest['model'] == point['model_id']
              and meta['tp']==meta['pp']==1 and meta['frequency_mhz']==2520
              and all(meta.get(k)==engine_identity[k] for k in ('model_hash','tokenizer_hash','image_digest')),
              'author native-forward table identity/protocol differs')
        _need(meta['source_revision']==PROFILE_SOURCE,'profile source outside reviewed range')
        _need(config['eco_prefill_csv']==csv['path'] and config['eco_profile_sha256']==csv['sha256'],
              'runner does not consume the exact inherited CSV')
        return proof
    profile_proof=gate('profile.author_lookup_protocol',profile)
    def mechanisms():
        completion=values['eco_mechanism_completion'];automatic=values['eco_automatic_completion']
        required=('automatic_split','automatic_merge','split_live_kv_ack','merge_live_kv_ack',
                  'automatic_park','policy_rotation','actual_hold','held_output_flush',
                  'continuous_complete_output','no_manual_membership','all_native_drains_acknowledged',
                  'controller_healthy','no_execution_or_cleanup_error')
        _need(completion.get('status')=='passed' and automatic.get('status')=='passed'
              and all(completion.get('checks',{}).get(k) is True and automatic.get('checks',{}).get(k) is True for k in required),
              'inherited automatic mechanisms lack complete raw checks')
        _need(completion.get('automatic_receipt_sha256')==inputs['eco_automatic_completion']['sha256'],
              'automatic completion checksum differs from outer receipt')
        reports=values['eco_mechanism_review'].get('reports',[])
        selected=[r for r in reports if r.get('sha256')==inputs['eco_automatic_completion']['sha256']]
        _need(len(selected)==1 and selected[0].get('status')=='mechanism_passed'
              and selected[0].get('checks') and all(selected[0]['checks'].values()),'independent mechanism review missing')
        caps=completion.get('capabilities',{})
        caps=list(caps.values()) if isinstance(caps,dict) else caps
        _need(caps and all(c.get('model_id')==point['model_id'] and c.get('tp')==1 and c.get('pp')==1
             and c.get('source_revision')==MECHANISM_SOURCE
             and all(c.get(k)==engine_identity[k] for k in ('model_hash','tokenizer_hash','image_digest')) for c in caps),
             'inherited native mechanism model/runtime identity differs')
    gate('mechanisms.inherited_original_periods',mechanisms)
    def source_check():
        profile_source=_source(inputs['eco_profile_source_manifest'],PROFILE_SOURCE)
        old=_source(inputs['eco_mechanism_source_manifest'],MECHANISM_SOURCE)
        current_ref=source_manifest or inputs.get('source_manifest')
        current=_source(current_ref);files=current['files'];base=Path(current_ref['path']).parent
        for name in PROTECTED:
            _need(files.get(name)==old['files'].get(name) and files.get(name), 'changed inherited mechanism code: '+name)
        wrappers=dict(REVIEWED_WRAPPERS)
        lifecycle_requested=any(value is not None for value in (
            point.get('eco_comparison_lifecycle'), engine_identity.get('eco_comparison_lifecycle'),
            config.get('eco_comparison_lifecycle')))
        if lifecycle_requested:
            _need(point.get('eco_comparison_lifecycle') == engine_identity.get('eco_comparison_lifecycle')
                  == config.get('eco_comparison_lifecycle') == LIFECYCLE_MODE,
                  'comparison lifecycle selection differs across point/config/identity')
            _need(inputs.get('eco_lifecycle_review',{}).get('sha256') == LIFECYCLE_REVIEW_SHA256,
                  'unreviewed lifecycle extension evidence')
            review=_bound(inputs.get('eco_lifecycle_review'))
            _need(review.get('schema') == 'ecoserve-comparison-lifecycle-review/v1'
                  and review.get('mode') == LIFECYCLE_MODE
                  and review.get('scope') == 'comparison_wrapper_lifecycle_cpu_review'
                  and review.get('formal_eligible') is False and review.get('hardware_qualified') is False
                  and review.get('base_runner_sha256') == REVIEWED_WRAPPERS['pdblend_baselines/ecoserve/run_native.py']
                  and review.get('protected_mechanism_source_sha256') == MECHANISM_SOURCE
                  and all(review.get('checks',{}).get(k) is True for k in LIFECYCLE_CHECKS)
                  and REVIEWED_LIFECYCLE_WRAPPERS
                  and review.get('source_files') == REVIEWED_LIFECYCLE_WRAPPERS,
                  'comparison lifecycle lacks its exact scoped source review')
            wrappers.update(REVIEWED_LIFECYCLE_WRAPPERS)
        for name,checksum in wrappers.items():
            _need(files.get(name)==checksum,'unreviewed launch/journal wrapper change: '+name)
        for name in (*PROTECTED,*wrappers):
            path=base/name
            _need(hashlib.sha256(path.read_bytes()).hexdigest()==files[name], 'executing source bytes differ: '+name)
        # Profile's full-forward collector is certified byte-for-byte by its
        # source inventory. Historical cancel/retention fixes do not alter the
        # protected native worker forward/CUDA measurement implementation.
        _need(profile_source['source_sha256']==values['eco_profile_manifest']['metadata']['source_revision'],
              'profile source inventory is not bound')
        runtime={k:v for k,v in files.items() if k.startswith(('pdblend_runtime/','pdblend/engine/'))}
        _need(digest(runtime)==engine_identity['runtime_source_sha256'],'current runtime fingerprint differs')
        result=dict(source_sha256=current['source_sha256'],mechanism_source_sha256=MECHANISM_SOURCE,
                    profile_source_sha256=PROFILE_SOURCE,protected_files=list(PROTECTED),reviewed_wrappers=wrappers)
        if lifecycle_requested:result['comparison_lifecycle']=LIFECYCLE_MODE
        return result
    continuity=gate('source.reviewed_compatibility',source_check)
    return dict(schema='ecoserve-author-lookup-inputs-v1',scope='author_lookup_protocol',
        preflight_ready=not failures,formal_eligible=False,full_profile_qualified=False,
        missing_gates=list(failures),gate_failures=failures,checked_gates=checks,config=config,
        profile_protocol=profile_proof,source_continuity=continuity,
        inherited_artifact_flags_unchanged=True,inputs_sha256=digest(inputs))
