"""Bind a three-model DistServe profile cohort before the original collector.

Preparation and CPU preflight never qualify concurrent GPU measurements. The
original NativeStageWave must observe all six engines, or serialize the cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from . import stage_collect
from ..native_profile import DIST_FREQS

MODELS = {'distserve-7b': ('Qwen2.5-7B-Instruct', 1),
          'distserve-14b': ('Qwen2.5-14B-Instruct', 1),
          'distserve-32b': ('Qwen2.5-32B-Instruct', 2)}
DATASETS = ('alpaca', 'sharegpt', 'longbench')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):value.update(block)
    return value.hexdigest()


def point_plan(corpus, model, tp):
    if (model, tp) not in MODELS.values():raise ValueError('unsupported model/TP pair')
    bindings, lengths, ends = [], [], []
    for dataset in DATASETS:
        path = Path(corpus).resolve()/(dataset+'.json')
        data = json.loads(path.read_text())
        if data.get('model_name') != model:raise ValueError('model-specific corpus required')
        if not data.get('calibration') or not data.get('tuning'):
            raise ValueError('independent calibration/tuning splits required')
        for row in data['calibration']+data['tuning']:
            n, out = row['input_tokens'], row['output_tokens']
            if type(n) is not int or type(out) is not int or not 1 <= n <= 7168 or not 1 <= out or n+out > 8192:
                raise ValueError('invalid independent corpus shape')
            lengths.append(n); ends.append(n+out)
        bindings.append(dict(dataset=dataset,path=str(path),sha256=sha(path),splits=['calibration','tuning']))
    lo, hi, mid = min(lengths), max(lengths), sorted(lengths)[len(lengths)//2]
    if not lo <= mid < hi:raise ValueError('independent shape envelope has no distinct midpoint/high point')
    upper = min(7808, max(ends)+128)
    pre = [(lo,), (hi,), (lo,lo), (lo,hi), (lo,)*4, (hi,lo,lo,lo), (2048,)*4]
    dec = [(lo,), (upper,), (lo,lo), (lo,upper), (lo,)*4, (upper,lo,lo,lo), (upper,)*4]
    bridges=[n for n in (4096,6144,7168) if lo<=n<=hi]
    dec += [(n,) for n in bridges if (n,) not in dec]
    hold_pre = [(mid,), (lo,hi), (2048,)*4]
    hold_dec = [(max(mid,lo+128),), (lo+128,upper-256), (upper-256,)*4]
    # Candidates above the native limit remain explicit unsupported receipts;
    # only measured windows enter a model-specific fit.
    points = [dict(frequency_mhz=f,role=role,purpose=purpose,lengths=list(shape),repeats=3)
        for f in DIST_FREQS for role, train, hold in [('prefill',pre,hold_pre),('decode',dec,hold_dec)]
        for purpose, shapes in [('training',train),('holdout',hold)] for shape in shapes]
    return dict(schema='distserve-targeted-stage-plan-v1',system='distserve',model_id=model,tp=tp,pp=1,
        selection_split='calibration',selection_splits=['calibration','tuning'],evaluation_used_for_selection=False,
        inputs=bindings,points=points,batch_candidates=[1,2,4],frequency_candidates=list(DIST_FREQS),
        shape_selection='bounded representative vertices from this model calibration/tuning lengths',
        selected_envelope=dict(minimum_prompt=lo,median_prompt=mid,maximum_prompt=hi,
                               maximum_requested_end=max(ends),decode_prompt_upper=upper,
                               predeclared_b1_decode_bridges=bridges),
        capacity_policy='each live engine checks max_num_seqs, full-prefill budget and 90% actual KV capacity; unsupported_engine is excluded from fitting',
        formal_eligible=False,limitations=[
            'Only actual measured per-request shape hulls can qualify; no maximum-shape shortcut.',
            'Short decode contexts may remain uncovered after settled measurement.',
            'Batches above four, other TP, PP and unmeasured shapes remain missing_profile.',
            'Collection completion does not establish holdout accuracy or trace/deployment qualification.'])


def _bound(root, ref):
    path = (root/ref['path']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path) != ref['sha256']:
        raise ValueError('profile cohort bound file differs')
    return path


def validate_cohort(path, member, argv):
    path = Path(path).resolve(); root = path.parent; value = json.loads(path.read_text())
    p = argparse.ArgumentParser(add_help=False)
    for name in ('model','gpus','point-plan','input-manifest','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--tp',type=int,required=True)
    p.add_argument('--preflight-only',action='store_true')
    args, _ = p.parse_known_args(argv)
    expected = json.loads(Path(args.input_manifest).read_text())
    if (value.get('schema')!='distserve-three-model-stage-cohort/v1'
            or expected.get('cohort_sha256')!=sha(path) or expected.get('cohort_member')!=member
            or member!=os.environ.get('PDBLEND_PROFILE_MEMBER') or member not in MODELS
            or value.get('source_sha256')!=expected.get('source_sha256')
            or value.get('image_digest')!=expected.get('image_digest')):
        raise ValueError('profile cohort/member/source binding differs')
    members = value.get('members',{})
    if set(members)!=set(MODELS):raise ValueError('complete three-model cohort required')
    for name,(model,tp) in MODELS.items():
        row=members[name]
        if (row.get('model_id'),row.get('tp'),row.get('pp'),row.get('gpu_count'))!=(model,tp,1,2*tp):
            raise ValueError('cohort native pair allocation differs')
        _bound(root,row['point_plan'])
    row = members[member]
    point_path = _bound(root,row['point_plan'])
    if (Path(args.point_plan).resolve()!=point_path or args.model!=row['model_id'] or args.tp!=row['tp']
            or [int(g) for g in args.gpus.split(',')]!=list(range(row['gpu_count']))):
        raise ValueError('collector arguments differ from bound member')
    wave_path = _bound(root,value['wave_spec'])
    live_wave = Path(os.environ['PDBLEND_PROFILE_WAVE'])/'wave.json'
    if sha(live_wave)!=value['wave_spec']['sha256']:raise ValueError('live profile cohort changed')
    wave = json.loads(wave_path.read_text())
    if (wave.get('cohort_id')!=value['cohort_id'] or wave.get('members')!=list(MODELS)
            or any(wave.get(key) is not True for key in ('coordinator','synchronize_parallel_windows','keep_peers_resident_until_all_done'))):
        raise ValueError('cohort synchronization/residency protocol differs')
    if value.get('startup_port_contract') or os.environ.get('PDBLEND_DIST_PORT_GUARD')=='1':
        from .stage_ports import CONTRACT
        if (os.environ.get('PDBLEND_DIST_PORT_GUARD')!='1' or wave.get('startup_port_contract')!=CONTRACT
                or value.get('startup_port_contract')!=CONTRACT):
            raise ValueError('cohort startup port protocol was not bound before launch')
    selected = json.loads(point_path.read_text())
    inputs = selected.get('inputs',[])
    if len(inputs)!=3 or [item.get('dataset') for item in inputs]!=list(DATASETS):
        raise ValueError('model corpus bindings incomplete')
    corpus = Path(inputs[0]['path']).parent
    if selected != point_plan(corpus,args.model,args.tp):
        raise ValueError('point plan differs from independently bound calibration/tuning corpus')
    stage_collect.load_point_plan(point_path,args.model,args.tp)
    return dict(schema='distserve-three-model-cohort-preflight/v1',status='cpu_preflight_passed',
        hardware_executed=False,formal_eligible=False,parallel_qualified=False,
        member=member,cohort_id=value['cohort_id'],cohort_sha256=sha(path),
        source_sha256=value['source_sha256'],image_digest=value['image_digest'],
        members=members,gpu_count=8,qualification='pending actual six-engine isolated/concurrent measurement',
        capacity='pending each live native engine; unsupported windows cannot enter fitting'),Path(args.out)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort-inputs',type=Path,required=True)
    parser.add_argument('--member',required=True)
    options, arguments=parser.parse_known_args(argv)
    if arguments and arguments[0]=='--':arguments=arguments[1:]
    receipt,out=validate_cohort(options.cohort_inputs,options.member,arguments)
    stage_collect.atomic_json(out/'cohort-preflight.json',receipt)
    return stage_collect.main(arguments)


if __name__=='__main__':raise SystemExit(main())
