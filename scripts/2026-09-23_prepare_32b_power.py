#!/usr/bin/env python3
"""CPU-only, model-owned 32B power tables; never fit or repair timing holdout."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import math
import statistics
from pathlib import Path

from pdblend.profile import power_calibration as pc
from pdblend.profile.calibration import _checkpoint_points,evaluate_holdout
from pdblend.profile.model import PerfModel
from pdblend.profile.power_table import KIND,validate

ROOT=Path(__file__).resolve().parents[1]
helper_spec=importlib.util.spec_from_file_location('power_training_comparison',ROOT/'scripts/2026-09-23_compare_7b_tp4_power.py')
helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)


def check_training(raw,path,tp):
    if (raw.get('system'),raw.get('model_id'),raw.get('tp'),raw.get('pp'),raw.get('holdout_independent'))!=('pdblend','Qwen2.5-32B-Instruct',tp,1,False):
        raise ValueError('requires only model-owned 32B PDBlend training')
    if sorted(raw['freqs'])!=list(pc.FREQUENCIES):raise ValueError('six frequencies required')
    expected_shapes=14 if tp==2 else 22
    if len(raw['decode'])!=6*expected_shapes:raise ValueError('original sparse matrix must remain complete')
    checked=[]
    for row in raw['decode']:
        if len(row['repeats'])!=3:raise ValueError('three training repeats required')
        for rep in row['repeats']:
            p=(path.parent/rep['samples_file']).resolve()
            if not p.is_relative_to(path.parent.resolve()) or pc.digest(p)!=rep['samples_sha256']:
                raise ValueError('training sample checksum mismatch')
            data=json.loads(p.read_text())
            c=row['context_tokens']+statistics.fmean((a+b)/2 for a,b in zip(data['start_token_counts'],data['end_token_counts']))
            w=statistics.fmean(sum(a) for _,a in data['power'])
            if not math.isclose(c,rep['effective_context_tokens'],abs_tol=1e-8) or not math.isclose(w,rep['power_w'],rel_tol=1e-9):
                raise ValueError('actual training power/context differs from raw')
            if rep['steady_window_s']<5 or rep['min_steps']<8 or len(data['power'])<2 or not data['frequency']:
                raise ValueError('training sampling gates failed')
            checked.append(dict(path=str(p),sha256=rep['samples_sha256']))
    return checked


def fit_candidate(candidate_dir,out):
    manifest=json.loads((candidate_dir/'manifest.json').read_text());base_path=candidate_dir/'candidate.json'
    raw_path=Path(manifest['training_raw']);raw=json.loads(raw_path.read_text());base=PerfModel.load(base_path)
    if pc.digest(raw_path)!=manifest['training_raw_sha256'] or pc.digest(base_path)!=manifest['candidate_sha256']:
        raise ValueError('original training/base manifest checksum mismatch')
    checks=check_training(raw,raw_path,base.tp)
    if (base.system,Path(base.model).name,base.tp,base.pp)!=('pdblend','Qwen2.5-32B-Instruct',raw['tp'],1):
        raise ValueError('base candidate identity mismatch')
    cv=helper.compare(raw['decode']);model=copy.deepcopy(base)
    for f in model.freqs:
        spec=dict(kind=KIND,batch_interpolation='linear',nodes=helper.make_nodes([r for r in raw['decode'] if r['freq_mhz']==f]),
            training_raw_sha256=pc.digest(raw_path),validation_status='training_only')
        validate(spec);model.decode_power_overrides[f]=spec
    model.quality['decode_power_calibration']=dict(status='training_only',independent_holdout=False,
        training_raw_sha256=pc.digest(raw_path),base_candidate_sha256=pc.digest(base_path))
    assert pc.timing_fields(model)==pc.timing_fields(base)
    pc.write_immutable(out/'candidate.json',json.loads(model.to_json()))
    summary=dict(schema=1,system='pdblend',model_id=raw['model_id'],tp=raw['tp'],pp=1,
        family='bounded_table_linear_batch',candidate_sha256=pc.digest(out/'candidate.json'),
        training_raw=str(raw_path),training_raw_sha256=pc.digest(raw_path),
        base_candidate=str(base_path),base_candidate_sha256=pc.digest(base_path),
        timing_changed=False,holdout_used_for_fit_or_selection=False,training_windows_checked=len(checks),
        raw_checksums=checks,comparisons=cv,formal_eligible=False,independent_power_holdout_required=True,
        selection='Predeclared bounded linear batch/context table; no model or threshold tuning on holdout.',
        coverage='Per-batch measured effective-context bands and their adjacent-batch intersections only; no rectangular union or extrapolation.',
        unsupported_batches='1 < batch < 4',timing_failures_unchanged=True,
        script_sha256=pc.digest(__file__),comparison_script_sha256=pc.digest(ROOT/'scripts/2026-09-23_compare_7b_tp4_power.py'))
    pc.write_immutable(out/'training-comparison.json',summary)
    return model,base,raw,raw_path,summary


def prepare_tp4(candidate_dir,out,original_holdout):
    model,base,raw,raw_path,report=fit_candidate(candidate_dir,out)
    original=Path(original_holdout);raw_old_path=original/'raw.json';old=json.loads(raw_old_path.read_text())
    completion=json.loads((original/'completion.json').read_text());original_manifest_path=candidate_dir/'manifest.json'
    previous=json.loads(original_manifest_path.read_text())
    if (not completion.get('complete') or completion.get('independent_holdout') is not True or
        completion.get('candidate_sha256')!=report['base_candidate_sha256'] or completion.get('raw_sha256')!=pc.digest(raw_old_path)):
        raise ValueError('original 32B timing archive is incomplete or belongs to another candidate')
    if any(old.get(k)!=raw.get(k) for k in ('system','model_id','model_hash','tokenizer_hash','tp','pp')):
        raise ValueError('32B timing identity mismatch')
    _checkpoint_points(old,original)
    timing=pc.timing_component(evaluate_holdout(old,base,original,expected_plan=previous['plan']))
    # Intentional: timing remains failed. This package only schedules independent
    # power; composite_audit will continue to fail until separate timing repair.
    pc.write_immutable(out/'original-timing-component-audit.json',timing)
    points=pc.reserve_points(raw,model,raw_path.parent)
    for p in points:
        p['reservation']['requested_target_was_clamped_to_training_domain']=True
        p['reservation']['high_batch_long_context_claimed']=False
    plan=dict(schema=1,purpose='independent_decode_power_holdout',system='pdblend',model_id=raw['model_id'],
        model_hash=raw['model_hash'],tokenizer_hash=raw['tokenizer_hash'],tp=4,pp=1,points=points,repeats=3,point_count=24,
        minimum_decode_window_seconds=504,independent_holdout=True,fit_performed=False,
        power_gate=dict(mape_max=.10,max_error_max=.15,each_independent_window_max_error=.10),
        full_reservation_inside_frozen_power_and_timing_domain=True,
        no_automatic_retry_of_prediction_failures=True,prompt_seed_rule='1000 * batch + request_index',
        qualification_frequency=2100,cpu_scheduling_proxy=pc.scheduling_proxy(raw,points),
        formal_eligible=False,energy_comparable=False,original_timing_recollection=False,
        original_timing_component_passed=timing['passed'],original_timing_failure_retained=True)
    pc.write_immutable(out/'power-plan.json',plan)
    inputs=dict(training_raw=raw_path,base_candidate=candidate_dir/'candidate.json',original_raw=raw_old_path,
        original_completion=original/'completion.json',original_manifest=original_manifest_path,
        preparation_script=Path(__file__).resolve(),training_comparison=out/'training-comparison.json')
    manifest=dict(schema=1,status='prepared_independent_power_validation',formal_eligible=False,
        system='pdblend',model_id=raw['model_id'],model_hash=raw['model_hash'],tokenizer_hash=raw['tokenizer_hash'],tp=4,pp=1,
        candidate_sha256=pc.digest(out/'candidate.json'),plan_sha256=pc.digest(out/'power-plan.json'),
        timing_component_sha256=pc.digest(out/'original-timing-component-audit.json'),
        inputs={k:dict(path=str(p.resolve()),sha256=pc.digest(p)) for k,p in inputs.items()},
        implementation_sha256=pc.implementation_hashes(),training_environment=raw['environment'],original_timing_environment=old['environment'],
        timing_fields_unchanged=True,timing_component_passed=timing['passed'],original_completion_status=completion.get('calibration_status'),
        required_remaining=['fresh_power_holdout','separate_training_only_timing_repair_and_fresh_B32_holdout',
                            'provenance_and_mixed_evidence','native_mechanisms','campaign_acceptance'])
    pc.write_immutable(out/'manifest.json',manifest)
    pc.load_package(out)
    return report,plan,timing


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    candidates=ROOT/'results/2026-09-22/three-model/calibration-candidates'
    two=next(candidates.glob('32b-tp2-*/manifest.json')).parent
    four=next(candidates.glob('32b-tp4-*/manifest.json')).parent
    _,_,_,_,r2=fit_candidate(two,a.out/'tp2-training-only')
    evidence=json.loads((ROOT/'results/2026-09-23/tp4-holdout-audit/32b.json').read_text())
    r4,plan,timing=prepare_tp4(four,a.out/'tp4-package',Path(evidence['artifact']))
    lines=['# 32B independent power-table candidates','',
        'Only original model-owned training raw entered fitting/CV. Original timing coefficients and all original timing failures remain unchanged. No GPU work or queue changes were performed.','',
        '| Topology | Checked windows | Training MAPE / max | Grouped shape CV | Grouped batch CV |','|---|---:|---:|---:|---:|']
    for report in (r2,r4):
        row=report['comparisons']['bounded_table_linear_batch'];values=[]
        for axis in ('training_resubstitution','shape','batch'):
            v=row[axis];values.append((f'{v["mape"]*100:.2f}% / {v["max_error"]*100:.2f}%' if v['supported'] else 'unsupported') + f' ({v["supported"]}/{v["total"]})')
        lines.append(f'| TP{report["tp"]} | {report["training_windows_checked"]} | '+ ' | '.join(values)+' |')
    lines+=['','CV keeps all three repeats of each frequency/batch/nominal-context shape in one fold. Unsupported boundary folds are counted explicitly; they are not zero-error predictions or coverage evidence.',
        '',f'TP4 power-only panel: 24 points / 72 windows. Full output reservations pass the unchanged timing domain and the new measured power domain. CPU serial-prefill work proxy + windows: {plan["cpu_scheduling_proxy"]["combined_proxy_seconds"]/60:.2f} min, excluding load/barrier/qualification/cleanup. This is not a measured batch runtime or guarantee.',
        '',f'Original TP4 timing component remains passed={timing["passed"]}; timing maximum relative error={timing["timing_max"]*100:.3f}%. Its failures are preserved in `tp4-package/original-timing-component-audit.json`. Fresh power success cannot make the composite calibration pass while timing remains failed.',
        '', 'B2/B3 and fractional batches between 1 and 4 remain missing. Sparse high-batch contexts stay bounded: TP4 B128/B256 have only the short/mid measured domain; TP2 B64 has only a very narrow single measured-context band. No union creates high-batch/long-context coverage. TP2 is training-only and receives no GPU job from this preparation.','']
    (a.out/'review.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output=str(a.out),tp4_timing_passed=timing['passed'],tp4_proxy_minutes=plan['cpu_scheduling_proxy']['combined_proxy_seconds']/60)))

if __name__=='__main__':main()
