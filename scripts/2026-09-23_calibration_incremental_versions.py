#!/usr/bin/env python3
"""Publish new CPU-verified component descriptors; never edit old registries."""
import argparse
import hashlib
import json
from pathlib import Path

from pdblend.profile.model import PerfModel
from pdblend.profile.power_calibration import write_immutable
from pdblend.profile.versions import load_version

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/2026-09-23'
BASE=ROOT/'results/2026-09-22/three-model/calibration-candidates'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def validated_report(path):
    report=read(path)
    if report.get('all_jobs_released') is not True:raise ValueError('incremental wave is not finalized')
    for row in report['members'].values():
        for item in row.get('evidence',{}).values():
            if sha(item['path'])!=item['sha256']:raise ValueError('final closeout evidence changed')
    return report


def identity_version(row):
    body={k:v for k,v in row.items() if k not in ('version_id','execution')}
    digest=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()[:20]
    row['version_id']=f"{row['model_id']}-tp{row['tp']}-pp{row['pp']}-{digest}"
    return row


def base_row(member,report_path,candidate_path):
    if member.get('version_creation_ready') is not True:raise ValueError('component not ready for a new version')
    candidate=read(candidate_path);raw=read(member['evidence']['raw.json']['path'])
    source=RESULTS/'incremental-profile-sources'/raw['environment']['source_hash']/'manifest.json'
    return dict(schema=1,system='pdblend',model_id=member['model_id'],tp=member['tp'],pp=1,
        model_hash=raw['model_hash'],tokenizer_hash=raw['tokenizer_hash'],sampling_complete=True,power_passed=True,
        effective_timing_passed=True,calibration_components_passed=True,full_profile_qualified=False,
        formal_eligible=False,energy_comparable=False,seeds=[701],single_seed=True,
        bounded_coverage=candidate['bounded_coverage'],decode_timing_domains={f:x['domain'] for f,x in candidate['decode_overrides'].items()},
        decode_power_domains={f:dict(kind=x['kind'],nodes=x['nodes']) for f,x in candidate.get('decode_power_overrides',{}).items()},
        environment=raw['environment'],evidence=dict(closeout=bound(report_path),power_candidate=bound(candidate_path),
            frozen_source=bound(source)),original_inputs={},raw_evidence=[],timing={},
        missing_gates=['full_profile_quality_audit','trace_shape_domain_coverage','native_mechanisms','campaign_acceptance'],
        limits=['No extrapolation, frequency interpolation or cross-system profile borrowing.',
                'Component qualification is separate from formal ranking.'])


def compose(report_path):
    report=validated_report(report_path);queue=read(ROOT/'results/2026-09-22/three-model/queue.json');rows=[]
    for member_name,dirname in [('7b-tp1-longctx','7b-tp1-4ddc49563cef6321c0b5'),
                               ('32b-tp2-longctx','32b-tp2-f31ed0be1462e4ef9564')]:
        member=report['members'][member_name];candidate_path=BASE/dirname/'candidate.json'
        row=base_row(member,report_path,candidate_path);root=Path(member['root']);long=read(root/'long-candidate/candidate.json')
        jobs=[j for j in queue['jobs'].values() if j['job_id'].startswith('holdout-'+dirname) and j['status']=='succeeded']
        if len(jobs)!=1:raise ValueError('unique original short holdout required')
        lease=max([x for x in queue['leases'].values() if x['job_id']==jobs[0]['job_id']],key=lambda x:x['claimed_at'])
        short=Path(lease['attempt_dir']);completion=read(short/'completion.json')
        if (completion.get('calibration_passed') is not True or completion.get('complete') is not True or
                completion['candidate_sha256']!=sha(candidate_path) or completion['raw_sha256']!=sha(short/'raw.json')):
            raise ValueError('original short component is not independently qualified')
        row.update(component_kind='short_and_exact_batch_long_union',timing=member['holdout'],
            long_domain=dict(exact_batches=long['exact_batches'],batch_interpolation_qualified=False,
                context_intervals={key:[nodes[0]['context'],nodes[-1]['context']] for key,nodes in long['nodes'].items()},
                composition='union_without_gap_interpolation'))
        row['limits']+=['Unmeasured short/long context gaps remain missing_profile.', 'Long domain accepts only declared exact batches.']
        row['evidence'].update(long_candidate=bound(root/'long-candidate/candidate.json'),short_completion=bound(short/'completion.json'),
            short_holdout_audit=bound(short/'holdout-audit.json'),long_completion=bound(root/'long-holdout/completion.json'),
            long_holdout_audit=bound(root/'long-holdout/holdout-audit.json'))
        prior=read(member['evidence']['package_manifest']['path'])['inputs']['raw.json']
        row['raw_evidence']=[bound(short/'raw.json'),bound(root/'raw.json'),bound(root/'long-holdout/raw.json'),prior,
            dict(bound(root/'long-candidate/endpoint-training-archive.json'),samples_root_binding=bound(root/'raw.json'))]
        rows.append(identity_version(row))
    member=report['members']['14b-tp4-power-mixed'];root=Path(member['root']);package=RESULTS/'14b-tp4-local-power-package-v2'
    manifest=read(package/'manifest.json');row=base_row(member,report_path,package/'candidate.json')
    row.update(component_kind='local_power_and_repaired_mixed',timing=member['timing'],power=member['power'],
        validated_power_batches=[1,128],unvalidated_other_power_shapes=True)
    row['limits']+=['Fresh power qualification covers B1/B128 only; other shapes remain unvalidated.']
    row['original_inputs']=manifest['inputs']
    original=Path(manifest['original_holdout']);prior=manifest['inputs']['prior_raw']
    derived=read(root/'mixed-repair/combined-timing-view.json')
    sources={'original-completed':bound(original/'raw.json'),'mixed-repair':bound(root/'mixed-repair/raw.json')}
    for name in derived.get('evidence_sources',{}):sources[name]=prior
    row['raw_evidence']=[bound(root/'raw.json'),bound(root/'mixed-repair/raw.json'),prior,bound(original/'raw.json'),
        dict(bound(root/'mixed-repair/combined-timing-view.json'),samples_root_binding=bound(original/'raw.json'),evidence_sources=sources)]
    row['evidence'].update(package_manifest=bound(package/'manifest.json'),local_completion=bound(root/'completion.json'),
        local_composite=bound(root/'composite-audit.json'),repaired_timing=bound(root/'mixed-repair/repaired-timing-audit.json'))
    rows.append(identity_version(row))
    return dict(schema=1,registry_kind='bounded_component_calibration_versions',versions=rows,source_binding=bound(__file__),
        seeds=[701],formal_eligible=False,energy_comparable=False,old_registry_unchanged=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closeout',type=Path,default=RESULTS/'incremental-wave-closeout-v1/audit.json')
    parser.add_argument('--out',type=Path,default=RESULTS/'calibration-incremental-versions-v1')
    args=parser.parse_args();registry=compose(args.closeout)
    # Validate in a temporary new path before publishing an immutable version.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        test=Path(directory)/'registry.json';test.write_text(json.dumps(registry))
        for row in registry['versions']:
            load_version(test,row['version_id'],system=row['system'],model_id=row['model_id'],tp=row['tp'],pp=1,usage='development')
    write_immutable(args.out/'registry.json',registry)
    print(json.dumps(dict(registry=str(args.out/'registry.json'),versions=[r['version_id'] for r in registry['versions']])))


if __name__=='__main__':main()
