#!/usr/bin/env python3
"""CPU-only immutable recipes for minimum-topology short panels and 14B long holdout."""
import argparse
import json
from pathlib import Path

from pdblend.profile import short_domain_collect as short
from pdblend.profile.long_holdout_only import prepare as prepare_long, load_package as load_long
from pdblend.profile.power_calibration import write_immutable

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/'results/2026-09-23'
CANDIDATES=ROOT/'results/2026-09-22/three-model/calibration-candidates'
BASES={'7b':'7b-tp1-4ddc49563cef6321c0b5','14b':'14b-tp1-a52c3dc4e13305790e9b','32b':'32b-tp2-f31ed0be1462e4ef9564'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-short',action='append',default=[],metavar='MODEL=PATH',
        help='Retain independently verified windows from an old 7b/14b/32b short directory')
    args=p.parse_args()
    resumed={}
    for value in args.resume_short:
        model,separator,path=value.partition('=')
        if not separator or model not in BASES or model in resumed or not path:
            raise ValueError('resume-short requires one unique 7b/14b/32b=PATH per model')
        resumed[model]=Path(path).resolve()
    out=args.out.resolve()
    if out.exists():raise FileExistsError('new immutable review directory required')
    report=json.loads((RESULTS/'incremental-wave-closeout-v1/audit.json').read_text())
    out.mkdir(parents=True);members={}
    # Bind the current collector/qualification code in a new immutable input
    # package. This rebuilds only CPU metadata from the same 36 training rows.
    long=out/'14b-long-package'
    prepare_long(training=report['members']['14b-tp1-longctx']['root'],out=long)
    lm,_,_=load_long(long)
    sources=['short_domain.py','short_domain_collect.py','sampling_guard.py','resident_long_holdout.py','resident_domain_job.py']
    for size,tp in (('7b',1),('14b',1),('32b',2)):
        key=f'{size}-tp{tp}-short-resident';training=Path(report['members'][f'{size}-tp{tp}-longctx']['root']);package=out/f'{size}-short-package'
        base=CANDIDATES/BASES[size]/'candidate.json';dataset=ROOT/'datasets/prepared'/f'2026-09-22-{size}-v1/manifest.json'
        manifest=short.prepare(base_candidate=base,dataset_manifest=dataset,identity_raw=training/'raw.json',
            out=package,resume_from=resumed.get(size))
        _,plan=short.load_package(package)
        argv=['python','-m','pdblend.profile.resident_domain_job','--model',f'/models/{manifest["model_id"]}',
            '--gpus',*[str(i) for i in range(tp)],'--base-port','{lease_port}','--out','{attempt_dir}',
            '--epochs-root','{coordinator}','--member',key,'--short-package',str(package)]
        roots=[str(package),str(training),str(base.parent),str(dataset.parent)]
        if size in resumed:
            # The inherited archive verifies its enclosing attempt manifest,
            # exact invocation, original package and all frozen source files.
            roots.append(str(resumed[size].parent))
            roots.extend(str(Path(path).parent) for path in manifest['inherited_archive']['files_sha256']
                         if Path(path).is_absolute())
        if size=='14b':
            argv+=['--long-package',str(long)];roots +=[str(long),lm['training']]
        seconds=sum(p['repeats']*(p['settle_s']+p['measure_s']) for phase in ('training','holdout') for p in plan[phase])
        recipe=dict(schema=1,kind='experimental_short_and_optional_qualified_long_resident',model_id=manifest['model_id'],
            system='pdblend',tp=tp,pp=1,gpu_count=tp,exclusive=False,max_attempts=2,timeout_s=7200,
            argv_template=argv,readonly_roots=sorted(set(roots)),required_receipts=['completion.json'],
            package_manifest_sha256=short.digest(package/'manifest.json'),
            inherited_short_archive=manifest.get('inherited_archive'),
            implementation_sha256={name:short.digest(ROOT/'src/pdblend/profile'/name) for name in sources},
            source_files=['src/pdblend/profile/'+name for name in sources],
            source_freeze_required=True,epochs_controller_required=True,cohort_fixed_membership=True,
            short_training_points=len(plan['training']),short_holdout_points=len(plan['holdout']),
            short_window_minimum_s=seconds,estimated_short_elapsed_s=[1050,1500],
            original_continuous_decode_power_protocol_passed=False,pure_decode_power_qualified=False,
            long_holdout_points=24 if size=='14b' else 0,long_training_points=0,
            experimental_sampling=True,formal_eligible=False,energy_comparable=False,enqueued=False,hardware_started=False)
        write_immutable(out/f'{key}.json',recipe)
        members[key]=dict(model_id=manifest['model_id'],tp=tp,kind='resident_short_with_optional_long',package=str(package),
            long_package=str(long) if size=='14b' else None,recipe=str(out/f'{key}.json'),readonly_roots=recipe['readonly_roots'],
            expected_training_points=len(plan['training']),expected_holdout_points=len(plan['holdout'])+(24 if size=='14b' else 0))
    write_immutable(out/'review.json',dict(schema=1,status='cpu_preflight_passed',members=members,seed=701,formal_eligible=False,
        hardware_started=False,experimental_short_sampling=True,original_14b_package_unchanged=True,
        limitations=['Short power is complete repeated-request workload power, not pure decode.',
            'HTTP SSE timing does not replace a native CUDA timing crosscheck.',
            'Each short decode repeat accumulates >=5s across segments; original continuous >=5s qualification remains false.',
            'Only exact B1 and output64 are sampled in the decode/mixed role.',
            'Timing and mixed-window power must pass a fresh holdout before even experimental consumption.',
            'No existing PerfModel domain is changed by measurement completion.']))
    print(json.dumps(dict(review=str(out/'review.json'),members=list(members)),indent=2))


if __name__=='__main__':main()
