"""Frozen paired-run evidence. No legacy result is silently promoted to formal."""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
from ecopadg.measure.power import instant_power_verified

REQUIRED_MECHANISMS={
    'distserve': ('independent_parallel_search','instance_search','phase_batching',
                 'kv_admission','interconnect_placement','output_correctness'),
    'dynamollm': ('length_prediction','nine_logical_pools','fragmentation',
                 'scale_inst_1800s','scale_shard_300s','scale_freq_5s',
                 'measured_reconfiguration','staggered_switch','output_correctness'),
    'ecoserve': ('engine_temporal_exclusion','rolling_activation','unified_constraints',
                'macro_split_merge','output_correctness'),
    'mixed': ('full_frequency','output_correctness'),
    'mixed_dvfs': ('feasible_energy_frequency','output_correctness')}
REQUIRED_MECHANISMS={k:v+('independent_calibration',) for k,v in REQUIRED_MECHANISMS.items()}
FREEZE_GROUPS=('source','model','image','profiles','traces','protocol')
FORMAL_SEEDS=(101,202,303)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024**2),b''):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_files(paths):
    return {str(Path(p).resolve()):sha256(p) for p in sorted(paths)}


def validate_freeze(manifest):
    return [p for p,digest in manifest.items() if not Path(p).is_file() or sha256(p)!=digest]


def freeze_bundle(groups,engine_image):
    """Freeze explicit artifact classes; a lone file is not a formal freeze.

    The image artifact is Docker's inspection JSON, and engine_image is its
    immutable sha256 ID. Model groups include weights, tokenizer and config.
    """
    if any(not groups.get(k) for k in FREEZE_GROUPS):
        raise ValueError('source/model/image/profiles/traces/protocol artifacts required')
    groups={k:sorted(str(Path(p).resolve()) for p in groups[k]) for k in FREEZE_GROUPS}
    files=freeze_files({p for paths in groups.values() for p in paths})
    return dict(schema=2,files=files,groups=groups,identities=_identities(groups,files,engine_image))


def _identities(groups,files,engine_image):
    identities={k+'_sha256':hashlib.sha256(json.dumps(
        {p:files[p] for p in groups[k]},sort_keys=True).encode()).hexdigest()
        for k in ('source','model','profiles')}
    identities['profile_sha256']=identities.pop('profiles_sha256')
    identities['engine_image']=engine_image
    return identities


def formal_freeze_gaps(bundle):
    if bundle.get('schema')!=2:
        return ['missing typed source/model/image/profile/trace/protocol freeze']
    files,groups=bundle.get('files',{}),bundle.get('groups',{})
    gaps=validate_freeze(files)
    for kind in FREEZE_GROUPS:
        if not groups.get(kind) or any(p not in files for p in groups[kind]):
            gaps.append('missing frozen artifact group: '+kind)
    if gaps:
        return gaps
    # All bytes were checked by validate_freeze above. Re-hashing the model a
    # second time here would double the multi-gigabyte I/O for every cell.
    expected=_identities(groups,files,bundle.get('identities',{}).get('engine_image',''))
    if expected!=bundle.get('identities'):
        gaps.append('provenance identities do not match artifact hashes')
    image=bundle.get('identities',{}).get('engine_image','')
    inspected=[]
    for p in groups['image']:
        try:
            document=json.loads(Path(p).read_text())
            inspected.extend(x.get('Id') for x in (document if isinstance(document,list) else [document]))
        except (ValueError,AttributeError):
            gaps.append('invalid image inspection artifact')
    if not image.startswith('sha256:') or len(image)!=71 or image not in inspected:
        gaps.append('engine image must match an inspected immutable image ID')
    return gaps


def matrix_gaps(cells):
    """Do not accept a hand-picked subset or omit the paired dynamic trace."""
    required={(d,l,s) for d in ('alpaca','sharegpt','longbench')
              for l in ('low','medium','near_saturation') for s in FORMAL_SEEDS}
    required|={('dynamic','changing',s) for s in FORMAL_SEEDS}
    keys=[(c.get('dataset'),c.get('load'),c.get('seed')) for c in cells]
    gaps=[]
    if set(keys)!=required or len(keys)!=len(required):
        gaps.append('requires 27 static cells and three paired 60-minute dynamic cells')
    for c in cells:
        if (c.get('split')!='formal' or c.get('n_requests',0)<1
                or (c.get('dataset')!='dynamic' and c['n_requests']<500)
                or (c.get('dataset')=='dynamic' and c.get('trace_duration_s')!=3600)):
            gaps.append('invalid cell size, split or dynamic duration')
            break
    return gaps


def baseline_gaps(evidence):
    gaps={}
    for baseline,mechanisms in REQUIRED_MECHANISMS.items():
        recorded=evidence.get(baseline,{})
        missing=[]
        for mechanism in mechanisms:
            proof=recorded.get(mechanism,{})
            if proof.get('passed') is not True or not proof.get('artifact'):
                missing.append(mechanism)
                continue
            artifact=Path(proof['artifact'])
            if not artifact.is_file() or sha256(artifact)!=proof.get('sha256'):
                missing.append(mechanism)
        if missing:
            gaps[baseline]=missing
    return gaps


COLLECTION_COMPONENTS=('cpu_contracts','certified_search','eco_hardware','dynamo_hardware',
    'controller_hardware','independent_calibration','hardware_phase_batching','real_kv_boundary')


def checked_mechanism_collection(evidence, registry_path, collection_path, freeze=None):
    """Verify the actual collector result, without inventing a common raw schema.

    Raw proofs differ by mechanism. The collector already recomputes those
    contracts; its final summary binds that checked registry by digest.
    """
    registry_path=Path(registry_path).resolve();collection_path=Path(collection_path).resolve()
    registry=json.loads(registry_path.read_text());collection=json.loads(collection_path.read_text())
    components=collection.get('components',{});sources=collection.get('source_files',{})
    if (registry!=evidence or baseline_gaps(registry)
            or collection.get('complete') is not True
            or collection.get('baseline_mechanisms_complete') is not True
            or collection.get('missing')!={}
            or collection.get('registry')!=str(registry_path)
            or collection.get('registry_sha256')!=sha256(registry_path)
            or not isinstance(components,dict)
            or any(not isinstance(components.get(k),dict)
                   or components[k].get('passed') is not True or components[k].get('error')
                   for k in COLLECTION_COMPONENTS)
            or not isinstance(sources,dict) or not sources or validate_freeze(sources)):
        raise ValueError('mechanism registry lacks an unchanged successful collector result')
    files={registry_path,collection_path}|{Path(p).resolve() for p in sources}
    if freeze is not None:
        protocol=set(freeze.get('groups',{}).get('protocol',()))
        frozen=freeze.get('files',{})
        if (not {str(registry_path),str(collection_path)}<=protocol
                or any(frozen.get(str(p))!=sha256(p) for p in files)):
            raise ValueError('mechanism registry, collection or source evidence is not frozen')
    return files


def formal_evidence_gaps(freeze, evidence, cells):
    """Bind report arguments to the evidence approved before formal execution."""
    links=freeze.get('formal_evidence',{})
    if not isinstance(links,dict) or any(not links.get(k) for k in
            ('mechanisms','mechanism_collection','expected_cells')):
        return ['missing frozen formal evidence links']
    try:
        expected=Path(links['expected_cells']).resolve()
        if (str(expected) not in freeze.get('groups',{}).get('protocol',())
                or freeze.get('files',{}).get(str(expected))!=sha256(expected)
                or json.loads(expected.read_text())!=cells):
            raise ValueError('expected cells differ from the frozen complete target')
        checked_mechanism_collection(evidence,links['mechanisms'],links['mechanism_collection'],freeze)
    except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
        return ['formal evidence binding: '+str(exc)]
    return []


def common_capacity(calibrations,required=tuple(REQUIRED_MECHANISMS),target=.99):
    """Use independent baseline calibration, not PDBlend's best point."""
    capacities={}
    for dataset in ('alpaca','sharegpt','longbench'):
        best=[]
        for baseline in required:
            rates=[]
            for result in calibrations:
                if result.get('dataset')!=dataset or result.get('system')!=baseline: continue
                proof=result.get('confirmation') or {}
                rate=result.get('capacity_rps')
                upper=result.get('infeasible_upper_rps')
                if (result.get('passed') and isinstance(rate,(int,float)) and math.isfinite(rate) and rate>0
                        and isinstance(upper,(int,float)) and math.isfinite(upper) and upper>rate
                        and proof.get('rate')==rate and proof.get('slo_attainment',0)>=target
                        and proof.get('validity')=='ok' and proof.get('split')=='calibration'
                        and proof.get('completed',0)>=128 and proof.get('completed')==proof.get('n_expected')):
                    rates.append(float(rate))
            if not rates:
                raise ValueError(f'no independently calibrated capacity: {dataset}/{baseline}')
            best.append(max(rates))
        capacities[dataset]=min(best)
    return capacities


def evaluation_matrix(capacities,seeds=(101,202,303)):
    return [dict(dataset=dataset,load=level,rate=capacities[dataset]*fraction,
                 seed=seed,n_requests=500,split='formal')
            for dataset in sorted(capacities) for level,fraction in
                (('low',.3),('medium',.6),('near_saturation',.9)) for seed in seeds]


def mean_ci95(values):
    """Paired independent runs, two-sided Student t interval (three seeds)."""
    if len(values)<3:
        return None,None
    # For the frozen three-seed first round. Never pretend requests are runs.
    if len(values)!=3:
        raise ValueError('first-round statistics require exactly three independent paired seeds')
    mean=statistics.mean(values)
    radius=4.30265272975*statistics.stdev(values)/math.sqrt(3)
    return mean-radius,mean+radius


def evaluate(rows,expected_cells,baseline_evidence,freeze):
    reasons=[]
    variants={r.get('variant') for r in rows if r.get('system')=='pdblend'}
    if variants and (len(variants)!=1 or not variants<={'pdblend-greedy','pdblend-joint','pdblend-dynamic'}):
        reasons.append('formal PDBlend candidate is missing or changes between runs')
    gaps=baseline_gaps(baseline_evidence)
    changed=formal_freeze_gaps(freeze)
    if gaps:
        reasons.append('baseline mechanisms incomplete')
    if changed:
        reasons.append('frozen inputs changed')
    reasons.extend(matrix_gaps(expected_cells))
    reasons.extend(formal_evidence_gaps(freeze,baseline_evidence,expected_cells))
    identities=freeze.get('identities',{})
    trace_cells={};trace_work={}
    for path in freeze.get('groups',{}).get('traces',[]):
        try:
            trace=json.loads(Path(path).read_text())
            if not isinstance(trace.get('requests'),list) or any(not isinstance(r,dict) for r in trace['requests']):
                raise ValueError('explicit request objects required')
            work=[r.get('output_len') for r in trace['requests']]
            if not work or any(type(n) is not int or n<=0 for n in work):
                raise ValueError('positive prescribed output lengths required')
            trace_work[freeze['files'][path]]=sum(work)
            trace_cells[freeze['files'][path]]=(trace['dataset'],trace['load'],trace['seed'],
                len(trace['requests']),trace['split'])
        except (OSError,ValueError,KeyError,TypeError):
            reasons.append('frozen trace lacks an explicit evaluation identity')
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['system'],row['dataset'],row['load'],row['seed'])].append(row)
    by_baseline={}
    for baseline in REQUIRED_MECHANISMS:
        energy_by_seed=defaultdict(lambda:defaultdict(list))
        slo_by_point=defaultdict(list)
        invalid=[]
        for cell in expected_cells:
            key=(cell['dataset'],cell['load'],cell['seed'])
            ours=grouped[('pdblend',*key)]
            base=grouped[(baseline,*key)]
            if len(ours)!=1 or len(base)!=1:
                invalid.append(dict(cell=cell,reason='missing or duplicate paired run'))
                continue
            a,b=ours[0],base[0]
            required=('trace_sha256','model_sha256','engine_image','measurement_schema',
                      'profile_sha256','source_sha256','n_expected','generated_tokens','expected_generated_tokens',
                      'power_mode','power_source_id','power_field_id')
            if (any(a.get(k) is None or a.get(k)!=b.get(k) for k in required)
                    or a.get('measurement_schema')!=2
                    or any(r.get('validity')!='ok' or r.get('completed')!=cell['n_requests']
                           or r.get('n_expected')!=cell['n_requests']
                           or r.get('gpu_count')!=8 or not math.isfinite(r.get('energy_j',0))
                           or r.get('energy_j',0)<=0
                           or type(r.get('generated_tokens')) is not int
                           or type(r.get('expected_generated_tokens')) is not int
                           or r.get('generated_tokens')!=trace_work.get(r.get('trace_sha256'))
                           or r.get('expected_generated_tokens')!=trace_work.get(r.get('trace_sha256'))
                           or not 0<=r.get('slo_attainment',-1)<=1
                           or r.get('split')!='formal' or r.get('formal_eligible') is not True
                           or not instant_power_verified(r)
                           or any(r.get(k)!=v for k,v in identities.items())
                           or trace_cells.get(r.get('trace_sha256'))!=(*key,cell['n_requests'],'formal')
                           or (cell['dataset']=='dynamic' and
                               (r.get('trace_duration_s')!=3600 or r.get('duration_s',0)<3600))
                           for r in (a,b))):
                invalid.append(dict(cell=cell,reason='workload, provenance or measurement mismatch'))
                continue
            ratio=a['energy_j']/b['energy_j']
            energy_by_seed[cell['seed']][cell['dataset']].append(1-ratio)
            slo_by_point[(cell['dataset'],cell['load'])].append(a['slo_attainment']-b['slo_attainment'])
        savings=[statistics.mean(statistics.mean(v) for v in datasets.values())
                 for datasets in energy_by_seed.values()]
        energy_ci=mean_ci95(savings)
        slo_ci={f'{d}/{l}':mean_ci95(v) for (d,l),v in slo_by_point.items()}
        enough=(not invalid and len(savings)==3 and len(slo_ci)==len({(c['dataset'],c['load']) for c in expected_cells}))
        passed=(enough and energy_ci[0] is not None and energy_ci[0]>=.05
                and all(ci[0] is not None and ci[0]>=-.01 for ci in slo_ci.values()))
        by_baseline[baseline]=dict(passed=passed,invalid_pairs=invalid,
            energy_saving_mean=statistics.mean(savings) if savings else None,
            energy_saving_ci95=energy_ci,slo_delta_ci95=slo_ci)
        if invalid:
            reasons.append(f'{baseline}: incomplete valid pairs')
    verdict=('evidence_insufficient' if reasons else
             'target_achieved' if all(b['passed'] for b in by_baseline.values()) else 'target_not_achieved')
    return dict(verdict=verdict,reasons=reasons,baseline_gaps=gaps,
                changed_frozen_files=changed,comparisons=by_baseline)
