"""Repair only four invalid 1500-MHz mixed windows on a resident 14B engine."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import power_calibration as pc
from .calibration import evaluate_holdout
from .identity import sha256_value
from .wave import atomic_json

EXPECTED={(1500,b,c) for b in (8,32) for c in (512,2048)}


def key(row):return row['freq_mhz'],row['batch'],row['chunk_tokens']


def original_view(manifest):
    root=Path(manifest['original_holdout'])
    combined=root/'combined-holdout.json'
    path=combined if combined.is_file() else root/'raw.json'
    raw=json.loads(path.read_text())
    roots={'original-completed':root}
    prior=Path(manifest['inputs']['prior_raw']['path']).parent
    for source,binding in raw.get('evidence_sources',{}).items():
        if binding['raw_sha256']!=pc.digest(prior/'raw.json'):
            raise ValueError('mixed reference inherited checkpoint changed')
        roots[source]=prior
    return raw,roots,path


def plan_repair(manifest):
    raw,_,path=original_view(manifest)
    bad=[r for r in raw['mixed'] if not r.get('valid')]
    if not bad:return None
    if (len(bad)!=4 or {key(r) for r in bad}!=EXPECTED or
        any(r.get('invalid_reason')!='missing_base_step_or_prefill' for r in bad) or
        len(raw['mixed'])!=12 or len({key(r) for r in raw['mixed']})!=12):
        raise ValueError('only the four inherited-reference mixed failures can be repaired by this panel')
    for _,batch,chunk in EXPECTED:
        base=[r for r in raw['decode'] if (r['freq_mhz'],r['batch'],r['context_tokens'])==(1500,batch,1024)]
        alone=[r for r in raw['prefill'] if (r['freq_mhz'],r['input_tokens'])==(1500,chunk)]
        if len(base)!=1 or len(alone)!=1 or base[0]['step_seconds']<=0 or alone[0]['seconds']<=0:
            raise ValueError('mixed repair reference is not complete and unique')
    return dict(schema=1,scope='four_invalid_1500MHz_mixed_points_only',points=[list(k) for k in sorted(EXPECTED)],
        reference_file=str(path),reference_sha256=pc.digest(path),original_failed_rows=copy.deepcopy(bad),
        remeasure_prefill_decode=False,formal_eligible=False,energy_comparable=False)


def checked_completed(*,package,out,manifest):
    out=Path(out);plan=manifest['mixed_repair']
    completion=json.loads((out/'completion.json').read_text())
    if not completion.get('complete'):return None
    binding=dict(package_manifest_sha256=pc.digest(Path(package)/'manifest.json'),reference_sha256=plan['reference_sha256'])
    if completion.get('binding')!=binding or pc.digest(out/'raw.json')!=completion.get('raw_sha256'):
        raise ValueError('mixed repair completed receipt belongs to another package/raw')
    raw=json.loads((out/'raw.json').read_text())
    if (raw.get('mixed_repair_binding')!=binding or
        raw.get('identity_sha256')!=sha256_value({k:v for k,v in raw.items() if k!='identity_sha256'})):
        raise ValueError('mixed repair raw identity or reference changed')
    reference,_,path=original_view(manifest)
    if str(path)!=plan['reference_file'] or pc.digest(path)!=plan['reference_sha256']:
        raise ValueError('mixed repair reference changed')
    if any(raw.get(k)!=reference.get(k) for k in ('system','model_id','model_hash','tokenizer_hash','tp','pp')):
        raise ValueError('mixed repair model identity differs')
    rows=raw['mixed']
    if len(rows)!=4 or {key(r) for r in rows}!=EXPECTED or not all(r.get('valid') for r in rows):
        raise ValueError('mixed repair contains missing/extra/invalid points')
    for row in rows:
        path=(out/row['samples_file']).resolve()
        if not path.is_relative_to(out.resolve()) or pc.digest(path)!=row['samples_sha256']:
            raise ValueError('mixed repair sample checksum/path mismatch')
        base=next(r for r in reference['decode'] if (r['freq_mhz'],r['batch'],r['context_tokens'])==(1500,row['batch'],1024))
        pre=next(r for r in reference['prefill'] if (r['freq_mhz'],r['input_tokens'])==(1500,row['chunk_tokens']))
        expected=dict(decode_evidence_source=base.get('evidence_source'),prefill_evidence_source=pre.get('evidence_source'),
            decode_reference_sha256=hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest(),
            prefill_reference_sha256=hashlib.sha256(json.dumps(pre,sort_keys=True).encode()).hexdigest())
        if row.get('reference_binding')!=expected or row['base_step_s']!=base['step_seconds'] or row['alone_prefill_s']!=pre['seconds']:
            raise ValueError('mixed point references differ from measured base rows')
    return completion,raw


def needs_collection(*,package,out,manifest):
    if manifest.get('mixed_repair') is None:return False
    if not (Path(out)/'completion.json').is_file():return True
    return checked_completed(package=package,out=out,manifest=manifest) is None


async def collect_existing(*,profiler,client,gpus,package,out,manifest):
    plan=manifest.get('mixed_repair')
    if plan is None:return dict(requested=False,complete=True)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    before=copy.deepcopy(profiler.raw)
    result=dict(requested=True,complete=False,status='running',formal_eligible=False,energy_comparable=False,
        original_failed_rows_preserved=True,scope=plan['scope'])
    try:
        reference,_,path=original_view(manifest)
        if str(path)!=plan['reference_file'] or pc.digest(path)!=plan['reference_sha256']:
            raise ValueError('mixed repair reference differs from immutable package')
        clone=copy.copy(profiler);clone.out_dir=out;clone.raw=copy.deepcopy(profiler.raw)
        clone.raw.update(prefill=[],decode=[],mixed=[],transfer=[],static={},
            config=dict(clone.raw['config'],power_only_holdout=False,mixed_repair_only=True),
            mixed_repair_binding=dict(package_manifest_sha256=pc.digest(Path(package)/'manifest.json'),
                reference_sha256=plan['reference_sha256']),parent_power_artifact_root=str(Path(profiler.out_dir).resolve()),
            parent_evidence_paths_are_relative_to='parent_power_artifact_root')
        for field in ('parallel_interference','external_interference','concurrency_environment','local_power_isolation',
                      'power_pending','evidence_bindings','identity_sha256'):
            clone.raw.pop(field,None)
        binding=copy.deepcopy(clone.raw['mixed_repair_binding'])
        if (out/'raw.json').is_file():
            clone.resume()
            if clone.raw.get('mixed_repair_binding')!=binding:raise ValueError('mixed repair resume belongs to another reference')
        await clone._mixed(client,gpus,freqs=(1500,),reference_raw=reference)
        clone._checkpoint()
        rows=clone.raw['mixed'];valid=[]
        if len(rows)!=4 or {key(r) for r in rows}!=EXPECTED:
            raise ValueError('mixed collector did not produce exactly the four requested points')
        for row in rows:
            if row.get('valid'):
                sample=(out/row['samples_file']).resolve()
                if not sample.is_relative_to(out.resolve()) or pc.digest(sample)!=row['samples_sha256']:
                    raise ValueError('mixed repair sample checksum/path mismatch')
                valid.append(row)
        result.update(complete=len(valid)==4,status='completed' if len(valid)==4 else 'inconclusive',
            raw_sha256=pc.digest(out/'raw.json'),measured_points=4,valid_points=len(valid),
            failures=[r for r in rows if not r.get('valid')],reference_sha256=plan['reference_sha256'],binding=binding)
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        atomic_json(out/'completion.json',result)
        raise
    finally:
        if profiler.raw!=before:raise ValueError('mixed followup changed protected power archive')
    atomic_json(out/'completion.json',result)
    return dict(result,completion=str(out/'completion.json'),receipt_sha256=pc.digest(out/'completion.json'))


def audit_repair(*,manifest,package,out):
    """Compose a new timing view; the original failed receipt is unchanged."""
    root=Path(out)/'mixed-repair';completion=json.loads((root/'completion.json').read_text())
    if not completion.get('complete'):
        result=dict(passed=False,complete=False,failures=[dict(metric='mixed_repair_incomplete',completion=completion)],
                    original_timing_unchanged=True,formal_eligible=False)
        atomic_json(root/'repaired-timing-audit.json',result)
        return result
    _,fresh=checked_completed(package=package,out=root,manifest=manifest)
    reference,roots,_=original_view(manifest)
    rows=fresh['mixed']
    if len(rows)!=4 or {key(r) for r in rows}!=EXPECTED or not all(r.get('valid') for r in rows):
        raise ValueError('mixed repair contains missing/extra/invalid points')
    view=copy.deepcopy(reference)
    for section in ('prefill','decode'):
        for row in view[section]:row.setdefault('evidence_source','original-completed')
    retained=[r for r in view['mixed'] if key(r) not in EXPECTED]
    if len(retained)!=8 or not all(r.get('valid') for r in retained):
        raise ValueError('eight original valid mixed points are required')
    view['mixed']=[dict(r,evidence_source='original-completed') for r in retained]+[
        dict(copy.deepcopy(r),evidence_source='mixed-repair') for r in rows]
    view.pop('identity_sha256',None);view.pop('new_raw_identity_sha256',None)
    view.update(scope='derived_timing_view_with_four_new_mixed_points',measurement_archive=False,formal_eligible=False)
    roots['mixed-repair']=root
    base=pc.PerfModel.load(manifest['inputs']['base_candidate']['path'])
    original_manifest=json.loads((Path(manifest['original_holdout'])/'frozen-fit.json').read_text())
    audit=pc.timing_component(evaluate_holdout(view,base,Path(out),expected_plan=original_manifest['plan'],evidence_roots=roots))
    atomic_json(root/'combined-timing-view.json',view)
    audit.update(complete=True,old_original_timing_passed=False,original_failed_rows_preserved=True,
        new_mixed_points=4,retained_valid_mixed_points=8,
        combined_view_sha256=pc.digest(root/'combined-timing-view.json'),evidence_roots={k:str(v) for k,v in roots.items()},
        repair_completion_sha256=pc.digest(root/'completion.json'),formal_eligible=False)
    atomic_json(root/'repaired-timing-audit.json',audit)
    return audit
