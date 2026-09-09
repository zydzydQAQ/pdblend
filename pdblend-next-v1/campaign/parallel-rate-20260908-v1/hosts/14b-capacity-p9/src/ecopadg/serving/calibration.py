"""Independent baseline capacity bracketing on explicitly prepared instances.

Probe and confirmation requests come only from the calibration split. A rate
is certified only after a separate full calibration-set confirmation and an
observed infeasible upper bracket. No PDBlend result chooses the baseline rate.
"""
import argparse
import asyncio
from dataclasses import dataclass,field
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

from .campaign import node_lease
from .cell import run_cell
from .datasets import make_trace
from .evidence import REQUIRED_MECHANISMS,freeze_files


def implementation_sources():
    """Inventory execution and measurement dependencies, including new files."""
    root=Path(__file__).resolve()
    paths=set(root.parents[1].rglob('*.py'))
    paths.add(root.parents[4]/'benchmarks/scripts/bench_vllm.py')
    fork=root.parents[4]/'vllm-pd-fork/vllm'
    if fork.is_dir(): paths.update(fork.rglob('*.py'))
    return paths


class InvalidCalibrationEvidence(ValueError):
    """A measurement failure is not evidence of a workload capacity boundary."""


def verified_admission_boundary(summary):
    """Only the measurement layer can certify explicit zero-token rejections.

    This can establish an upper delivery boundary, never a feasible rate or an
    equal-work energy result. Unknown failures remain invalid measurements.
    """
    completed=summary.get('completed');rejected=summary.get('admission_rejections')
    expected=summary.get('n_expected')
    return bool(summary.get('validity')=='invalid_work' and summary.get('capacity_observation_valid') is True
        and summary.get('gpu_count')==8 and summary.get('power_mode')=='instant'
        and summary.get('power_source_verified') is True and not summary.get('runtime_error')
        and not summary.get('sampling_error')
        and all(type(v) is int for v in (completed,rejected,expected))
        and completed>=0 and rejected>0 and completed+rejected==expected)


def capacity_observation_error(summary, *, allow_admission_rejection=False):
    rejection=allow_admission_rejection and verified_admission_boundary(summary)
    if summary.get('validity') != 'ok' and not rejection:
        return 'invalid measurement: '+str(summary.get('validity'))
    if summary.get('runtime_error') or summary.get('sampling_error'):
        return 'runtime or power sampling failure'
    completed, expected = summary.get('completed'), summary.get('n_expected')
    if not rejection and (not isinstance(completed, int) or isinstance(completed, bool) or completed <= 0
            or completed != expected):
        return 'incomplete or inconsistent output work'
    if not rejection and ('generated_tokens' in summary or 'expected_generated_tokens' in summary) and (
            summary.get('generated_tokens') != summary.get('expected_generated_tokens')):
        return 'generated output work differs from the requested work'
    attainment = summary.get('slo_attainment')
    if (not isinstance(attainment, (int, float)) or isinstance(attainment, bool)
            or not math.isfinite(attainment) or not 0 <= attainment <= 1):
        return 'invalid SLO attainment measurement'
    return None


@dataclass
class RateSearch:
    initial_rate: float
    max_trials: int=7
    target: float=.99
    lower: float=0.
    upper: float|None=None
    observations: list=field(default_factory=list)
    failure: str|None=None

    def __post_init__(self):
        if not math.isfinite(self.initial_rate) or self.initial_rate<=0 or self.max_trials<1 or not 0<self.target<=1:
            raise ValueError('finite positive rate, trial count and SLO target required')

    def record(self,rate,summary):
        if self.failure: raise InvalidCalibrationEvidence(self.failure)
        error=capacity_observation_error(summary,allow_admission_rejection=True)
        if error:
            from .measurement import finite_json
            self.failure=error
            self.observations.append(dict(rate=rate,feasible=None,measurement_valid=False,
                failure=error,summary=finite_json(summary)))
            raise InvalidCalibrationEvidence(error)
        rejected=verified_admission_boundary(summary)
        feasible=not rejected and summary['slo_attainment']>=self.target
        self.observations.append(dict(rate=rate,feasible=feasible,measurement_valid=True,
            boundary_kind='explicit_admission_rejection' if rejected else 'complete_work_slo',summary=summary))
        if feasible: self.lower=max(self.lower,rate)
        else:
            self.upper=rate if self.upper is None else min(self.upper,rate)
            # A longer confirmation can invalidate a short probe's rate.
            self.lower=max((o['rate'] for o in self.observations
                            if o['feasible'] and o['rate']<self.upper),default=0.)
        return feasible

    def next_rate(self):
        if self.failure: raise InvalidCalibrationEvidence(self.failure)
        if len(self.observations)>=self.max_trials: return None
        if not self.observations: return self.initial_rate
        if self.upper is None: return self.lower*2
        if not self.lower: return self.upper/2
        if self.upper<=self.lower*1.25: return None
        return (self.lower+self.upper)/2


async def calibrate(args, *, before_cell=None):
    manifest=json.loads(args.manifest.read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    source=freeze_files(implementation_sources())
    (args.out/'source.before.json').write_text(json.dumps(source,indent=2))
    results=[];failure=None
    for entry in manifest['entries']:
        dataset=entry['dataset'];config_path=Path(entry['config'])
        config=json.loads(config_path.read_text());system=config['strategy']
        if system not in REQUIRED_MECHANISMS: raise ValueError('capacity calibration requires an independent baseline')
        corpus=json.loads((Path(manifest['corpus'])/(dataset+'.json')).read_text())
        records=corpus['calibration']
        if len(records)<128: raise ValueError('confirmation requires at least 128 independent calibration examples')
        shuffled=list(records);random.Random(11).shuffle(shuffled)
        search=RateSearch(entry['initial_rate'],manifest.get('max_trials',7),manifest.get('target',.99))
        prefix=f'{system}-{dataset}'
        async def cell(rate,confirm,index):
            selected=records if confirm else shuffled[:manifest.get('probe_requests',64)]
            seed=22 if confirm else 11
            trace=make_trace(selected,rate,seed,dataset=dataset,split='calibration',load='capacity')
            name=prefix+('-confirm-' if confirm else '-probe-')+str(index)
            path=args.out/(name+'.trace.json');path.write_text(json.dumps(trace))
            options=SimpleNamespace(config=config_path,strategy=None,trace=path,out=args.out/name,
                split='calibration',dataset=dataset,load='capacity',seed=seed,freeze=None,mechanisms=None,
                timeout=manifest.get('request_timeout_s',120),slo_ttft_s=None,slo_tpot_s=None)
            # Optional between-cell preparation restores an explicit physical
            # layout after a dynamic baseline changed its instances. It is
            # outside run_cell's serving window; in-cell changes stay inside.
            if before_cell is not None:
                await before_cell(options)
            summary=await run_cell(options)
            return dict(summary,rate=rate,trace=str(path),artifact=str(options.out/'summary.json'))
        confirmed=None
        try:
            while (rate:=search.next_rate()) is not None:
                search.record(rate,await cell(rate,False,len(search.observations)))
            if search.lower and search.upper is not None and search.upper>search.lower:
                for attempt in range(2):
                    rate=search.lower if attempt==0 else search.upper*.8
                    result=await cell(rate,True,attempt)
                    if search.record(rate,result): confirmed=result;break
        except BaseException as exc:
            failure=exc
        upper_observation=next((o['summary'] for o in search.observations
            if o['rate']==search.upper and o.get('measurement_valid') and o['feasible'] is False),None)
        result=dict(system=system,dataset=dataset,config=str(config_path),
            passed=failure is None and confirmed is not None,capacity_rps=confirmed['rate'] if confirmed and failure is None else None,
            infeasible_upper_rps=search.upper,confirmation=confirmed,observations=search.observations,
            infeasible_upper_observation=upper_observation,
            infeasible_upper_reason=('admission_rejection' if verified_admission_boundary(upper_observation)
                else 'joint_slo_attainment') if upper_observation else None,
            scope='single explicitly prepared physical layout; other TP/layout candidates need separate calibration')
        if failure is not None: result['failure']=repr(failure)
        results.append(result)
        (args.out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
        if failure is not None: break
    unchanged=source==freeze_files(implementation_sources())
    if not unchanged:
        for result in results:
            result['passed']=False
            result['invalid_reason']='implementation changed during capacity calibration'
        (args.out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    complete=dict(passed=failure is None and unchanged and all(r['passed'] for r in results),source_unchanged=unchanged,
                  results=results,manifest=manifest,
                  source_scope='ecopadg, shared benchmark client and available vLLM fork; inventory and hashes')
    if failure is not None: complete['failure']=repr(failure)
    (args.out/'summary.json').write_text(json.dumps(complete,indent=2,allow_nan=False))
    if failure is not None: raise failure
    return complete


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    with node_lease(): asyncio.run(calibrate(args))


if __name__=='__main__': main()
