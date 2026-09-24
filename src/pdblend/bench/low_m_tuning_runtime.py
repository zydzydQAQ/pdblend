"""GPU execution entry for independent low-M tuning on a root-owned lease.

The evaluation dispatcher and its seed-701 guards are deliberately untouched.
This adapter shares only native startup/reset/drain and the raw meter with it.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import json
from pathlib import Path

from .capacity_floor_v2 import validate_manifest, validate_trial
from .comparison_acceptance import _need, _bound
from .comparison_campaign import binding
from .comparison_runtime import NativeResidentAdapter
from .comparison_metrics import canonical_outcomes, reduce_comparison
from .comparison_metering import summarize_comparison
from .low_m_tuning import TrialPlanner, source_inventory
from .resident_session import digest, write_new


class LowMTuningAdapter(NativeResidentAdapter):
    """A dedicated experiment adapter; cannot consume evaluation point groups."""
    async def start(self, group):
        _need(group.get('scope') == 'independent_low_m_tuning/v2' and 'points' not in group,
              'dedicated tuning group required; evaluation groups are not accepted')
        _bound(group['tuning_manifest'])
        manifest = validate_manifest(group['tuning_manifest']['path'])
        _need(digest(source_inventory()) == manifest['context']['algorithm_source_sha256'],
              'tuning algorithm changed after manifest freeze')
        _need(group['model_id'] == manifest['identity']['model_id'], 'tuning group model differs')
        from pdblend.profile.query.versions import load_profile
        loaded = load_profile(manifest['profile']['path'], system='pdblend',model_id=group['model_id'],
                              tp=manifest['identity']['tp'],pp=manifest['identity']['pp'],usage='development')
        from pdblend.planner.transitions import identity
        _need(identity(loaded.model) == manifest['identity'], 'tuning selected profile differs')
        self.tuning_manifest, self.tuning_manifest_ref, self.tuning_model = manifest,group['tuning_manifest'],loaded.model
        # Empty *observation* list requests shared native startup only. All tuning
        # trace/candidate checks above replace no evaluation check or receipt.
        result = await super().start(dict(group, points=[]))
        self.tuning_group = group
        return result

    async def register_point(self, point):
        raise ValueError('tuning uses frozen manifest trial IDs, never generated evaluation points')

    async def execute_trial(self, trial_id, out):
        from .client import Request
        from .run import _point
        from pdblend.online.policies import get_policy
        from pdblend.planner.pool import Plan, SLO
        manifest = self.tuning_manifest
        _need(digest(source_inventory()) == manifest['context']['algorithm_source_sha256'],
              'tuning runtime source changed before execution')
        matches = [trial for trial in manifest['trials'] if trial['id'] == trial_id]
        _need(len(matches) == 1, 'unknown tuning trial')
        trial = matches[0]; trace = _bound(trial['trace']); out = Path(out)
        out.mkdir(parents=True,exist_ok=False)
        reset = await self.reset(dict(system='pdblend',name=trial_id))
        write_new(out/'reset.json',reset)
        requests = [Request(**row) for row in trace['requests']]
        slots = len(self.specs); ceiling = manifest['recovery_policy']['safety_max_freq']
        # Startup and recovery use the canonical four M instances. The explicit
        # trial layout can begin only after real requests validate startup.
        generations = {spec.generation for spec in self.specs}
        _need(len(generations)==1,'tuning startup native generations differ')
        initial = Plan(**manifest['startup_plan'],power_w=float('inf'),ttft_s=float('inf'),tpot_s=float('inf'),
            tp=manifest['identity']['tp'],pp=manifest['identity']['pp'],generation=next(iter(generations)),
            pool_id=getattr(self.specs[0],'pool_id',''),
            profile_key=json.dumps(manifest['identity']['profile_key'],sort_keys=True,separators=(',',':')))
        self.pdblend_boundary.begin_window()
        raw = await _point(self.fleet,self.gpus,self.tuning_model,get_policy('pdblend'),SLO(**trial['slo']),
            requests,[],out,self.base_port+80,10.,300.,sampling_seed=trial['seed'],initial_plan=initial,
            planning_trace=requests,observation_duration_s=150.,comparison_record_tokens=True,
            comparison_wait_initial_plan=True,pdblend_runtime=manifest['recovery_policy'],
            planner_factory=lambda model,cfg:TrialPlanner(model,cfg,trial),
            tuning_scope='independent_low_m_tuning/v2')
        self.last_drain = await self.pdblend_boundary.finish_window(raw)
        write_new(out/'native-result.json',raw); write_new(out/'native-drain.json',self.last_drain)
        await asyncio.sleep(.3)
        from pdblend.results.power_archive import write_power_archive
        snapshot = self.monitor.snapshot()
        write_power_archive(out/'power-snapshot.json',snapshot)
        meter = summarize_comparison(snapshot,gpu_uuids=self.identity['fleet_gpu_uuids'],
            origin_s=raw['service_started_s'],tail_end_s=self.last_drain['tail_end_s'],duration_s=150.)
        write_new(out/'comparison-metering.json',meter)
        outcomes = _bound(binding(out/'outcomes.jsonl'),journal=True)
        canonical = canonical_outcomes('pdblend',trace,outcomes,service_started_s=raw['service_started_s'])
        metrics = reduce_comparison(trace,canonical,service_started_s=raw['service_started_s'],duration_s=150.,
                                     slo=(trial['slo']['ttft_s'],trial['slo']['tpot_s']))
        write_new(out/'comparison-requests.json',metrics.pop('request_metrics'))
        write_new(out/'comparison-metrics.json',metrics)
        names = dict(requests='comparison-requests.json',outcomes='outcomes.jsonl',controller='controller.jsonl',
            native_result='native-result.json',drain='native-drain.json',native_cleanup='native-cleanup.json',
            metering='comparison-metering.json',power='power-snapshot.json',frequencies='freq.jsonl',
            frequency_readings='frequency-readings.jsonl',reset='reset.json')
        names.update(routes='routes.jsonl',transition_measurements='transition-measurements.json')
        receipt = dict(kind='pdblend_low_m_trial_v2',trial_id=trial_id,manifest=self.tuning_manifest_ref,
            executed_context=manifest['context'],actual_plan=trial['plan'],engine_identity=self.identity,
            artifacts=dict({name:binding(out/filename) for name,filename in names.items()},
                           startup=binding(self.out/'qualification.json')),hardware_executed=True,
            selection_split='tuning',evaluation_used_for_selection=False,formal_eligible=False)
        write_new(out/'trial-receipt.json',receipt)
        try:
            verdict = dict(accepted=True,metrics=validate_trial(out/'trial-receipt.json',manifest=manifest,
                                manifest_ref=self.tuning_manifest_ref))
        except (ValueError,KeyError,TypeError) as exc:
            verdict = dict(accepted=False,error=str(exc))
        write_new(out/'verdict.json',verdict)
        return verdict


async def run_group(group_path, output, *, base_port=8700, trial_ids=None):
    """Called only by the queue owner inside its existing exclusive GPU lease."""
    group = _bound(binding(Path(group_path))); output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    adapter = LowMTuningAdapter(output,base_port=base_port)
    results, failure, cleanup, interrupted, ids = {}, None, None, False, []
    try:
        await adapter.start(group)
        ids = trial_ids or [trial['id'] for trial in adapter.tuning_manifest['trials']]
        for trial_id in ids:
            # A root-owned sentinel is observed only between complete trials.
            # Every prior trial has already persisted its terminal cohort,
            # native drain, energy, raw receipt and acceptance verdict.
            if (output/'stop-after-window').exists():
                interrupted = True
                break
            results[trial_id] = await adapter.execute_trial(trial_id,output/'trials'/trial_id)
    except BaseException as exc:
        failure = exc
    finally:
        try:
            cleanup = await adapter.close()
        except BaseException as exc:
            cleanup = dict(passed=False,process_cleanup_verified=False,errors=[repr(exc)])
            if failure is None: failure = exc
        write_new(output/'cleanup.json',cleanup)
        complete = failure is None and not interrupted and cleanup.get('passed') is True
        status = 'passed' if complete else 'interrupted' if interrupted and failure is None else 'failed'
        write_new(output/'completion.json',dict(scope=group['scope'],status=status,
            complete=complete,results=results,cleanup=cleanup,cleanup_receipt=binding(output/'cleanup.json'),
            group=binding(Path(group_path)),remaining_trial_ids=[trial_id for trial_id in ids if trial_id not in results],
            continuation_required=bool(interrupted),
            all_trials_qualified=complete and bool(results) and all(row.get('accepted') is True for row in results.values()),
            error=None if failure is None else repr(failure),formal_eligible=False))
    if failure is not None: raise failure
    if cleanup.get('passed') is not True: raise RuntimeError('tuning native session cleanup failed')
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--base-port',type=int,default=8700)
    parser.add_argument('--trial-ids',nargs='+')
    args = parser.parse_args()
    results = asyncio.run(run_group(args.group,args.output,base_port=args.base_port,trial_ids=args.trial_ids))
    print(json.dumps(dict(trials=len(results),accepted=sum(row['accepted'] for row in results.values()))))


if __name__ == '__main__':
    main()
