#!/usr/bin/env python3
"""Single-owner scheduler and CSV publisher for the frozen saturation round."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import runpy
from pathlib import Path
import time

# Activate the hash-bound host analysis before importing any pdblend module.
runpy.run_path(str(Path(__file__).with_name('2026-09-24_host_analysis_bootstrap.py')))['activate'](sys.argv)

from pdblend.bench import comparison_campaign as cc
from pdblend.bench.comparison_hash_cache import UnchangedFileHashes
from pdblend.bench.comparison_jobs import resident_job
from pdblend.bench.resident_session import digest, write_new
from pdblend.bench.single_observation_slo_boundary import read_extension_manifest
from pdblend.experimentation.lease import GPULeaseQueue

ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT / 'results/2026-09-22/three-model/queue.json'
CSV = ROOT / 'results/compare.csv'
TERMINAL = {'succeeded', 'failed', 'cancelled', 'blocked'}


class PriorityHandoff(Exception):
    """The user transferred execution to repairs before a new frozen round."""


def ensure_round_active(package):
    path = Path(package) / 'orchestration/priority-coordination.json'
    if path.is_file():
        record = read(path)
        if record.get('status') in {'awaiting_priority_clarification', 'user_confirmed_repairs_first'}:
            raise PriorityHandoff(f'round scheduling held by {path}')


def read(path):
    return json.loads(Path(path).read_text())


def module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem.replace('-', '_'), path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def preserve_published_metrics(previous, current):
    """A publication can add observations but cannot rewrite measured history."""
    by_receipt = {}
    for row in current:
        key = row.get('receipt_sha256')
        if key:
            values = {k: row.get(k, '') for k in cc.METRIC_FIELDS}
            if key in by_receipt and by_receipt[key] != values:
                raise ValueError('one receipt has conflicting metric rows')
            by_receipt[key] = values
    for row in previous:
        key = row.get('receipt_sha256')
        if not key:
            continue
        if key not in by_receipt:
            raise ValueError('publication would drop a historical receipt: ' + key)
        if by_receipt[key] != {k: row.get(k, '') for k in cc.METRIC_FIELDS}:
            raise ValueError('publication would change historical metrics: ' + key)


class Runner:
    def __init__(self, package, old_launch, *, defer_cold_export_until_idle=False):
        self.package = Path(package).resolve()
        ensure_round_active(self.package)
        self.out = self.package / 'orchestration'; self.out.mkdir(exist_ok=True)
        self.lock = (self.out / 'owner.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.campaign = self.package / 'campaign.json'
        self.queue = GPULeaseQueue(QUEUE)
        self.roots = []
        argv = read(old_launch)['argv']
        for i, arg in enumerate(argv):
            if arg == '--sessions':
                self.roots.append(Path(argv[i+1]))
        self.jobs = read(self.package / 'jobs.json')
        self.state = dict(run_id=self.package.name, started_s=time.time(), phase='starting',
                          model=None, jobs=[], current_campaign=str(self.campaign))
        checkpoint = self.out / 'status.json'
        if checkpoint.is_file():
            previous = read(checkpoint)
            if previous.get('run_id') != self.package.name:
                raise ValueError('orchestration checkpoint belongs to another round')
            self.state.update(previous)
            self.campaign = Path(self.state['current_campaign'])
            read(self.campaign)
        self.state.setdefault('completed_models', [])
        self.state.setdefault('baseline_campaigns', {})
        self.state.setdefault('extension_manifests', [])
        self.fingerprint = None
        self.defer_cold_export_until_idle = defer_cold_export_until_idle
        self.cold_export_pending = True
        self.manifests = self.state['extension_manifests']
        self.historical_manifests = read(self.campaign).get('historical_extension_manifests', [])
        cc._WATCH_DIGEST_CACHE = UnchangedFileHashes()
        self.baselines = module(ROOT / 'scripts/2026-09-24_prepare_boundary_baselines.py')
        registry_path = self.out / 'historical-baseline-registry.json'
        self.baseline_registry_ref = cc.binding(registry_path) if registry_path.is_file() else None
        self.events = self.out / 'events.jsonl'
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name) == os.getpid():
                continue
            try:
                cmd = proc.joinpath('cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if b'pdblend.bench.comparison_campaign' in cmd and str(CSV).encode() in cmd:
                raise RuntimeError('another comparison CSV writer is still active')

    def emit(self, event, **fields):
        row = dict(event=event, at_s=time.time(), **fields)
        with self.events.open('a') as stream:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
        self.state.update(updated_s=time.time(), current_campaign=str(self.campaign))
        path = self.out / 'status.tmp'
        path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + '\n')
        os.replace(path, self.out / 'status.json')
        print(json.dumps(row, sort_keys=True), flush=True)

    def enqueue(self, job):
        ensure_round_active(self.package)
        self.queue.enqueue(job['job_id'], job['payload'], priority=job['priority'],
                           max_attempts=job['max_attempts'])
        if job['job_id'] not in self.state['jobs']:
            self.state['jobs'].append(job['job_id'])
        self.emit('enqueued', job_id=job['job_id'], system=job['payload'].get('system'))

    def publish(self, *, force=False):
        state = read(QUEUE)
        if self.cold_export_pending and self.defer_cold_export_until_idle:
            active = [lease['job_id'] for lease in state['leases'].values()
                      if lease.get('status') == 'active']
            if active:
                if not self.state.get('csv_cold_export_deferred'):
                    self.state['csv_cold_export_deferred'] = True
                    self.emit('cold_export_deferred_until_lease_release', jobs=active)
                return
        inputs = cc.comparison_watch_inputs(self.campaign, state, session_roots=self.roots)
        roots = inputs['session_roots']
        paths = {Path(inputs['campaign'])}
        for root in roots:
            root = Path(root)
            for pattern in ('**/windows/*/receipt.json', '**/completion.json', '**/extensions/latest.json'):
                paths.update(root.glob(pattern))
        fingerprint = tuple((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(paths))
        if not force and fingerprint == self.fingerprint:
            return
        old_bytes = CSV.read_bytes() if CSV.is_file() else None
        previous = list(csv.DictReader(old_bytes.decode().splitlines())) if old_bytes else []
        pending = self.out / 'compare.pending.csv'
        # History is for publication only; active continuation never consumes it.
        manifests = list(self.historical_manifests)
        manifests.extend(ref for ref in self.manifests if ref not in manifests)
        result = cc.export(inputs['campaign'], pending, session_roots=roots,
            analysis_policy='all_recorded_windows/v1', extension_manifests=manifests)
        if self.state.get('explicit_rejection_sidecars'):
            module(ROOT / 'scripts/2026-09-25_apply_explicit_rejection_boundaries.py').annotate_temporary_csv(
                pending, self.state['explicit_rejection_sidecars'])
        with pending.open() as stream:
            preserve_published_metrics(previous, list(csv.DictReader(stream)))
        if (CSV.read_bytes() if CSV.is_file() else None) != old_bytes:
            raise RuntimeError('another writer changed the comparison CSV during export')
        os.replace(pending, CSV)
        result['csv_sha256'] = hashlib.sha256(CSV.read_bytes()).hexdigest()
        self.cold_export_pending = False
        self.state.pop('csv_cold_export_deferred', None)
        self.fingerprint = fingerprint
        self.emit('csv_published', result=result)

    def wait(self, ids):
        prior = None
        while True:
            ensure_round_active(self.package)
            queue = read(QUEUE)
            statuses = {i: queue['jobs'][i]['status'] for i in ids}
            self.publish()
            if statuses != prior:
                self.emit('job_status', statuses=statuses); prior = statuses
            if all(s in TERMINAL for s in statuses.values()):
                return queue
            time.sleep(10)

    def run_baseline(self, job):
        return module(ROOT / 'scripts/2026-09-24_baseline_unstarted_recovery.py').run_baseline(self, job)

    def pd(self, original):
        job = original
        argv = original['payload']['argv']
        attempts = [Path(argv[i+1]) for i, value in enumerate(argv) if value == '--previous']
        while True:
            self.enqueue(job)
            queue = self.wait([job['job_id']])
            entry = queue['jobs'][job['job_id']]
            lease = queue['leases'].get(entry.get('lease_id'), {})
            if not lease:
                leases = [v for v in queue['leases'].values() if v['job_id'] == job['job_id']]
                lease = max(leases, key=lambda v: v['claimed_at'], default={})
            session = Path(lease['attempt_dir']) / 'session' if lease else None
            if session is None or not (session / 'completion.json').is_file():
                self.emit('pd_session_failed_without_completion', job_id=job['job_id'])
                return None
            attempts.append(session)
            report = read(session / 'completion.json')
            manifest_ref = report.get('extension_manifest')
            if manifest_ref and manifest_ref not in self.manifests:
                self.manifests.append(manifest_ref)
                self.emit('pd_manifest_recorded', job_id=job['job_id'], manifest=manifest_ref)
            if not report.get('continuation_required') or not report.get('complete'):
                return manifest_ref
            # Same source/profile/policy; previous completed receipts are skipped.
            # Only the new immutable lease identity and resume roots change.
            group_path = Path(original['payload']['argv'][original['payload']['argv'].index('--group')+1])
            group = read(group_path)
            group['session_id'] += '-continuation-' + str(len(attempts))
            path = self.out / 'continuations' / (group['session_id'] + '.json')
            if path.is_file():
                if read(path) != group:
                    raise ValueError('continuation identity changed on resume')
            else:
                write_new(path, group)
            execution = cc.load_bound(read(self.package / 'campaign.json')['execution_inputs'])
            job = resident_job(group, path, root=ROOT, source=execution['source'], image=execution['image_digest'],
                verification=execution['model_verification']['path'], campaign=self.campaign, priority=3000)
            job['payload'].update(system='pdblend', model_id=group['model_id'], run_id=self.package.name,
                observation_scope='pdblend_profile_unqualified_evaluation/v1', result_policy='all_recorded_windows/v1')
            for previous in attempts:
                job['payload']['argv'] += ['--previous', str(previous)]
            self.emit('pd_continuation_prepared', previous_sessions=[str(p) for p in attempts])

    def selection(self, model, manifest_ref, *, generation=None, supersedes=None):
        datasets = {d: dict(lower=None, upper=None, status='session_execution_obstruction')
                    for d in ('alpaca', 'sharegpt', 'longbench')}
        if manifest_ref:
            value = read_extension_manifest(manifest_ref)['manifest']
            if value['model_id'] != model or value['run_id'] != self.package.name:
                raise ValueError('completed boundary belongs to another frozen round')
            for dataset, state in value['boundaries'].items():
                # A nonmonotonic series still gets its observed endpoint pair;
                # it explicitly does not establish one unique capacity bound.
                accepted = state.get('status') == 'bracketed'
                datasets[dataset] = dict(state,
                    unique_boundary=accepted and not state.get('nonmonotonic'),
                    lower=state.get('passed_lower_point') if accepted else None,
                    upper=state.get('failed_upper_point') if accepted else None)
        suffix = '' if generation is None else '-v' + str(generation)
        path = self.out / ('boundary-selection-' + model.split('-')[1].lower() + suffix + '.json')
        value = dict(schema='observed-boundary-selection/v1', run_id=self.package.name,
            model_id=model, manifest=manifest_ref, datasets=datasets,
            baseline_outcomes_used_for_selection=False)
        if supersedes is not None:
            cc.load_bound(supersedes)
            value['supersedes_selection'] = supersedes
        if self.state.get('explicit_rejection_sidecars'):
            value = module(ROOT / 'scripts/2026-09-25_apply_explicit_rejection_boundaries.py').apply_selection(
                value, self.state['explicit_rejection_sidecars'], cc.load_bound(cc.binding(self.campaign)))
        if path.exists():
            if read(path) != value:
                raise ValueError('published boundary selection changed')
        else:
            write_new(path, value)
        return path

    def prepare_baselines(self, model, selection):
        def completed(out):
            marker = read(out / 'prepared.json')
            selected = cc.load_bound(marker['selection'])
            if selected != read(selection):
                raise ValueError('prepared baseline selection changed')
            return cc.load_bound(marker['campaign']), cc.load_bound(marker['jobs'])

        recorded = self.state['baseline_campaigns'].get(model)
        if recorded:
            out = Path(recorded).parent
            prepared, jobs = completed(out)
            return out, prepared, jobs
        base = self.package / ('baselines-' + model.split('-')[1].lower())
        out, recovery = base, 0
        while out.exists():
            if (out / 'prepared.json').is_file():
                prepared, jobs = completed(out)
                break
            # Preserve the partial preparation for diagnosis. Its absolute
            # artifact bindings cannot be repaired by renaming the directory.
            self.emit('partial_baseline_preparation_preserved', path=str(out))
            recovery += 1
            out = base.with_name(base.name + f'-recovery-{recovery:04d}')
        else:
            options = {}
            if getattr(self, 'baseline_registry_ref', None) is not None:
                options['historical_baseline_registry'] = self.baseline_registry_ref
            prepared, jobs = self.baselines.prepare(self.campaign, selection, out, **options)
        self.state['baseline_campaigns'][model] = str(out / 'campaign.json')
        return out, prepared, jobs

    def audit_results(self):
        verifier = module(ROOT / 'scripts/2026-09-24_verify_baseline_completion.py')
        auditor = module(ROOT / 'scripts/2026-09-24_audit_saturation_completion.py')
        with CSV.open() as stream:
            receipts = sorted({row['receipt_path'] for row in csv.DictReader(stream)
                               if row.get('receipt_path') and row['system'] != 'pdblend'})
        stamp = str(time.time_ns())
        gaps = self.out / ('baseline-completion-' + stamp + '.json')
        write_new(gaps, verifier.verify(self.package / 'campaign.json', receipts=receipts))
        report = auditor.audit(self.package, CSV, gap_review=cc.binding(gaps))
        path = self.out / ('completion-audit-' + stamp + '.json')
        write_new(path, report)
        self.state['completion_audit'] = cc.binding(path)
        return report

    def run(self):
        # A manually started recovery can already have a new sibling manifest
        # before this scheduler is restarted. Authorize its branch before the
        # first export, retaining a later combined checkpoint if one exists.
        for model, ref in self.state.get('execution_recovery_plans', {}).items():
            progress = self.state.get('execution_recovery_state', {}).get(ref['sha256'], {})
            if progress.get('complete'):
                continue
            plan = cc.load_bound(ref)
            if plan.get('run_id') != self.package.name or plan.get('model_id') != model:
                raise ValueError('recovery publication belongs to another round')
            if progress.get('combined_prepared'):
                marker = cc.load_bound(progress['combined_prepared'])
                self.campaign = Path(marker['campaign']['path'])
            else:
                self.campaign = Path(plan['recovery_campaign']['path'])
            break
        self.publish(force=True)
        for job in self.jobs:
            model = job['payload']['model_id']
            recovery_ref = self.state.get('execution_recovery_plans', {}).get(model)
            if recovery_ref:
                recovery = module(ROOT / 'scripts/2026-09-24_pd_execution_recovery.py')
                recovery.recover_model(self, recovery_ref)
                continue
            if model in self.state['completed_models']:
                continue
            self.state.update(model=model, phase='pdblend_original_and_boundary')
            self.emit('model_started', model_id=model)
            if model in self.state['baseline_campaigns']:
                selection = self.out / ('boundary-selection-' + model.split('-')[1].lower() + '.json')
                read(selection)
            else:
                manifest = self.pd(job)
                selection = self.selection(model, manifest)
            self.state['phase'] = 'baseline_supplements_and_endpoints'
            out, prepared, jobs = self.prepare_baselines(model, selection)
            self.campaign = out / 'campaign.json'
            self.emit('baselines_prepared', model_id=model, windows=sum(len(g['points']) for g in prepared['groups']),
                      groups=len(jobs), selection=str(selection))
            for baseline in jobs:
                override = self.state.get('baseline_recovery_jobs', {}).get(baseline['job_id'])
                if override:
                    baseline = cc.load_bound(override)
                self.run_baseline(baseline)
            self.publish(force=True)
            self.state['completed_models'].append(model)
            self.emit('model_complete', model_id=model)
        self.publish(force=True)
        report = self.audit_results()
        self.state['phase'] = ('complete_with_obstructions' if report['has_obstructions'] else 'complete') if report[
            'all_scope_accounted'] else 'needs_attention'
        self.state['finished_s'] = time.time()
        self.emit('round_execution_finished', csv=str(CSV), jobs=self.state['jobs'],
                  completion_audit=self.state.get('completion_audit'), phase=self.state['phase'])
        if not (self.out / 'worker.stop').exists():
            write_new(self.out / 'worker.stop', dict(reason='all scheduled matrix leases finished', at_s=time.time()))
        scope_path = QUEUE.parent / 'active-execution-scope.json'
        if scope_path.is_file():
            scope = read(scope_path)
            if scope.get('allowed_run_ids') == [self.package.name]:
                scope.update(status='released', released_s=time.time(), final_phase=self.state['phase'])
                scope_path.write_text(json.dumps(scope, indent=2, sort_keys=True) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--previous-export-launch', type=Path, required=True)
    parser.add_argument('--defer-cold-export-until-idle', action='store_true', default=True,
                        help='Cold evidence rehash is deferred until all active GPU leases end (default).')
    args = parser.parse_args()
    runner = None
    try:
        runner = Runner(args.package, args.previous_export_launch,
                        defer_cold_export_until_idle=args.defer_cold_export_until_idle)
        runner.run()
    except PriorityHandoff as exc:
        if runner is not None:
            runner.state['phase'] = 'priority_coordination'
            runner.emit('priority_handoff_observed', reason=str(exc))
        else:
            print(json.dumps(dict(event='priority_handoff_observed', reason=str(exc))), flush=True)
    except Exception as exc:
        if runner is not None:
            runner.state['phase'] = 'needs_attention'
            runner.emit('orchestration_error', error=f'{type(exc).__name__}: {exc}')
        raise


if __name__ == '__main__':
    main()
