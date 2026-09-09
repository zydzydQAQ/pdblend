"""Continue only a proved clean boundary after a pre-GPU observer-type mismatch.

No signal or experiment is repeated. The failed coordinator and its status are
retained. The original measurement/restoration code is imported unchanged.
"""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import os
import time
import priority as p

ROOT = Path(__file__).resolve().parent
PAUSE = ROOT / 'attempt-001'
OLD_MANIFEST_SHA = 'a8f7dc8231ce88751bf22513534cd54113ee5277a7f22c258fc8b85d483d97e8'


def inspect_boundary():
    p.verify(ROOT / 'priority-manifest.json', OLD_MANIFEST_SHA)
    old = p.read(PAUSE / 'status.json')
    p.require(old['phase'] == 'failed' and old['hardware_started'] is False
              and old['error'] == "RuntimeError('scale was not a clean intentional boundary stop')",
              'only the pre-GPU observer mismatch can be continued')
    p.require(not (PAUSE / 'restore-pdb').exists() and not (PAUSE / 'repeats').exists(), 'hardware stage was attempted')
    for pid in (727215, 667430, 584282, 523564):
        p.require(not (p.process(pid) or {}).get('alive'), 'old process remains alive')
    scale, handoff = p.read(p.SCALE / 'status.json'), p.read(p.HANDOFF / 'status.json')
    intent = p.read(PAUSE / 'boundary-stop-intent.json')
    p.require(intent['supervisor']['pid'] == scale['pid'] == 584282
              and intent['handoff']['pid'] == handoff['pid'] == 523564
              and intent['direct_child_signals'] is False, 'boundary signal attribution differs')
    p.require(scale['phase'] == 'stopped' and not scale['complete']
              and scale.get('error') == "ValueError('scale STOP: no successor')", 'not the exact clean boundary stop')
    p.require(handoff['phase'] == 'stopped' and not handoff['complete'] and handoff['scale_exitcode'] == 1
              and handoff.get('scale_boundary_stop_sent_s') >= intent['issued_s'], 'handoff terminal differs')
    for step in scale['steps']:
        p.require(step.get('complete') and step.get('exitcode') == 0 and step.get('verified_new_checkpoint')
                  and not step.get('deadline_interrupt_s') and not step.get('unconfirmed_child'), 'failed scale child')
        p.require(not (p.process(step['pid']) or {}).get('alive'), 'scale child not exited')
    p.require(scale['steps'][-1]['pid'] == intent['current_step']['pid'], 'another baseline cell started after stop')
    wrapper = p.load(p.V1 / 'B_watch.py', 'boundary_resume_old_wrapper')
    execution, runner, _ = wrapper.original_modules()
    contract = p.load(p.C / 'scale-only-continuation-B32B-v1/contract.py', 'boundary_resume_scale_contract')
    checked = contract.check_spec(p.read(p.SPEC), p.RELEASE, p.RELEASE_SHA)
    groups = checked['groups']
    p.require(all(g['group']['system'] == 'ecoserve' or not g['pending'] for g in groups), 'non-Eco pending work')
    eco = next(g for g in groups if g['group']['system'] == 'ecoserve')
    proof = dict(schema='B-clean-stopped-scale-prefix-v2', observed_s=time.time(), complete_prefix=True,
        whole_baseline192_complete=False, original_spec=p.ref(p.SPEC), original_release=p.ref(p.RELEASE),
        scale_status=p.ref(p.SCALE / 'status.json'), handoff_status=p.ref(p.HANDOFF / 'status.json'),
        boundary_intent=p.ref(PAUSE / 'boundary-stop-intent.json'), all_children_clean=True,
        previous_pre_gpu_observer=p.ref(PAUSE / 'status.json'),
        groups=[dict(system=g['group']['system'], binding=g['group'].get('scale_binding'),
                     completed=g['reused'], pending_ids=[r['cell_id'] for r in g['pending']]) for g in groups],
        previous_baseline_binding=eco['group']['scale_binding'])
    return execution, runner, eco['current'], proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    def verify():
        p.require(p.sha(args.manifest) == args.manifest_sha256, 'resume manifest changed')
        for path, digest in p.read(args.manifest)['files'].items():
            p.require(p.sha(path) == digest, 'resume source changed: ' + path)
        p.verify(ROOT / 'priority-manifest.json', OLD_MANIFEST_SHA)
    verify()
    execution, runner, previous, proof = inspect_boundary()
    if not args.run:
        print(json.dumps(dict(cpu_only=True, clean_prefix=True,
            pending=len(next(g['pending_ids'] for g in proof['groups'] if g['system'] == 'ecoserve')))))
        return
    p.require(not args.out.exists(), 'fresh physical stage required; no retry')
    args.out.mkdir(parents=True)
    state = dict(schema='B-priority-stage-v2', pid=os.getpid(), started_s=time.time(), phase='clean_boundary_verified',
                 hardware_started=False, complete=False, automatic_retries=False, previous_pre_gpu_attempt=p.ref(PAUSE / 'status.json'))
    def update(**items):
        state.update(items, updated_s=time.time())
        p.write(args.out / 'status.json', state)
    update()
    try:
        with (p.ABL / 'watcher-32b.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            execution, runner, previous, proof = inspect_boundary()
            p.write(args.out / 'prefix-proof.json', proof, True)
            asyncio.run(p.experiments(execution, runner, previous, proof, args.out, update, verify))
        update(phase='handoff_ready', complete=True, continuation_complete=False)
    except BaseException as exc:
        update(phase='failed', error=repr(exc), needs_attention=True)
        raise
    finally:
        update(finished_s=time.time())


if __name__ == '__main__':
    main()
