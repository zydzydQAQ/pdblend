"""Measured original-PDB restoration after the preserved B16 return transaction."""
import argparse
import asyncio
import copy
import importlib.util
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import protocol as p

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

async def restore(args, state):
    hold = load(ROOT / 'hold.py', 'improvement_preserved_B_handoff')
    handoff = hold.validate_handoff(args.ready)
    prior = p.checked(handoff['fresh_restoration_binding'])
    declaration = p.read(p.REPO / 'campaign/main-rate-rerun-v1/declarations.json')
    source = next(c for c in declaration['cells'] if c['model'] == '32b')['source_binding']
    parent = p.checked(source)
    execution = load(p.REPO / 'campaign/pdblend-ablation-20260908-v1/execution.py',
                     'improvement_frozen_physical_restore')
    common = execution.load_common(parent['host_release'])
    from ecopadg.serving.campaign import node_lease
    p.need('PDBLEND_NODE_LOCK_FD' not in os.environ, 'fresh non-inherited node lease required')
    p.need(not args.out.exists(), 'fresh restoration output required')
    args.out.mkdir(parents=True)
    p.write(args.out / 'source-handoff.json', dict(ready=p.ref(args.ready), original_pdb=source,
            returned_baseline=handoff['fresh_restoration_binding']), exclusive=True)
    with node_lease():
        hold.validate_handoff(args.ready)
        state.update(phase='measured_restore', node_lease_held=True)
        p.write(args.out / 'status.json', state)
        fresh = await execution.restore_core(common, parent, prior, args.out / 'restoration')
        raw_status = p.read(args.out / 'restoration/status.json')
        p.need(raw_status['complete'] and raw_status['correctness']['passed'], 'restore qualification incomplete')
        improved = copy.deepcopy(fresh)
        improved.update(deadline_s=p.DEADLINE, improvement_preparation=True,
            prior_restoration_binding=p.ref(args.out / 'restoration/binding.json'),
            deadline_scope='new authorized improvement batch; original restoration record retains its deadline')
        common.GLOBAL_DEADLINE = p.DEADLINE
        common.validate_binding(improved)
        p.write(args.out / 'binding.json', improved, exclusive=True)
        state.update(phase='pdb_ready_for_improvement', complete=True,
            binding=p.ref(args.out / 'binding.json'),
            ordinary_qualification=p.ref(args.out / 'restoration/correctness/status.json'),
            restored_source_unchanged=True, extra_restore_energy_j=raw_status['setup_and_correctness_energy_j'],
            serving_performance_started=False)
    state['node_lease_held'] = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ready', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        hold = load(ROOT / 'hold.py', 'improvement_preserved_B_handoff')
        hold.validate_handoff(args.ready)
        print({'cpu_only': True, 'handoff_ready': True})
        return
    p.need(not args.out.exists(), 'existing restoration evidence is immutable')
    state = dict(schema=1, pid=os.getpid(), started_s=time.time(), phase='starting', complete=False)
    async def bounded():
        current = asyncio.current_task()
        interrupted = False
        def stop():
            nonlocal interrupted
            if not interrupted:
                interrupted = True
                current.cancel()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop)
        await restore(args, state)
    try:
        asyncio.run(bounded())
    except BaseException as exc:
        state.update(phase='failed', error=repr(exc))
        raise
    finally:
        if args.out.exists():
            state.update(finished_s=time.time(), node_lease_held=False)
            p.write(args.out / 'status.json', state)

if __name__ == '__main__':
    main()
