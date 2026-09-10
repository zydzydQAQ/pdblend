"""Path/scoped predecessor adapter; original PDB restore and qualification unchanged."""
import argparse
import asyncio
import json
import signal
from pdb_only_support import HERE, UNIFORM, p, predecessor, source_contract


def original():
    module = p.load(UNIFORM / 'prepare_node.py', 'B_pdb_only_original_preparer')
    module.HERE = HERE
    module.predecessor = predecessor
    return module


def run_phase(phase):
    manifest = source_contract()
    module = original()
    if phase == 'pdb-restore':
        async def controlled():
            task = asyncio.current_task()
            stopped = False
            def cancel():
                nonlocal stopped
                if not stopped:
                    stopped = True
                    task.cancel()
            for sig in (signal.SIGINT, signal.SIGTERM):
                asyncio.get_running_loop().add_signal_handler(sig, cancel)
            return await module.restore_pdb()
        return asyncio.run(controlled())
    if phase == 'pdb-spec':
        path = module.make_pdb_spec()
        spec = p.read(path)
        spec['files'].update(manifest['files'])
        spec['files'][str(HERE / 'manifest.json')] = p.sha(HERE / 'manifest.json')
        p.save(path, spec)
        module.g.validate(spec)
        return str(path)
    assert phase == 'pdb-freeze'
    return module.freeze_pdb()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pdb-restore', 'pdb-spec', 'pdb-freeze'))
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source_contract()
    print(json.dumps(run_phase(args.phase) if args.run else dict(passed=True, cpu_only=True)))
