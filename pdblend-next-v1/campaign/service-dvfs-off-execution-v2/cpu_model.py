"""Run under one model's actual frozen host; no NVML, HTTP, or GPU requests."""
import asyncio
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

import adapter


def run(package):
    package = Path(package).resolve()
    source = Path(adapter.read(package / 'binding.json')['source_package'])
    protected = {str(source / p): adapter.sha(source / p)
                 for p in adapter.read(source / 'package-manifest.json')['files']}
    count = 0
    with tempfile.TemporaryDirectory(prefix='_dvfs_cpu_', dir=package.parent) as name:
        root = Path(name)
        for p in package.rglob('*'):
            if p.is_file() and p.name != 'package-manifest.json' and '__pycache__' not in p.parts:
                target = root / p.relative_to(package)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, target)
        spec = adapter.read(root / 'runspec.json')
        for row in spec['cells']:
            row['controller_config'] = str(root / 'inputs/controller.fixed.json')
        (root / 'runspec.json').write_text(json.dumps(spec))
        adapter.immutable(root / 'package-manifest.json', dict(files={
            str(p.relative_to(root)): adapter.sha(p) for p in root.rglob('*') if p.is_file()}))
        b = adapter.bind(root)
        b.package_check()
        assert b.ROOT == root and b.CONFIG == root / 'inputs/controller.fixed.json'
        assert b.live.__globals__['ROOT'] == source
        assert b.CELL.__module__.startswith('_dvfs_cell_')
        count += 1

        freeze = copy.deepcopy(adapter.read(source / 'freeze.json'))
        freeze['policy_config_sha256'] = adapter.sha(b.CONFIG)
        adapter.immutable(root / 'freeze.json', freeze)
        row = spec['cells'][0]
        limits = dict(arrival_window_s=300, issued_s=0, latest_arrival_epoch_s=10**12,
                      cell_execution_deadline_s=10**12+420, restore_deadline_s=10**12+540)
        status = dict(cells=[])
        # This is the real imported Cell constructor and its real b.write.
        cell = b.CELL(b, None, row, freeze, status, 'cpu-status.json', None, limits=limits)
        assert cell.operation == root / 'operations' / row['cell_id']
        assert cell.out == root / 'cells' / row['cell_id']
        assert (root / 'receipts' / (row['cell_id']+'.json')).is_file()
        assert (root / 'cpu-status.json').is_file()
        assert not (source / 'operations' / row['cell_id']).exists()
        count += 1

        child = adapter.load(root / 'child.py', '_dvfs_child_cpu')
        if b.BINDING['model'] == '7b':
            core = child.core()
            assert core.ROOT == root and core.CONFIG == b.CONFIG
            core.package_check()
            assert core.event_paths.__globals__['ROOT'] == source
            arguments = lambda r: core.cell_args(r, root / 'cells' / r['cell_id'])
            assert core.verify_actual_config.__closure__ is not None
        else:
            assert child.ROOT == root and child.HOST == b.HOST
            arguments = child.cell_arguments
        count += 1

        from ecopadg.serving import cell as host_cell
        assert Path(host_cell.__file__).resolve() == b.HOST / 'src/ecopadg/serving/cell.py'
        class BeforeNetwork(Exception):
            pass
        for dataset in ('alpaca', 'sharegpt', 'longbench'):
            selected = next(r for r in spec['cells'] if r['dataset'] == dataset)
            args = arguments(selected)
            assert args.config == b.CONFIG and args.trace == Path(selected['trace'])
            assert args.out == root / 'cells' / selected['cell_id']
            args.out = root / ('cpu-controller-'+dataset)
            seen = []
            def constructor(config):
                seen.append(copy.deepcopy(config))
                raise BeforeNetwork()
            with patch.object(host_cell, 'Controller', constructor):
                try:
                    asyncio.run(host_cell.run_cell(args))
                except BeforeNetwork:
                    pass
                else:
                    raise AssertionError('actual host failed to reach Controller')
            expected = adapter.expected_actual(adapter.read(b.CONFIG), selected, args.out)
            assert seen == [expected]
            b.verify_actual_config(selected, args.out)
            assert expected['dvfs'] is False
            original = adapter.read(source / 'inputs/controller.fixed.json')
            assert adapter.read(b.CONFIG) == dict(original, dvfs=False)
            count += 1

        # Real child epoch gate: an expired epoch raises before delegating to
        # the actual benchmark's header/worker setup. No session is created.
        called = []
        if b.BINDING['model'] == '14b':
            gate = child.EpochGate(lambda *a: called.append(a),
                                   {'latest_arrival_epoch_s': 1}, lambda r: None)
            invoke = lambda: gate('url', child.bench_vllm.EVALUATION_V3, 2, 2)
        elif b.BINDING['model'] == '32b':
            epoch = adapter.load(root / 'epoch.py', '_dvfs_cpu_epoch')
            gate = epoch.EpochGuard({'issued_s': 0, 'latest_arrival_epoch_s': 1},
                                   root / 'epoch.json', lambda *a: called.append(a))
            invoke = lambda: gate('url', 'evaluation-v3', 2, 2)
        else:
            gate = child.EpochGate({'latest_arrival_epoch_s': 1}, lambda value: None)
            invoke = lambda: gate.check('url', 'evaluation-v3', 2, 2)
        try:
            invoke()
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError('late actual benchmark epoch accepted')
        assert called == []
        count += 1

        # Unverified preflight never acquires control/clock ownership. Real
        # cleanup must still record its result in the independent output root.
        async def deny(*args, **kwargs):
            raise AssertionError('unauthorized HTTP/control')
        b.http = deny
        result = asyncio.run(cell.cleanup())
        assert result['complete'] is True and result['mutations_started'] is False
        assert (cell.operation / 'outer-cleanup.json').is_file()
        count += 1
        queue = adapter.load(Path(adapter.__file__).parent/'ablation_queue.py', '_cpu_ablation_queue')
        try:
            asyncio.run(queue.sweep(b,max_cells=1))
        except RuntimeError as exc:
            assert 'independent node lease' in str(exc)
        else:
            raise AssertionError('execution without owning the node lease accepted')
        count += 1
        with patch.object(sys,'argv',[str(root/'run.py'),'--phase','prepare']),patch.dict(os.environ,{'PDBLEND_NODE_LOCK_FD':'999999'}):
            try:
                adapter.main(b)
            except RuntimeError as exc:
                assert 'inherited lease' in str(exc)
            else:
                raise AssertionError('inherited main/scale lease accepted')
        count += 1
    adapter.file_refs(protected)
    return dict(model=adapter.read(package/'binding.json')['model'], cpu_cases_passed=count,
                original_frozen_files_unchanged=True, gpu_executed=False)


if __name__ == '__main__':
    print(json.dumps(run(sys.argv[1])))
