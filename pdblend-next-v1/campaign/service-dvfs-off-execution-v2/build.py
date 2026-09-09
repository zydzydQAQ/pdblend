"""Create three tiny immutable adapters without copying traces or frameworks."""
import json
from pathlib import Path
import shutil

from adapter import candidate_row, immutable, read, require, selected_sources, sha

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
PACKAGES = {
    '14b': ('A14B-deadline-matrix-v1', 'A14B-service-dvfs-off-v2'),
    '32b': ('B32B-main-scale-fixed-window-v1', 'B32B-service-dvfs-off-v2'),
    '7b': ('C7B-deadline-matrix-v1', 'C7B-service-dvfs-off-v2'),
}
WRAPPER = '''"""Independent service-DVFS-off binding; no GPU action on import/check."""
import hashlib
import json
from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parent
_BINDING = json.loads((_ROOT / 'binding.json').read_text())
for _path, _digest in _BINDING['adapter_files'].items():
    if hashlib.sha256(Path(_path).read_bytes()).hexdigest() != _digest:
        raise RuntimeError('ablation adapter source changed: ' + _path)
sys.path.insert(0, str(Path(_BINDING['adapter'])))
import adapter as _adapter
_BOUND = _adapter.bind(_ROOT)
globals().update({k: v for k, v in vars(_BOUND).items() if not k.startswith('__')})
if __name__ == '__main__':
    _adapter.main(_BOUND)
'''


def build(model, source_name, output_name):
    source, output = CAMPAIGN / source_name, CAMPAIGN / output_name
    require(not output.exists(), 'new output required: ' + str(output))
    output.mkdir()
    (output / 'inputs').mkdir()
    source_spec = read(source / 'runspec.json')
    old = read(source / 'inputs/controller.fixed.json')
    require(old['dvfs'] is True, 'source already disabled')
    config = output / 'inputs/controller.fixed.json'
    immutable(config, dict(old, dvfs=False))
    rows = [candidate_row(row, i, config, sha(config))
            for i, row in enumerate(selected_sources(source_spec), 1)]
    spec = dict(schema=1, phase='service_dvfs_off', model=model, cells=rows,
        protocol_id=source_spec['protocol_id'], measurement_schema=3, split='development',
        execute_baselines=False, formal_eligible=False, execution_complete=False,
        deadline_scope=source_spec['deadline_scope'], execution_budget=source_spec['execution_budget'],
        sole_policy_change={'dvfs': False}, source_package=str(source),
        source_package_manifest_sha256=sha(source / 'package-manifest.json'),
        original_temporal_gate_unchanged=True, additional_actual_gpu_validation=False,
        timing_scope='18 full300s cells; admit startup+420s+restore per cell under original global deadline; no completion promise')
    immutable(output / 'runspec.json', spec)
    (output / 'run.py').write_text(WRAPPER)
    # Retain the existing child except for one passive context around run_cell.
    child = (source / 'child.py').read_text()
    child = child.replace('import asyncio\n', 'import asyncio\nfrom clock_commands import capture_clock_commands\n', 1)
    lines = child.splitlines(keepends=True)
    matches = [i for i, line in enumerate(lines) if 'result=await run_cell(' in line]
    require(len(matches) == 1, 'expected one original host invocation')
    index = matches[0]
    indent = lines[index][:len(lines[index])-len(lines[index].lstrip())]
    lines[index:index+1] = [indent+'with capture_clock_commands(operation):\n', '    '+lines[index]]
    (output / 'child.py').write_text(''.join(lines))
    shutil.copyfile(ROOT / 'clock_commands.py', output / 'clock_commands.py')
    if model == '32b':
        shutil.copyfile(source / 'epoch.py', output / 'epoch.py')
    freeze = read(source / 'freeze.json')
    files = {str(source / p): h for p, h in read(source / 'package-manifest.json')['files'].items()}
    files.update({str(source / p): sha(source / p) for p in ('package-manifest.json', 'freeze.json')})
    host = Path(freeze['controller_release'])
    files[str(host / 'manifest.json')] = sha(host / 'manifest.json')
    files[str(Path(freeze['engine_release']) / 'manifest.json')] = sha(Path(freeze['engine_release']) / 'manifest.json')
    files[old['profiles']] = sha(old['profiles'])
    queue = CAMPAIGN / 'fixed-window-queue-v1/queue.py'
    files[str(queue)] = sha(queue)
    adapter_files = {str(ROOT / p): sha(ROOT / p) for p in ('adapter.py', 'ablation_queue.py', 'clock_commands.py')}
    files.update(adapter_files)
    immutable(output / 'binding.json', dict(schema=1, model=model, source_package=str(source),
        source_files=files, adapter=str(ROOT), adapter_files=adapter_files,
        source_queue=str(queue), host_release=str(host), execution_reuse='original Cell code imported by absolute path',
        original_child_sha256=sha(source / 'child.py'), child_sha256=sha(output/'child.py'),
        child_change='passive ClockOwner intent and actual hardware set/reset observation only', inherited_lease_allowed=False,
        scope='PDB continuous service command policy only; parking/idle unlock unchanged'))
    immutable(output / 'package-manifest.json', dict(schema=1, model=model,
        files={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
        frozen_for_cpu_review=True, gpu_executed=False))
    return dict(package=str(output), package_manifest_sha256=sha(output / 'package-manifest.json'),
                cells=len(rows), source_package=str(source), copied_trace_bytes=0)


if __name__ == '__main__':
    print(json.dumps([build(model, *names) for model, names in PACKAGES.items()], indent=2))
