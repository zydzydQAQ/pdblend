"""Serialize owned spare clock transitions with admissions; keep uncertainty barriers."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import time
from build_capacity_p5 import default_equivalent

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locked(source):
    for start, end in (
        ('        if clocks:\n', '\n    async def start('),
        ('        if self.controller.backend.clocks:\n', '        cid = (await self.command(args, limit, 30)).strip()')):
        if source.count(start) != 1:
            raise ValueError('capacity clock block differs: ' + start)
        a = source.index(start)
        b = source.index(end, a)
        block = source[a:b]
        first, rest = block.split('\n', 1)
        # The action lock covers only proof and physical clock operation.
        # Docker launch, readiness, numerical checks and workload keep running.
        replacement = first + '\n            async with self.controller.action_lock:\n' + ''.join(
            '    ' + line if line.strip() else line for line in rest.splitlines(keepends=True))
        source = source[:a] + replacement + source[b:]
    ast.parse(source)
    return source


def build():
    shared = None
    for model in ('7b', '14b', '32b'):
        parent = ROOT / 'hosts' / f'{model}-capacity-p5'
        out = ROOT / 'hosts' / f'{model}-capacity-p6'
        if out.exists():
            raise FileExistsError(out)
        manifest = json.loads((parent / 'manifest.json').read_text())
        for name, expected in manifest['files'].items():
            if sha(parent / name) != expected:
                raise ValueError('frozen P5 changed: ' + name)
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if name == 'capacity_backend.py':
                dest.write_text(locked((parent / name).read_text()))
            else:
                shutil.copyfile(parent / name, dest)
        files = {name: sha(out / name) for name in sorted(manifest['files'])}
        if shared is not None and files != shared:
            raise ValueError('cross-model source differs')
        shared = files
        p4 = ROOT / 'hosts' / f'{model}-fixed-p4'
        runtime = 'src/ecopadg/serving/runtime.py'
        proof = default_equivalent((p4 / runtime).read_text(), (out / runtime).read_text())
        result = dict(schema=6, model=model, implementation_series='parallel-p6', created_s=time.time(),
            files=files, parent_manifest=dict(path=str(parent / 'manifest.json'), sha256=sha(parent / 'manifest.json')),
            frozen_references={str(p): sha(p) for p in (Path(__file__), ROOT / 'build_capacity_p5.py')},
            common_controller_sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            identical_source_files_all_models=True, feature='capacity_integration_v1', default_enabled=False,
            changed_parent_files=['capacity_backend.py'], added_files=[], gpu_qualified=False,
            capacity_clock_operations_own_action_lock=True, lock_order=['action', 'clock', 'state'],
            slow_engine_start_never_holds_action_lock=True, uncertainty_barriers_unchanged=True,
            capacity_disabled_complete_p4_ast_sha256=proof,
            calibration_required_at_start=True, workload_end_restores_initial_layout=True)
        (out / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
        print(model, sha(out / 'manifest.json'), result['common_controller_sha256'])


if __name__ == '__main__':
    build()
