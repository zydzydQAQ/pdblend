"""Freeze CPU-only observer adaptation after actual-process and contract checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import build

ROOT = Path(__file__).resolve().parent

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path): return dict(path=str(path), sha256=sha(path))
def write(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False); handle.write('\n')

def main():
    assert not (ROOT / 'manifest.json').exists()
    ancestry = build.build()
    files = {str(p): sha(p) for p in sorted(ROOT.glob('*.py'))}
    results = []
    for argv, count in [
        ([sys.executable, str(ROOT / 'test_hooks.py'), '-v'], 13),
        ([sys.executable, str(ROOT / 'test_real_spawn.py'), '-v'], 3),
        ([sys.executable, str(ROOT / 'test_source_contract.py'), '-v'], 4),
        ([sys.executable, '-m', 'pytest', '-q', str(build.PARENT / 'test_dynamic_ownership.py')], 16),
    ]:
        start = time.time()
        result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=45)
        results.append(dict(argv=argv, exit_code=result.returncode, tests=count,
                            stdout=result.stdout, stderr=result.stderr, elapsed_s=time.time()-start))
        assert result.returncode == 0, result.stdout + result.stderr
    assert all(sha(p) == digest for p, digest in files.items())
    cpu = dict(schema='isolated-fixed100-measurement-CPU-v1', passed=True, tests_passed=36,
               gpu_actions=False, actual_separate_process_transport_tested_with_CPU_backend=True,
               source_files=files, ancestry=ancestry, results=results, created_s=time.time())
    write(ROOT / 'cpu-validation.json', cpu)
    adapted = dict(schema=2, parent_manifest=ref(build.PARENT / 'dynamic_measurement.manifest.json'),
                   original_source=ref(build.PARENT / 'dynamic_measurement.py'),
                   adapted_source=str(ROOT / 'dynamic_measurement.py'),
                   sha256=sha(ROOT / 'dynamic_measurement.py'),
                   protocol_unchanged=True, eight_gpu_energy_unchanged=True,
                   measurement_adapter=ancestry['adapter'], global_deadline=None,
                   changes=['isolated read-only original PowerSampler transport',
                            'asynchronous primary observer ready before unchanged cell startup',
                            'per-cell sampler UUID ownership and complete raw/exit evidence'],
                   builder_sha256=sha(ROOT / 'build.py'))
    write(ROOT / 'dynamic_measurement.manifest.json', adapted)
    adapter = json.loads(build.ADAPTER.read_text())
    manifest_files = {**files, **adapter['files'], str(build.ADAPTER): build.ADAPTER_SHA,
                      str(ROOT/'cpu-validation.json'): sha(ROOT/'cpu-validation.json'),
                      str(ROOT/'dynamic_measurement.manifest.json'): sha(ROOT/'dynamic_measurement.manifest.json'),
                      str(build.PARENT/'manifest.json'): sha(build.PARENT/'manifest.json'),
                      str(build.PARENT/'test_dynamic_ownership.py'): sha(build.PARENT/'test_dynamic_ownership.py')}
    manifest = dict(schema='dynamic-fixed100-isolated-power-v2', files=manifest_files,
                    parent=ref(build.PARENT/'manifest.json'), measurement_adapter=ancestry['adapter'],
                    control_source_unchanged=True, dynamic_ownership_unchanged=True,
                    request_window_s=100, request_timeout_s=120, cleanup_bound_s=90,
                    global_deadline=None, cpu_validation=ref(ROOT/'cpu-validation.json'),
                    actual_gpu_qualification_not_inferred=True)
    write(ROOT/'manifest.json', manifest)
    print(json.dumps(dict(manifest=ref(ROOT/'manifest.json'), cpu=ref(ROOT/'cpu-validation.json'), tests=36)))

if __name__ == '__main__': main()
