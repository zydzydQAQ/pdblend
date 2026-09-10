"""CPU helpers for immutable, fresh-node calibration declarations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
U = HERE.parent
A = U.parent
ROOT = A.parent
HOST = ROOT / 'hosts/14b-capacity-p12'
DRIVER = HERE / 'driver'
sys.path[:0] = [str(DRIVER), str(HOST / 'src'), str(HOST), '/root/workspace/pdblend/.runtime-deps']
from capacity_executor import fixed as checked, require as need, sha
from capacity_certificate import ref


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return ref(path)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def add(files, reference):
    need(sha(reference['path']) == reference['sha256'], 'changed immutable reference')
    files[reference['path']] = reference['sha256']


def tree_files(path):
    return {str(p): sha(p) for p in Path(path).rglob('*') if p.is_file() and '__pycache__' not in p.parts}


def frozen_sources():
    files = tree_files(DRIVER)
    for name in ('fresh_support.py', 'producer.py', 'audit.py', 'verify.py', 'prepare_adapter.py'):
        path = HERE / name
        need(path.is_file(), 'producer implementation incomplete: ' + name)
        files[str(path)] = sha(path)
    for base, relative in [(HOST, True), (U / 'isolated-power', False), (U / 'meter-runtime', False)]:
        reference = ref(base / 'manifest.json')
        manifest = checked(reference)
        add(files, reference)
        for path, digest in manifest['files'].items():
            actual = base / path if relative else Path(path)
            need(sha(actual) == digest, 'unchanged execution source differs')
            files[str(actual)] = digest
    for name in ('bootstrap.py', 'power_selftest.py', 'verify_power.py'):
        add(files, ref(U / name))
    return files


def no_live_pid(pid, ticks=None):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[0] == 'Z' or (ticks is not None and fields[19] != str(ticks))
    except FileNotFoundError:
        return True
