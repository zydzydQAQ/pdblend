"""Small immutable-reference helpers for the uniform-rate continuation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT.parents[1]


def need(value, why):
    if not value:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_text())


def checked(reference):
    need(sha(reference['path']) == reference['sha256'], 'frozen reference changed: ' + reference['path'])
    return read(reference['path'])


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def load(reference, name):
    if isinstance(reference, dict):
        need(sha(reference['path']) == reference['sha256'], 'module reference changed')
        path = Path(reference['path'])
    else:
        path = Path(reference)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def process_identity(pid):
    try:
        parts = Path('/proc', str(pid), 'stat').read_text().rsplit(') ', 1)[1].split()
        return dict(pid=int(pid), state=parts[0], startticks=parts[19])
    except FileNotFoundError:
        return None


def active_owner(state):
    actual = process_identity(state['pid'])
    return bool(actual and actual['state'] != 'Z' and
                (not state.get('startticks') or actual['startticks'] == state['startticks']))
