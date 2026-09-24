"""Select a bound CPU analysis snapshot without reloading systemd or GPUs."""
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_source(package):
    package = Path(package).resolve()
    selector = package / 'host-analysis/active-source.json'
    if not selector.is_file():
        return None
    record = json.loads(selector.read_text())
    if record.get('schema') != 'matrix-host-analysis-selection/v1':
        raise ValueError('unknown host analysis selector')
    ref = record['source_manifest']
    manifest_path = Path(ref['path']).resolve()
    source = manifest_path.parent
    if not source.is_relative_to(package / 'host-analysis/sources'):
        raise ValueError('host analysis snapshot is outside the matrix package')
    if sha(manifest_path) != ref['sha256']:
        raise ValueError('host analysis manifest changed')
    manifest = json.loads(manifest_path.read_text())
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob('*.py'))}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':'),
                                        allow_nan=False).encode()).hexdigest()
    if files != manifest['files'] or identity != manifest['source_sha256'] or source.name != identity:
        raise ValueError('host analysis Python inventory changed')
    return source


def activate(argv):
    if '--package' not in argv:
        return None
    source = resolve_source(argv[argv.index('--package') + 1])
    if source is None:
        return None
    for name, module in list(sys.modules.items()):
        if name == 'pdblend' or name.startswith('pdblend.'):
            path = getattr(module, '__file__', None)
            if path is None or not Path(path).resolve().is_relative_to(source):
                raise RuntimeError('host analysis must be selected before importing PDblend')
    sys.path.insert(0, str(source))
    os.environ['PYTHONPATH'] = str(source)
    return source
