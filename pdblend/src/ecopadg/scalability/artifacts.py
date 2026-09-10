"""Small, explicit artifact operations shared by the experiment driver."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if not replace:
        with path.open('x') as handle:
            handle.write(data)
    else:
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(data)
        temp.replace(path)


def source_files():
    """Freeze only actual package/benchmark code, never the huge result tree."""
    package = Path(__file__).resolve().parents[1]
    files = set(package.rglob('*.py'))
    benchmark = package.parents[2] / 'benchmarks' / 'scripts' / 'bench_vllm.py'
    if benchmark.is_file():
        files.add(benchmark)
    return sorted(files)
