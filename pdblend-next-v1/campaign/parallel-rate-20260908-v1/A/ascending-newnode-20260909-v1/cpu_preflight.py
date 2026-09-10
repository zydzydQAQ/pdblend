"""Read-only staged-file, identity and CPU-container checks; no GPU work."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_item(item, root, key):
    path = root / item[key]
    if not path.is_file():
        return {'path': str(path), 'error': 'missing'}
    if path.stat().st_size != item['bytes']:
        return {'path': str(path), 'error': 'size_mismatch'}
    if sha(path) != item['sha256']:
        return {'path': str(path), 'error': 'sha256_mismatch'}
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path)
    parser.add_argument('--cpu-container', action='store_true')
    args = parser.parse_args()
    state = dict(schema='newnode14B-cpu-preflight-v1', passed=False,
        started_at_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        GPU_work_started=False, hardware_qualification_granted=False, checks={}, errors=[])
    try:
        identity = json.loads((HERE / 'node-identity.json').read_text())
        actual_host = socket.gethostname()
        assert actual_host == identity['actual_hostname'], 'Wrong physical host'
        raw = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version',
                              '--format=csv,noheader'], check=True, capture_output=True, text=True).stdout
        actual_gpus = []
        for line in raw.strip().splitlines():
            index, uuid, bus, name, memory, driver = [s.strip() for s in line.split(',')]
            actual_gpus.append(dict(index=int(index), uuid=uuid, pci_bus_id=bus, name=name, memory=memory, driver=driver))
        assert [(g['index'], g['uuid']) for g in actual_gpus] == [(g['index'], g['uuid']) for g in identity['GPUs']]
        state['checks']['node_identity'] = dict(passed=True, hostname=actual_host, GPUs=actual_gpus)
        for label, name, key in [('portable_files', 'portable-manifest-v2.json', 'path_relative_to_workspace'),
                                 ('model_files', 'model-manifest.json', 'name')]:
            manifest = json.loads((HERE / name).read_text())
            root = Path('/root/workspace') if label == 'portable_files' else Path(manifest['model_root'])
            with ThreadPoolExecutor(max_workers=4) as executor:
                failures = [r for r in executor.map(lambda item: verify_item(item, root, key), manifest['files']) if r]
            state['checks'][label] = dict(passed=not failures, checked_files=len(manifest['files']),
                checked_bytes=sum(f['bytes'] for f in manifest['files']), manifest_sha256=sha(HERE / name), failures=failures)
            assert not failures, label + ' failed'
        sys.path.insert(0, '/root/workspace/pdblend/.runtime-deps')
        import aiohttp
        import numpy
        import pynvml
        state['checks']['host_dependencies'] = dict(passed=True, python=sys.version,
            aiohttp=aiohttp.__version__, numpy=numpy.__version__, pynvml_import=True)
        image = 'sha256:0bb51d143b7fcaaea2e794dd6e207cf4165a4f21522a2e932a4bd4a117074bc2'
        inspected = subprocess.run(['docker', 'image', 'inspect', '--format', '{{.Id}}', image],
                                    check=True, capture_output=True, text=True).stdout.strip()
        assert inspected == image
        state['checks']['PDB_image'] = dict(passed=True, id=inspected)
        if args.cpu_container:
            code = "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['torch','vllm','transformers','safetensors']}))"
            result = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint', 'python3', image,
                                     '-c', code], check=True, capture_output=True, text=True, timeout=120)
            state['checks']['CPU_container_versions'] = dict(passed=True, versions=json.loads(result.stdout),
                GPU_device_access_requested=False, model_server_started=False)
        state['passed'] = True
    except BaseException as exc:
        state['errors'].append(repr(exc))
    state['finished_at_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    output = json.dumps(state, indent=2, ensure_ascii=False) + '\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(output)
    print(output)
    return 0 if state['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
