"""Run the unchanged validator, adding only durable HTTP-dispatch observations."""
import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent


async def main():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    path = ROOT / 'budget_validation.frozen.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['validator_sha256']
    spec = importlib.util.spec_from_file_location('unaltered_budget_validator', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    validator = module.Validation(SimpleNamespace(ports=[33500, 33501],
        runtime_dir=ROOT.parent/'B32B-engine-v3-candidate-v2/runtime', out=ROOT/'validation'))
    original_http = validator.http
    with (ROOT / 'dispatch.jsonl').open('x', buffering=1) as journal:
        async def observe(port, method, route, body=None, **kwargs):
            journal.write(json.dumps(dict(port=port, method=method, route=route, request=body,
                request_id=kwargs.get('request_id'), label=kwargs.get('label'), started_s=time.time())) + '\n')
            return await original_http(port, method, route, body, **kwargs)
        validator.http = observe
        passed = await validator.run()
    print(json.dumps(dict(passed=passed)), flush=True)
    return passed


if __name__ == '__main__':
    raise SystemExit(0 if asyncio.run(main()) else 1)
