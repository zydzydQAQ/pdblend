"""Only adapt measurement transport in the frozen fixed100 dynamic executor."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent / 'dynamic-execution-until-complete-001'
ADAPTER = ROOT.parent / 'isolated-power-v2' / 'manifest.json'
ADAPTER_SHA = 'af64c1ba352fcc72f5bd74f8f6870bd70e347c76185ee404a9d2119b733cca85'
PINS = {
    'dynamic_measurement.py': 'af0e86fb356667d437ecf29085787babcc8d0212f10cec37656d413c42ed00d9',
    'dynamic_child.py': '4669358b2740602dc7bd53422137aa4d2c12653d19f614f2ead63c03cd761358',
    'dynamic_ownership.py': 'e925f22bbc99b8881891465fe436ef5afaf179abaa03c7ddb3ceecc02b23b0b6',
    'protocol.py': '51f6d11dd895ded4640e11e0dd91aa52a02cbf5bd4a2a6be24b283546e6d450a',
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)

def build():
    assert sha(ADAPTER) == ADAPTER_SHA
    for name, digest in PINS.items():
        assert sha(PARENT / name) == digest
    source = (PARENT / 'dynamic_measurement.py').read_text()
    source = once(source, 'import dynamic_ownership as ownership\n',
                  'import dynamic_ownership as ownership\nimport sampler_hooks\n')
    source = once(source, '    from ecopadg.measure.power import PowerSampler, trapezoid_energy\n', '')
    source = once(source, "    write(operation / 'identity.before.json', before)\n", """    write(operation / 'identity.before.json', before)
    host_manifest, adapter_manifest = sampler_hooks.references(binding)
    sampler_root = output / 'isolated-samplers'
    prior_samplers = sampler_hooks.directories(sampler_root)
    sampler_hooks.install(sampler_root, host_manifest, adapter_manifest)
    from ecopadg.measure.power import PowerSampler, trapezoid_energy
""")
    source = once(source, "        capacity_identity=capacity['identity'], initial_instances=binding['instances'])",
                  "        capacity_identity=capacity['identity'], initial_instances=binding['instances'],\n"
                  "        host_manifest=host_manifest, adapter_manifest=adapter_manifest)")
    source = once(source, '    sampler.start()\n    try:\n',
                  '    sampler.start()\n    try:\n        await asyncio.to_thread(sampler.wait_ready)\n')
    source = once(source, "    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)\n",
                  "    receipt['measurement_adapter'] = adapter_manifest\n"
                  "    sampler = PowerSampler(range(8), interval=.02, backend=hardware, sample_clocks=True)\n")
    source = once(source, "        receipt['measurement_valid'] = bool(failure is None", """        try:
            owned_samplers = sampler_hooks.directories(sampler_root) - prior_samplers
            owned_samplers.update(sampler_hooks.directories(operation / 'isolated-samplers'))
            isolated_files = sampler_hooks.completed_artifacts(owned_samplers, host_manifest, adapter_manifest)
            receipt.setdefault('dynamic_artifacts', {}).update(isolated_files)
            receipt['isolated_sampler_directories'] = sorted(map(str, owned_samplers))
            receipt['isolated_samplers'] = sampler_hooks.sampler_references(owned_samplers)
            receipt['isolated_sampler_evidence_complete'] = True
        except BaseException as exc:
            errors.append('isolated sampler ownership/completion: ' + repr(exc))
            receipt['isolated_sampler_evidence_complete'] = False
        receipt['measurement_valid'] = bool(failure is None""")
    child = (PARENT / 'dynamic_child.py').read_text()
    child = once(child, '    from ecopadg.serving.cell import run_cell\n', '')
    child = once(child, "    original_request = aiohttp.ClientSession._request\n", """    import sampler_hooks
    adapter = sampler_hooks.install(operation / 'isolated-samplers', job['host_manifest'], job['adapter_manifest'])
    from ecopadg.serving import cell as cell_module
    original_request = aiohttp.ClientSession._request
""")
    child = once(child, '            result = await run_cell(args)\n',
                 '            result = await sampler_hooks.prepared_primary(args, cell_module, adapter)\n')
    outputs = {'dynamic_measurement.py': source.encode(), 'dynamic_child.py': child.encode(),
               **{name: (PARENT / name).read_bytes() for name in ('dynamic_ownership.py', 'protocol.py')}}
    for name, data in outputs.items():
        path = ROOT / name
        if path.exists():
            assert not (ROOT / 'manifest.json').exists(), 'cannot change a frozen candidate'
            path.write_bytes(data)
        else:
            path.write_bytes(data)
    return dict(parent_files={str(PARENT / name): digest for name, digest in PINS.items()},
                adapter=dict(path=str(ADAPTER), sha256=ADAPTER_SHA),
                generated_files={str(ROOT / name): sha(ROOT / name) for name in outputs})

if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
