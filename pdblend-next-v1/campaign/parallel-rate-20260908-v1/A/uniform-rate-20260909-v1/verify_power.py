"""Recompute new-A observer qualification from saved raw, without GPU access."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import power_selftest as p


def checked(reference):
    assert p.sha(reference['path']) == reference['sha256'], 'saved evidence changed'
    return p.read(reference['path'])


def verify(reference):
    state = checked(reference)
    assert state['schema'] == 'new-A-isolated-power-qualification-v1' and state['passed'] is True
    assert state['node'] == 'Anew20260909' and state['model'] == '14b'
    assert state['read_only'] is True and state['hardware_writes'] is False and state['new_requests'] == 0
    sources = checked(state['source_manifest'])
    assert sources and all(p.sha(path) == digest for path, digest in sources.items())
    identity = checked(state['identity'])
    p.validate_identity(p.read(p.IDENTITY), state['hostname'], identity['GPUs'])
    assert identity['hostname'] == state['hostname']
    assert state['host_manifest'] == p.ref(p.HOST / 'manifest.json')
    sys.path.insert(0, str(p.METER))
    from capacity_certificate import raw_measurement, close
    result = state['result']
    raw = raw_measurement(result['receipt'])
    assert raw['measurement_start_s'] >= state['started_s']
    assert raw['measurement_end_s'] <= state['finished_s']
    assert raw['duration_s'] >= 8 and close(result['energy_j'], raw['energy_j'])
    terminal = checked(state['terminal'])
    assert terminal['complete'] and not terminal['errors']
    spec = importlib.util.spec_from_file_location('newA_saved_sampler_audit', p.HOOKS)
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    directories = {Path(row['directory']) for row in terminal['isolated_samplers']}
    assert len(directories) == 1
    verified = hooks.completed_artifacts(directories, state['host_manifest'], p.ref(p.ADAPTER))
    assert verified == terminal['artifacts']
    assert result['measurement_adapter'] == p.ref(p.ADAPTER)
    assert result['isolated_samplers'] == terminal['isolated_samplers']
    return dict(passed=True, independently_recomputed=True, qualification=reference,
                node=state['node'], hostname=state['hostname'], identity=state['identity'],
                host_manifest=state['host_manifest'], energy_j=raw['energy_j'],
                duration_s=raw['duration_s'], raw_energy_recomputed=True,
                ordinary_frequency_capacity_qualification_granted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualification', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(p.ref(args.qualification))))
