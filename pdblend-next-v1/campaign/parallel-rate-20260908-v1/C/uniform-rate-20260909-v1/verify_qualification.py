"""Supplement the inherited C gate with raw all-eight power/identity replay."""
import csv
import json
import math
from pathlib import Path
import sys

import support as p


def verify(reference):
    saved = p.checked(reference)
    spec = p.checked(saved['spec'])
    source = Path(spec['inherited_qualification_sources'])
    p.load(source / 'cold_restore.py', 'cold_restore')
    q = p.load(source / 'qualify.py', 'qualify')
    inherited = p.load(source / 'verify_qualification.py', 'uniform_inherited_C_qualification')
    result = inherited.verify(reference)
    binding = p.checked(result['binding'])
    state = p.checked(saved['status'])
    directory = Path(saved['status']['path']).parent
    host = Path(binding['host_release'])
    sys.path[:0] = [str(host / 'src'), str(host), '/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.measurement import power_evidence
    from ecopadg.measure.power import trapezoid_energy
    from ecopadg.metrics import clip_power_window
    frozen = dict(saved['source_files'], **saved['files'])
    for name in ('ecopadg.serving.measurement', 'ecopadg.measure.power', 'ecopadg.metrics'):
        actual = str(Path(sys.modules[name].__file__).resolve())
        p.need(frozen.get(actual) == p.sha(actual), 'unbound measurement module: ' + actual)
    with (directory / 'power/power.csv').open() as stream:
        samples = [(float(r['t_s']), [float(r['gpu%d_w' % i]) for i in range(8)])
                   for r in csv.DictReader(stream)]
    metadata = [json.loads(line) for line in (directory / 'power/power_metadata.jsonl').read_text().splitlines()]
    measurement = state['measurement']
    observed = power_evidence(samples, p.read(directory / 'power/power_source.json'), metadata)
    p.need(observed == measurement['power_evidence'] and observed['power_source_verified'],
           'raw power-source evidence differs')
    energy = trapezoid_energy(clip_power_window(samples, measurement['measurement_start_s'],
                                              measurement['measurement_end_s'], pad_s=0))
    p.need(math.isclose(energy, measurement['energy_j'], rel_tol=1e-10, abs_tol=1e-6),
           'all-eight raw energy differs')
    for side in ('before', 'after'):
        values = p.read(directory / ('identity.' + side + '.json'))
        p.need(len(values) == len(binding['instances']), 'qualification instance count differs')
        for actual, instance in zip(values, binding['instances']):
            c = actual['container']
            p.need(c['Id'] == instance['container']['id'] and c['Image'] == instance['container']['image']
                   and c['State']['Running'] and c['State']['StartedAt'] == instance['container']['StartedAt'],
                   'qualification process identity differs')
            p.need(all(actual['provenance'].get(k) == v for k, v in instance['provenance'].items()),
                   'qualification native source differs')
            q.legacy_idle(actual['runtime'], instance)
    result.update(raw_eight_GPU_power_recomputed=True, energy_j=energy,
                  before_after_native_identity_recomputed=True, qualification_validator=p.ref(__file__))
    return result


if __name__ == '__main__':
    print(json.dumps(verify(p.ref(sys.argv[1]))))
