"""Add the freshly observed topology dependency to the fixed qualifier closure."""
from pathlib import Path

import bootstrap as b
import power_selftest as p
import verify_domain2100

TOPOLOGY = Path('/root/workspace/pdblend/new-results/campaigns/three-pool-v2/interconnect.txt')
TOPOLOGY_SHA = '00d907ee5584f1dbfda45f181a18987df3c0ed2f236dc764a7653e7defeea7ff'


def verify(reference):
    result = verify_domain2100.verify(reference)
    observation = p.read(p.HERE / 'topology-observation.json')
    assert observation['actual_measurement'] and observation['hostname'] == 'iZwz9274emxme9019d2sjgZ'
    assert observation['path'] == str(TOPOLOGY) and observation['sha256'] == TOPOLOGY_SHA
    assert p.sha(TOPOLOGY) == p.sha(p.HERE / 'topology.observed.txt') == TOPOLOGY_SHA
    binding = b.checked(result['binding'])
    for path in binding['configs'].values():
        config = p.read(path)
        assert config['interconnect'] == str(TOPOLOGY)
        assert config['max_service_frequency_mhz'] == 2100 and config['allow_pd'] is False
    for path in [TOPOLOGY, p.HERE / 'topology-observation.json', p.HERE / 'topology.observed.txt', Path(__file__)]:
        result['files'][str(path)] = p.sha(path)
    result['fresh_topology_sha256'] = TOPOLOGY_SHA
    return result
