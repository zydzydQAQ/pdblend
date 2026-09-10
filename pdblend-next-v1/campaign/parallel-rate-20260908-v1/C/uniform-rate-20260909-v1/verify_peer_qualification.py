"""Independently join fresh frequency/cancel qualification and peer restoration."""
import csv
import math
from pathlib import Path

import support as p


def verify(reference):
    qualification = p.checked(reference)
    p.need(qualification['schema'] == 'C-peer-restored-uniform-qualification-v1', 'unknown peer qualification')
    for path, digest in qualification['files'].items():
        p.need(p.sha(path) == digest, 'peer qualification artifact changed')
    result = p.load(qualification['base_validator'], 'C_uniform_base_q').verify(qualification['base_qualification'])
    state = p.checked(qualification['peer_setup'])
    p.need(state['complete'] and state['finished_s'] and not state['node_lease_held'] and not state.get('error'),
           'peer restore has not completed')
    p.need(state['binding'] == result['binding'] and state['base_qualification'] == qualification['base_qualification'],
           'peer operation qualified another binding')
    p.need(state['failure_diagnosis']['no_performance_request_issued'] and state['same_policy_and_host_source'],
           'startup attempt not independently diagnosed')
    plan = p.checked(state['historical_peer_plan'])
    peer = plan['register']['body']['id']
    p.need(state['registered'] == dict(id=peer, registered=[dict(id=peer, rank=0)])
           and state['prepared'] == dict(ready=[dict(peers=[peer], rank=0)]), 'native peer ACK differs')
    p.need(all(x['complete'] for x in state['restoration'].values()) and state['measurement']['measurement_valid'],
           'peer cleanup/power invalid')
    report = p.load(p.ROOT.parent / 'main-slo-improvement-v7/report.py', 'C_peer_independent_integration')
    directory = Path(qualification['peer_setup']['path']).parent
    with (directory / 'power/power.csv').open() as stream:
        powers = list(csv.DictReader(stream))
    measurement = state['measurement']
    energy = sum(report.integrate(powers, measurement['measurement_start_s'], measurement['measurement_end_s'],
                                  ['gpu%d_w' % i for i in range(8)]))
    p.need(math.isclose(energy, measurement['energy_j'], rel_tol=1e-10, abs_tol=1e-6), 'peer setup raw power differs')
    result.update(qualification=reference, peer_setup=qualification['peer_setup'], peer_setup_energy_j=energy,
                  original_peer_mapping_restored=True)
    return result
