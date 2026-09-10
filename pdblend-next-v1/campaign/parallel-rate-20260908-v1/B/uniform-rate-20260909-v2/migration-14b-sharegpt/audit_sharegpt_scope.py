"""Compare the executed ShareGPT configuration with its frozen parent and A reference.

This establishes implementation scope, not cross-host performance equivalence.
Profiles and qualified physical recovery costs remain specific to each host.
"""
import argparse
from pathlib import Path
import pipeline_v4 as m
p = m.p
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def compare_configs(a, b, parent):
    shared = ('strategy', 'allow_pd', 'dvfs', 'dynamic_pools', 'slow_topology',
              'candidate_label', 'max_service_frequency_mhz', 'slo_ttft_s', 'slo_tpot_s',
              'max_pending', 'output_prior', 'decision_budget_s', 'power_mode',
              'manage_clocks', 'park_idle', 'idle_domain_reacquire_v1')
    p.need(all(a.get(k) == b.get(k) for k in shared), 'ShareGPT mechanism differs from the executed A reference')
    p.need(a.get('capacity_integration_v1') is not True and b.get('capacity_integration_v1') is False,
           'ShareGPT capacity mode differs from the original implementation')
    for config in (a, b, parent):
        geometry = [(i['tp'], i['gpus'], i['role']) for i in config['instances']]
        p.need(geometry == [(1, [6], 'mixed'), (1, [7], 'mixed')], 'original ShareGPT geometry changed')
    allowed = {'instances', 'port', 'journal', 'slo_tpot_s', 'slo_protocol'}
    p.need(b.get('slo_protocol') in (None, 'per-dataset-slo-v1') and a.get('slo_protocol') == b.get('slo_protocol'),
           'measurement SLO protocol changed')
    changed = {k for k in set(parent) | set(b) if parent.get(k) != b.get(k)}
    p.need(changed <= allowed and parent['slo_tpot_s'] == .1 and b['slo_tpot_s'] == .15,
           'unexplained deviation from frozen B P12 configuration')
    # Enriched physical identity is permitted; endpoint and native geometry are fixed.
    for old, new in zip(parent['instances'], b['instances']):
        for k in ('id', 'tp', 'gpus', 'role', 'port', 'kv_port', 'url', 'engine_config'):
            p.need(old.get(k) == new.get(k), 'B parent native identity changed: ' + k)
    return dict(shared_fields={k: b.get(k) for k in shared},
                parent_changed_fields=sorted(changed),
                cross_host_different_fields=sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)))


def compare_sources(a_host, b_host):
    def sources(root):
        return {str(x.relative_to(root)): p.sha(x) for x in root.rglob('*.py') if '__pycache__' not in x.parts}
    a, b = sources(a_host), sources(b_host)
    changed = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    p.need(changed == ['benchmarks/scripts/bench_vllm.py'], 'control source differs from original A ShareGPT')
    return dict(python_files=len(a), identical_control_and_support_files=len(a)-1,
                changed_sources=changed, a_sources=a, b_sources=b)


def build():
    a_dir = ROOT/'A/uniform-rate-20260909-v1/pipeline-003/0001-pdblend-sharegpt-r0p25/measurement'
    b_dir = HERE/'pipeline-007/0001-pdblend-sharegpt-r0.25-rep1-normal/measurement'
    a_runtime = next(a_dir.glob('results/cells/*repeat1/runtime_config.json'))
    b_runtime = next(b_dir.glob('results/cells/*/runtime_config.json'))
    a_binding = ROOT/'A/uniform-rate-20260909-v1/idle-qualification-002/binding.json'
    b_binding = HERE/'pipeline-007/meter-pdblend/binding.json'
    parent_path = ROOT/'B/distributed-14b-v1/pdb-p12-release-001/binding.json'
    a, b, parent = [p.checked(p.ref(x)) for x in (a_binding, b_binding, parent_path)]
    parent_config = Path(parent['configs']['sharegpt'])
    configs = compare_configs(p.checked(p.ref(a_runtime)), p.checked(p.ref(b_runtime)), p.checked(p.ref(parent_config)))
    sources = compare_sources(Path(a['host_release']), Path(b['host_release']))
    references = [p.ref(x) for x in (a_runtime, b_runtime, a_binding, b_binding, parent_path, parent_config, Path(__file__))]
    files = {x['path']: x['sha256'] for x in references}
    for prefix, values in ((a['host_release'], sources.pop('a_sources')), (b['host_release'], sources.pop('b_sources'))):
        files.update({str(Path(prefix)/k): v for k, v in values.items()})
    return dict(schema='B14-ShareGPT-original-implementation-scope-audit-v1', passed=True,
                node='B', model='14b', dataset='sharegpt', config_comparison=configs,
                source_comparison=sources, files=files,
                conclusion='Original ShareGPT uses two mixed TP1 instances with DVFS and idle recovery. Full capacity integration belongs to the separately qualified Alpaca configuration.',
                host_specific_differences=['Historical candidate profiles and frequency costs', 'Qualified idle reacquisition timeout: A 2s, B original 1.5s', 'B original fresh clock-failure confirmation and peer setup policy'],
                performance_comparability='B five systems measured on the same physical host; A performance observations are not reused.')


def verify(reference):
    proof = p.checked(reference)
    p.need(proof == build(), 'ShareGPT scope proof no longer matches immutable source evidence')
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); p.need(not args.out.exists(), 'new immutable audit output required')
    p.save(args.out, build()); print(p.ref(args.out))
