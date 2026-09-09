"""B-only baseline192 gate using the qualified temporal release contract.

Missing future publication or verifier files mean pending, never permission to
skip baselines. No imports of serving runtime, hardware APIs or A/C scale code.
"""
import importlib.util
import json
from pathlib import Path
import socket
import time

import readiness as shared

ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent
PUBLICATION = 'B32B-baseline-completion.gate.json'
KIND = 'B32B-baseline-192-terminal-publication-v1'
RELEASE_KIND = 'model-main-release-per-cell-qualified-temporal-v2'
HOSTNAME = 'iZwz9i5bte3xkpmcoes3t2Z'
CONTRACT = 'scale-only-continuation-B32B-v1/contract.py'
CONTRACT_SHA = '482dba053bb9fa95ec895c7bcac6c51311b0b04e6887fcfb346be42fb62fff2b'
QUALIFIED_RELEASE = 'B32B-qualified-main-release-v1/release.py'
QUALIFIED_RELEASE_SHA = '39e408a369c9e7eec9e60c8978524b9a31013d8f76230f3d690baabdc79b0a22'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def read_ref(ref, campaign):
    require(isinstance(ref, dict) and set(ref) == {'path', 'sha256'}, 'explicit path/SHA reference required')
    path = Path(ref['path'])
    require(path.is_absolute() and campaign.resolve() in path.resolve().parents,
            'publication evidence must be inside this campaign')
    digest = ref['sha256']
    require(isinstance(digest, str) and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest),
            'invalid publication SHA')
    require(shared.sha(path) == digest, 'published evidence changed: ' + str(path))
    return shared.read(path)


def baseline_processes(proc_root=Path('/proc')):
    rows = {r['pid']: r for r in shared.process_rows(proc_root)}
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            args = [a.decode(errors='replace') for a in path.read_bytes().split(b'\0') if a]
        except (FileNotFoundError, ProcessLookupError):
            continue
        if not args or 'python' not in Path(args[0]).name or any('\n' in a for a in args):
            continue
        scripts = [a for a in args[1:] if 'campaign/' in a and a.endswith('.py')]
        if any(('/B32B-ecoserve-qualified-main-launch-' in '/' + a.lstrip('/') and Path(a).name == 'launch.py')
               or ('/B32B-baseline-main-first-sequence-' in '/' + a.lstrip('/') and Path(a).name == 'supervise.py')
               or ('/B32B-main-scale-bridge-' in '/' + a.lstrip('/') and Path(a).name in ('bridge.py', 'watch.py'))
               for a in scripts):
            rows[int(path.parent.name)] = dict(pid=int(path.parent.name), argv=args)
    return list(rows.values())


def load_contract(campaign=CAMPAIGN, package=ROOT):
    manifest = shared.read(package / 'execution-manifest-b32.json')
    dependencies = manifest['gate_dependencies']
    require(dependencies.get(str(campaign / CONTRACT)) == CONTRACT_SHA and
            dependencies.get(str(campaign / QUALIFIED_RELEASE)) == QUALIFIED_RELEASE_SHA,
            'B-only qualified verifier pins missing or changed')
    require(not any('/scale-only-continuation-v3/' in p for p in dependencies),
            'A/C scale verifier must not authorize B')
    for path, digest in dependencies.items():
        require(shared.sha(path) == digest, 'B baseline verifier unavailable/changed: ' + path)
    path = campaign / CONTRACT
    spec = importlib.util.spec_from_file_location('ablation_B32B_qualified_scale_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inspect_readiness(model='32b', *, package=ROOT, campaign=CAMPAIGN,
                      proc_root=Path('/proc'), hostname=None,
                      process_reader=baseline_processes, contract_loader=load_contract):
    result = dict(model=model, observed_s=time.time(), ready=False, reasons=[], evidence={},
        hardware_actions=False, requires_fresh_node_lease_recheck=True,
        publication_path=str(package / PUBLICATION))
    try:
        require(model == '32b', 'B-only readiness adapter rejects other models')
        require((hostname or socket.gethostname()) == HOSTNAME, 'B gate must run on its actual model host')
        path = package / PUBLICATION
        require(path.is_file(), 'pending B publication: ' + str(path) + '; requires all baseline main120 + scale72, not Eco main30 alone')
        publication_sha = shared.sha(path)
        gate = shared.read(path)
        require(gate.get('schema') == 1 and gate.get('kind') == KIND and gate.get('model') == '32b'
                and gate.get('hostname') == HOSTNAME, 'wrong B publication schema/model/host')
        require(gate.get('baseline_systems') == list(shared.BASELINES), 'publication must cover exactly four baselines')
        groups = gate.get('scale_parents')
        require(isinstance(groups, list) and bool(groups), 'no actual scale parent references; Eco main30 is insufficient')
        release = read_ref(gate['release'], campaign)
        require(release.get('kind') == RELEASE_KIND, 'B requires its explicit qualified-temporal release')
        parents = []
        specs = []
        for group in groups:
            spec = read_ref(group['spec'], campaign)
            read_ref(group['status'], campaign)
            require(spec.get('release') == gate['release'], 'scale spec refers to a different release')
            specs.append(spec)
            parents.append(shared.terminal_parent(Path(group['spec']['path']), Path(group['status']['path']),
                gate['release']['sha256'], proc_root))
        live = process_reader(proc_root)
        result['evidence']['live_queues'] = live
        require(not live, 'B baseline serving driver or automatic successor parent is still alive')
        contract = contract_loader(campaign, package)
        contract.released.verify_release(gate['release']['path'], gate['release']['sha256'],
            expected_model='32b', deep=True)
        records = release['models']['32b']['records']
        rows = [r['row'] for r in records if r['row']['system'] in shared.BASELINES]
        require(len(rows) == 120 and all(r['phase'] == 'main' for r in rows), 'all baseline main120 required')
        scale_count = 0
        for spec in specs:
            checked = contract.check_spec(spec, gate['release']['path'], gate['release']['sha256'])
            require(not checked['pending'], 'actual B scale checkpoints still pending')
            for group in checked['groups']:
                if group['group']['system'] in shared.BASELINES:
                    require(not group['pending'], 'baseline scale group remains pending')
                    require(all(r['phase'] == 'scale' for r in group['rows']), 'non-scale rows in scale proof')
                    rows.extend(group['rows'])
                    scale_count += len(group['rows'])
        require(scale_count == 72, 'all four baseline scale72 required; PDB reuse18 is not baseline work')
        shared.validate_domain(rows, '32b')
        require(shared.sha(path) == publication_sha, 'publication changed during deep verification')
        read_ref(gate['release'], campaign)
        after = []
        for group in groups:
            read_ref(group['spec'], campaign)
            read_ref(group['status'], campaign)
            after.append(shared.terminal_parent(Path(group['spec']['path']), Path(group['status']['path']),
                gate['release']['sha256'], proc_root))
        require(after == parents and not process_reader(proc_root), 'B parent/process state changed during verification')
        result.update(ready=True, publication_sha256=publication_sha)
        result['evidence'].update(parents=parents, release=gate['release'],
            actual_completion=dict(raw_verified=True, verified_baseline_main=120,
                verified_baseline_scale=72, total=192,
                work_and_slo_not_completion_gates=True))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError, ImportError) as exc:
        result['reasons'].append(str(exc))
    return result
