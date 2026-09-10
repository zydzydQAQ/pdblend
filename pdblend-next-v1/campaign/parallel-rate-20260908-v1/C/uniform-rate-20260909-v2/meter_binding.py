"""Append-only collector substitution around an independently qualified native binding."""
import argparse
import copy
from pathlib import Path
import shutil
import sys

ROOT = Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0, str(ROOT / 'common/uniform-rate-20260909-v2'))
import support as p
CLIENT = 'benchmarks/scripts/bench_vllm.py'


def native(reference, verifier=None):
    b = p.checked(reference)
    if verifier:
        result = p.load(verifier, 'native_baseline_qualification').verify(reference)
        p.need(result['passed'] and result['independently_recomputed'], 'native qualification failed')
        p.need(result['binding'] == reference, 'different native binding')
    else:
        p.need(b['model'] == '7b' and 'fresh_native_qualification' in b, 'C fresh native qualification required')
        evidence = b['fresh_native_qualification']
        old = p.checked(evidence['selected_source'])
        restart = p.checked(b['fresh_restart'])
        fresh = p.checked(restart['binding'])
        p.need(restart['complete'] and not restart['node_lease_held'] and not p.active_owner(restart), 'restart owner active')
        p.need(p.checked(restart['restoration'])['correctness']['passed'], 'ordinary correctness failed')
        p.need(b['instances'] == fresh['instances'] and b['configs'] == old['configs']
               and b['host_release'] == old['host_release'] and b['system'] == old['system'], 'native policy changed')
        helper = p.load(ROOT / 'B/baseline-return-after-external-source-v1/execution.py', 'native_source_loader')
        helper.load_common(b['host_release'])
        from ecopadg.serving.measurement import power_evidence
        audit = p.load(p.REPO / 'campaign/AC-baseline-binding-v2/gate_evidence.py', 'native_raw_gate')
        strategy = 'dynamollm-resident' if b['system'] == 'dynamollm' else b['system']
        proof, files = audit.audit(Path(evidence['gate']['path']).parent, b['instances'], strategy, power_evidence)
        p.need(proof == b['mechanism_proof'], 'fresh mechanism proof differs from raw')
        p.need(all(b['files'].get(x) == h for x, h in files.items()), 'native raw closure absent')
    for path, digest in b['files'].items():
        p.need(p.sha(path) == digest, 'native source/evidence changed: ' + path)
    return b


def verify(reference):
    q = p.checked(reference)
    original = native(q['native_binding'], q.get('native_validator'))
    b = p.checked(q['binding'])
    parent = Path(original['host_release'])
    host = Path(b['host_release'])
    prior = p.checked(q['parent_manifest'])
    manifest = p.checked(q['host_manifest'])
    collector = p.checked(q['collector_manifest'])
    p.need(q['parent_manifest'] == p.ref(parent / 'manifest.json'), 'wrong source parent')
    p.need(manifest['parent_release'] == str(parent) and manifest['parent_manifest_sha256'] == q['parent_manifest']['sha256'], 'source lineage differs')
    p.need(set(prior['files']) == set(manifest['files']), 'source file set changed')
    p.need([n for n in prior['files'] if prior['files'][n] != manifest['files'][n]] == [CLIENT], 'only collector may change')
    p.need(prior['files'][CLIENT] == collector['predecessor']['sha256'], 'unexpected client parent')
    p.need(manifest['files'][CLIENT] == collector['collector']['sha256'] == p.sha(host / CLIENT), 'wrong added collector')
    for name, digest in manifest['files'].items():
        p.need(p.sha(host / name) == digest, 'runtime copy changed: ' + name)
    expected = copy.deepcopy(original)
    expected.update(host_release=str(host), output=str(Path(q['binding']['path']).parent / 'results'))
    expected['files'].update({str(host / name): digest for name, digest in manifest['files'].items()})
    for ref in (q['native_binding'], q['parent_manifest'], q['host_manifest'], q['collector_manifest'], p.ref(__file__)):
        expected['files'][ref['path']] = ref['sha256']
    if q.get('native_validator'):
        expected['files'][q['native_validator']['path']] = q['native_validator']['sha256']
    p.need(b == expected, 'qualified binding contains an undeclared change')
    return dict(passed=True, independently_recomputed=True, binding=q['binding'])


def prepare(binding_ref, out, native_validator=None):
    p.need(not out.exists(), 'fresh collector qualification directory required')
    original = native(binding_ref, native_validator)
    parent = Path(original['host_release'])
    prior = p.read(parent / 'manifest.json')
    collector_ref = p.ref(ROOT / 'common/token-evidence-v2/manifest.json')
    collector = p.checked(collector_ref)
    p.need(prior['files'][CLIENT] == collector['predecessor']['sha256'], 'unexpected client parent')
    host = out / 'runtime'
    for name, digest in prior['files'].items():
        p.need(p.sha(parent / name) == digest, 'parent runtime source differs')
        (host / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(collector['collector']['path'] if name == CLIENT else parent / name, host / name)
    manifest = dict(schema='uniform-token-evidence-runtime-v2', parent_release=str(parent),
                    parent_manifest_sha256=p.sha(parent / 'manifest.json'), collector_manifest=collector_ref,
                    changed_runtime_files=[CLIENT], files={name:p.sha(host / name) for name in prior['files']})
    p.save(host / 'manifest.json', manifest)
    b = copy.deepcopy(original)
    b.update(host_release=str(host), output=str(out / 'results'))
    b['files'].update({str(host / name): digest for name, digest in manifest['files'].items()})
    refs = (binding_ref, p.ref(parent / 'manifest.json'), p.ref(host / 'manifest.json'), collector_ref, p.ref(__file__))
    for ref in refs + ((native_validator,) if native_validator else ()):
        b['files'][ref['path']] = ref['sha256']
    p.save(out / 'binding.json', b)
    q = dict(schema='uniform-token-evidence-qualification-v2', native_binding=binding_ref,
             native_validator=native_validator, binding=p.ref(out / 'binding.json'),
             parent_manifest=refs[1], host_manifest=refs[2], collector_manifest=collector_ref)
    p.save(out / 'qualified.json', q)
    verify(p.ref(out / 'qualified.json'))
    return p.ref(out / 'qualified.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--binding', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--native-validator', type=Path)
    args = parser.parse_args()
    print(prepare(p.ref(args.binding), args.out, p.ref(args.native_validator) if args.native_validator else None))
