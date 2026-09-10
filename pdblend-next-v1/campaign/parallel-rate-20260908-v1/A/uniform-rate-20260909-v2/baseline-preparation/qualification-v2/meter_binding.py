"""Bind a qualified runtime whether its collector is old or already the new frozen collector."""
import argparse
import copy
from pathlib import Path
import sys
ROOT=Path('/root/workspace/pdblend-next-v1/campaign/parallel-rate-20260908-v1')
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p
ORIGINAL=ROOT/'C/uniform-rate-20260909-v2/meter_binding.py'
CLIENT='benchmarks/scripts/bench_vllm.py'
SCHEMA='uniform-existing-token-evidence-qualification-v2'


def native(binding_ref,validator_ref):
    b=p.checked(binding_ref)
    result=p.load(validator_ref,'native_qualified_existing_collector').verify(binding_ref)
    p.need(result['passed'] and result['independently_recomputed'] and result['binding']==binding_ref,'native qualification failed')
    files=dict(b['files']);files.update(result.get('files',{}))
    for path,digest in files.items():p.need(p.sha(path)==digest,'native evidence/source changed: '+path)
    return b,files


def verify(reference):
    q=p.checked(reference)
    if q.get('schema')!=SCHEMA:return p.load(ORIGINAL,'original_collector_binding').verify(reference)
    original,files=native(q['native_binding'],q['native_validator'])
    binding=p.checked(q['binding']);manifest=p.checked(q['host_manifest']);collector=p.checked(q['collector_manifest'])
    host=Path(original['host_release'])
    p.need(q['host_manifest']==p.ref(host/'manifest.json'),'native source lineage differs')
    p.need(manifest['files'][CLIENT]==collector['collector']['sha256']==p.sha(host/CLIENT),'existing collector is not the exact new source')
    for name,digest in manifest['files'].items():p.need(p.sha(host/name)==digest,'qualified host source changed: '+name)
    expected=copy.deepcopy(original);expected.update(output=str(Path(q['binding']['path']).parent/'results'))
    expected['files'].update(files)
    for ref in (q['native_binding'],q['native_validator'],q['host_manifest'],q['collector_manifest'],p.ref(__file__)):
        expected['files'][ref['path']]=ref['sha256']
    p.need(binding==expected,'undeclared change while binding existing collector')
    return dict(passed=True,independently_recomputed=True,binding=q['binding'],files=expected['files'])


def prepare(binding_ref,out,native_validator=None):
    p.need(native_validator and not Path(out).exists(),'fresh output and explicit native validator required')
    out=Path(out);original=p.checked(binding_ref);host=Path(original['host_release']);manifest_ref=p.ref(host/'manifest.json')
    manifest=p.checked(manifest_ref);collector_ref=p.ref(ROOT/'common/token-evidence-v2/manifest.json');collector=p.checked(collector_ref)
    current=manifest['files'][CLIENT]
    if current==collector['predecessor']['sha256']:
        return p.load(ORIGINAL,'original_collector_binding').prepare(binding_ref,out,native_validator)
    p.need(current==collector['collector']['sha256'],'unrecognized collector source')
    original,files=native(binding_ref,native_validator)
    binding=copy.deepcopy(original);binding.update(output=str(out/'results'));binding['files'].update(files)
    for reference in (binding_ref,native_validator,manifest_ref,collector_ref,p.ref(__file__)):
        binding['files'][reference['path']]=reference['sha256']
    p.save(out/'binding.json',binding)
    p.save(out/'qualified.json',dict(schema=SCHEMA,native_binding=binding_ref,native_validator=native_validator,
        binding=p.ref(out/'binding.json'),host_manifest=manifest_ref,collector_manifest=collector_ref,
        collector_already_present=True,controller_source_changed=False))
    reference=p.ref(out/'qualified.json');verify(reference);return reference

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--native-validator',type=Path,required=True);a=ap.parse_args()
    print(prepare(p.ref(a.binding),a.out,p.ref(a.native_validator)))
