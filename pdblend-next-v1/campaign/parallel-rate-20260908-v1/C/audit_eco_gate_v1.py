"""Recompute the eight-TP1 Eco gate solely from frozen source and raw files."""
import collections,hashlib,importlib.util,json,sys,time
from pathlib import Path
C=Path(__file__).resolve().parent;REPO=C.parents[2]
READER=REPO/'campaign/AC-baseline-binding-v2/gate_evidence.py'
READER_SHA='b84a113563f1b064be5ef7a8cbf2006b0790f3b60bb1be3851ce66114dd9e9e6'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def audit(binding_ref):
    assert sha(binding_ref['path'])==binding_ref['sha256'];b=read(binding_ref['path'])
    assert b['system']=='ecoserve' and len(b['instances'])==8 and [i['gpus'] for i in b['instances']]==[[j] for j in range(8)]
    assert all(i['tp']==1 for i in b['instances'])
    for path,h in b['files'].items():assert sha(path)==h,'changed binding input '+path
    host=Path(b['host_release']);manifest=host/'manifest.json';m=read(manifest)
    assert b['files'][str(manifest)]==sha(manifest)
    for name,h in m['files'].items():assert sha(host/name)==h
    assert sha(READER)==READER_SHA
    s=importlib.util.spec_from_file_location('original_ac_gate_raw_reader',READER);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
    sys.path[:0]=[str(host/'src'),str(host),'/root/workspace/pdblend/.runtime-deps']
    from ecopadg.serving.measurement import power_evidence
    gate=Path(b['correctness_evidence']);proof,files=g.audit(gate,b['instances'],'ecoserve',power_evidence)
    assert proof==b['mechanism_proof'] and all(b['files'].get(p)==h for p,h in files.items())
    checks=read(gate/'checks/checks.json');counts=collections.Counter(r['label'] for r in checks['requests'])
    assert len(checks['requests'])==55 and len({r['request_id'] for r in checks['requests']})==55
    return dict(schema='AC-eight-TP1-Eco-raw-gate-recomputation-v1',passed=True,cpu_only=True,gpu_actions=False,
        binding=binding_ref,source=ref(manifest),gate_status=ref(gate/'status.json'),reader=ref(READER),auditor=ref(__file__),
        actual_request_records=55,request_label_counts=dict(counts),request_ids=[r['request_id'] for r in checks['requests']],
        actual_mechanism=proof,files=files,actual_identity_verified_from_before_after=True,live_layout_not_accessed=True,
        scope='original eight-TP1 deterministic ordinary/PD/cancel/temporal protocol; no general numerical correctness claim')
if __name__=='__main__':
    result=audit(ref(Path(sys.argv[1])));out=Path(sys.argv[2]);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=True,requests=result['actual_request_records'],labels=result['request_label_counts'],physical=result['actual_mechanism']['physical'],output=ref(out))))
