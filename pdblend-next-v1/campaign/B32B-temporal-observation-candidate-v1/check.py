"""Default is CPU/hash only; no execution, stage or deployment mode."""
import ast,hashlib,json
from pathlib import Path
from build_candidate import ROOT,PARENT,patch
from pdblend_diagnostics import validate_spec

def check():
    c=json.loads((ROOT/'source-contract.json').read_text())
    for path,digest in c['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    for path in c['source_provenance_refs']:
        assert Path(path).is_file(),path
    assert patch(PARENT.read_text())==(ROOT/'model_runner.py').read_text()
    for p in ROOT.glob('*.py'):ast.parse(p.read_text())
    specs=list((ROOT/'specs').glob('*.json'));assert len(specs)==4
    for p in specs:
        s=validate_spec(json.loads(p.read_text()));r=s['requests']
        assert len({x['request_uuid'] for x in r})==len(r)
        assert set(s['request_ids']) <= {x['request_uuid'] for x in r}
        for x in r:
            b=x['body'];assert b['max_tokens']==64 and b['temperature']==0 and b['top_p']==1 and b['seed']==0 and b['ignore_eos'] is True and b['stream'] is False
            assert b['prompt']==([9707,1879,13]*65)[:x['prompt_length']]
        assert s['new_hardware_authorized'] is False and s['all_pairs_exact_single_oracle'] is True
    m=ROOT/'manifest.json'
    if m.exists():
        for path,digest in json.loads(m.read_text())['files'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    return dict(cpu_hash_check_passed=True,source_refs=len(c['files']),future_specs=len(specs),
                hardware_executed=False,capture_complete=False,serving_fix=False,
                deployment_ready=False,current_baseline_modified=False)
if __name__=='__main__':print(json.dumps(check(),indent=2))
