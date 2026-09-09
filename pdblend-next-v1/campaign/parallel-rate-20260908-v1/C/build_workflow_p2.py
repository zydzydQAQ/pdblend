"""Declare the same twelve complete workloads for the corrected p2 implementation."""
import copy
import hashlib
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent

def main():
    out=HERE/'workflow-p2';out.mkdir(exist_ok=False)
    old=json.loads((HERE/'work-declaration.json').read_text())
    d=copy.deepcopy(old);d.update(schema='parallel-rate-C-work-p2',implementation_series='parallel-rate-p2',created_s=time.time(),
        prior_failed_p1_retained=True,reason='Final idle-admission current-clock plan gate with persistent physical uncertainty')
    for c in d['cells']:c['cell_id']=c['cell_id'].replace('parallel-rate-p1-','parallel-rate-p2-')
    f=out/'work-declaration.json';f.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    digest=hashlib.sha256(f.read_bytes()).hexdigest()
    olddigest=hashlib.sha256((HERE/'work-declaration.json').read_bytes()).hexdigest()
    (out/'runner.py').write_text((HERE/'runner.py').read_text().replace(olddigest,digest))
    (out/'protocol.py').write_text((HERE/'protocol.py').read_text().replace('REPO = ROOT.parents[2]','REPO = ROOT.parents[3]'))
    prepare=(HERE/'prepare_release.py').read_text().replace("'observed_first_admission_frequency_v1': True", "'observed_idle_admission_frequency_v2': True")
    (out/'prepare_release.py').write_text(prepare)
    (out/'operate.py').write_text((HERE/'operate.py').read_text())
    (out/'lineage.json').write_text(json.dumps(dict(prior_declaration_sha256=olddigest,declaration_sha256=digest,
        unchanged_workloads=True,new_implementation_due_to_reproduced_diagnostic_wrapper_bypass=True),indent=2)+'\n')
    print(json.dumps(dict(workflow=str(out),declaration_sha256=digest,cells=len(d['cells']))))

if __name__=='__main__':main()
