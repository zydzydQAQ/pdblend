"""Six existing CPU-only raw checks, including a valid incomplete B SG2 point.

Does not generate a host proof or a global release.
"""
import json
from pathlib import Path
import barrier as b

def record(binding_path,dataset=None):
    binding=b.read(binding_path);out=Path(binding['output']);rows=[]
    for path in (out/'checkpoints').glob('*.json'):
        cp=b.read(path)
        if cp['row']['phase']=='main' and (dataset is None or cp['row']['dataset']==dataset):rows.append((path,cp))
    path,cp=max(rows,key=lambda x:x[1]['row']['rate_rps']);row=cp['row'];receipt=b.read(cp['receipt']);s=receipt['summary'];config=binding['configs'][row['dataset']]
    return dict(row=row,checkpoint=str(path),checkpoint_sha256=b.sha(path),output=str(out),binding=str(binding_path),binding_sha256=b.sha(binding_path),
        config=config,config_sha256=binding['files'][config],measurement_valid=True,work_complete=s['work_complete'],n_expected=row['n_requests'],
        good_requests=s['good_requests'],completed_work_requests=s['completed_work_requests'],energy_j=s['energy_j'],
        implementation_variant=s['implementation_variant'],receipt_sha256=cp['receipt_sha256'],
        actual_config_sha256=cp['artifacts'][str(out/'cells'/row['cell_id']/'runtime_config.json')],finished_s=receipt['finished_s'],child_pid=receipt['child_pid'])

if __name__=='__main__':
    campaign=Path(__file__).resolve().parent.parent
    cases=[('A14B-five-system100-v1/pdblend/binding.json',None),('C7B-five-system100-v1/pdblend-r2/binding.json',None),
        ('B32B-five-system100-v1/binding.pdblend.r2.json','sharegpt'),
        ('A14B-five-system100-baselines-v1/bindings/mixed-resident/binding.json',None),
        ('C7B-five-system100-baselines-v1/bindings/mixed-resident/binding.json',None),
        ('B32B-baseline-sequence-v1/attempt-001/bindings/mixed/binding.json',None)]
    results=[]
    for path,dataset in cases:
        r=record(campaign/path,dataset);reader=b.Reader();result=b.verify_record(r,reader);reader.stable()
        results.append(dict(cell_id=r['row']['cell_id'],binding=str(campaign/path),result=result,files=reader.files))
        print(json.dumps(dict(cell_id=r['row']['cell_id'],passed=True,work_complete=result['work_complete'],raw_files=len(reader.files))),flush=True)
    out=Path(__file__).with_name('actual-validation.json')
    b.write_new(out,dict(cpu_only=True,gpu_executed=False,global_release_created=False,real_raw_cases=results))
