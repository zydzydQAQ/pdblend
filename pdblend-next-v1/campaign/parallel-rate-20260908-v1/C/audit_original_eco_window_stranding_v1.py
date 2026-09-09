"""Read-only request/window/tail-power proof for original C Eco failures."""
import csv,hashlib,json,time
from pathlib import Path
C=Path(__file__).resolve().parent;REPO=C.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
def main():
    snapshot=REPO/'campaign/five-system-results-v4/actual-snapshot-006/results.json'
    points=[]
    for p in read(snapshot)['points']:
        if not(p['model']=='7b' and p['system']=='ecoserve' and p['phase']=='main' and p['failed_requests']):continue
        cp=read(p['checkpoint_path']);directory=Path(p['receipt_path']).parents[2]/'cells'/p['cell_id']
        for path,h in cp['artifacts'].items():assert sha(path)==h
        bench=list(csv.DictReader((directory/'bench.csv').open()));events=[json.loads(s) for s in (directory/'control.jsonl').read_text().splitlines()]
        adm={str(e.get('client_request_id')):e for e in events if e.get('kind')=='admission'};windows={}
        for e in events:
            for action in e.get('plan',{}).get('windows',[]):windows.setdefault(action['instance_id'],[]).append(dict(at_s=e['at_s'],admit_prefill=action['admit_prefill'],client_request_id=e.get('client_request_id')))
        power=list(csv.DictReader((directory/'power.csv').open()));evidence=[]
        for b in bench:
            if b['success']=='1':continue
            assert b['error']=='request_hard_timeout' and b['request_timeout']=='True' and b['n_text_chunks']=='0' and not b['first_token_s']
            a=adm[b['request_id']];routes=a['plan']['routes'];assert len(routes)==1;rid=routes[0]['decode_id'];assert routes[0]['prefill_id']==rid
            after=[w for w in windows[rid] if a['at_s']<w['at_s']<float(b['finish_s'])];assert after and after[-1]['admit_prefill'] is False
            end=float(b['finish_s']);start=end-30;g=int(rid.removeprefix('base100cr'))
            tail=[float(v[f'gpu{g}_util_pct']) for v in power if start<=float(v['t_s'])<=end];assert tail
            tailmean=sum(tail)/len(tail);assert tailmean<1,'target GPU not idle; separate diagnosis required'
            evidence.append(dict(request_id=b['request_id'],controller_request_id=a['request_id'],rid=rid,admission_s=a['at_s'],last_window=after[-1],
                no_reopening_before_timeout=True,first_token=None,chunks=0,timeout_s=end,deadline_s=float(b['request_deadline_s']),target_gpu=g,tail30_mean_util_pct=tailmean,tail_samples=len(tail)))
        assert len(evidence)==p['failed_requests']==p['request_timeouts']
        points.append(dict(cell_id=p['cell_id'],checkpoint=ref(p['checkpoint_path']),receipt=ref(p['receipt_path']),raw_files=cp['artifacts'],
            scientific_eligible_false=True,scientific_comparison_eligible=False,raw_arithmetic_verified=True,raw_energy_retained=True,
            reason='Every failed request was admitted before an unreopened temporal OFF on its target; no first token, and target GPU last30s mean utilization below1% before the original timeout.',
            classification='ecoserve_temporal_prefill_window_liveness_defect',failed_request_evidence=evidence,original_energy_j=p['energy_j']))
    assert len(points)==6
    prior=read(C/'eco-drain37-v1/quarantine.json');new=[p for p in prior['points'] if p['cell_id'].startswith('parallel-rate-')];assert len(new)==1
    out=C/'eco-drain37-v1/quarantine-002.json';assert not out.exists()
    out.write_text(json.dumps(dict(schema='baseline-engineering-quarantine-v1',created_s=time.time(),points=points+new,source=ref(__file__),original_snapshot=ref(snapshot),supersedes_readonly_quarantine=ref(C/'eco-drain37-v1/quarantine.json'),no_original_raw_modified=True),indent=2)+'\n')
    print(json.dumps(dict(output=ref(out),count=len(points)+1,original_failed_requests=sum(len(p['failed_request_evidence']) for p in points))))
if __name__=='__main__':main()
