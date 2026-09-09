"""Rescore retained observations against newly supplied SLO; no replay or energy alteration."""
import csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'B32B-budget-paired-longbench-v1'
POLICY=dict(protocol='user-slo-v4',scope='new user supplied thresholds, raw-only rescoring',
    thresholds={'alpaca':dict(ttft_s=1.,tpot_s=.1),'sharegpt':dict(ttft_s=5.,tpot_s=.15),'longbench':dict(ttft_s=15.,tpot_s=.2)},
    joint_attainment_target=.9,existing_source_files_are_read_only=True)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[];inputs={}
    for budget in (8192,2048):
        out=SOURCE/f'cell-longbench-budget{budget}';summary_path=out/'summary.json';bench_path=out/'bench.csv'
        summary=json.loads(summary_path.read_text());rows=list(csv.DictReader(bench_path.open()))
        inputs[str(summary_path)]=sha(summary_path);inputs[str(bench_path)]=sha(bench_path)
        expected=int(summary['n_expected']);assert len(rows)==expected==64
        requests=[]
        for r in rows:
            success=r['success']=='1';ttft=float(r['ttft_s']);tpot=float(r['tpot_s']);good=success and ttft<=15 and tpot<=.2
            requests.append(dict(idx=int(r['idx']),success=success,ttft_s=ttft,tpot_s=tpot,original_slo_ok=r['slo_ok']=='1',new_slo_ok=good,
                input_tokens=int(r['input_tokens'] or 0),output_tokens=int(r['generated_tokens'] or 0)))
        good=sum(x['new_slo_ok'] for x in requests);duration=summary['measurement_end_s']-summary['measurement_start_s']
        records.append(dict(budget=budget,dataset='longbench',source_dir=str(out),original_protocol='TTFT5/TPOT0.1/joint0.9',
            original_slo_attainment=summary['slo_attainment'],new_protocol='TTFT15/TPOT0.2/joint0.9',completed=summary['completed'],n_expected=expected,
            good_requests=good,slo_attainment=good/expected,slo_feasible=good/expected>=.9,energy_j=summary['energy_j'],
            energy_unchanged=True,energy_per_good_request_j=summary['energy_j']/good if good else None,
            goodput_measurement_rps=good/duration,measurement_duration_s=duration,measurement_valid=summary['measurement_valid'],
            work_complete=summary['work_complete'],failed_slo_request_indices=[x['idx'] for x in requests if not x['new_slo_ok']],
            requests=requests,execution_used_original_5s_0p1_planner=True,execution_replayed=False))
    result=dict(schema_version=1,policy=POLICY,source_sha256=inputs,records=records,
        limitation='Rescored observed traces retain the original runtime admission/frequency decisions made using5s/0.1s. This is not a replay under a newly configured planner.',
        original_temporal_gate='failed and unrepaired',energy_recomputed_or_changed=False)
    (ROOT/'policy.json').write_text(json.dumps(POLICY,indent=2)+'\n');(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    for p,h in inputs.items():assert sha(Path(p))==h
    for r in records:print(json.dumps({k:v for k,v in r.items() if k!='requests'}))
if __name__=='__main__':main()
