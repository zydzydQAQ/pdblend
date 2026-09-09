import asyncio,copy,hashlib,importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parent

def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('b_matrix',ROOT/'run.py');c=load('b_contract',ROOT/'contract.py')

def test_complete_matrix_matches_frozen_generator_source():
 rows=c.validate(b.read(ROOT/'runspec.json'),b.read(ROOT/'inputs/source-sweep-manifest.json'),b.read(ROOT/'source-contract.json'),b.read,b.sha)
 assert len(rows)==171 and {r['seed'] for r in rows}=={701,1701,2701}
 assert min(r['n_requests'] for r in rows)>=1000 and min(r['trace_duration_s'] for r in rows)>=300
 assert len({r['controller_config'] for r in rows})==1
 source=[r['source_cell'] for r in rows]
 for dataset in ('alpaca','sharegpt','longbench'):
  rates={x['rate_rps'] for x in source if x['dataset']==dataset};assert len(rates)==19
  for rate in rates:
   matches=[x for x in source if x['dataset']==dataset and x['rate_rps']==rate]
   assert len(matches)==3 and len({x['content_pairing_sha256'] for x in matches})==1
   assert len({x['n_requests'] for x in matches})==1
 assert all(x['within_trace_resampling'] and x['unique_selected_pool_records']==256 and not x['formal_eligible'] for x in source)


def test_execution_and_cleanup_are_exact_reviewed_r2_bytes():
 other=ROOT.parent/'B32B-load-screen-user-slo-v1-r2/execution.py'
 assert b.sha(ROOT/'execution.py')==b.sha(other)=='faaa80fdfcb6e406596a5bfe8487d7e3e944342472cacb62ee173ac1a1490a26'
 cfg=b.read(b.CONFIG);assert cfg['output_prior']==211 and not cfg['allow_pd'] and not cfg['dynamic_pools'] and not cfg['slow_topology']
 assert cfg['scheduler_budget_ablation']==dict(schema_version=1,max_num_batched_tokens=8192,max_num_seqs=32)
 assert b.read(ROOT/'source-contract.json')['requests_exact'] is None


@pytest.mark.parametrize('dataset',['alpaca','sharegpt','longbench'])
def test_actual_controller_receives_new_user_slo_before_any_network(tmp_path,monkeypatch,dataset):
 from ecopadg.serving import cell
 child=load('b_matrix_child',ROOT/'child.py');row=next(x for x in b.read(ROOT/'runspec.json')['cells'] if x['dataset']==dataset)
 args=child.cell_arguments(row);args.out=tmp_path/'out'
 seen=[]
 class BeforeNetwork(Exception):pass
 def constructor(config):seen.append(copy.deepcopy(config));raise BeforeNetwork()
 monkeypatch.setattr(cell,'Controller',constructor)
 with pytest.raises(BeforeNetwork):asyncio.run(cell.run_cell(args))
 assert seen and seen[0]['slo_ttft_s']==row['slo_ttft_s'] and seen[0]['slo_tpot_s']==row['slo_tpot_s']
 assert seen[0]['output_prior']==211 and seen[0]['scheduler_budget_ablation']['max_num_batched_tokens']==8192
