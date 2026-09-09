import importlib.util,json,ast
from pathlib import Path
import pytest
from types import SimpleNamespace
ROOT=Path('/root/workspace/pdblend-next-v1/campaign/main-slo-improvement-v6/B')

def load(path):
 spec=importlib.util.spec_from_file_location(path.parent.name.replace('-','_')+'_'+path.stem,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('name,batch,inp,outp,kv',[('profile-D6-long7168-001',6,7168,512,46080),('profile-D16-mid2048-001',16,2048,512,40960)])
def test_full_context_capacity_with_sequential_prefill(name,batch,inp,outp,kv):
 p=ROOT/name;m=load(p/'capacity.py')
 assert (m.BATCH,m.INPUT,m.OUTPUT,m.REQUIRED_KV_TOKENS)==(batch,inp,outp,kv)
 r=dict(timestamp=100.,generation=7,acknowledged_generation=7,acknowledged_generations=[7],observed_control_generation=7,scheduler_budget_pending=None,scheduler_io=[{'controls':{'runtime':{'generation':7,'error':None}}}],scheduler_budget_effective={'max_num_batched_tokens':8192,'max_num_seqs':32},scheduler_budget={'schema_version':1,'max_num_batched_tokens':8192,'max_num_seqs':32},scheduler_budget_limits={'max_num_batched_tokens':8192,'max_num_seqs':32,'max_model_len':8192,'max_num_partial_prefills':1},role='mixed',mode='continuous',accepting=True,admit_prefill=True,admit_decode=True,transport_healthy=True,active=0,running=0,waiting=0,transfer_buffered_tensors=0,transfer_inflight_receives=0,transfer_inflight_sends=0,kv_allocations={},transfer_allocations={},free_kv_tokens=47104,total_kv_tokens=47104)
 proof=m.validate_runtime_capacity(r,now=100.)
 assert proof['required_kv_tokens']==kv and proof['initial_prefill_token_sum']>8192
 with pytest.raises(RuntimeError,match='free KV'):m.validate_runtime_capacity(dict(r,free_kv_tokens=kv-1),now=100.)
 with pytest.raises(RuntimeError):m.validate_runtime_capacity(dict(r,scheduler_budget_effective={'max_num_batched_tokens':16384,'max_num_seqs':32}),now=100.)
 tree=ast.parse((p/'observation.frozen.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arguments');ns=dict(Path=Path,SimpleNamespace=SimpleNamespace,ROOT=p);exec(compile(ast.Module(body=[fn],type_ignores=[]),'arguments','exec'),ns)
 args=ns['arguments']();assert args.input_patterns==[[inp]] and args.output_pattern==[outp] and args.batches==[batch] and args.repeats==3 and args.frequencies==[1500,2520] and args.budgets==[8192]
 declaration=json.loads((p/'declaration.json').read_text());assert declaration['full_decode_steps_per_request']==511 and declaration['required_kv_tokens']==kv
 assert len(declaration['phases'])==6 and all(x['arrival_offsets_s']==[0.]*batch for x in declaration['phases'])
