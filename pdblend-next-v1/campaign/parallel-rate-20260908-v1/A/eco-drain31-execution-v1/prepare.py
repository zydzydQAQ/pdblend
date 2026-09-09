"""Freeze only an explicitly scoped, restored and fresh-qualified A Eco queue."""
import argparse,copy,json,time
from pathlib import Path
from contract import *

def prepare(scope_ref,binding_ref,out):
 need(not out.exists(),'new immutable execution release required')
 package=read(HERE/'manifest.json')
 for path,h in package['files'].items():need(sha(path)==h,'draft execution source changed')
 scope,base,proof=validate_scope(scope_ref,binding_ref)
 need(sha(COMMON)==COMMON_SHA,'original fixed100 execution changed')
 files={**base['files'],**package['files']}
 references=[ref(HERE/'manifest.json'),scope_ref,binding_ref,scope['logical_declaration'],scope['final_pdb_terminal_audit'],scope['final_pdb_release'],ref(COMMON),ref(COMMON.parent/'child.py'),ref(COMMON.parent/'manifest.json'),*scope.get('append_declarations',[]),*proof['qualification']['original_policy_references']]
 for reference in references:files[reference['path']]=reference['sha256']
 for row in scope['required_cells']:
  need(sha(row['trace_path'])==row['trace_sha256'],'original declared trace changed');files[row['trace_path']]=row['trace_sha256']
 rules=dict(schema='A-Eco-fixed100-arrival-engineering-v1',scope=scope_ref,arrival_window_s=100,request_timeout_s=120,cleanup_local_budget_s=90,all8gpu_power=True,max_dispatch_lateness_s=1.,p99_dispatch_lateness_s=.1,max_handler_lateness_s=1.,normal_low_SLO_is_valid=True,any_request_failure_stops_queue=True,no_automatic_capacity_failure_waiver=True)
 write(out/'execution-rules.json',rules);files[str(out/'execution-rules.json')]=sha(out/'execution-rules.json')
 qualified=copy.deepcopy(base);qualified.update(output=str(out/'unused'),deadline_s=None,campaign_lifecycle='until_declared_complete_v1',files=files,formal_eligible=False)
 write(out/'binding.json',qualified)
 release=dict(schema='A-Eco-scoped-qualified-execution-release-v1',created_s=time.time(),ready_for_gpu=True,model='14b',hostname=base['hostname'],host_release=str(HOST),host_manifest=ref(HOST/'manifest.json'),logical_declaration=scope['logical_declaration'],declaration=scope_ref,binding=ref(out/'binding.json'),qualification_binding=binding_ref,proof=proof,execution_rules=ref(out/'execution-rules.json'),common_executor=ref(COMMON),source_package=ref(HERE/'manifest.json'),files=files,required_count=len(scope['required_cells']),logical_count=31,deadline_s=None,campaign_lifecycle='until_declared_complete_v1')
 write(out/'release.json',release);return ref(out/'release.json')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--scope',type=Path,required=True);parser.add_argument('--scope-sha256',required=True);parser.add_argument('--binding',type=Path,required=True);parser.add_argument('--binding-sha256',required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();print(json.dumps(prepare(dict(path=str(args.scope.resolve()),sha256=args.scope_sha256),dict(path=str(args.binding.resolve()),sha256=args.binding_sha256),args.out.resolve())))
