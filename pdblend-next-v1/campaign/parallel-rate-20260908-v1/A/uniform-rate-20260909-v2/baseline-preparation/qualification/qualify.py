"""New physical node legacy bootstrap -> native gate -> candidate gate -> binding proofs."""
import argparse,copy,json,os,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p

def tree(path):return {str(x):p.sha(x) for x in Path(path).rglob('*') if x.is_file() and '__pycache__' not in x.parts}

def child(argv,out,state):
 state['current_argv']=argv;p.save(out/'status.json',state)
 with (out/(state['phase']+'.log')).open('xb') as log:
  process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  state['child']=dict(pid=process.pid,startticks=p.process_identity(process.pid)['startticks'],argv=argv);p.save(out/'status.json',state)
  interrupted=False
  def stop(sig,frame):
   nonlocal interrupted
   interrupted=True
   if process.poll() is None:process.send_signal(sig)
  for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
  code=process.wait();state['child'].update(exitcode=code,finished_s=time.time());p.save(out/'status.json',state)
 p.need(code==0 and not interrupted,'qualification child failed/stopped; keep its immutable raw')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bootstrap',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--node',required=True);ap.add_argument('--hostname',required=True);ap.add_argument('--profile',type=Path,required=True);ap.add_argument('--datasets',nargs='+');ap.add_argument('--policy-adapter',type=Path,default=HERE.parent/'policy_adapter.py');ap.add_argument('--run',action='store_true');a=ap.parse_args()
 p.need(a.run and not a.out.exists(),'fresh explicit qualification required')
 bootref=p.ref(a.bootstrap);boot=p.checked(bootref);profileref=p.ref(a.profile)
 source_manifest=p.ref(HERE/'source-files.json');sources=p.checked(source_manifest)
 for file,digest in sources.items():p.need(p.sha(file)==digest,'qualification source changed: '+file)
 p.need(boot['model']=='14b' and boot['hostname']==a.hostname and boot.get('fresh_node_native_identity') and not boot.get('old_node_qualification_inherited'),'fresh 14B physical bootstrap required')
 hetero=any(i['tp']==2 for i in boot['instances']);datasets=a.datasets or (['sharegpt'] if a.node=='B' else ['longbench'] if hetero else ['alpaca','longbench'])
 p.need(not hetero or datasets==['longbench'],'only measured original LB heterogeneous layout')
 a.out.mkdir(parents=True);deployment_bootstrap=bootref
 boot=copy.deepcopy(boot);boot['files'].update(sources);boot['files'][source_manifest['path']]=source_manifest['sha256'];boot['files'][deployment_bootstrap['path']]=deployment_bootstrap['sha256']
 a.bootstrap=a.out/'input-binding.json';p.save(a.bootstrap,boot);bootref=p.ref(a.bootstrap)
 state=dict(schema='uniform-v2-fresh-legacy-qualification-owner',node=a.node,model='14b',hostname=a.hostname,pid=os.getpid(),startticks=p.process_identity(os.getpid())['startticks'],started_s=time.time(),complete=False,node_lease_held=False,bootstrap=bootref,deployment_bootstrap=deployment_bootstrap,profile=profileref,phase='native',datasets=datasets)
 p.save(a.out/'status.json',state)
 try:
  runtime={p.read(i['engine_config'])['runtime_dir'] for i in boot['instances']};p.need(len(runtime)==1,'native runtime layout differs')
  gate=HERE/('heterogeneous_gate.py' if hetero else 'resident_gate.py')
  child([sys.executable,'-B',str(gate),'--binding',str(a.bootstrap),'--runtime-dir',runtime.pop(),'--out',str(a.out/'native'),'--run'],a.out,state)
  state['phase']='frequency';p.save(a.out/'status.json',state)
  child([sys.executable,'-B',str(HERE/'frequency.py'),'--binding',str(a.bootstrap),'--profile',str(a.profile),'--out',str(a.out/'frequency'),'--run']+([] if hetero else ['--retain-weights']),a.out,state)
  import verify_frequency
  frequency=verify_frequency.verify(a.out/'frequency',bootref,profileref);p.save(a.out/'frequency-audit.json',frequency)
  sys.path[:0]=[str(Path(boot['host_release'])/'src'),str(Path(boot['host_release'])),'/root/workspace/pdblend/.runtime-deps']
  from ecopadg.serving.measurement import power_evidence
  audit=p.load(ROOT.parent/'AC-baseline-binding-v2/gate_evidence.py','new_node_legacy_original_audit')
  adapter=p.load(a.policy_adapter,'new_node_legacy_policy_adapter');bindings={};state['phase']='derive';p.save(a.out/'status.json',state)
  measured=p.read(a.out/'frequency/status.json');topology=measured['topology'];retained=measured.get('retained_weights',{}).get('manifest')
  for system in (['distserve'] if hetero else ['mixed','distserve','dynamollm','ecoserve']):
   use=[d for d in datasets if not (d=='longbench' and system=='distserve' and not hetero)]
   if not use:continue
   proof,raw=audit.audit(a.out/'native',boot['instances'],system,power_evidence,hetero=hetero)
   # Pure configuration adapter freezes platform/source differences separately.
   policy_bindings={};binding=None
   for dataset in use:
    prepared=adapter.build(bootref,dataset,system,a.out/system/'policy'/dataset,profileref,topology_ref=topology,retained_weights_ref=retained)
    policy_bindings[dataset]=prepared['binding'];candidate=p.checked(prepared['binding'])
    if binding is None:binding=copy.deepcopy(candidate)
    else:
     p.need(candidate['instances']==binding['instances'] and candidate['host_release']==binding['host_release'],'dataset policy physical/source identity differs')
     binding['configs'].update(candidate['configs']);binding['files'].update(candidate['files'])
   p.need(binding['instances']==boot['instances'],'policy mapping changed actual native members')
   evidence=dict(schema='fresh-node-legacy-qualified-binding-v1',node=a.node,model='14b',bootstrap=bootref,deployment_bootstrap=deployment_bootstrap,profile=profileref,
    native_gate=str(a.out/'native'),frequency_gate=str(a.out/'frequency'),frequency_audit=p.ref(a.out/'frequency-audit.json'),heterogeneous=hetero,
    mechanism_proof=proof,policy_bindings=policy_bindings,policy_adapter=p.ref(a.policy_adapter),
    qualifier=p.ref(__file__),validator=p.ref(HERE/'verify.py'),frequency_validator=p.ref(HERE/'verify_frequency.py'),source_equivalence=p.ref(HERE/'native-source-equivalence.json'),owner_status=str(a.out/'status.json'),datasets=use,old_node_qualification_inherited=False)
   binding.update(output_correctness_verified=True,correctness_gate_required_before_performance=False,correctness_evidence=str(a.out/'native'),mechanism_proof=proof,fresh_legacy_qualification=evidence)
   binding['files'].update(raw);binding['files'].update(frequency['files']);binding['files'].update(tree(HERE))
   for ref in (bootref,profileref,*policy_bindings.values(),p.ref(a.out/'frequency-audit.json'),p.ref(a.policy_adapter)):
    binding['files'][ref['path']]=ref['sha256']
   if retained:binding.setdefault('large_inputs',{}).update(measured['retained_weights']['rank_files'])
   target=a.out/system/'binding.json';p.save(target,binding);bindings[system]=p.ref(target)
  p.save(a.out/'bindings.json',bindings);state.update(complete=True,phase='qualified',bindings=p.ref(a.out/'bindings.json'))
 except BaseException as exc:state.update(error=repr(exc),phase='stopped_failure');raise
 finally:state.update(finished_s=time.time(),node_lease_held=False);p.save(a.out/'status.json',state)
if __name__=='__main__':main()
