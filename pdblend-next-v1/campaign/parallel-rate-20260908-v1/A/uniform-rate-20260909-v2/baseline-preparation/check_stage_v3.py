"""Check all future dataset/system handoffs before the first baseline measurement."""
import argparse,signal,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'common/uniform-rate-20260909-v2'))
import support as p

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--definition',type=Path,required=True);a=ap.parse_args();definition=p.read(a.definition)
 process=subprocess.Popen([sys.executable,'-B',str(HERE/'stage_cached_v3.py'),'--definition',str(a.definition)])
 def stop(sig,frame):
  if process.poll() is None:process.send_signal(sig)
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
 p.need(process.wait()==0,'fresh baseline creation/qualification failed')
 actual={};expected={d+':'+s for d,s in definition['groups']}
 p.need(set(definition['handoffs'])==expected,'declared future dataset/system handoff set differs')
 for key,path in definition['handoffs'].items():
  dataset,system=key.split(':');handoff=p.checked(p.ref(path));q=p.checked(handoff['qualification']);binding=p.checked(q['binding'])
  p.need(handoff['node']=='Anew20260909' and handoff['model']==binding['model']=='14b' and handoff['system']==binding['system']==system,'handoff system/model changed')
  p.need(dataset in binding['configs'] and binding['hostname']=='iZwz9274emxme9019d2sjgZ','actual handoff missing dataset or wrong physical host')
  config=p.read(binding['configs'][dataset]);p.need(config['max_service_frequency_mhz']==2100 and config['comparison_system']==system,'wrong actual platform/system config')
  p.need(binding['files'].get(binding['configs'][dataset])==p.sha(binding['configs'][dataset]),'actual dataset config not frozen')
  actual[key]=dict(handoff=p.ref(path),binding=q['binding'],configuration=p.ref(binding['configs'][dataset]))
 p.save(Path(definition['out'])/'handoff-matrix-verification.json',dict(passed=True,cpu_only=True,node='Anew20260909',model='14b',groups=actual,all_declared_groups_present=True))
if __name__=='__main__':main()
