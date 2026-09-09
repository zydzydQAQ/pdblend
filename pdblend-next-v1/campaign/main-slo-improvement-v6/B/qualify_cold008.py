"""Read completed physical data and derive three-repeat transition bounds only."""
from pathlib import Path
import sys,json,hashlib,time
R=Path(__file__).resolve().parent
CODE=R/'cold-code-008'
sys.path[:0]=[str(CODE),'/root/workspace/pdblend-next-v1/releases/io-v3-runtime/src','/root/workspace/pdblend-next-v1/releases/io-v3-runtime','/root/workspace/pdblend/.runtime-deps']
from capacity_certificate import derive_group
from capacity_executor import durable,sha

def ref(p):return dict(path=str(p),sha256=sha(p))
def main():
 out=R/'cold-calibration-008';status=json.loads((out/'status.json').read_text());inventory=json.loads((out/'inventory.json').read_text())
 assert status['complete'] is True and status['phase']=='measured_complete' and status['completed']==[1,2,3] and not status['failed'] and status['clock_restore_complete'] is True
 assert inventory['complete'] is True and inventory['transition_inflight'] is False
 assert {i['id'] for i in inventory['active_instances']}==set(inventory['initial_ids'])
 launch=json.loads((R/'cold-launch-008.json').read_text());assert not Path('/proc',str(launch['pid'])).exists()
 binding=R/'cold-calibration-binding-008.json';identity=json.loads(binding.read_text())['identity']
 output=R/'cold-qualified-008';output.mkdir(exist_ok=False)
 result=dict(schema='capacity-cold-three-repeat-qualification-v1',observed_s=time.time(),production_ready=False,layout_capacity_calibrated=False,matched_savings_calibrated=False,binding=ref(binding),status=ref(out/'status.json'),inventory=ref(out/'inventory.json'),bounds=[])
 for operation in ['restore_cold','remove']:
  suffix='restore' if operation=='restore_cold' else 'remove'
  group=dict(schema='capacity-evidence-group-v1',kind='transition',identity=identity,capacity_binding=ref(binding),operation=operation,gpus=[4,5],members=[dict(result=ref(out/f'cycle-{i}-{suffix}.json'),inventory=ref(out/'inventory.json')) for i in (1,2,3)])
  p=output/(operation+'.json');durable(p,group)
  kind,bound,raw=derive_group(ref(p),identity)
  result['bounds'].append(dict(group=ref(p),bound=bound,raw=raw))
 result['passed']=True;durable(output/'qualification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
