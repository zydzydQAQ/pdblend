"""Real C count structures + actual planner; synthetic demand/certificates are CPU-only."""
import asyncio, copy, dataclasses, json, unittest, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parent))
import capacity_planner as P
import capacity_runtime as R
from capacity_executor import sha

async def replay_service(instances, raw, tp):
 s=R.CapacityService.__new__(R.CapacityService)
 observed=[]
 for i in instances:
  v=raw[i['id']]
  observed.append(SimpleNamespace(instance_id=i['id'],gpus=i['gpus'],timestamp_s=v['timestamp'],
   generation=v['generation'],role=v['role'],accepting=v['accepting'],requests=[],waiting=v['waiting'],
   running=v['running'],kv_allocations=v['kv_allocations'],reserved_kv_tokens=0,
   transfer_allocations=v['transfer_allocations'],reserved_transfer_bytes=0))
 s.controller=SimpleNamespace(state=SimpleNamespace(snapshot=SimpleNamespace(instances=observed)),
  backend=SimpleNamespace(instances={i['id']:i for i in instances},last=raw,topology_version=1))
 s.module=P;s.identity=P.Identity('1'*64,'2'*64,'sha256:'+'3'*64,'4'*64,tp)
 s.inventory=SimpleNamespace(value=dict(known_instances={i['id']:{'changed_s':raw[i['id']]['timestamp']} for i in instances},initial_ids=[i['id'] for i in instances],transition_inflight=False))
 s.last_spares={};s.planner=SimpleNamespace(policy=P.Policy(min_residents=2))
 async def gpu_state(gpus):
  return [dict(gpu=g,at_s=R.time.time()-.001,free_bytes=100_000_000_000,process_pids=[]) for g in gpus]
 s.backend=SimpleNamespace(gpu_state=gpu_state)
 return s

class UnknownResident(unittest.IsolatedAsyncioTestCase):
 async def test_actual_C_snapshot_preserves_unknown_and_original_protection(self):
  path=Path('/root/workspace/pdblend-next-v1/campaign/C7B-batch16-context-current100-v4/identity.after.json')
  rows=json.loads(path.read_text())['instances'];raw={v['provenance']['instance_id']:v['runtime'] for v in rows};before=copy.deepcopy(raw)
  ins=[dict(id=x['provenance']['instance_id'],role='mixed',tp=1,gpus=[int(x['provenance']['cuda_visible_devices'])],url='unused') for x in rows]
  service=await replay_service(ins,raw,1);snap=await service.snapshot()
  self.assertEqual(raw,before);self.assertEqual(len(snap.residents),2)
  self.assertTrue(all(x.inflight_transfers is None and not x.transfer_counts_known and not x.removable for x in snap.residents))
  self.assertTrue(all(x.transport_healthy for x in snap.residents))

 def setup_domain(self, *, third=False):
  now=1300.;identity=P.Identity('1'*64,'2'*64,'sha256:'+'3'*64,'4'*64,1)
  ev=P.Evidence(identity,'5'*64);domain='6'*64
  two=((6,),(7,));three=((5,),(6,),(7,))
  layouts=[P.LayoutBound(two,domain,2.,ev),P.LayoutBound(three,domain,4.,ev)]
  transitions=[P.TransitionBound('restore_cold',(5,),5.,100.,ev,1000),P.TransitionBound('remove',(5,),2.,20.,ev)]
  savings=[P.SavingsBound(three,two,domain,0.,1.,100.,ev)]
  planner=P.CapacityPlanner(identity,layouts,transitions,savings,P.Policy(min_residents=2))
  originals=tuple(P.Resident('c'+str(g),(g,),now,5,1000.,inflight_transfers=None,transfer_counts_known=False,removable=False) for g in (6,7))
  residents=originals+((P.Resident('new',(5,),now,6,1100.),) if third else ())
  spares=() if third else (P.Spare((5,),now,1260.,100000),)
  snapshot=P.Snapshot(identity,2,residents,spares)
  return planner,snapshot,domain,now

 def test_grow_with_two_unknown_retained(self):
  p,s,d,n=self.setup_domain();v=p.choose(s,P.Demand(n,100.,3.,3.1,d),P.State(),n)
  self.assertEqual(v.reason,'restore_proposed');self.assertEqual(v.proposal.gpus,(5,))
  self.assertEqual(v.proposal.energy_claim,'capacity_recovery_no_energy_gain_claim')
  self.assertTrue(P.revalidate(v.proposal,s,n))

 def shrink(self, mutate=None):
  p,s,d,n=self.setup_domain(third=True)
  if mutate:s=mutate(s)
  state=P.State(low_since_s=n-61,layout_groups=P.groups(i.gpus for i in s.residents))
  result=p.choose(s,P.Demand(n,100.,.1,.2,d),state,n)
  return result,s,n

 def test_shrink_only_known_new_owner(self):
  result,s,n=self.shrink();self.assertEqual(result.reason,'remove_proposed')
  self.assertEqual(result.proposal.remove_id,'new');self.assertTrue(P.revalidate(result.proposal,s,n))
  forged=dataclasses.replace(result.proposal,remove_id='c6',gpus=(6,),source_generation=5)
  self.assertFalse(P.revalidate(forged,s,n))

 def test_unknown_new_owner_cannot_be_removed(self):
  def mutate(s):return dataclasses.replace(s,residents=s.residents[:2]+(dataclasses.replace(s.residents[2],inflight_transfers=None,transfer_counts_known=False),))
  result,_,_=self.shrink(mutate);self.assertIsNone(result.proposal)

 def test_stale_original_blocks_removal(self):
  def mutate(s):return dataclasses.replace(s,residents=(dataclasses.replace(s.residents[0],observed_at_s=1200.),)+s.residents[1:])
  result,_,_=self.shrink(mutate);self.assertEqual(result.reason,'remaining_capacity_state_unverified')

 def test_unknown_never_encoded_as_zero_and_malformed_rejected(self):
  self.assertEqual(R.transfer_observation(dict(transfer_inflight_sends=None,transfer_inflight_receives=0,transfer_inflight_sends_observed=False)),(None,False))
  self.assertEqual(R.transfer_observation(dict(transfer_inflight_sends=0,transfer_inflight_receives=0,transfer_inflight_sends_observed=False)),(None,False))
  with self.assertRaises(ValueError):R.transfer_observation(dict(transfer_inflight_sends=True,transfer_inflight_receives=0))
  with self.assertRaises(ValueError):P.Resident('bad',(5,),1.,1,1.,inflight_transfers=0,transfer_counts_known=False)

if __name__=='__main__':unittest.main()


def test_planner_loader_keeps_versions_separate_and_rejects_legacy_unknown_contract(tmp_path):
 original=Path('/root/workspace/pdblend-next-v1/campaign/capacity-controller-v2/planner.py')
 with __import__('pytest').raises(ValueError,match='unknown transfer'):
  R.load_planner(dict(path=str(original),sha256=sha(original)))
 first=Path(P.__file__)
 copied=tmp_path/'planner.py';copied.write_text(first.read_text()+'\n# independent source version\n')
 a=R.load_planner(dict(path=str(first),sha256=sha(first)))
 b=R.load_planner(dict(path=str(copied),sha256=sha(copied)))
 assert a is not b and a.__name__ != b.__name__


def test_isolation_candidate_can_only_propose_unknown_extra_pending_actual_stop_barrier():
 fixture=UnknownResident()
 p,s,domain,now=fixture.setup_domain(third=True)
 extra=dataclasses.replace(s.residents[2],inflight_transfers=None,transfer_counts_known=False,
                           isolated_transport_candidate=True)
 snapshot=dataclasses.replace(s,residents=s.residents[:2]+(extra,))
 state=P.State(low_since_s=now-61,layout_groups=P.groups(i.gpus for i in snapshot.residents))
 result=p.choose(snapshot,P.Demand(now,100.,.1,.2,domain),state,now)
 assert result.proposal is not None and result.proposal.remove_id=='new'
 assert extra.inflight_transfers is None and extra.transfer_counts_known is False
 assert all(not v.removable for v in snapshot.residents[:2])
 no_proof=dataclasses.replace(extra,isolated_transport_candidate=False)
 snapshot=dataclasses.replace(snapshot,residents=s.residents[:2]+(no_proof,))
 assert p.choose(snapshot,P.Demand(now,100.,.1,.2,domain),state,now).proposal is None


def test_backend_factory_rejects_unqualified_legacy_or_forged_source_before_constructing():
 import pytest
 with pytest.raises(ValueError,match='unguarded legacy'):
  R.physical_backend(None,{'native_kind':'legacy_sync_put'},None)
 with pytest.raises(ValueError,match='unknown physical'):
  R.physical_backend(None,{'backend_kind':'anything'},None)
 with pytest.raises(ValueError,match='source must match'):
  R.physical_backend(None,dict(backend_kind='guarded_legacy_self_only_v1',
   source_composition={'sources':{'backend':{'path':'one','sha256':'1'*64}}},
   backend_source={'path':'two','sha256':'2'*64}),None)
