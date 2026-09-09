import asyncio
from types import SimpleNamespace
import pytest

from ecopadg.serving.topology import InstanceSpec,TopologyManager,validate_layout


def test_topology_rejects_gpu_overlap_extra_gpus_and_port_collisions():
    a=InstanceSpec('a',2,(0,1),18000,19000)
    validate_layout([a],tuple(range(8)))
    with pytest.raises(ValueError): validate_layout([a,InstanceSpec('b',1,(1,),18001,19100)],range(8))
    with pytest.raises(ValueError): validate_layout([InstanceSpec('b',1,(8,),18001,19100)],range(8))
    with pytest.raises(ValueError): validate_layout([a,InstanceSpec('b',1,(2,),19000,19100)],range(8))


def test_failed_replacement_restores_fresh_peer_identity_and_keeps_other_instance():
    async def run():
        old=InstanceSpec('a',2,(0,1),18000,19000)
        other=InstanceSpec('b',1,(2,),18001,19100)
        new=InstanceSpec('new',1,(0,),18002,19200)
        calls=[];commits=[];events=[]
        class Lifecycle:
            async def start(self,spec,peers,retained_weights):
                calls.append(('start',spec.instance_id))
                assert retained_weights=='cache'
                if spec.instance_id=='new': raise RuntimeError('injected preparation failure')
            async def stop(self,spec): calls.append(('stop',spec.instance_id))
        class Journal:
            async def emit(self,event): events.append(event)
        async def freeze(ids,value): calls.append(('freeze',tuple(ids),value))
        async def commit(ids,added): commits.append((ids,added))
        class Manager(TopologyManager):
            async def request(self,spec,path,payload=None,timeout=60):
                calls.append((path,spec.instance_id))
                return dict(generation=0,retained_weights='cache')
            async def verify(self,spec): return dict(instance_id=spec.instance_id)
        manager=Manager(SimpleNamespace(clocks=None),Lifecycle(),[old,other],tuple(range(8)),
                        Journal(),freeze=freeze,commit=commit)
        with pytest.raises(RuntimeError,match='injected'):
            await manager.reconfigure(('a',),(new,),savings_lower_j=200,cost_upper_j=100)
        assert ('stop','b') not in calls
        assert len(commits)==1
        restored=commits[0][1][0]
        assert restored.instance_id.startswith('ar') and restored.instance_id!='a'
        assert restored.tp==2 and restored.gpus==(0,1)
        assert events[-1]['kind']=='topology_rollback' and events[-1]['recovered']
        assert calls[-1]==('freeze',('a',),False)
    asyncio.run(run())


def test_failed_cleanup_never_reuses_unknown_gpu_or_unfreezes_routes():
    async def run():
        old=InstanceSpec('a',1,(0,),18000,19000)
        new=InstanceSpec('new',1,(0,),18001,19100)
        calls=[];events=[]
        class Lifecycle:
            async def start(self,spec,*args):
                calls.append(('start',spec.instance_id))
                raise RuntimeError('new worker unhealthy')
            async def stop(self,spec):
                calls.append(('stop',spec.instance_id))
                if spec.instance_id=='new': raise RuntimeError('cannot confirm replacement stopped')
        class Journal:
            async def emit(self,event): events.append(event)
        async def freeze(ids,value): calls.append(('freeze',value))
        async def commit(*args): raise AssertionError('must not commit uncertain topology')
        class Manager(TopologyManager):
            async def request(self,*args,**kwargs): return dict(generation=0)
        manager=Manager(SimpleNamespace(clocks=None),Lifecycle(),[old],range(8),Journal(),
                        freeze=freeze,commit=commit)
        with pytest.raises(RuntimeError,match='cannot confirm'):
            await manager.reconfigure(('a',),(new,),savings_lower_j=200,cost_upper_j=100,
                                      retained_weights='cache')
        assert ('freeze',False) not in calls
        assert [c for c in calls if c[0]=='start']==[('start','new')]
        assert events[-1]['kind']=='topology_recovery_failed'
    asyncio.run(run())


def cache(tmp_path):
    import json
    root=tmp_path/'weights';root.mkdir()
    (root/'rank-0.safetensors').write_bytes(b'test-only-cache')
    (root/'manifest.json').write_text(json.dumps(dict(schema=1,complete=True,tp=1,
        model_config_sha256='model',ranks=[dict(rank=0,file='rank-0.safetensors',sha256='a'*64)])))
    return str(root)


def manager_fixture(specs,*,fail_commit_journal=False):
    calls=[];events=[];commits=[]
    backend=SimpleNamespace(clocks=None,instances={s.instance_id:s.endpoint() for s in specs})
    class Lifecycle:
        async def start(self,spec,*args):calls.append(('start',spec.instance_id))
        async def stop(self,spec):calls.append(('stop',spec.instance_id))
    class Journal:
        async def emit(self,event):
            events.append(event)
            if event['kind']=='topology_commit' and fail_commit_journal:
                raise RuntimeError('injected commit journal failure')
    async def freeze(ids,value):calls.append(('freeze',tuple(ids),value))
    async def commit(ids,added):
        commits.append((ids,added))
        assert set(ids)<=set(backend.instances)
        backend.instances={k:v for k,v in backend.instances.items() if k not in ids}
        backend.instances.update({s.instance_id:s.endpoint() for s in added})
    class Manager(TopologyManager):
        async def request(self,spec,path,payload=None,timeout=60):
            calls.append((path,spec.instance_id))
            return dict(generation=0,retained_weights='cache')
        async def verify(self,spec):return dict(instance_id=spec.instance_id)
    manager=Manager(backend,Lifecycle(),specs,range(8),Journal(),freeze=freeze,commit=commit)
    return manager,backend,calls,commits,events


def test_pure_addition_keeps_live_instances_and_requires_complete_cache(tmp_path):
    async def run():
        old=InstanceSpec('old',1,(0,),18000,19000)
        new=InstanceSpec('new',1,(1,),18001,19100)
        manager,backend,calls,commits,events=manager_fixture([old])
        with pytest.raises(ValueError,match='available complete retained-weight'):
            await manager.reconfigure((),(new,),savings_lower_j=200,cost_upper_j=100,retained_weights='missing')
        assert not calls
        answer=await manager.reconfigure((),(new,),savings_lower_j=200,cost_upper_j=100,retained_weights=cache(tmp_path))
        assert answer['committed'] and set(manager.specs)=={'old','new'}
        assert set(backend.instances)=={'old','new'}
        assert ('stop','old') not in calls and ('/drain','old') not in calls
        assert ('start','new') in calls and commits[0][0]==()
    asyncio.run(run())


def test_pure_removal_drains_only_removed_replica_and_keeps_other_serving(tmp_path):
    async def run():
        old=InstanceSpec('old',1,(0,),18000,19000)
        survivor=InstanceSpec('live',1,(1,),18001,19100)
        manager,backend,calls,commits,events=manager_fixture([old,survivor])
        await manager.reconfigure(('old',),(),savings_lower_j=200,cost_upper_j=100,retained_weights=cache(tmp_path))
        assert set(manager.specs)==set(backend.instances)=={'live'}
        assert ('/drain','old') in calls and ('stop','old') in calls
        assert ('/drain','live') not in calls and ('stop','live') not in calls
        assert not any(c[0]=='start' for c in calls)
    asyncio.run(run())


def test_empty_transaction_or_loss_of_all_capacity_or_extra_gpu_cannot_mutate(tmp_path):
    async def run():
        old=InstanceSpec('old',1,(0,),18000,19000)
        manager,backend,calls,commits,events=manager_fixture([old])
        retained=cache(tmp_path)
        for removed,added in [((),()),(('old',),()),((),(InstanceSpec('extra',1,(8,),18001,19100),))]:
            with pytest.raises(ValueError):
                await manager.reconfigure(removed,added,savings_lower_j=200,cost_upper_j=100,retained_weights=retained)
        assert not calls and set(manager.specs)=={'old'}
    asyncio.run(run())


@pytest.mark.parametrize('direction',['add','remove'])
def test_add_remove_rollback_after_route_commit_restores_only_changed_replica(tmp_path,direction):
    async def run():
        old=InstanceSpec('old',1,(0,),18000,19000)
        other=InstanceSpec('live',1,(1,),18001,19100)
        new=InstanceSpec('new',1,(2,),18002,19200)
        manager,backend,calls,commits,events=manager_fixture([old,other],fail_commit_journal=True)
        with pytest.raises(RuntimeError,match='commit journal failure'):
            await manager.reconfigure(() if direction=='add' else ('old',),
                (new,) if direction=='add' else (),savings_lower_j=200,cost_upper_j=100,retained_weights=cache(tmp_path))
        assert 'live' in backend.instances and ('stop','live') not in calls
        assert len(backend.instances)==len(manager.specs)==2
        if direction=='add':
            assert set(backend.instances)=={'old','live'}
            assert ('stop','new') in calls and ('stop','old') not in calls
        else:
            restored=next(i for i in backend.instances if i!='live')
            assert restored.startswith('oldr') and restored!='old'
        assert events[-1]['kind']=='topology_rollback'
    asyncio.run(run())
