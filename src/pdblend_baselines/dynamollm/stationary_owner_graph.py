"""Original CUDA allocation ownership and handoff planning, without GPU execution.

File binding, OS process liveness, reported allocation ownership and consumer
ACKs remain distinct evidence. This CPU graph cannot attest CUDA state, free an
allocation, restore KV, start a target or authorize routing. Imported IPC aliases
always resolve to the producer; only an explicit direct-receive allocation may
become another root. No dense target weight is reconstructed here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .gpu_weights import _segments
from .stationary_ipc import digest, need, process_matches, validate_descriptor
from .stationary_memory import owner_inventory_from_kv_receipt
from .stationary_tensors import tensor_plan


def read_bound(ref):
    """Hash the complete artifact before selecting a rank within its wrapper."""
    raw = Path(ref['path']).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == ref['sha256'], 'bound owner evidence bytes changed')
    value = json.loads(raw)
    for key in ref.get('json_path', []):
        need(isinstance(key, str) or type(key) is int and key >= 0, 'explicit evidence JSON path required')
        value = value[key]
    return value


def _process(value):
    need(set(value) == {'pid', 'start_ticks', 'boot_id'} and type(value['pid']) is int
         and value['pid'] > 0 and type(value['start_ticks']) is int and value['start_ticks'] > 0
         and isinstance(value['boot_id'], str) and value['boot_id'], 'exact process start/boot identity required')
    return digest(value)


def _root(value):
    value = deepcopy(value)
    value['root_id'] = digest(value)
    return value


class OriginalOwnerGraph:
    """Evidence-bound roots plus actual export/ACK lifecycle metadata.

    A successful plan reports geometry, not hardware qualification. A future
    target controller must also obtain fresh original-owner tensor attestations,
    physical memory admission, completed transport and native output evidence.
    """
    def __init__(self, plan, released_owner_refs, *, allow_cpu_oracle=False):
        need(plan['plan_sha256'] == digest({k:v for k,v in plan.items() if k != 'plan_sha256'}),
             'original owner tensor plan changed')
        need(len(released_owner_refs) == len(plan['source_gpu_uuids']), 'complete original owner graph required')
        self.plan = deepcopy(plan)
        self.base_refs = deepcopy(released_owner_refs)
        self.allow_cpu_oracle = allow_cpu_oracle
        self.roots, self.received_refs, self.exports, self.acks, self.retired = {}, [], {}, {}, {}
        self.quarantined = set()
        self.retired_receive_owners = {}
        ranks = set()
        for ref in self.base_refs:
            receipt = read_bound(ref)
            inventory = owner_inventory_from_kv_receipt(plan, receipt, evidence=ref,
                allow_cpu_oracle=allow_cpu_oracle)
            rank = inventory['source_rank']
            need(rank not in ranks, 'duplicate original owner rank')
            ranks.add(rank)
            _process(inventory['process'])
            need(type(receipt.get('generation')) is int and receipt['generation'] >= 0,
                 'original owner native generation required')
            for name, allocation in inventory['parameters'].items():
                axis, segments = _segments(name, plan['source_shapes'][name],
                    len(plan['source_gpu_uuids']), rank, plan['geometry'])
                row = _root(dict(parameter=name, axis=axis, segments=segments,
                    process=inventory['process'], gpu_uuid=inventory['gpu_uuid'],
                    tensor=receipt['original_weight_identity'][name], allocation=allocation,
                    ownership='original_dense_source', evidence=ref,
                    generation=receipt['generation'], cpu_oracle=receipt['cpu_oracle']))
                self.roots[row['root_id']] = row
        need(ranks == set(range(len(plan['source_gpu_uuids']))), 'original owner rank missing')

    def _verify_files(self):
        for ref in self.base_refs:
            read_bound(ref)
        for ref in self.received_refs:
            read_bound(read_bound(ref)['handoff_ref'])
        for row in self.exports.values():
            read_bound(row['packet_ref']);read_bound(row['plan_ref'])
        for ref in self.acks.values():
            read_bound(ref)
        for ref in self.retired_receive_owners.values():
            read_bound(read_bound(ref)['host_pid_binding_ref'])

    def verify_live_roots(self, *, process_alive=process_matches):
        self._verify_files()
        checked = {}
        for root in self.roots.values():
            key = _process(root['process'])
            if key in self.retired_receive_owners:
                need(root['ownership']=='direct_cuda_receive_allocation' and not process_alive(root['process']),
                     'retired receive owner has restarted or contains an original source root')
                continue
            need(key not in self.quarantined, 'owner quarantined after uncertain consumer release')
            if key not in checked:
                need(process_alive(root['process']), 'original allocation owner exited or restarted')
                checked[key] = deepcopy(root['process'])
        return dict(bound_artifacts_verified=True, live_processes=list(checked.values()),
            allocation_observation_scope='bound_reported_original_storage_not_live_CUDA_revalidation',
            fresh_owner_tensor_revalidation_required=True, hardware_qualified=False)

    def plan_handoff(self, target_gpu_uuids, target_shapes, *, process_alive=process_matches):
        """Cover canonical Qwen intervals using original producers, preferring local roots."""
        checks = self.verify_live_roots(process_alive=process_alive)
        tp = len(target_gpu_uuids)
        need(tp in (1,2,4) and len(set(target_gpu_uuids)) == tp
             and all(g.startswith('GPU-') for g in target_gpu_uuids), 'explicit unique target UUIDs required')
        need(set(target_shapes) == set(self.plan['source_shapes']), 'complete target parameter inventory required')
        tensor_plan(source_gpus=self.plan['source_gpu_uuids'],target_gpus=target_gpu_uuids,
            source_shapes=self.plan['source_shapes'],target_shapes=target_shapes,geometry=self.plan['geometry'])
        routes = []
        for rank,gpu in enumerate(target_gpu_uuids):
            for name,shape in sorted(target_shapes.items()):
                axis, segments = _segments(name,shape,tp,rank,self.plan['geometry'])
                candidates = [r for r in self.roots.values() if r['parameter'] == name and r['axis'] == axis
                              and _process(r['process']) not in self.retired_receive_owners]
                for local_start,global_start,length in segments:
                    cursor,end = global_start,global_start+length
                    while cursor < end:
                        available = [(r,sl,sg,size) for r in candidates for sl,sg,size in r['segments']
                            if sg <= cursor < sg+size]
                        need(available, 'original owner graph has a missing canonical fragment: '+name)
                        available.sort(key=lambda x:(x[0]['gpu_uuid'] != gpu,
                            x[0]['ownership'] != 'original_dense_source',x[0]['root_id']))
                        root,sl,sg,size = available[0]
                        # Split before any new local root begins; retaining a
                        # remote prefix must not hide the following local span.
                        stops = [s for r in candidates if r['gpu_uuid'] == gpu
                                 for _,s,_ in r['segments'] if cursor < s < end]
                        stop = min([end,sg+size,*stops])
                        count = stop-cursor
                        elements = count if axis is None else count*math.prod(n for i,n in enumerate(shape) if i != axis)
                        routes.append(dict(parameter=name,target_rank=rank,target_gpu_uuid=gpu,
                            target_shape=list(shape),axis=axis,target_offset=local_start+cursor-global_start,
                            canonical_offset=cursor,length=count,source_offset=sl+cursor-sg,
                            source_root_id=root['root_id'],source_owner=deepcopy(root['process']),
                            source_gpu_uuid=root['gpu_uuid'],source_tensor=deepcopy(root['tensor']),
                            kind='original_owner_ipc_export' if root['gpu_uuid'] == gpu else 'direct_missing_fragment_transfer',
                            bytes=elements*2))
                        cursor = stop
        value = dict(schema='dynamo-original-owner-handoff-plan/v1',
            owner_graph_sha256=digest(self.snapshot()),target_gpu_uuids=list(target_gpu_uuids),
            target_shapes=deepcopy(target_shapes),geometry=deepcopy(self.plan['geometry']),routes=routes,
            planned_retained_bytes=sum(r['bytes'] for r in routes if r['kind']=='original_owner_ipc_export'),
            planned_transfer_bytes=sum(r['bytes'] for r in routes if r['kind']=='direct_missing_fragment_transfer'),
            required_live_owner_root_ids=sorted({r['source_root_id'] for r in routes}),checks=checks,
            dense_weight_created=False,executed=False,target_served=False,formal_eligible=False)
        value['handoff_sha256'] = digest(value)
        return value

    def add_received_root(self, receipt_ref):
        """Register a NEW directly received allocation, never an imported IPC alias.

        This contract has no GPU producer yet. The future private transport must
        write the real receiver-process/allocator observation and exact handoff
        route. Reading a declaration does not hardware-qualify that transport.
        """
        value = read_bound(receipt_ref)
        need(value.get('schema') == 'dynamo-direct-receive-owner-root/v1'
             and value.get('ownership') == 'direct_cuda_receive_allocation'
             and value.get('ipc_imported') is False and value.get('completed') is True
             and value.get('cuda_synchronized') is True
             and value.get('host_weight_staging_bytes') == 0
             and value.get('retained_fragment_copy_bytes') == 0, 'new root must be an owned direct-receive allocation')
        need(value.get('cpu_oracle') is False or self.allow_cpu_oracle and value.get('cpu_oracle') is True,
             'CPU receive oracle requires explicit opt-in')
        handoff = read_bound(value['handoff_ref'])
        need(handoff['handoff_sha256'] == digest({k:v for k,v in handoff.items() if k != 'handoff_sha256'}),
             'received root handoff plan changed')
        route = value['route']
        need(type(route.get('length')) is int and route['length'] > 0
             and type(route.get('bytes')) is int and route['bytes'] > 0,
             'positive actual received fragment bounds required')
        need(route in handoff['routes'] and route['kind']=='direct_missing_fragment_transfer'
             and route['source_root_id'] in self.roots, 'received root has no original donor route')
        donor = self.roots[route['source_root_id']]
        need(route['source_owner']==donor['process'] and route['source_gpu_uuid']==donor['gpu_uuid']
             and route['source_tensor']==donor['tensor']
             and value['gpu_uuid']==route['target_gpu_uuid']
             and handoff['target_gpu_uuids'][route['target_rank']]==route['target_gpu_uuid']
             and handoff['geometry']==self.plan['geometry']
             and handoff['target_shapes'][route['parameter']]==route['target_shape'],
             'received root owner/physical placement differs')
        need(route['parameter']==donor['parameter'] and route['axis']==donor['axis']
             and any(start<=route['canonical_offset']
                 and route['canonical_offset']+route['length']<=start+length
                 and local+route['canonical_offset']-start==route['source_offset']
                 for local,start,length in donor['segments']), 'received route differs from original owned canonical shard')
        axis,segments=_segments(route['parameter'],route['target_shape'],len(handoff['target_gpu_uuids']),
                               route['target_rank'],self.plan['geometry'])
        need(axis==route['axis'] and any(start<=route['canonical_offset']
             and route['canonical_offset']+route['length']<=start+length
             and local+route['canonical_offset']-start==route['target_offset']
             for local,start,length in segments), 'received route does not cover the declared target shard')
        _process(value['process'])
        need(type(value.get('generation')) is int and value['generation']>=0,
             'received allocation owner epoch required')
        tensor,allocation = value['tensor'],value['allocation']
        need(all(type(allocation.get(k)) is int and allocation[k]>0 for k in
                 ('storage_ptr','storage_bytes','segment_address','allocation_bytes'))
             and all(type(tensor.get(k)) is int and tensor[k]>0 for k in ('storage_ptr','data_ptr'))
             and type(tensor.get('storage_offset')) is int and tensor['storage_offset']==0,
             'actual receive allocation addresses and sizes must be positive integers')
        expected = list(route['target_shape'])
        if route['axis'] is not None:expected[route['axis']] = route['length']
        stride = [math.prod(expected[i+1:]) for i in range(len(expected))]
        need(tensor['shape']==expected and tensor['stride']==stride and tensor['storage_offset']==0
             and tensor['storage_ptr']==tensor['data_ptr']==allocation['storage_ptr']
             and allocation['storage_bytes']==route['bytes']==math.prod(expected)*2
             and allocation['segment_address']<=tensor['storage_ptr']
             and tensor['storage_ptr']+route['bytes']<=allocation['segment_address']+allocation['allocation_bytes'],
             'direct receive must own exactly one fragment within its real allocator segment')
        need(value['cpu_oracle'] is True or isinstance(tensor.get('device'),str)
             and tensor['device'].startswith('cuda:') and tensor['device'][5:].isdigit(),
             'actual receive tensor must use its CUDA process address space')
        need(value['gpu_uuid']!=donor['gpu_uuid'] and value['process']!=donor['process'],
             'direct receive must have a different actual GPU producer')
        row = _root(dict(parameter=route['parameter'],axis=route['axis'],
            segments=[(0,route['canonical_offset'],route['length'])],process=value['process'],gpu_uuid=value['gpu_uuid'],
            tensor=tensor,allocation=allocation,ownership='direct_cuda_receive_allocation',evidence=receipt_ref,
            donor_root_id=donor['root_id'],generation=value['generation'],cpu_oracle=value['cpu_oracle']))
        need(row['root_id'] not in self.roots, 'duplicate receive root')
        # The same process/storage cannot acquire a second claimed owner root.
        need(not any(r['process']==row['process'] and r['tensor']['storage_ptr']==tensor['storage_ptr']
                     for r in self.roots.values()), 'allocation already has an owner root')
        for old in self.roots.values():
            if old['process']!=row['process'] or old['gpu_uuid']!=row['gpu_uuid']:continue
            before=old['allocation']
            need(allocation['storage_ptr']+allocation['storage_bytes']<=before['storage_ptr']
                 or before['storage_ptr']+before['storage_bytes']<=allocation['storage_ptr'],
                 'new received storage overlaps an existing original allocation root')
            same_segment=allocation['segment_address']==before['segment_address']
            need(same_segment and allocation['allocation_bytes']==before['allocation_bytes']
                 or not same_segment and (allocation['segment_address']+allocation['allocation_bytes']<=before['segment_address']
                     or before['segment_address']+before['allocation_bytes']<=allocation['segment_address']),
                 'received allocator segments overlap or report inconsistent full backing bytes')
        self.roots[row['root_id']] = row;self.received_refs.append(deepcopy(receipt_ref))
        return deepcopy(row)

    def register_export(self, packet_ref, plan_ref):
        """Join existing StationaryOwner packets to their original storage roots."""
        packet,plan = read_bound(packet_ref),read_bound(plan_ref)
        need(packet['packet_sha256']==digest({k:v for k,v in packet.items() if k!='packet_sha256'})
             and plan['plan_sha256']==digest({k:v for k,v in plan.items() if k!='plan_sha256'})
             and packet['plan_sha256']==plan['plan_sha256'], 'owner IPC packet/plan changed')
        need(packet['export_id'] not in self.exports, 'duplicate owner export')
        _process(packet['owner']);_process(packet['consumer'])
        need(packet['owner']!=packet['consumer'] and packet['views'], 'distinct owner and consumer identities required')
        expected=[p for p in plan['pieces'] if p['kind']=='retain_on_gpu'
                  and p['source_rank']==packet['source_rank'] and p['target_rank']==packet['target_rank']]
        need([v['piece'] for v in packet['views']]==expected, 'IPC export has missing or duplicated retained fragments')
        roots = []
        for row in packet['views']:
            p,d = row['piece'],row['descriptor']
            need(p in plan['pieces'] and p['kind']=='retain_on_gpu'
                 and p['source_rank']==packet['source_rank'] and p['target_rank']==packet['target_rank']
                 and p['source_gpu_uuid']==p['target_gpu_uuid']==packet['gpu_uuid'], 'foreign IPC retained piece')
            validate_descriptor(d,gpu_uuid=packet['gpu_uuid'],piece=p,source=packet['source_storage'][p['parameter']])
            matches=[r for r in self.roots.values() if r['process']==packet['owner']
                and r['gpu_uuid']==packet['gpu_uuid'] and r['parameter']==p['parameter']
                and r['tensor']['storage_ptr']==d['source_storage_ptr']]
            need(len(matches)==1, 'IPC consumer alias cannot act as an original allocation producer')
            root=matches[0];a=root['allocation']
            need(all(packet['source_storage'][p['parameter']].get(k)==root['tensor'][k]
                     for k in ('shape','stride','storage_offset','storage_ptr')),
                 'IPC view geometry differs from the original allocation observation')
            need(d['storage_size_bytes']==a['storage_bytes'] and d['allocation_bytes']==a['allocation_bytes']
                 and d['source_storage_ptr']-d['storage_offset_bytes']==a['segment_address'],
                 'IPC descriptor does not map the complete original root allocation')
            need(packet['generation']==root['generation'], 'original producer generation differs')
            roots.append(root['root_id'])
        self.exports[packet['export_id']]=dict(packet_ref=deepcopy(packet_ref),plan_ref=deepcopy(plan_ref),
            owner=packet['owner'],consumer=packet['consumer'],root_ids=sorted(set(roots)))

    def acknowledge_release(self, ack_ref):
        ack=read_bound(ack_ref);key=ack['export_id']
        need(key in self.exports and key not in self.acks, 'unknown or duplicate consumer ACK')
        packet=read_bound(self.exports[key]['packet_ref'])
        need(ack['packet_sha256']==packet['packet_sha256'] and ack['consumer']==packet['consumer']
             and ack['gpu_uuid']==packet['gpu_uuid'] and ack.get('cuda_synchronized') is True
             and ack.get('views_released') is True, 'consumer clean ACK binding differs')
        need(_process(ack['consumer']) not in self.retired, 'consumer already retired or crashed')
        self.acks[key]=deepcopy(ack_ref)

    def retire_consumer(self, consumer, *, process_alive=process_matches):
        self._verify_files();key=_process(consumer)
        need(not process_alive(consumer), 'consumer must actually exit before owner restore')
        edges=[(k,e) for k,e in self.exports.items() if e['consumer']==consumer]
        need(edges and key not in self.retired, 'unknown or already retired consumer')
        clean=all(k in self.acks for k,_ in edges)
        if not clean:self.quarantined.update(_process(e['owner']) for _,e in edges)
        self.retired[key]=dict(consumer=deepcopy(consumer),clean=clean,process_observed_gone=True,
            owner_process_isolation_required=not clean)
        return deepcopy(self.retired[key])

    def require_owner_restore(self, owner, *, process_alive=process_matches):
        self.verify_live_roots(process_alive=process_alive)
        need(any(r['process']==owner and r['ownership']=='original_dense_source' for r in self.roots.values())
             and process_alive(owner), 'restore requires a live original dense source owner')
        edges=[(k,e) for k,e in self.exports.items() if e['owner']==owner]
        need(all(k in self.acks and self.retired.get(_process(e['consumer']),{}).get('clean') is True
                 and not process_alive(e['consumer']) for k,e in edges),
             'all target consumers need clean ACK and actual exit before rollback')
        return dict(owner=deepcopy(owner),graph_release_preconditions_passed=True,
            physical_owner_weight_release_credit_bytes=0,source_KV_restored=False,
            fresh_owner_tensor_revalidation_required=True,target_served=False,formal_eligible=False)

    def retire_received_owner(self, cleanup_ref, *, process_alive=process_matches):
        """Keep historical roots, but stop routing to a disposed receive owner.

        Original source roots cannot be retired via this path. GPU cleanup is
        a separately bound observation; OS identity must also be gone now.
        """
        receipt=read_bound(cleanup_ref);owner=receipt['process'];key=_process(owner)
        roots=[r for r in self.roots.values() if r['process']==owner]
        need(roots and key not in self.retired_receive_owners
             and all(r['ownership']=='direct_cuda_receive_allocation' for r in roots),
             'original source allocation cannot be retired as a temporary receive owner')
        need(not process_alive(owner), 'receive allocation owner has not actually exited')
        need(receipt.get('schema')=='dynamo-owned-process-isolation/v1'
             and receipt.get('process_gone') is True and receipt.get('compute_pid_absent') is True
             and receipt.get('gpu_uuids')==sorted({r['gpu_uuid'] for r in roots})
             and receipt.get('observations'), 'bound receive-owner cleanup observation required')
        # NVML returns host PIDs even when /proc in this process uses container
        # PIDs. An independently bound host observer must join the identities.
        binding=read_bound(receipt['host_pid_binding_ref'])
        host_pid=binding.get('host_pid');chain=binding.get('namespace_pid_chain',[])
        need(binding.get('schema')=='dynamo-nvml-host-pid-binding/v1'
             and binding.get('process')==owner and binding.get('observer_pid_namespace')=='host'
             and type(host_pid) is int and host_pid>0 and chain and chain[0]==host_pid and chain[-1]==owner['pid']
             and all(type(p) is int and p>0 for p in chain)
             and binding.get('host_proc_start_ticks')==owner['start_ticks']
             and binding.get('host_boot_id')==owner['boot_id'],
             'explicit host PID/start/boot and container namespace mapping required for NVML cleanup')
        need(all(o.get('source')=='NVML_compute_processes' and o.get('gpu_uuid') in receipt['gpu_uuids']
                 and o.get('pid_namespace')=='host' and isinstance(o.get('compute_pids'),list)
                 and all(type(p) is int and p>0 for p in o['compute_pids'])
                 and host_pid not in o['compute_pids']
                 for o in receipt['observations'])
             and {o['gpu_uuid'] for o in receipt['observations']}==set(receipt['gpu_uuids']),
             'receive-owner compute PID is absent only with complete actual UUID observations')
        edges=[(k,e) for k,e in self.exports.items() if e['owner']==owner]
        need(all(k in self.acks and self.retired.get(_process(e['consumer']),{}).get('clean') is True
                 and not process_alive(e['consumer']) for k,e in edges),
             'receive owner still backs a downstream target consumer')
        self.retired_receive_owners[key]=deepcopy(cleanup_ref)

    def snapshot(self):
        return dict(schema='dynamo-original-owner-graph/v1',original_plan=deepcopy(self.plan),
            released_owner_refs=deepcopy(self.base_refs),received_root_refs=deepcopy(self.received_refs),
            roots=deepcopy(self.roots),exports=deepcopy(self.exports),acks=deepcopy(self.acks),
            retired_consumers=deepcopy(self.retired),quarantined_owners=sorted(self.quarantined),
            retired_receive_owners=deepcopy(self.retired_receive_owners),
            cpu_oracle=self.allow_cpu_oracle,hardware_qualified=False,target_served=False,formal_eligible=False)

    @classmethod
    def restore_snapshot(cls, ref, *, allow_cpu_oracle=False, process_alive=process_matches):
        value=read_bound(ref)
        need(value['schema']=='dynamo-original-owner-graph/v1'
             and value['cpu_oracle']==allow_cpu_oracle, 'owner graph snapshot scope differs')
        graph=cls(value['original_plan'],value['released_owner_refs'],allow_cpu_oracle=allow_cpu_oracle)
        for item in value['received_root_refs']:graph.add_received_root(item)
        for edge in value['exports'].values():graph.register_export(edge['packet_ref'],edge['plan_ref'])
        for ack in value['acks'].values():graph.acknowledge_release(ack)
        for row in value['retired_consumers'].values():graph.retire_consumer(row['consumer'],process_alive=process_alive)
        for cleanup in value['retired_receive_owners'].values():
            graph.retire_received_owner(cleanup,process_alive=process_alive)
        need(digest(graph.snapshot())==digest(value), 'owner graph lost roots, references or release history')
        graph.verify_live_roots(process_alive=process_alive)
        return graph
