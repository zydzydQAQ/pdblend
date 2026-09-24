"""Opt-in Dynamo worker extension for source KV teardown/rebuild only."""
from .stationary_ipc import need, cuda_uuid_observation
from .stationary_kv import SourceKvWorkspace
from .stationary_worker import DynamoStationaryWorkerExtension


class DynamoStationaryKvWorkerExtension(DynamoStationaryWorkerExtension):
    def dynamo_stationary_operation(self, operation, payload=None):
        payload = dict(payload or {})
        key = payload.get('transaction_id')
        workspaces = getattr(self, '_dynamo_stationary_kv', None)
        if workspaces is None:
            workspaces = self._dynamo_stationary_kv = {}
        if operation == 'release' and key in workspaces:
            need(workspaces[key].status == 'closed', 'restore KV and retire execution guards before releasing owner lease')
        if operation not in ('release_kv', 'restore_kv', 'kv_workspace_status', 'close_kv_workspace'):
            result = super().dynamo_stationary_operation(operation, payload)
            if key is not None:
                result.setdefault('transaction_id', key)
            if operation == 'describe':
                import torch
                uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
                observed = cuda_uuid_observation(uuid, torch_uuid_type=getattr(torch._C, '_CUuuid', None))
                result['gpu_uuid'] = observed['canonical']
                result['cuda_uuid_observation'] = observed
            elif operation == 'pin':
                result['cuda_uuid_identity'] = self._dynamo_stationary_owners[key].codec.uuid_identity
            return result
        owners = getattr(self, '_dynamo_stationary_owners', {})
        need(key in owners, 'source KV requires the existing pinned original storage owner')
        owner = owners[key]
        need(type(payload.get('expected_generation')) is int
             and payload['expected_generation'] == owner.lease.generation
             == getattr(self, '_native_generation', None), 'KV operation native generation differs')
        if operation == 'release_kv':
            need(key not in workspaces, 'source KV transaction already exists')
            workspaces[key] = SourceKvWorkspace.for_native_worker(self, owner)
            result = workspaces[key].release(payload['native_scheduler_drain'])
        else:
            need(key in workspaces, 'source KV workspace has not been detached')
            workspace = workspaces[key]
            if operation == 'restore_kv':
                result = workspace.restore(payload['native_scheduler_drain'])
            elif operation == 'close_kv_workspace':
                workspace.close()
                result = workspace.receipt()
            else:
                result = workspace.receipt()
        return dict(result, rank=self.rank, transaction_id=key, serving_qualified=False)
