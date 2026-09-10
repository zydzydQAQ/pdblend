"""Independent restoration of retained engines after a failed controller task."""
import asyncio
from pathlib import Path
import aiohttp
from capacity_executor import durable, require


def resource_barrier(controller, service, initial, capacity_cleanup_complete):
    require(capacity_cleanup_complete, 'unknown capacity cleanup prevents retained restoration')
    require(not controller.active and not controller.request_tasks, 'controller work still owns retained resources')
    require(not controller.backend.inflight_actions, 'physical engine operation remains active')
    tasks = [getattr(controller, name, None) for name in
             ('dispatch_task', 'telemetry_task', 'role_task', 'slow_task', 'pd_slow_control_task', 'capacity_task')]
    require(all(task is None or task.done() for task in tasks), 'control task still runs')
    clocks = getattr(controller, 'clock_owner', None) or getattr(controller.backend, 'clocks', None)
    require(clocks is not None and all(handle.closed for handle in clocks.files)
            and not clocks.pending_physical_commands and not clocks.physical_command_uncertainty
            and not clocks.applied, 'clock owner is not closed or physical state remains unknown')
    if service is not None:
        inventory = service.inventory.value
        require(inventory['complete'] and inventory['transition_inflight'] is False
                and inventory['active_instances'] == initial, 'capacity inventory has not returned to exact initial2')
        require(all(row.get('state') in ('stopped', 'stopped_after_failure')
                    for iid, row in inventory['known_instances'].items() if iid not in inventory['initial_ids']),
                'an owned extra process has not been stopped')
    return True


async def restore_retained(controller, service, common, original, out, *, capacity_cleanup_complete):
    resource_barrier(controller, service, original['instances'], capacity_cleanup_complete)
    async with aiohttp.ClientSession(trust_env=False) as session:
        # This checks exact Docker identities/provenance and actual native
        # requests, KV, transfers and acknowledgements before any new command.
        before = await common.identity(session, original)
        replies = await asyncio.wait_for(asyncio.gather(*(
            common.resume(session, instance, instance['restore_budget_tokens'])
            for instance in original['instances'])), 120)
        after = await common.identity(session, original)
    result = dict(schema='independent-retained-original-restoration-v1', passed=True,
        capacity_cleanup_complete=True, original_controller_failure_not_waived=True,
        before=before, replies=replies, after=after, experiment_requests_sent=0)
    durable(Path(out) / 'retained-restoration.json', result)
    return result
