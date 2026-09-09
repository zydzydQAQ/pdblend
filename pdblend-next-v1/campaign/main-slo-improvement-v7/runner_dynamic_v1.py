"""Dynamic stages using the same frozen screening loop and workload declaration."""
import copy
from pathlib import Path
import runner as base
import protocol as p
from dynamic_ownership import inherited_lease, LOCK

def point_binding(original, release, cell, output, out):
    p.need(cell['arm'] == 'dynamic' and release.get('dynamic_qualified') is True,
           'this executor requires a physically qualified dynamic arm')
    config = p.checked(release['configs']['dynamic'][cell['dataset']])
    p.need(config.get('capacity_integration_v1') is True, 'physical capacity lifecycle is disabled')
    inventory = out/'inventories'/(cell['cell_id']+'.json')
    job = output/'operations'/cell['cell_id']/'job.json'
    lease = inherited_lease()
    authority = dict(schema='capacity-parent-lease-authority-v1',
        fd=lease['fd'], holder_pid=lease['holder_pid'], holder_start_ticks=lease['holder_start_ticks'],
        lock_path=str(LOCK), lock_device=lease['device'], lock_inode=lease['inode'],
        capacity_inventory_path=str(inventory), expected_job_path=str(job),
        invocation=p.ref(release['release_path']))
    authority_path = out/'authorities'/(cell['cell_id']+'.json')
    p.write(authority_path, authority, exclusive=True)
    config.update(capacity_inventory_path=str(inventory), capacity_lease_authority=p.ref(authority_path))
    config_path = out/'configs'/(cell['cell_id']+'.json')
    p.write(config_path, config, exclusive=True)
    binding = copy.deepcopy(original)
    binding.update(host_release=release['host_release'], deadline_s=p.DEADLINE, output=str(output),
        unchanged_pdb_policy=False, formal_eligible=False,
        improvement=dict(arm='dynamic', repeat=cell['repeat'], original_cell_id=cell['original_cell_id'],
                         implementation=release['implementation_id']))
    binding['configs'] = {cell['dataset']:str(config_path)}
    binding['files'].update(release['files'])
    for reference in (p.ref(config_path), p.ref(authority_path), cell['trace'], release['declaration']):
        binding['files'][reference['path']] = reference['sha256']
    return binding

base.point_binding = point_binding

if __name__ == '__main__':
    base.main()
