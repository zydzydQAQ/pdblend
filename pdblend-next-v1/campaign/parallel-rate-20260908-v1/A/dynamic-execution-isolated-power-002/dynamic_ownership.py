"""Durable, cell-specific identities for dynamic measurement and failed cleanup."""
import asyncio
import json
import os
from pathlib import Path
import time

LOCK = Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock')

def need(ok, message):
    if not ok:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text())

def start_ticks(pid):
    return int(Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])

def inherited_lease():
    wanted = LOCK.stat()
    found = []
    for path in Path('/proc/self/fd').iterdir():
        try:
            s = path.stat()
            info = Path('/proc/self/fdinfo', path.name).read_text()
            if (s.st_dev, s.st_ino) == (wanted.st_dev, wanted.st_ino) and 'FLOCK' in info and 'WRITE' in info:
                found.append(int(path.name))
        except OSError:
            pass
    need(len(found) == 1, 'one actual exclusive parent lease descriptor required')
    return dict(fd=found[0], holder_pid=os.getpid(), holder_start_ticks=start_ticks(os.getpid()),
                device=wanted.st_dev, inode=wanted.st_ino)

def verify_inherited(lease):
    need(os.getppid() == lease['holder_pid'] and start_ticks(os.getppid()) == lease['holder_start_ticks'],
         'measurement parent identity changed')
    s = os.fstat(lease['fd']); wanted = LOCK.stat()
    need((s.st_dev, s.st_ino) == (wanted.st_dev, wanted.st_ino) == (lease['device'], lease['inode']),
         'inherited descriptor is not the exact node lease')
    info = Path('/proc', str(os.getppid()), 'fdinfo', str(lease['fd'])).read_text()
    need('FLOCK' in info and 'WRITE' in info, 'parent no longer holds its actual exclusive lease')

def inventory(path, initial, *, child_pid=None, identity=None):
    value = read(path)
    need(value['schema'] == 'capacity-live-inventory-v1', 'unknown dynamic inventory')
    if child_pid is not None:
        need(value['pid'] == child_pid, 'inventory belongs to another child')
    if identity is not None:
        need(value['identity'] == identity, 'inventory model/source identity changed')
    original = {i['id']: i for i in initial}
    need(set(value['initial_ids']) == set(original), 'initial instance ownership changed')
    known = value['known_instances']
    need(set(original) <= set(known), 'retained original instance missing')
    for iid, i in known.items():
        need(iid == i['id'] and type(i['port']) is int, 'invalid dynamic endpoint identity')
        if iid in original:
            need(i.get('owner_kind') == 'retained_original'
                 and all(i[k] == original[iid][k] for k in ('gpus', 'tp', 'url', 'port', 'container')),
                 'retained original process rewritten')
        else:
            need(i.get('owner_kind') == 'created_for_cell' and i.get('owner_id')
                 and i['container_name'] == 'pdb-v2-'+iid and iid.startswith('cap-'+i['owner_id']+'-'),
                 'dynamic extra lacks exact durable ownership')
            need(i['tp'] == initial[0]['tp'] and len(i['gpus']) == i['tp']
                 and set(i['gpus']) <= set(range(8))
                 and not set(i['gpus']) & {g for old in initial for g in old['gpus']},
                 'dynamic extra overlaps retained instances or changes TP')
    need(len({i['port'] for i in known.values()}) == len(known), 'ambiguous engine port ownership')
    need(all(i['id'] in known for i in value['active_instances']), 'published unowned identity')
    return value

def validate_terminal(value, initial):
    need(value.get('complete') is True and value.get('transition_inflight') is False,
         'dynamic controller did not finish measured cleanup')
    need({i['id'] for i in value['active_instances']} == {i['id'] for i in initial},
         'dynamic cell did not actually return to initial two')
    need(not any(e['kind'] in ('transition_failed', 'rollback_failed') for e in value['events']),
         'failed physical transaction retained; measurement cannot certify dynamic implementation')

def transition_artifacts(value):
    """Freeze every referenced physical power/memory record into the cell receipt."""
    import hashlib
    artifacts = {}
    commits = {e['transaction'] for e in value['events'] if e['kind'] == 'physical_commit'}
    measured = set()
    for event in value['events']:
        if event['kind'] != 'transition_measurement':
            continue
        need(event.get('measurement_valid') is True and event.get('gpu_indices') == list(range(8)),
             'invalid dynamic transition power evidence')
        measured.add(event['transaction'])
        reference = event['receipt']
        raw = read(reference['path'])
        need(hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() == reference['sha256'],
             'transition receipt changed')
        artifacts[reference['path']] = reference['sha256']
        for path, digest in raw['artifacts'].items():
            need(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, 'raw transition artifact changed')
            artifacts[path] = digest
    need(commits <= measured, 'physical commit lacks complete eight-GPU energy evidence')
    return artifacts

async def command(args, deadline):
    remaining = min(25., deadline-time.time())
    need(remaining > 0, 'outer dynamic cleanup budget exhausted')
    child = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await asyncio.wait_for(child.communicate(), remaining)
    except BaseException:
        if child.returncode is None:
            child.kill()
            await child.wait()
        raise
    return child.returncode, out.decode(), err.decode()

async def remove_owned_extras(common, session, value, binding, deadline, hardware):
    """After child exit, cancel its logged work and stop only exact owned extras."""
    results = []
    involved_gpus = set()
    for iid, instance in value['known_instances'].items():
        if instance['owner_kind'] == 'retained_original':
            continue
        record = dict(instance_id=iid, complete=False, policy_benefit_claim=False)
        results.append(record)
        involved_gpus.update(instance['gpus'])
        code, out, err = await command(['docker', 'inspect', instance['container_name']], deadline)
        if code:
            need('No such object' in err or 'No such container' in err, 'cannot inspect owned dynamic extra: '+err)
            record['container_absent'] = True
        else:
            rows = json.loads(out); need(len(rows) == 1, 'ambiguous physical identity')
            actual = rows[0]
            need(actual['Name'].lstrip('/') == instance['container_name']
                 and actual['Config'].get('Labels', {}).get('pdblend.capacity.owner') == instance['owner_id']
                 and actual['Image'] == binding['identity']['engine_image'], 'physical container ownership changed')
            old = instance.get('container')
            if old:
                need(actual['Id'] == old['id'], 'owned container ID changed')
            if actual['State']['Running']:
                if old:
                    need(actual['State']['Pid'] == old['host_pid']
                         and actual['State']['StartedAt'] == old['StartedAt'], 'owned process was replaced')
                # If startup never produced a runtime, no route was published.
                published = any(e.get('kind') == 'routing_commit' and iid in e.get('added', []) for e in value['events'])
                if published:
                    record['native_cleanup'] = await common.restore(session, instance)
                    need(record['native_cleanup']['complete'], 'owned extra native cleanup incomplete')
                code, _, err = await command(['docker', 'stop', '--time', '10', actual['Id']], deadline)
                need(code == 0, 'owned dynamic extra stop failed: '+err)
            code, out, err = await command(['docker', 'inspect', actual['Id']], deadline)
            need(code == 0, 'post-stop identity unavailable: '+err)
            after = json.loads(out)[0]
            need(not after['State']['Running'] and after['State']['Pid'] == 0, 'owned dynamic extra still runs')
            record.update(container_id=actual['Id'], physically_stopped=True)
    # A later owned replica can legally reuse a stopped historical replica's
    # GPUs. Stop all exact owned identities before checking shared GPU release.
    for gpu in sorted(involved_gpus):
        procs = hardware._nvml.nvmlDeviceGetComputeRunningProcesses(hardware._handle(gpu))
        need(not procs, 'dynamic GPU still has a physical process')
    for record in results:
        record['complete'] = True
    return results
