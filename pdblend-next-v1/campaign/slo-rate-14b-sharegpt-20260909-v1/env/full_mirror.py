"""Plan now, mirror only a terminal idle node. No GPU mutation or remote deletion."""
from pathlib import Path
import argparse, collections, datetime, fcntl, hashlib, json, os, shlex, subprocess, sys, time

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import monitor
HOSTS=monitor.HOSTS
WORKSPACE=Path('/root/workspace')

DISCOVER=r'''
import json,hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]);node=sys.argv[2];home=root/node
refs={};origins={};seeds=[];errors=[]
def add(path,digest,origin):
    if isinstance(path,str) and path.startswith('/root/workspace/') and isinstance(digest,str) and len(digest)==64:
        refs.setdefault(path,set()).add(digest);origins.setdefault(path,set()).add(origin)
def walk(value,origin):
    if isinstance(value,list):
        for item in value:walk(item,origin)
    elif isinstance(value,dict):
        if isinstance(value.get('path'),str):add(value['path'],value.get('sha256'),origin)
        for k,v in value.items():
            if k.startswith('/root/workspace/'):
                add(k,v if isinstance(v,str) else v.get('sha256') if isinstance(v,dict) else None,origin)
            if k in ('receipt','trace') and isinstance(v,str):add(v,value.get(k+'_sha256'),origin)
            walk(v,origin)
patterns=['run-*/cell-*/measurement/results/checkpoints/*.json',
 'fixed-qualification/qualified.json','pdb-qualification/qualified.json',
 'baseline*/qualification/*/binding.json','baseline*/ready.json',
 'baseline*/registered-bootstrap.json','baseline*/bootstrap/binding.json',
 'baseline*/deployment/deployment-receipt.json','pdb-ready.json',
 'retirement/status.json','cold-bootstrap/status.json','environment-failed-*.json']
for pattern in patterns:
    for path in sorted(home.glob(pattern)):
        try:
            raw=path.read_bytes()
            if len(raw)>16*1024*1024:raise ValueError('metadata seed unexpectedly large')
            value=json.loads(raw)
            if '/checkpoints/' in str(path) and not value.get('finished_s'):continue
            digest=hashlib.sha256(raw).hexdigest();add(str(path),digest,str(path));walk(value,str(path));seeds.append(dict(path=str(path),sha256=digest))
        except Exception as exc:errors.append(dict(path=str(path),error=repr(exc)))
print(json.dumps(dict(refs={p:sorted(v) for p,v in refs.items()},origins={p:sorted(v) for p,v in origins.items()},seeds=seeds,errors=errors)))
'''

STAT=r'''
import json,stat,sys
from pathlib import Path
out=[]
for item in json.load(sys.stdin):
    path=Path(item['path']);row=dict(item)
    try:
        assert path.is_absolute() and path.is_relative_to('/root/workspace')
        assert not path.is_symlink() and path.resolve().is_relative_to('/root/workspace')
        value=path.stat();assert stat.S_ISREG(value.st_mode)
        row.update(remote_size=value.st_size,remote_mtime_ns=value.st_mtime_ns,remote_status='available')
    except Exception as exc:row.update(remote_status='unavailable',remote_error=repr(exc),remote_size=None)
    out.append(row)
print(json.dumps(out))
'''

GUARD=r'''
import json,sys,subprocess,hashlib,time
from pathlib import Path
root=Path(sys.argv[1]);node=sys.argv[2];path=Path(sys.argv[3]);assert path.is_relative_to(root/node)
raw=path.read_bytes();state=json.loads(raw)
def active(value):
    p=Path('/proc')/str(value.get('pid',0))/'stat'
    if not p.exists():return False
    bits=p.read_text().rsplit(') ',1)[1].split()
    return bits[0] not in ('Z','X') and (not value.get('startticks') or str(value['startticks'])==bits[19])
assert state['node']==node and state.get('complete') and state.get('five_system_complete') and state.get('phase')=='complete','node formal grid incomplete'
assert state.get('finished_s') and not state.get('node_lease_held') and not active(state),'node supervisor not terminal'
for name in ('child','unsettled_child'):
    child=state.get(name)
    if child:assert child.get('finished_s') and not active(child) and not child.get('physical_lease_release_not_certified'),'measurement descendant not terminal'
last=state.get('last_cell_status')
if last:
    content=Path(last['path']).read_bytes();assert hashlib.sha256(content).hexdigest()==last['sha256'];d=json.loads(content)
    assert d.get('complete') and d.get('cleanup_complete') and d.get('finished_s') and not d.get('node_lease_held'),'last cell cleanup incomplete'
result=subprocess.run(['fuser',str(Path('/root/workspace/pdblend/new-results/campaigns/node-experiment.lock'))],capture_output=True,text=True)
assert result.returncode==1 and not result.stdout.strip(),'original node lock has an open owner or cannot be checked'
print(json.dumps(dict(passed=True,checked_s=time.time(),node=node,terminal=dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest()),node_lock_has_owner=False,grid_complete=True)))
'''

def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):d.update(block)
    return d.hexdigest()

def ref(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)

def remote(node,script,*args,payload=None):
    words=['nice','-n','15','python3','-B','-c',script,*map(str,args)]
    run=subprocess.run(monitor.SSH+['root@'+HOSTS[node],shlex.join(words)],
        input=json.dumps(payload) if payload is not None else None,text=True,capture_output=True,check=True)
    return json.loads(run.stdout)

def excluded(path):
    p=Path(path);parts={x.lower() for x in p.parts}
    if not p.is_absolute() or not p.is_relative_to(WORKSPACE) or '..' in p.parts:return 'outside_workspace'
    if p.suffix.lower() in ('.safetensors','.bin','.pt','.pth'):
        return 'weight_binary_payload_excluded_from_transfer'
    if parts & {'models','weights','retained_weights','weight_cache','weight-cache'} and p.suffix.lower()!='.json':
        return 'non_JSON_weight_directory_payload_excluded_from_transfer'
    if '__pycache__' in parts or p.suffix=='.pyc':return 'ephemeral_python_cache'
    return None

def plan(node,index,out,terminal_ref=None):
    # This is only a local CPU cache lock, not the remote experiment lease.
    # Concurrent node plans share expensive existing-model SHA results safely.
    cache_dir=ROOT/'staging';cache_dir.mkdir(parents=True,exist_ok=True)
    with (cache_dir/'full-mirror-local-sha-cache.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        cache_path=cache_dir/'full-mirror-local-sha-cache.json'
        cache=json.loads(cache_path.read_text()) if cache_path.exists() else {}
        result=_plan(node,index,out,terminal_ref,cache)
        save(cache_path,cache)
        return result

def _plan(node,index,out,terminal_ref,cache):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    snapshot=Path(index).read_bytes();snapshot_path=out/'report-hash-index.snapshot.json'
    if snapshot_path.exists():raise ValueError('plan output already exists; choose a fresh directory')
    snapshot_path.write_bytes(snapshot)
    declared=remote(node,DISCOVER,ROOT,node)
    expected={p:set(v) for p,v in declared['refs'].items()};origins=declared['origins']
    if terminal_ref:
        expected.setdefault(terminal_ref['path'],set()).add(terminal_ref['sha256'])
    # Other physical nodes are never copied through this node. Missing shared
    # context is probed on this host and preserved as unavailable if absent.
    outside=[]
    for row in json.loads(snapshot):
        p=row['path']
        own=p.startswith(str(ROOT/node)+'/')
        other=any(p.startswith(str(ROOT/x)+'/') for x in ('A','B','C') if x!=node)
        if own or (p in expected and not other):
            expected.setdefault(p,set()).add(row['sha256']);origins.setdefault(p,[]).append('report-hash-index')
        elif row['status']!='verified':
            outside.append(dict(path=p,sha256=row['sha256'],reason='not declared by this node; no cross-host fetch'))
    rows=[];probes=[];conflicts=[]
    for path,digests in sorted(expected.items()):
        if any(path.startswith(str(ROOT/x)+'/') for x in ('A','B','C') if x!=node):continue
        if len(digests)!=1:
            conflicts.append(dict(path=path,reason='declared_SHA_conflict',expected_sha256=sorted(digests)));continue
        digest=next(iter(digests));item=dict(path=path,sha256=digest)
        reason=excluded(path)
        if reason in ('outside_workspace','ephemeral_python_cache'):
            item.update(local_status='excluded',exclusion=reason);probes.append(item);continue
        p=monitor.safe_path(path)
        if p.exists():
            stamp=monitor.stat_key(p);known=cache.get(str(p))
            actual=known['sha256'] if known and known.get('stat')==stamp else sha(p)
            cache[str(p)]=dict(sha256=actual,stat=stamp)
            if actual!=digest:conflicts.append(dict(path=path,reason='local_SHA_conflict_existing_preserved',expected_sha256=digest,actual_sha256=actual))
            else:rows.append(dict(item,local_status='verified',local_size=p.stat().st_size,local_stat=stamp,payload_excluded_from_transfer=bool(reason)))
        elif reason:probes.append(dict(item,local_status='excluded',exclusion=reason))
        else:probes.append(dict(item,local_status='missing'))
    # Stat only; large remote bytes are neither read nor hashed during planning.
    checked=remote(node,STAT,payload=probes)
    for item in checked:
        parts={x.lower() for x in Path(item['path']).parts}
        if (item['local_status']=='missing' and item.get('remote_size',0) and item['remote_size']>8*1024**2
            and parts & {'models','weights','retained_weights','weight_cache','weight-cache'}):
            item.update(local_status='excluded',exclusion='weight_JSON_metadata_exceeds_8MiB_limit')
    rows.extend(checked)
    todo=[r for r in rows if r['local_status']=='missing' and r.get('remote_status')=='available' and r['remote_size']<=1024**3]
    oversized=[r for r in rows if r['local_status']=='missing' and r.get('remote_size',0) and r['remote_size']>1024**3]
    blocked=[r for r in rows if r.get('remote_status')=='unavailable' and r['local_status']!='excluded']
    groups=collections.defaultdict(lambda:dict(files=0,bytes=0))
    for r in todo:
        path=Path(r['path']);group=path.relative_to(ROOT/node).parts[0] if path.is_relative_to(ROOT/node) else 'shared_context'
        groups[group]['files']+=1;groups[group]['bytes']+=r['remote_size']
    result=dict(schema='slo14-full-evidence-mirror-plan-v1',node=node,created_s=time.time(),
        report_index=ref(snapshot_path),tool=ref(__file__),transfer_helper=ref(ROOT/'monitor.py'),
        remote_discovery_seeds=declared['seeds'],discovery_errors=declared['errors'],
        expected_SHA_source='Pinned report references plus finalized remote checkpoint and qualification declarations; planning does not rehash large remote files.',
        files=rows,conflicts=conflicts,unavailable=blocked,oversized=oversized,index_missing_outside_node_scope=outside,
        missing_transfer_files=len(todo),missing_transfer_bytes=sum(r['remote_size'] for r in todo),groups=dict(groups),
        excluded_count=sum(r['local_status']=='excluded' for r in rows),
        large_transfer_started=False,node_grid_completion_required_for_transfer=True,
        incomplete_plan_while_grid_active=True,
        exclusions='Weight binary payloads (safetensors/bin/pt/pth and non-JSON weight-directory contents) are never transferred. Declared JSON manifest/owner/hash metadata inside models/weights directories is allowed up to 8 MiB. Existing local same-SHA files, including weight payloads, are reused after SHA verification; missing payloads stay excluded.',
        installation='Missing local files only, verified SHA and exclusive hard-link install; local different SHA preserved; remote originals retained.')
    save(out/'plan.json',result)
    return result

def summary(value):
    return dict(node=value['node'],missing_files=value['missing_transfer_files'],
        estimated_GiB=round(value['missing_transfer_bytes']/1024**3,3),groups=value['groups'],
        excluded=value['excluded_count'],conflicts=len(value['conflicts']),unavailable=len(value['unavailable']),
        oversized=len(value['oversized']),discovery_errors=len(value['discovery_errors']))

def execute(node,index,out,terminal,bandwidth):
    out=Path(out)
    first=remote(node,GUARD,ROOT,node,terminal)
    value=plan(node,index,out,terminal_ref=first['terminal'])
    save(out/'initial-node-terminal.json',first)
    value['incomplete_plan_while_grid_active']=False
    value['node_terminal_guard']=ref(out/'initial-node-terminal.json')
    save(out/'plan.json',value)
    if value['conflicts'] or value['oversized'] or value['discovery_errors']:
        raise ValueError('mirror plan has conflicts/oversized files/discovery errors; existing files preserved, see plan.json')
    todo=[r for r in value['files'] if r['local_status']=='missing' and r.get('remote_status')=='available']
    worker=monitor.Mirror(out/'sha-cache.json',bandwidth=int(bandwidth*1024**2))
    worker.cache.update({r['path']:dict(sha256=r['sha256'],stat=r['local_stat'])
        for r in value['files'] if r['local_status']=='verified'})
    state=dict(schema='slo14-full-evidence-mirror-status-v1',node=node,pid=os.getpid(),started_s=time.time(),
        complete=False,plan=ref(out/'plan.json'),terminal=first['terminal'],downloaded_files=0,downloaded_bytes=0)
    save(out/'status.json',state)
    try:
        for offset in range(0,len(todo),32):
            assert ref(ROOT/'monitor.py')==value['transfer_helper'],'transfer helper changed after planning'
            proof=remote(node,GUARD,ROOT,node,terminal)
            assert proof['terminal']==first['terminal'],'node terminal changed during mirror'
            batch=[dict(path=r['path'],sha256=r['sha256']) for r in todo[offset:offset+32]]
            worker.fetch(node,batch)
            save(out/'sha-cache.json',worker.cache)
            state.update(downloaded_files=worker.downloaded,downloaded_bytes=worker.bytes,errors=worker.errors)
            save(out/'status.json',state)
            if worker.errors:raise ValueError('remote/local SHA or transfer failure; retained originals and already verified files preserved')
        checked=[]
        for item in value['files']:
            if item['local_status']=='excluded' or item.get('remote_status')=='unavailable':continue
            reference=dict(path=item['path'],sha256=item['sha256'])
            if not worker.present(reference):raise ValueError('expected file absent after mirror: '+item['path'])
            checked.append(reference)
        save(out/'verified-files.json',checked)
        state.update(complete=True,verified_files=ref(out/'verified-files.json'),excluded=value['excluded_count'],
            unavailable=value['unavailable'],scope_complete=not value['unavailable'],weight_reconstruction_complete=False,
            note='Expected non-weight file hashes verified. Excluded weight payloads and unavailable references remain explicit; no remote deletion or GPU action.')
    except BaseException as exc:
        state.update(error=repr(exc));raise
    finally:
        state['finished_s']=time.time();save(out/'status.json',state)
    print(json.dumps(state))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','run'])
    parser.add_argument('--node',choices=['A','C'],required=True)
    parser.add_argument('--index',type=Path,default=ROOT/'reports/current/raw-hash-index.json')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--terminal',type=Path)
    parser.add_argument('--bandwidth-mib',type=float,default=16.)
    args=parser.parse_args()
    if not 1<=args.bandwidth_mib<=64:parser.error('bandwidth must be 1..64 MiB/s')
    if args.action=='plan':
        value=plan(args.node,args.index,args.out);print(json.dumps(dict(plan=ref(args.out/'plan.json'),**summary(value))))
    else:
        if args.terminal is None:parser.error('run requires the final per-node terminal status path')
        execute(args.node,args.index,args.out,args.terminal,args.bandwidth_mib)

if __name__=='__main__':main()
