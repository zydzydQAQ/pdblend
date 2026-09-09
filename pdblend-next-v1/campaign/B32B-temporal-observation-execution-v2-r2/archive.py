"""Stream complete original runtime prefixes; never edit a serving file."""
import hashlib,json,os,time
from pathlib import Path
import common as c
MAX_FILE_BYTES=1024**3
MAX_TOTAL_BYTES=2*1024**3
CHUNK=4*1024**2

def paths(binding):
    result={}
    for instance in binding['instances']:
        cfg=c.read(instance['engine_config']);base=Path(cfg['runtime_dir'])/(instance['id']+'.control.json')
        c.require(instance['tp']==2,'original B TP2 runtime namespace required')
        result[instance['id']]={'control':base,'events':base.with_suffix('.events.jsonl'),
            'kv_rank_0':Path(str(base)+'.kv.0.jsonl'),'kv_rank_1':Path(str(base)+'.kv.1.jsonl')}
    c.require(len(result)==4,'four original runtime namespaces required')
    return result

def guard(end_s,mono_end):
    c.require(time.time()<end_s and time.monotonic()<mono_end,'runtime archive deadline exhausted')

def prefix_digest(path,size,end_s,mono_end):
    h=hashlib.sha256();left=size
    with Path(path).open('rb') as f:
        while left:
            guard(end_s,mono_end);block=f.read(min(CHUNK,left));c.require(block,'runtime prefix was truncated');h.update(block);left-=len(block)
    return h.hexdigest()

def copy_exact(source,dest,end_s,mono_end):
    c.require(source.is_file() and not source.is_symlink(),'missing/symlink original runtime file')
    before=source.stat();c.require(before.st_size<=MAX_FILE_BYTES,'full runtime file exceeds archive cap; never trim')
    h=hashlib.sha256();remaining=before.st_size
    with source.open('rb') as inp,dest.open('xb') as out:
        c.require(os.fstat(inp.fileno()).st_ino==before.st_ino,'runtime file replaced before copy')
        while remaining:
            guard(end_s,mono_end);block=inp.read(min(CHUNK,remaining));c.require(block,'runtime file truncated during copy')
            h.update(block);out.write(block);remaining-=len(block)
        out.flush();os.fsync(out.fileno())
    after=source.stat()
    c.require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==
              (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'runtime source changed while snapshotting')
    c.require(c.sha(dest)==h.hexdigest(),'archived runtime bytes differ');guard(end_s,mono_end)
    if source.name.endswith('.jsonl'):
        with dest.open('rb') as f:
            if before.st_size:f.seek(-1,2);c.require(f.read()==b'\n','original owner event tail incomplete')
    return dict(source=str(source),archive=str(dest),size=before.st_size,sha256=h.hexdigest(),
        source_device=before.st_dev,source_inode=before.st_ino,source_mtime_ns=before.st_mtime_ns)

def verify_stopped(inspections,binding):
    expected={i['container']['id'] for i in binding['instances']}
    c.require(len(inspections)==4 and {x['Id'] for x in inspections}==expected,'original container set differs before archive')
    c.require(all(x['State'].get('Running') is False and x['State'].get('Pid')==0 for x in inspections),
        'all four original engine processes must actually stop before final archive')

def capture(binding,out,end_s,*,stopped=None,initial=None):
    out=Path(out);c.require(not out.exists(),'new full runtime snapshot directory required')
    mono_end=time.monotonic()+max(0.,end_s-time.time());guard(end_s,mono_end)
    if stopped is not None:verify_stopped(stopped,binding)
    source=paths(binding);allpaths=[p for group in source.values() for p in group.values()]
    c.require(sum(p.stat().st_size for p in allpaths)<=MAX_TOTAL_BYTES,'full runtime set exceeds cap; never trim')
    out.mkdir();report=dict(schema=1,started_s=time.time(),complete=False,original_processes_stopped=stopped is not None,instances={})
    c.write(out/'archive.json',report)
    try:
        for iid,group in source.items():
            report['instances'][iid]={}
            for kind,p in group.items():report['instances'][iid][kind]=copy_exact(p,out/(iid+'.'+kind),end_s,mono_end)
            if initial is not None:
                prefix_hashes={}
                for kind in ('events','kv_rank_0','kv_rank_1'):
                    old=initial['instances'][iid][kind];current=report['instances'][iid][kind]
                    c.require(current['source']==old['source'] and current['size']>=old['size'],'original event prefix path/size changed')
                    c.require(prefix_digest(Path(current['source']),old['size'],end_s,mono_end)==old['sha256'],
                        'original event prefix changed between pre-control and stopped snapshots')
                    prefix_hashes[kind]=old['sha256']
                report['instances'][iid]['initial_prefix_sha256']=prefix_hashes
            c.write(out/'archive.json',report)
        report.update(complete=True,finished_s=time.time());c.write(out/'archive.json',report);return report
    except BaseException as exc:
        report.update(error=repr(exc),finished_s=time.time());c.write(out/'archive.json',report);raise

def verify_prefix_after_restore(binding,snapshot,native,end_s):
    c.require(snapshot.get('complete') is True and snapshot.get('original_processes_stopped') is True,'full stopped archive missing')
    mono_end=time.monotonic()+max(0.,end_s-time.time());result={}
    for iid,group in paths(binding).items():
        old=snapshot['instances'][iid];event=group['events'];current=event.stat()
        c.require(str(event)==old['events']['source'] and current.st_size>=old['events']['size'],'restored event prefix truncated/path changed')
        digest=prefix_digest(event,old['events']['size'],end_s,mono_end)
        c.require(digest==old['events']['sha256'],'restored event prefix content changed')
        for kind in ('control','events','kv_rank_0','kv_rank_1'):
            c.require(c.sha(old[kind]['archive'])==old[kind]['sha256'],'immutable stopped runtime archive changed')
        kv_prefixes={}
        for kind in ('kv_rank_0','kv_rank_1'):
            p=group[kind];size=p.stat().st_size
            c.require(str(p)==old[kind]['source'] and size>=old[kind]['size'],'restored KV event prefix truncated/path changed')
            h=prefix_digest(p,old[kind]['size'],end_s,mono_end)
            c.require(h==old[kind]['sha256'],'restored KV event prefix content changed')
            kv_prefixes[kind]=dict(prefix_sha256=h,old_bytes=old[kind]['size'],current_bytes=size,appended_bytes=size-old[kind]['size'])
        control=c.read(group['control']);after=native['instances'][iid]['after']
        c.require(all(control.get(k)==after.get(k) for k in ('generation','role','mode','admit_prefill','admit_decode')),
            'restored control file differs from final actual native ACK')
        result[iid]=dict(kv_rank_prefixes=kv_prefixes,event_prefix_sha256=digest,old_event_bytes=old['events']['size'],current_event_bytes=current.st_size,
            appended_event_bytes=current.st_size-old['events']['size'],stopped_control_sha256=old['control']['sha256'],
            restored_control_sha256=c.sha(group['control']),restored_control=control,final_actual_generation=after['generation'])
    guard(end_s,mono_end)
    return dict(complete=True,checked_s=time.time(),instances=result,
        interpretation='events append; old control reset is retained in the immutable stopped archive')
