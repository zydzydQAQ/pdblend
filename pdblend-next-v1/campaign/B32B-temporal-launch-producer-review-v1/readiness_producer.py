"""Pure-read addition to the frozen diagnostic main90 gate; no hardware code."""
import hashlib
import json
import math
from pathlib import Path

PROTOCOL='per-dataset-slo-five-system-fixed-window-v1'
DEADLINE=1788872770.0400891
NODE='iZwz9i5bte3xkpmcoes3t2Z'
SOURCE_SHA='4b9494c6b0a38cb9d44dbc490530d88e9a0eec76b44854f6e40db0328bbfe5ed'
SYSTEMS=('mixed','dynamollm','distserve')


def require(ok,why):
    if not ok:raise RuntimeError(why)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def pid_live(pid):
    require(type(pid) is int and pid>0,'actual positive PID required')
    try:
        root=Path('/proc')/str(pid);stat=(root/'stat').read_text()
        return bool((root/'cmdline').read_bytes()) and stat[stat.rfind(')')+2:].split()[0]!='Z'
    except FileNotFoundError:return False


def verify_main_producers(proof_path,*,live=pid_live):
    """Require exactly one terminal actual producer per original main CP.

    The caller MUST also run frozen common.main_gate: that gate owns full raw,
    native, actual container and parent-supervisor checks. This only adds the
    missing CP -> completed[] -> binding/source/time relation. No release or
    modified receipt is written. A skipped CP is never execution.
    """
    files={}
    def read(path,expected=None):
        path=str(Path(path).resolve());data=Path(path).read_bytes();digest=hashlib.sha256(data).hexdigest()
        require(expected is None or digest==expected,'producer input SHA differs: '+path)
        require(path not in files or files[path]==digest,'producer input changed during read: '+path)
        files[path]=digest;return json.loads(data)
    def exited(pid,label):
        require(type(pid) is int and pid>0 and not live(pid),label+' remains live or has no actual PID')
    proof=read(proof_path)
    require(proof.get('model')=='32b' and proof.get('hostname')==NODE and proof.get('protocol_id')==PROTOCOL
        and proof.get('deadline_s')==DEADLINE,'wrong producer proof model/protocol/deadline')
    require(proof.get('source_sha256')==SOURCE_SHA,'producer proof source is not the frozen original B declaration')
    source_path=str(Path(proof['source_manifest']).resolve());source=read(source_path,SOURCE_SHA)
    require(source.get('model')=='32b' and source.get('protocol_id')==PROTOCOL,'wrong original source domain')
    records=[]
    for system in SYSTEMS:
        group=proof['baseline_systems'][system]
        require(group.get('complete') is True and group.get('completed')==30,'three actual main30 groups required')
        bp=str(Path(group['binding']).resolve());binding=read(bp);binding_sha=files[bp]
        require(binding.get('model')=='32b' and binding.get('hostname')==NODE and binding.get('system')==system
            and binding.get('protocol_id')==PROTOCOL and binding.get('deadline_s')==DEADLINE
            and binding['files'].get(source_path)==SOURCE_SHA,'actual producer binding/source differs')
        output=Path(binding['output']);rows=[r for r in source['cells'] if r['system']==system and r['phase']=='main']
        declared={r['cell_id']:r for r in rows};headers={r['cell_id']:r for r in group['records']}
        require(len(rows)==len(declared)==len(headers)==len(group['records'])==30 and set(headers)==set(declared),'main CP domain differs')
        invocations=[]
        for path in sorted((output/'invocations').glob('*.json')):
            inv=read(path)
            if inv.get('phase')!='main' or inv.get('system')!=system:continue
            require(inv.get('complete') is True and not inv.get('error') and finite(inv.get('started_s'))
                and finite(inv.get('finished_s')) and inv['started_s']<=inv['finished_s']<=DEADLINE,
                'related main invocation is not cleanly terminal')
            exited(inv.get('pid'),'main runner') # Includes a live skip-only invocation.
            require(isinstance(inv.get('completed'),list) and len(inv['completed'])==len(set(inv['completed']))
                and set(inv['completed'])<=set(declared),'invocation completed IDs are not the exact original main domain')
            invocations.append((str(path.resolve()),inv))
        require(invocations,'actual main producer invocation missing')
        for cid,row in declared.items():
            require(row.get('model')=='32b' and row.get('slo_scale')==1.,'wrong original main row')
            cp_path=output/'checkpoints'/(cid+'.json');header=headers[cid]
            require(header['checkpoint']==str(cp_path),'foreign producer checkpoint path')
            cp=read(cp_path,header['checkpoint_sha256'])
            require(cp['row']==row and cp.get('measurement_valid') is True,'producer CP changed or invalid')
            receipt_path=output/'operations'/cid/'receipt.json'
            require(cp['receipt']==str(receipt_path) and cp['receipt_sha256']==header['receipt_sha256'],'foreign producer receipt')
            receipt=read(receipt_path,cp['receipt_sha256'])
            require(receipt.get('measurement_valid') is True and receipt.get('child_stopped') is True,'producer measurement/child not terminal')
            exited(receipt.get('child_pid'),'main HTTP child')
            candidates=[(p,v) for p,v in invocations if cid in v['completed']]
            require(len(candidates)==1,'CP must have exactly one actual completed-list producer; skip is not execution')
            ip,inv=candidates[0]
            require(inv.get('binding_sha256')==binding_sha and inv.get('manifest_sha256')==SOURCE_SHA
                and inv.get('protocol_id')==PROTOCOL,'CP producer binding/source SHA differs')
            require(row['dataset'] in inv.get('selected_datasets',('alpaca','sharegpt','longbench')),'CP outside producer dataset selection')
            require(all(finite(v) for v in (receipt.get('started_s'),receipt.get('finished_s'),cp.get('completed_s')))
                and inv['started_s']<=receipt['started_s']<=receipt['finished_s']<=cp['completed_s']<=inv['finished_s']<=DEADLINE,
                'CP/receipt measurement lies outside actual producer interval/deadline')
            records.append(dict(cell_id=cid,system=system,checkpoint=str(cp_path),checkpoint_sha256=files[str(cp_path.resolve())],
                receipt_sha256=cp['receipt_sha256'],binding=bp,binding_sha256=binding_sha,invocation=ip,
                invocation_sha256=files[ip],execution_manifest_sha256=SOURCE_SHA,producer_pid=inv['pid']))
    require(len(records)==90,'exact90 actual producers required')
    for path,h in files.items():require(sha(path)==h,'producer evidence changed after verification: '+path)
    return dict(records=records,files=files,actual_main_producers=90,global_scale_released=False,
        scope='additional producer/source/time/process check; frozen full main_gate remains mandatory')
