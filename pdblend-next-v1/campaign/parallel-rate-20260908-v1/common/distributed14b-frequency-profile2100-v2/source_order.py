"""Read-only source identity prerequisite for metadata-order reconstruction."""
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(path=None):
    path=Path(path) if path else ROOT/'source-order-contract.json'
    return json.loads(path.read_text()),sha(path)


async def capture(gate,live,contract_path=None):
    """Uses the caller's existing async Docker command wrapper, no engine write.

    live must first pass the caller's original complete model/container/source
    gate. The helper process reads bytes; it imports no vLLM or CUDA modules.
    """
    expected,digest=contract(contract_path);paths=list(expected['files'])
    script='import hashlib,json; paths='+repr(paths)+'; print(json.dumps({p:hashlib.sha256(open(p,"rb").read()).hexdigest() for p in paths}))'
    records={}
    for item in live['instances']:
        source=item['provenance'];instance=source['instance_id'];container=item['container']
        logger=expected['engine_logger_source']
        require(source['source_files_at_import'].get(logger)==expected['engine_logger_sha256'],
            'actual imported engine logger source differs from reviewed ordering contract')
        values=json.loads(await gate.command('docker','exec',container['Id'],'python3','-c',script,timeout=12))
        require(values=={p:v['sha256'] for p,v in expected['files'].items()},'actual scheduler/default/sampling source differs')
        records[instance]=dict(instance_id=instance,container_id=container['Id'],image_id=container['Image'],
            container_started_at=container['State']['StartedAt'],engine_pid=source['pid'],
            engine_logger_sha256=expected['engine_logger_sha256'],files=values,
            contract_sha256=digest,observed_s=time.time(),read_only=True,
            loaded_engine_import_sha_verified=True,scheduler_semantics='reviewed immutable source plus live filesystem hashes; not direct KV state')
    return records


def validate_pair(before,after,instance_id,*,measurement_start_s,measurement_end_s,contract_path=None):
    expected,digest=contract(contract_path);a,b=before[instance_id],after[instance_id]
    for row in (a,b):
        require(row.get('instance_id')==instance_id and row.get('read_only') is True
            and row.get('loaded_engine_import_sha_verified') is True
            and row.get('contract_sha256')==digest
            and row.get('engine_logger_sha256')==expected['engine_logger_sha256']
            and row.get('files')=={p:v['sha256'] for p,v in expected['files'].items()},'source-order proof missing or inconsistent')
    keys=('instance_id','container_id','image_id','container_started_at','engine_pid','engine_logger_sha256','files','contract_sha256')
    require(all(a.get(k)==b.get(k) and a.get(k) is not None for k in keys),'container/process/source changed during observations')
    require(a['observed_s']<=measurement_start_s<measurement_end_s<=b['observed_s'],'source before/after does not enclose full work')
    return dict(verified=True,contract_sha256=digest,instance_id=instance_id,
        before_observed_s=a['observed_s'],after_observed_s=b['observed_s'],
        source_order_inferred_from_verified_code=True,physical_kv_directly_observed=False)
