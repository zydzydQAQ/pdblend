"""Read-only live identity checks for model-specific evaluation-v3 plans."""
import asyncio
import json
from pathlib import Path, PurePosixPath
import re


def _serving_sources(group):
    result={}
    for name,digest in group.items():
        path=PurePosixPath(name)
        if path.parent.parts[-3:]==('src','ecopadg','serving') and path.suffix=='.py':
            if path.name in result:
                raise ValueError('ambiguous frozen serving source path')
            result[path.name]=digest
    if not result or 'engine.py' not in result:
        raise ValueError('frozen serving source evidence missing')
    return result


def _host_model_path(model_path,mounts):
    path=PurePosixPath(model_path)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('live model path must be absolute')
    matches=[]
    for mount in mounts:
        destination=PurePosixPath(mount.get('Destination',''))
        try: relative=path.relative_to(destination)
        except ValueError: continue
        if mount.get('Type')=='bind' and mount.get('Source'):
            matches.append((len(destination.parts),Path(mount['Source']).joinpath(*relative.parts).resolve()))
    if not matches:
        raise ValueError('live model is not bound to frozen host model files')
    return max(matches,key=lambda item:item[0])[1]


def verify_live_record(raw,inspection,instance,model_spec,container_name):
    """Pure validation, so wrong model/image/source cases need no Docker/GPU."""
    if (not re.fullmatch(r'(pdb-v2-|pdb-next-)[A-Za-z0-9_-]+',container_name)
            or inspection.get('Name','').lstrip('/')!=container_name
            or inspection.get('State',{}).get('Running') is not True):
        raise ValueError('unapproved or non-running experiment container')
    image=model_spec['identities']['engine_image']
    if inspection.get('Image')!=image or not re.fullmatch(r'sha256:[0-9a-f]{64}',image):
        raise ValueError('live immutable engine image differs from plan')
    sources=_serving_sources(model_spec['groups']['source'])
    if set(sources)!={path.name for path in Path(__file__).parent.glob('*.py')}:
        raise ValueError('plan must freeze every serving source module')
    actual=_serving_sources(raw.get('source_files_at_import',{}))
    if actual!=sources:
        raise ValueError('live imported serving source differs from plan')
    if (raw.get('instance_id')!=instance['id'] or raw.get('tp')!=instance['tp']
            or raw.get('engine_version')!='0.9.2' or raw.get('dtype')!='bfloat16'
            or raw.get('max_model_len')!=8192
            or raw.get('cuda_visible_devices')!=','.join(map(str,instance['gpus']))
            or raw.get('kv_capacity_fixture') is not False
            or raw.get('num_gpu_blocks_override') is not None):
        raise ValueError('live engine execution configuration differs from plan')
    model_files=model_spec['groups']['model']
    roots={Path(p).resolve().parent for p in model_files if Path(p).name=='config.json'}
    if len(roots)!=1:
        raise ValueError('frozen model must have one configuration root')
    model_root=next(iter(roots))
    mounted_model=_host_model_path(raw.get('model',''),inspection.get('Mounts',[]))
    if mounted_model!=model_root:
        raise ValueError('live model mount differs from frozen model')
    index=model_root/'model.safetensors.index.json'
    try: weights=set(json.loads(index.read_text())['weight_map'].values())
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('frozen model shard index unavailable') from exc
    if not weights or any(Path(w).is_absolute() or '..' in Path(w).parts for w in weights):
        raise ValueError('invalid model weight shard identities')
    required={str(model_root/name) for name in weights|
        {'config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json'}}
    if not required<=set(model_files):
        raise ValueError('frozen model weights or tokenizer incomplete')
    return dict(instance_id=instance['id'],container_name=container_name,image_id=image,
                model_root=str(model_root),raw=raw)


async def _inspect_container(name):
    process=await asyncio.create_subprocess_exec('docker','inspect',name,
        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try: stdout,stderr=await asyncio.wait_for(process.communicate(),10)
    except BaseException:
        if process.returncode is None:
            process.kill(); await process.wait()
        raise
    if process.returncode:
        raise ValueError('container inspection failed: '+stderr.decode()[:300])
    data=json.loads(stdout)
    if not isinstance(data,list) or len(data)!=1:
        raise ValueError('container inspection is not a single identity')
    return data[0]


async def verify_live_engines_v3(backend,plan_path,model,config):
    plan=json.loads(Path(plan_path).read_text())
    spec=plan['models'][model]
    records=[]; owned=[]
    for instance_id,instance in tuple(backend.instances.items()):
        if instance.get('id')!=instance_id:
            raise ValueError('backend instance id differs from declared instance')
        name=instance.get('container_name') or config.get('container_names',{}).get(instance_id)
        if not isinstance(name,str) or not re.fullmatch(r'(pdb-v2-|pdb-next-)[A-Za-z0-9_-]+',name):
            raise ValueError('explicit scoped container_name required for every live instance')
        gpus=instance.get('gpus',[])
        if len(gpus)!=instance.get('tp') or any(type(g) is not int or not 0<=g<8 for g in gpus):
            raise ValueError('invalid instance GPU assignment')
        owned.extend(gpus)
        raw=await backend.json(instance_id,'/provenance')
        inspection=await _inspect_container(name)
        records.append(verify_live_record(raw,inspection,instance,spec,name))
    if not records or len(owned)!=len(set(owned)):
        raise ValueError('empty or overlapping live engine deployment')
    return records
